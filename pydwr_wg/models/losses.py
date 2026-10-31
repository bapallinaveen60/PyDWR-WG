"""
Composite Loss Formulations for PTA-InNet.
Integrates Focal-Jaccard segmentation, Huber reconstruction, spatial gradient penalties,
and differential phase (PhiDP) path-integral physical constraints.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    """
    Focal Loss to combat severe class imbalance between localized ground clutter spikes
    and extensive precipitation echoes.
    """

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, reduction: str = "mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        bce = F.binary_cross_entropy(pred, target, reduction="none")
        p_t = pred * target + (1 - pred) * (1 - target)
        alpha_t = self.alpha * target + (1 - self.alpha) * (1 - target)
        focal_weight = alpha_t * (1.0 - p_t) ** self.gamma
        loss = focal_weight * bce

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


class JaccardLoss(nn.Module):
    """
    Jaccard / Intersection-over-Union (IoU) Loss for binary clutter mask segmentation.
    """

    def __init__(self, smooth: float = 1e-5):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        intersection = torch.sum(pred * target, dim=(1, 2, 3))
        union = torch.sum(pred + target, dim=(1, 2, 3)) - intersection
        iou = (intersection + self.smooth) / (union + self.smooth)
        return (1.0 - iou).mean()


class SpatialGradientLoss(nn.Module):
    """
    L1 spatial gradient loss to suppress boundary discontinuities along the edges
    of inpainted terrain sectors.
    """

    def __init__(self):
        super().__init__()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # Radial range gradient: dy (dim -2)
        dy_pred = torch.abs(pred[:, :, 1:, :] - pred[:, :, :-1, :])
        dy_target = torch.abs(target[:, :, 1:, :] - target[:, :, :-1, :])
        loss_y = torch.mean(torch.abs(dy_pred - dy_target))

        # Azimuth circular gradient: dx (dim -1)
        # Wrap around for circular boundary
        dx_pred = torch.abs(pred[:, :, :, 1:] - pred[:, :, :, :-1])
        dx_target = torch.abs(target[:, :, :, 1:] - target[:, :, :, :-1])
        loss_x = torch.mean(torch.abs(dx_pred - dx_target))

        return loss_y + loss_x


class PhysicalPhaseLoss(nn.Module):
    """
    Differential phase path-integral physical constraint.
    Ensures that for blocked sector [r1, r2], the integrated specific differential phase
    matches the downstream measured PhiDP accumulation:

    L_phys = (1 / Nrays) * sum_theta max(0, Delta_PhiDP - 2 * sum(Kdp * dr) - epsilon)
    """

    def __init__(self, range_step_km: float = 0.1, epsilon: float = 0.5):
        super().__init__()
        self.dr = range_step_km
        self.epsilon = epsilon

    def forward(
        self,
        zh_clean_pred: torch.Tensor,
        zdr_corrupted: torch.Tensor,
        phidp_observed: torch.Tensor,
        blockage_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Calculates physical propagation phase consistency loss.

        Parameters
        ----------
        zh_clean_pred : torch.Tensor
            Predicted reflectivity [B, 1, Nr, Na] in dBZ.
        zdr_corrupted : torch.Tensor
            Differential reflectivity [B, 1, Nr, Na] in dB.
        phidp_observed : torch.Tensor
            Observed PhiDP [B, 1, Nr, Na] in degrees.
        blockage_mask : torch.Tensor
            Binary indicator of terrain blockage [B, 1, Nr, Na].
        """
        # Neural/empirical mapping psi(ZH, ZDR) -> expected KDP [deg/km]
        # In warm monsoonal rain: KDP approx 10^(0.08 * ZH - 0.28 * ZDR - 2.8)
        kdp_inferred = 10.0 ** (0.08 * torch.clamp(zh_clean_pred, 10.0, 55.0)
                                - 0.28 * torch.clamp(zdr_corrupted, 0.2, 3.0) - 2.8)
        kdp_inferred = torch.clamp(kdp_inferred, 0.0, 8.0)

        # Integrated two-way phase across range: 2 * sum(Kdp * dr)
        phase_accum_pred = torch.cumsum(2.0 * kdp_inferred * self.dr, dim=2)

        # Downstream constraint: compare observed PhiDP vs accumulated phase
        # Focus on gates directly downstream of blocked mountain ridges
        downstream_mask = (blockage_mask > 0.3)
        diff = torch.abs(phase_accum_pred - phidp_observed) - self.epsilon
        penalty = F.relu(diff)

        if downstream_mask.sum() > 0:
            loss = torch.sum(penalty * downstream_mask) / (downstream_mask.sum() + 1e-6)
        else:
            loss = penalty.mean()

        return loss


class TotalPTALoss(nn.Module):
    """
    Composite loss function balancing semantic segmentation, Huber reconstruction,
    spatial smoothness, and physical phase constraints:

    L_total = lambda1 * L_seg + lambda2 * L_rec + lambda3 * L_grad + lambda4 * L_phys
    """

    def __init__(
        self,
        lambda_seg: float = 1.0,
        lambda_rec: float = 2.5,
        lambda_grad: float = 0.8,
        lambda_phys: float = 1.5,
        range_step_km: float = 0.1,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
    ):
        super().__init__()
        self.lambda_seg = lambda_seg
        self.lambda_rec = lambda_rec
        self.lambda_grad = lambda_grad
        self.lambda_phys = lambda_phys

        self.focal = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
        self.jaccard = JaccardLoss()
        self.huber = nn.HuberLoss(delta=1.0)
        self.grad = SpatialGradientLoss()
        self.phys = PhysicalPhaseLoss(range_step_km=range_step_km)

    def forward(
        self,
        predictions: dict,
        target_clutter_mask: torch.Tensor,
        target_zh_clean: torch.Tensor,
        zdr_corrupted: torch.Tensor,
        phidp_observed: torch.Tensor,
        bbf: torch.Tensor,
    ) -> dict:
        """
        Computes composite loss and individual components.
        """
        pred_clutter = predictions["clutter_mask"]
        pred_zh = predictions["zh_clean"]

        # 1. Semantic segmentation loss
        l_focal = self.focal(pred_clutter, target_clutter_mask)
        l_jacc = self.jaccard(pred_clutter, target_clutter_mask)
        l_seg = l_focal + l_jacc

        # 2. Huber reconstruction loss across valid meteorological gates
        # Valid gates: targets with authentic rain (ZH > 0 dBZ)
        valid_mask = (target_zh_clean > 0.0).float()
        l_rec = self.huber(pred_zh * valid_mask, target_zh_clean * valid_mask)

        # 3. Spatial gradient penalty
        l_grad = self.grad(pred_zh, target_zh_clean)

        # 4. Differential phase physical propagation loss
        l_phys = self.phys(
            zh_clean_pred=pred_zh,
            zdr_corrupted=zdr_corrupted,
            phidp_observed=phidp_observed,
            blockage_mask=bbf,
        )

        total_loss = (
            self.lambda_seg * l_seg
            + self.lambda_rec * l_rec
            + self.lambda_grad * l_grad
            + self.lambda_phys * l_phys
        )

        return {
            "total_loss": total_loss,
            "loss_seg": l_seg,
            "loss_rec": l_rec,
            "loss_grad": l_grad,
            "loss_phys": l_phys,
        }
