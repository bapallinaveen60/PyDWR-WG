import unittest
import torch
from pydwr_wg.models.losses import (
    FocalLoss,
    JaccardLoss,
    SpatialGradientLoss,
    PhysicalPhaseLoss,
    TotalPTALoss,
)

class TestLosses(unittest.TestCase):
    def test_focal_loss(self):
        fl = FocalLoss()
        pred = torch.tensor([[[[0.9, 0.1]]]])
        target = torch.tensor([[[[1.0, 0.0]]]])
        loss = fl(pred, target)
        self.assertTrue(loss.item() >= 0.0)

    def test_jaccard_loss(self):
        jl = JaccardLoss()
        pred = torch.tensor([[[[1.0, 0.0], [0.0, 1.0]]]])
        target = torch.tensor([[[[1.0, 0.0], [0.0, 1.0]]]])
        loss = jl(pred, target)
        self.assertAlmostEqual(loss.item(), 0.0, places=4)

    def test_total_pta_loss(self):
        loss_module = TotalPTALoss()
        b, nr, na = 2, 32, 16

        preds = {
            "clutter_mask": torch.sigmoid(torch.randn(b, 1, nr, na)),
            "zh_clean": torch.randn(b, 1, nr, na) * 20.0 + 20.0,
        }
        target_mask = torch.randint(0, 2, (b, 1, nr, na)).float()
        target_zh = torch.randn(b, 1, nr, na) * 20.0 + 20.0
        zdr = torch.randn(b, 1, nr, na) * 1.0 + 0.5
        phidp = torch.cumsum(torch.abs(torch.randn(b, 1, nr, na)), dim=2)
        bbf = torch.rand(b, 1, nr, na)

        loss_dict = loss_module(
            predictions=preds,
            target_clutter_mask=target_mask,
            target_zh_clean=target_zh,
            zdr_corrupted=zdr,
            phidp_observed=phidp,
            bbf=bbf,
        )

        self.assertIn("total_loss", loss_dict)
        self.assertIn("loss_seg", loss_dict)
        self.assertIn("loss_rec", loss_dict)
        self.assertIn("loss_grad", loss_dict)
        self.assertIn("loss_phys", loss_dict)
        self.assertTrue(loss_dict["total_loss"].item() > 0.0)

if __name__ == "__main__":
    unittest.main()
