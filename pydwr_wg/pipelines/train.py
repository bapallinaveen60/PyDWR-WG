"""
Multi-GPU Training Pipeline for PTA-InNet.
Integrates AdamW optimization, Cosine Annealing learning rate schedule, and composite physical loss.
"""

import os
import time
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from typing import Dict, Any, Optional
from pydwr_wg.models.pta_innet import PTAInNet
from pydwr_wg.models.losses import TotalPTALoss
from pydwr_wg.simulation.scip_pipeline import SCIPPipeline
from pydwr_wg.io.radar_loader import RadarDataLoader
from pydwr_wg.io.dem_loader import DEMLoader
from pydwr_wg.preprocessing.ray_tracing import compute_partial_beam_blockage

class SyntheticSCIPDataset(Dataset):
    """
    On-the-fly semi-synthetic clutter dataset using SCIP pipeline for scalable training.
    """

    def __init__(self, n_samples: int = 100, radar_site: str = "mumbai"):
        self.n_samples = n_samples
        self.radar_loader = RadarDataLoader()
        self.dem_loader = DEMLoader(radar_lat=18.9180, radar_lon=72.8250)
        self.dem_polar = self.dem_loader.generate_western_ghats_profile()
        self.bbf_polar, self.h_beam_agl = compute_partial_beam_blockage(
            self.dem_polar, elevation_deg=0.5, radar_altitude_m=60.0
        )
        self.scip = SCIPPipeline()

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        # Generate pristine marine sweep
        pristine = self.radar_loader.generate_synthetic_marine_sweep()

        # Inject clutter and blockage
        x_tensor, m_clut, zh_clean, r_clean = self.scip.inject_clutter_and_blockage(
            pristine_sweep=pristine,
            dem_polar=self.dem_polar,
            bbf_polar=self.bbf_polar,
            h_beam_agl=self.h_beam_agl,
        )

        # Extract channels needed for physical loss
        zdr_norm = x_tensor[1:2]
        phidp_norm = x_tensor[3:4]
        bbf = x_tensor[8:9]

        return {
            "x": torch.from_numpy(x_tensor),
            "target_clutter_mask": torch.from_numpy(m_clut).unsqueeze(0),
            "target_zh_clean": torch.from_numpy(zh_clean).unsqueeze(0),
            "target_rain_rate": torch.from_numpy(r_clean).unsqueeze(0),
            "zdr_norm": torch.from_numpy(zdr_norm),
            "phidp_norm": torch.from_numpy(phidp_norm),
            "bbf": torch.from_numpy(bbf),
        }


def train_pta_innet(
    epochs: int = 5,
    batch_size: int = 2,
    learning_rate: float = 2.0e-4,
    n_samples: int = 10,
    device: Optional[str] = None,
    save_checkpoint_path: str = "checkpoints/pta_innet_best.pt",
) -> Dict[str, Any]:
    """
    Executes training loop with AdamW, cosine scheduler, and composite loss.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Initializing PTA-InNet training on device: {device}", flush=True)

    os.makedirs(os.path.dirname(save_checkpoint_path), exist_ok=True)

    # Initialize Model & Loss
    model = PTAInNet(in_channels=10).to(device)
    criterion = TotalPTALoss(
        lambda_seg=1.0,
        lambda_rec=2.5,
        lambda_grad=0.8,
        lambda_phys=1.5,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1.0e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1.0e-6)

    # DataLoaders
    train_dataset = SyntheticSCIPDataset(n_samples=n_samples)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    history = {"train_loss": [], "loss_seg": [], "loss_rec": [], "loss_phys": []}
    best_loss = float("inf")

    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        start_time = time.time()

        for batch_idx, batch in enumerate(train_loader):
            x = batch["x"].to(device)
            target_mask = batch["target_clutter_mask"].to(device)
            target_zh = batch["target_zh_clean"].to(device)
            zdr_norm = batch["zdr_norm"].to(device)
            phidp_norm = batch["phidp_norm"].to(device)
            bbf = batch["bbf"].to(device)

            optimizer.zero_grad()
            preds = model(x)

            loss_dict = criterion(
                predictions=preds,
                target_clutter_mask=target_mask,
                target_zh_clean=target_zh,
                zdr_corrupted=zdr_norm * 12.0 - 4.0,  # Denormalize ZDR [-4, 8] dB
                phidp_observed=phidp_norm * 360.0,    # Denormalize PhiDP
                bbf=bbf,
            )

            loss = loss_dict["total_loss"]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += loss.item()
            print(f"  Epoch {epoch+1} | Batch {batch_idx+1}/{len(train_loader)}: Loss = {loss.item():.4f}", flush=True)

        scheduler.step()
        avg_loss = epoch_loss / max(len(train_loader), 1)
        history["train_loss"].append(avg_loss)
        elapsed = time.time() - start_time

        print(f"Epoch [{epoch+1}/{epochs}] - Avg Loss: {avg_loss:.4f} - Time: {elapsed:.2f}s", flush=True)

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": best_loss,
                },
                save_checkpoint_path,
            )

    print(f"Training completed. Best model checkpoint saved to: {save_checkpoint_path}", flush=True)
    return history
