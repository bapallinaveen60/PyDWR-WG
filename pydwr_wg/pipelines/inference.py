"""
Operational Inference Pipeline for PTA-InNet.
De-clutters, inpaints, and estimates surface rainfall rates across polar radar sweeps.
"""

import os
import torch
import numpy as np
from typing import Dict, Any, Optional
from pydwr_wg.models.pta_innet import PTAInNet
from pydwr_wg.constants import CHANNEL_BOUNDS

class PTAInferencePipeline:
    """
    Production inference engine for real-time de-cluttering and precipitation reconstruction.
    """

    def __init__(
        self,
        checkpoint_path: Optional[str] = None,
        device: Optional[str] = None,
    ):
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.model = PTAInNet(in_channels=10).to(self.device)

        if checkpoint_path and os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            state_dict = checkpoint.get("model_state_dict", checkpoint)
            self.model.load_state_dict(state_dict)
            print(f"Loaded PTA-InNet checkpoint from: {checkpoint_path}")
        else:
            print("Running PTAInferencePipeline with initialized weights.")

        self.model.eval()

    def process_polar_tensor(
        self,
        x_tensor: np.ndarray,
    ) -> Dict[str, np.ndarray]:
        """
        Executes de-cluttering and inpainting on a 10-channel normalized input tensor [10, Nr, Na].

        Returns
        -------
        Dict[str, np.ndarray]
            - 'clutter_mask': Clutter probability mask [Nr, Na]
            - 'zh_clean': Reconstructed reflectivity [Nr, Na] in dBZ
            - 'rain_rate': Surface precipitation rate [Nr, Na] in mm/h
        """
        if x_tensor.ndim == 3:
            x_in = torch.from_numpy(x_tensor).unsqueeze(0).to(self.device)
        else:
            x_in = torch.from_numpy(x_tensor).to(self.device)

        with torch.no_grad():
            preds = self.model(x_in)

        clutter_mask = preds["clutter_mask"].squeeze().cpu().numpy()
        zh_clean = preds["zh_clean"].squeeze().cpu().numpy()
        rain_rate = preds["rain_rate"].squeeze().cpu().numpy()

        return {
            "clutter_mask": clutter_mask,
            "zh_clean": zh_clean,
            "rain_rate": rain_rate,
        }
