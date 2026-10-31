"""
Polarimetric Terrain-Aware Inpainting Network (PTA-InNet).
Dual-decoder encoder-decoder architecture with circular azimuth padding and multi-head attention.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple
from pydwr_wg.models.layers import PolarConvBlock, AttentionGate, CircularAzimuthPad2d

class MultiHeadRadialAttention(nn.Module):
    """
    Multi-Head Attention along the radar beam path to capture long-range correlation
    between downstream unblocked differential phase and blocked near-range sectors.
    """

    def __init__(self, embed_dim: int = 1024, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"

        self.q_proj = nn.Conv2d(embed_dim, embed_dim, kernel_size=1)
        self.k_proj = nn.Conv2d(embed_dim, embed_dim, kernel_size=1)
        self.v_proj = nn.Conv2d(embed_dim, embed_dim, kernel_size=1)
        self.out_proj = nn.Conv2d(embed_dim, embed_dim, kernel_size=1)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.GroupNorm(8, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        residual = x

        q = self.q_proj(x).view(b, self.num_heads, self.head_dim, h * w).permute(0, 1, 3, 2)
        k = self.k_proj(x).view(b, self.num_heads, self.head_dim, h * w).permute(0, 1, 3, 2)
        v = self.v_proj(x).view(b, self.num_heads, self.head_dim, h * w).permute(0, 1, 3, 2)

        scale = 1.0 / (self.head_dim ** 0.5)
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * scale
        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_probs = self.dropout(attn_probs)

        out = torch.matmul(attn_probs, v)  # [b, heads, hw, head_dim]
        out = out.permute(0, 1, 3, 2).contiguous().view(b, c, h, w)
        out = self.out_proj(out)

        return self.norm(residual + out)


class PTAInNet(nn.Module):
    """
    Polarimetric Terrain-Aware Inpainting Network (PTA-InNet) for radar de-cluttering
    and low-level warm orographic precipitation reconstruction.
    """

    def __init__(
        self,
        in_channels: int = 10,
        encoder_channels: Tuple[int, ...] = (64, 128, 256, 512, 1024),
        num_groups: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        c1, c2, c3, c4, c5 = encoder_channels

        # --- Hierarchical 5-Stage Encoder ---
        self.enc1 = PolarConvBlock(in_channels, c1, num_groups)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.enc2 = PolarConvBlock(c1, c2, num_groups)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.enc3 = PolarConvBlock(c2, c3, num_groups)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.enc4 = PolarConvBlock(c3, c4, num_groups)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.enc5 = PolarConvBlock(c4, c5, num_groups)

        # Bottleneck Self-Attention
        self.bottleneck_attn = MultiHeadRadialAttention(c5, num_heads=8, dropout=dropout)

        # --- Decoder 1: Semantic Clutter Segmentation Decoder ---
        self.up_seg4 = nn.ConvTranspose2d(c5, c4, kernel_size=2, stride=2)
        self.att_seg4 = AttentionGate(f_gate=c4, f_x=c4, f_int=c3)
        self.conv_seg4 = PolarConvBlock(c5, c4, num_groups)

        self.up_seg3 = nn.ConvTranspose2d(c4, c3, kernel_size=2, stride=2)
        self.att_seg3 = AttentionGate(f_gate=c3, f_x=c3, f_int=c2)
        self.conv_seg3 = PolarConvBlock(c4, c3, num_groups)

        self.up_seg2 = nn.ConvTranspose2d(c3, c2, kernel_size=2, stride=2)
        self.att_seg2 = AttentionGate(f_gate=c2, f_x=c2, f_int=c1)
        self.conv_seg2 = PolarConvBlock(c3, c2, num_groups)

        self.up_seg1 = nn.ConvTranspose2d(c2, c1, kernel_size=2, stride=2)
        self.att_seg1 = AttentionGate(f_gate=c1, f_x=c1, f_int=c1 // 2)
        self.conv_seg1 = PolarConvBlock(c2, c1, num_groups)

        self.out_seg = nn.Sequential(
            nn.Conv2d(c1, 1, kernel_size=1),
            nn.Sigmoid(),
        )

        # --- Decoder 2: Polarimetric Inpainting and Reconstruction Decoder ---
        self.up_rec4 = nn.ConvTranspose2d(c5, c4, kernel_size=2, stride=2)
        self.att_rec4 = AttentionGate(f_gate=c4, f_x=c4, f_int=c3)
        self.conv_rec4 = PolarConvBlock(c5, c4, num_groups)

        self.up_rec3 = nn.ConvTranspose2d(c4, c3, kernel_size=2, stride=2)
        self.att_rec3 = AttentionGate(f_gate=c3, f_x=c3, f_int=c2)
        self.conv_rec3 = PolarConvBlock(c4, c3, num_groups)

        self.up_rec2 = nn.ConvTranspose2d(c3, c2, kernel_size=2, stride=2)
        self.att_rec2 = AttentionGate(f_gate=c2, f_x=c2, f_int=c1)
        self.conv_rec2 = PolarConvBlock(c3, c2, num_groups)

        self.up_rec1 = nn.ConvTranspose2d(c2, c1, kernel_size=2, stride=2)
        self.att_rec1 = AttentionGate(f_gate=c1, f_x=c1, f_int=c1 // 2)
        self.conv_rec1 = PolarConvBlock(c2, c1, num_groups)

        # Heads for reconstructed ZH and surface rain rate
        self.head_zh = nn.Conv2d(c1, 1, kernel_size=1)
        self.head_rain = nn.Sequential(
            nn.Conv2d(c1, 1, kernel_size=1),
            nn.ReLU(inplace=True),  # Rain rate >= 0 mm/h
        )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass of PTA-InNet.

        Parameters
        ----------
        x : torch.Tensor
            Normalized multi-channel polar tensor [B, 10, Nr, Na].

        Returns
        -------
        Dict[str, torch.Tensor]
            - 'clutter_mask': Clutter probability mask [B, 1, Nr, Na] in [0, 1]
            - 'zh_clean': Inpainted, attenuation-corrected reflectivity [B, 1, Nr, Na] in dBZ
            - 'rain_rate': Surface precipitation rate [B, 1, Nr, Na] in mm/h
        """
        orig_h, orig_w = x.shape[2], x.shape[3]

        # Pad dynamically to multiples of 16 for clean 4-level pooling
        pad_h = (16 - (orig_h % 16)) % 16
        pad_w = (16 - (orig_w % 16)) % 16
        if pad_h > 0 or pad_w > 0:
            x_in = F.pad(x, (0, pad_w, 0, pad_h), mode="reflect")
        else:
            x_in = x

        # Encoder path
        s1 = self.enc1(x_in)
        p1 = self.pool1(s1)

        s2 = self.enc2(p1)
        p2 = self.pool2(s2)

        s3 = self.enc3(p2)
        p3 = self.pool3(s3)

        s4 = self.enc4(p3)
        p4 = self.pool4(s4)

        b5 = self.enc5(p4)
        b5 = self.bottleneck_attn(b5)

        # Decoder 1: Segmentation
        d_seg4 = self.up_seg4(b5)
        att_s4 = self.att_seg4(gate=d_seg4, skip=s4)
        d_seg4 = self.conv_seg4(torch.cat([d_seg4, att_s4], dim=1))

        d_seg3 = self.up_seg3(d_seg4)
        att_s3 = self.att_seg3(gate=d_seg3, skip=s3)
        d_seg3 = self.conv_seg3(torch.cat([d_seg3, att_s3], dim=1))

        d_seg2 = self.up_seg2(d_seg3)
        att_s2 = self.att_seg2(gate=d_seg2, skip=s2)
        d_seg2 = self.conv_seg2(torch.cat([d_seg2, att_s2], dim=1))

        d_seg1 = self.up_seg1(d_seg2)
        att_s1 = self.att_seg1(gate=d_seg1, skip=s1)
        d_seg1 = self.conv_seg1(torch.cat([d_seg1, att_s1], dim=1))

        clutter_prob = self.out_seg(d_seg1)

        # Decoder 2: Reconstruction & Inpainting
        d_rec4 = self.up_rec4(b5)
        att_r4 = self.att_rec4(gate=d_rec4, skip=s4)
        d_rec4 = self.conv_rec4(torch.cat([d_rec4, att_r4], dim=1))

        d_rec3 = self.up_rec3(d_rec4)
        att_r3 = self.att_rec3(gate=d_rec3, skip=s3)
        d_rec3 = self.conv_rec3(torch.cat([d_rec3, att_r3], dim=1))

        d_rec2 = self.up_rec2(d_rec3)
        att_r2 = self.att_rec2(gate=d_rec2, skip=s2)
        d_rec2 = self.conv_rec2(torch.cat([d_rec2, att_r2], dim=1))

        d_rec1 = self.up_rec1(d_rec2)
        att_r1 = self.att_rec1(gate=d_rec1, skip=s1)
        d_rec1 = self.conv_rec1(torch.cat([d_rec1, att_r1], dim=1))

        # Denormalize ZH: [-10, 75] dBZ
        zh_norm_pred = self.head_zh(d_rec1)
        zh_clean = zh_norm_pred * 85.0 - 10.0
        rain_rate = self.head_rain(d_rec1)

        # Crop back to original shape if padded
        if pad_h > 0 or pad_w > 0:
            clutter_prob = clutter_prob[:, :, :orig_h, :orig_w]
            zh_clean = zh_clean[:, :, :orig_h, :orig_w]
            rain_rate = rain_rate[:, :, :orig_h, :orig_w]

        return {
            "clutter_mask": clutter_prob,
            "zh_clean": zh_clean,
            "rain_rate": rain_rate,
        }
