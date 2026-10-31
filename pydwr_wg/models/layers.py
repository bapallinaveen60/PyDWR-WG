"""
Specialized PyTorch Layers for Polar Radar Processing.
Implements circular azimuth padding, depthwise separable convolutions, and attention gates.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class CircularAzimuthPad2d(nn.Module):
    """
    Applies zero padding along the radial range dimension (H) and circular periodic padding
    along the azimuthal angle dimension (W), preserving continuity across the 360 -> 0 meridian.
    """

    def __init__(self, pad_range: int = 1, pad_azimuth: int = 1):
        super().__init__()
        self.pad_range = pad_range
        self.pad_azimuth = pad_azimuth

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: [B, C, H (range), W (azimuth)]
        # 1. Circular padding along azimuth (dim -1)
        if self.pad_azimuth > 0:
            left = x[:, :, :, -self.pad_azimuth:]
            right = x[:, :, :, :self.pad_azimuth]
            x = torch.cat([left, x, right], dim=-1)

        # 2. Zero padding along range (dim -2)
        if self.pad_range > 0:
            x = F.pad(x, (0, 0, self.pad_range, self.pad_range), mode="constant", value=0.0)

        return x


class SeparableConv2d(nn.Module):
    """
    Depthwise Separable Convolution block to reduce parameterization on polar radar sweeps.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        dilation: int = 1,
        bias: bool = False,
    ):
        super().__init__()
        pad = (kernel_size - 1) // 2 * dilation
        self.depthwise = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=0,
            dilation=dilation,
            groups=in_channels,
            bias=bias,
        )
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)
        self.circular_pad = CircularAzimuthPad2d(pad_range=pad, pad_azimuth=pad)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.circular_pad(x)
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x


class PolarConvBlock(nn.Module):
    """
    Dual-separable convolution block with Group Normalization and LeakyReLU activation.
    GroupNorm is chosen over BatchNorm to handle high meteorological variability between sweeps.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_groups: int = 8,
        negative_slope: float = 0.1,
    ):
        super().__init__()
        # Ensure num_groups divides out_channels
        g1 = num_groups if out_channels % num_groups == 0 else 1
        self.conv1 = SeparableConv2d(in_channels, out_channels, kernel_size=3)
        self.gn1 = nn.GroupNorm(g1, out_channels)
        self.act1 = nn.LeakyReLU(negative_slope=negative_slope, inplace=True)

        self.conv2 = SeparableConv2d(out_channels, out_channels, kernel_size=3)
        self.gn2 = nn.GroupNorm(g1, out_channels)
        self.act2 = nn.LeakyReLU(negative_slope=negative_slope, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act1(self.gn1(self.conv1(x)))
        x = self.act2(self.gn2(self.conv2(x)))
        return x


class AttentionGate(nn.Module):
    """
    Attention Gate to filter encoder skip features before concatenation with decoder features.
    Suppresses background non-meteorological noise and emphasizes boundary-layer orographic rain.
    """

    def __init__(self, f_gate: int, f_x: int, f_int: int):
        super().__init__()
        self.w_gate = nn.Sequential(
            nn.Conv2d(f_gate, f_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.GroupNorm(1, f_int),
        )
        self.w_x = nn.Sequential(
            nn.Conv2d(f_x, f_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.GroupNorm(1, f_int),
        )
        self.psi = nn.Sequential(
            nn.Conv2d(f_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.GroupNorm(1, 1),
            nn.Sigmoid(),
        )
        self.act = nn.ReLU(inplace=True)

    def forward(self, gate: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        # Resize gate to match skip spatial dimensions if necessary
        if gate.shape[2:] != skip.shape[2:]:
            gate = F.interpolate(gate, size=skip.shape[2:], mode="bilinear", align_corners=False)

        g_feat = self.w_gate(gate)
        x_feat = self.w_x(skip)
        alpha = self.psi(self.act(g_feat + x_feat))
        return skip * alpha
