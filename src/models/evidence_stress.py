"""
Evidence Stress Module for SEFA-Forensics.
Applies controlled multi-band and compression-resilient forensic perturbations to force
the frozen foundation model to reveal latent forgery residuals.
"""

import math
from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class EvidenceStressModule(nn.Module):
    """
    Multi-Band and Compression-Resilient Evidence Stress Generator.
    Applies multi-scale spectral residual, boundary phase, and noise-floor perturbations
    designed to remain robust under JPEG compression and social media transcoding.
    """
    def __init__(
        self,
        stress_strength: float = 0.15,
        modes: Optional[List[str]] = None,
        blend_weight: float = 0.5,
    ):
        super().__init__()
        self.stress_strength = stress_strength
        self.modes = modes or ["multi_band_spectral", "boundary_phase", "compression_residual", "noise_floor"]
        self.blend_weight = blend_weight

        # 1. Multi-Scale Gaussian kernels (fine scale sigma=1.0, medium scale sigma=2.5)
        k_sz1 = 5
        coords1 = torch.arange(k_sz1).float() - (k_sz1 - 1) / 2.0
        grid1 = coords1.repeat(k_sz1, 1)
        k2d_1 = torch.exp(-(grid1 ** 2 + grid1.t() ** 2) / (2 * 1.0 ** 2))
        k2d_1 = (k2d_1 / k2d_1.sum()).view(1, 1, k_sz1, k_sz1).repeat(3, 1, 1, 1)
        self.register_buffer("kernel_fine", k2d_1)

        k_sz2 = 9
        coords2 = torch.arange(k_sz2).float() - (k_sz2 - 1) / 2.0
        grid2 = coords2.repeat(k_sz2, 1)
        k2d_2 = torch.exp(-(grid2 ** 2 + grid2.t() ** 2) / (2 * 2.5 ** 2))
        k2d_2 = (k2d_2 / k2d_2.sum()).view(1, 1, k_sz2, k_sz2).repeat(3, 1, 1, 1)
        self.register_buffer("kernel_coarse", k2d_2)

        # 2. Sobel filters for edge-phase gradient stress
        sobel_x = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]).view(1, 1, 3, 3).repeat(3, 1, 1, 1)
        sobel_y = torch.tensor([[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]]).view(1, 1, 3, 3).repeat(3, 1, 1, 1)
        self.register_buffer("sobel_x", sobel_x)
        self.register_buffer("sobel_y", sobel_y)

        # 3. Learnable / Adaptive stress band modulation coefficients (3 params)
        self.band_weights = nn.Parameter(torch.tensor([0.45, 0.35, 0.20]))

    def _multi_band_spectral_stress(self, x: torch.Tensor) -> torch.Tensor:
        """
        Decomposes image into High, Mid, and Low bands.
        Injects controlled stress across mid-frequencies so forensic traces survive even if
        extreme JPEG compression eliminates the highest frequencies.
        """
        low_pass_coarse = F.conv2d(x, self.kernel_coarse, padding=4, groups=3)
        low_pass_fine = F.conv2d(x, self.kernel_fine, padding=2, groups=3)

        high_band = x - low_pass_fine
        mid_band = low_pass_fine - low_pass_coarse

        w = F.softmax(self.band_weights, dim=0)
        # Composite multi-band residual
        spectral_stress = w[0] * high_band + w[1] * mid_band
        return torch.clamp(x + self.stress_strength * spectral_stress, 0.0, 1.0)

    def _boundary_phase_stress(self, x: torch.Tensor) -> torch.Tensor:
        """
        Perturbs edge transition phases to challenge seamless generative inpainting boundaries.
        """
        gx = F.conv2d(x, self.sobel_x, padding=1, groups=3)
        gy = F.conv2d(x, self.sobel_y, padding=1, groups=3)
        grad_mag = torch.sqrt(gx ** 2 + gy ** 2 + 1e-8)
        grad_norm = grad_mag / (grad_mag.max() + 1e-6)

        # Target boundary transition zone
        seam_zone = 4.0 * grad_norm * (1.0 - grad_norm)
        return torch.clamp(x + self.stress_strength * seam_zone * (torch.rand_like(x) - 0.5) * 0.3, 0.0, 1.0)

    def _noise_floor_stress(self, x: torch.Tensor) -> torch.Tensor:
        """
        Injects calibrated multi-scale sensor noise to expose latent diffusion scheduler footprint.
        """
        noise = torch.randn_like(x) * (self.stress_strength * 0.04)
        return torch.clamp(x + noise, 0.0, 1.0)

    def forward(self, x: torch.Tensor, mode: Optional[str] = None) -> torch.Tensor:
        """
        Apply evidence stress to input batch x [B, 3, H, W].
        """
        min_v = x.min()
        max_v = x.max()
        is_normalized = (min_v < 0) or (max_v > 1.0)
        x_unit = (x - min_v) / (max_v - min_v + 1e-8) if is_normalized else x

        if mode == "high_pass_residual" or mode == "multi_band_spectral":
            stressed = self._multi_band_spectral_stress(x_unit)
        elif mode == "gradient_phase" or mode == "boundary_phase":
            stressed = self._boundary_phase_stress(x_unit)
        elif mode == "noise_floor":
            stressed = self._noise_floor_stress(x_unit)
        else:
            # Multi-Modal Composite Stress
            s_spec = self._multi_band_spectral_stress(x_unit)
            s_phase = self._boundary_phase_stress(x_unit)
            s_noise = self._noise_floor_stress(x_unit)
            stressed = 0.5 * s_spec + 0.3 * s_phase + 0.2 * s_noise

        if is_normalized:
            stressed = stressed * (max_v - min_v + 1e-8) + min_v

        return stressed

    @staticmethod
    def compute_latent_residual(
        clean_features: torch.Tensor,
        stressed_features: torch.Tensor,
        mode: str = "absolute"
    ) -> torch.Tensor:
        """
        Computes discriminative residual: Delta Z = |Z(x) - Z(S(x))|
        """
        diff = clean_features - stressed_features
        if mode == "absolute":
            return torch.abs(diff)
        elif mode == "squared":
            return diff ** 2
        elif mode == "normalized":
            norm_clean = torch.norm(clean_features, dim=-1, keepdim=True) + 1e-6
            return torch.abs(diff) / norm_clean
        else:
            return diff
