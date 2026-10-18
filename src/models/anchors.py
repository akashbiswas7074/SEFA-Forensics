"""
Forensic Anchor Bank, Boundary Guided Refiner, and Multi-Scale Router for SEFA-Forensics.
Combines intrinsic anchor units with adaptive multi-quantile pooling and spatial boundary refinement.
"""

import math
from typing import Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F


class ForensicAnchorBank(nn.Module):
    """
    Maintains calibrated forensic anchor indices and sensitivity metadata.
    Anchors represent sparse internal channels/units within transformer blocks
    that exhibit maximal discriminative sensitivity under multi-band evidence stress.
    """
    def __init__(
        self,
        num_anchors: int = 64,
        target_layers: Optional[List[int]] = None,
        feature_dim: int = 384,
    ):
        super().__init__()
        self.num_anchors = num_anchors
        self.target_layers = target_layers or [3, 6, 9, 11]
        self.feature_dim = feature_dim

        anchors_per_layer = max(1, num_anchors // len(self.target_layers))
        self.anchors_per_layer = anchors_per_layer

        init_indices = torch.stack([
            torch.linspace(0, feature_dim - 1, anchors_per_layer).long()
            for _ in range(len(self.target_layers))
        ])
        self.register_buffer("anchor_indices", init_indices)
        self.register_buffer("anchor_weights", torch.ones(len(self.target_layers), anchors_per_layer))
        self.register_buffer("is_calibrated", torch.tensor(False, dtype=torch.bool))

        self.category_names = [
            "Diffusion High-Frequency Spectral Residual",
            "Boundary Phase Incoherence",
            "Latent Denoising Variance Discrepancy",
            "Semantic Texture Artifact",
        ]

    def update_anchors(
        self,
        calibrated_indices: torch.Tensor,
        calibrated_weights: torch.Tensor,
    ):
        self.anchor_indices.copy_(calibrated_indices)
        self.anchor_weights.copy_(calibrated_weights)
        self.is_calibrated.copy_(torch.tensor(True))

    def extract_anchors(
        self,
        layer_features: Dict[int, torch.Tensor],
        residual_features: Optional[Dict[int, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        acts_list = []
        residuals_list = []

        for idx, layer_idx in enumerate(self.target_layers):
            if layer_idx not in layer_features:
                continue

            feat = layer_features[layer_idx]  # [B, N, D]
            indices = self.anchor_indices[idx]  # [anchors_per_layer]
            w = (self.anchor_weights[idx] / (self.anchor_weights[idx].mean() + 1e-6)).unsqueeze(0).unsqueeze(0)

            selected_act = feat[..., indices] * w
            acts_list.append(selected_act)

            if residual_features is not None and layer_idx in residual_features:
                res = residual_features[layer_idx]
                selected_res = res[..., indices] * w
                residuals_list.append(selected_res)
            else:
                residuals_list.append(torch.zeros_like(selected_act))

        all_acts = torch.cat(acts_list, dim=-1)
        all_res = torch.cat(residuals_list, dim=-1)
        return all_acts, all_res


class ProgressiveLocalizationDecoder(nn.Module):
    """
    Progressive High-Resolution Localization Decoder (<12k parameters).
    Decodes spatial anomaly grid and multi-quantile patch priors into a high-resolution 224x224 mask:
    16x16 (ch=32) + 16x16 patch logit prior -> intermediate 1/4 res (ch=16) -> full res (ch=16) + Sobel gradient -> full res (ch=1).
    Bridges the ViT patch quantization bottleneck to produce crisp, boundary-aligned localization heatmaps.
    """
    def __init__(self, in_channels: int = 32, hidden_dim: int = 16):
        super().__init__()
        self.patch_proj = nn.Conv2d(1, hidden_dim, kernel_size=3, padding=1)
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels + hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.GELU(),
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.GELU(),
        )
        self.final_conv = nn.Sequential(
            nn.Conv2d(hidden_dim + 1, hidden_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden_dim, 1, kernel_size=3, padding=1),
        )
        # Initialize final conv with slight negative bias for clean authentic background
        nn.init.constant_(self.final_conv[-1].bias, -0.5)

    def forward(
        self,
        feat_grid: torch.Tensor,
        image: torch.Tensor,
        patch_logits: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        feat_grid: [B, in_channels, H_p, W_p] spatial feature grid from router
        image: [B, 3, H, W] normalized input image
        patch_logits: [B, 1, H_p, W_p] optional multi-quantile patch prior logits
        Returns:
            pred_mask: [B, 1, H, W] in [0, 1]
            logits: [B, 1, H, W] continuous localization logits
        """
        B, _, H, W = image.shape
        h_mid, w_mid = max(14, H // 4), max(14, W // 4)

        # Step 1: Intermediate upsampling to 1/4 resolution (e.g. 56x56) with patch prior injection
        x1_f = F.interpolate(feat_grid, size=(h_mid, w_mid), mode="bilinear", align_corners=False)
        if patch_logits is not None:
            p_mid = F.interpolate(patch_logits, size=(h_mid, w_mid), mode="bilinear", align_corners=False)
        else:
            p_mid = torch.zeros(B, 1, h_mid, w_mid, device=feat_grid.device, dtype=feat_grid.dtype)
        p_feat = self.patch_proj(p_mid)
        x1 = self.block1(torch.cat([x1_f, p_feat], dim=1))

        # Step 2: Full resolution upsampling (H, W)
        x2 = F.interpolate(x1, size=(H, W), mode="bilinear", align_corners=False)
        x2 = self.block2(x2)

        # Grayscale Sobel gradient for high-frequency boundary guidance
        gray = 0.2989 * image[:, 0:1] + 0.5870 * image[:, 1:2] + 0.1140 * image[:, 2:3]
        gx = torch.abs(gray[:, :, :, 1:] - gray[:, :, :, :-1])
        gx = F.pad(gx, (0, 1, 0, 0))
        gy = torch.abs(gray[:, :, 1:, :] - gray[:, :, :-1, :])
        gy = F.pad(gy, (0, 0, 0, 1))
        edge = torch.clamp(gx + gy, 0.0, 1.0)

        cat_feat = torch.cat([x2, edge], dim=1)
        logits = self.final_conv(cat_feat)
        return torch.sigmoid(logits), logits


class BoundaryGuidedRefiner(nn.Module):
    """
    Backwards-compatible wrapper around ProgressiveLocalizationDecoder.
    """
    def __init__(self, in_channels: int = 1, hidden_dim: int = 16):
        super().__init__()
        self.refiner = nn.Sequential(
            nn.Conv2d(in_channels + 1, hidden_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden_dim, 1, kernel_size=3, padding=1),
        )
        nn.init.zeros_(self.refiner[-1].weight)
        nn.init.zeros_(self.refiner[-1].bias)

    def forward(self, coarse_logits: torch.Tensor, image: torch.Tensor) -> torch.Tensor:
        gray = 0.2989 * image[:, 0:1] + 0.5870 * image[:, 1:2] + 0.1140 * image[:, 2:3]
        grad_x = torch.abs(gray[:, :, :, 1:] - gray[:, :, :, :-1])
        grad_x = F.pad(grad_x, (0, 1, 0, 0))
        grad_y = torch.abs(gray[:, :, 1:, :] - gray[:, :, :-1, :])
        grad_y = F.pad(grad_y, (0, 0, 0, 1))
        edge_guidance = torch.clamp(grad_x + grad_y, 0.0, 1.0)

        cat_feat = torch.cat([coarse_logits, edge_guidance], dim=1)
        residual = self.refiner(cat_feat)
        refined_logits = coarse_logits + residual
        return torch.sigmoid(refined_logits)


class AnchorRouter(nn.Module):
    """
    Ultra-Lightweight Multi-Quantile Router with Unified Classifier (<0.15% parameters).
    Combines:
    1. Spatial depthwise 3x3 convolution across the 16x16 patch grid to detect local residual gradients
    2. Adaptive multi-scale patch quantile scores (micro, medium, macro tampering)
    3. Multi-statistic anomaly aggregation (mean, max, std, top-10% quantile)
    4. DINOv2 CLS token projection for global image-level forensic discrimination
    5. Unified classifier avoiding scale interference and gradient cancellation
    """
    def __init__(
        self,
        total_anchors: int = 64,
        hidden_dim: int = 64,
        cls_dim: int = 384,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.total_anchors = total_anchors
        self.hidden_dim = hidden_dim

        in_dim = total_anchors * 2
        h2 = hidden_dim // 2

        self.mlp_in = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, h2),
            nn.GELU(),
        )

        # Spatial Depthwise 3x3 convolution across patch grid (preserves parameter efficiency)
        self.spatial_dw = nn.Sequential(
            nn.Conv2d(h2, h2, kernel_size=3, padding=1, groups=h2),
            nn.BatchNorm2d(h2),
            nn.GELU(),
        )

        # Patch-level localization projection (outputs continuous logits)
        self.patch_head = nn.Linear(h2, 1)

        # DINOv2 CLS token projection
        self.cls_proj = nn.Sequential(
            nn.Linear(cls_dim, h2),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Unified global classification head fusing CLS + Patch statistics + Multi-scale top-k anomaly scores
        # Features: mean (h2) + max (h2) + std (h2) + top-10% (h2) + CLS (h2) + multi-scale anomaly (3) = h2*5 + 3
        self.classifier = nn.Sequential(
            nn.Linear(h2 * 5 + 3, h2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(h2, 1),
        )

    def count_trainable_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(
        self,
        anchor_acts: torch.Tensor,
        anchor_residuals: torch.Tensor,
        spatial_hw: Tuple[int, int],
        cls_token: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, N, K = anchor_acts.shape
        H_p, W_p = spatial_hw

        # Concatenate anchor activations and stress residuals
        x = torch.cat([anchor_acts, anchor_residuals], dim=-1)  # [B, N, 2K]
        feat = self.mlp_in(x)                                   # [B, N, h2]

        # Apply spatial depthwise context across neighbor patches
        feat_grid = feat.view(B, H_p, W_p, -1).permute(0, 3, 1, 2)
        feat_grid = feat_grid + self.spatial_dw(feat_grid)
        feat = feat_grid.permute(0, 2, 3, 1).reshape(B, N, -1)

        # Patch localization continuous logits: [B, 1, H_p, W_p]
        patch_logits_flat = self.patch_head(feat)               # [B, N, 1]
        patch_logits = patch_logits_flat.view(B, H_p, W_p, 1).permute(0, 3, 1, 2)

        # ADAPTIVE MULTI-QUANTILE POOLING:
        p_flat = patch_logits_flat.squeeze(-1)
        k1 = max(1, int(0.04 * N))  # Micro-retouching (Top 4%)
        k2 = max(1, int(0.15 * N))  # Medium inpainting (Top 15%)
        k3 = max(1, int(0.40 * N))  # Macro inpainting (Top 40%)

        s1 = torch.topk(p_flat, k=k1, dim=1)[0].mean(dim=1, keepdim=True)
        s2 = torch.topk(p_flat, k=k2, dim=1)[0].mean(dim=1, keepdim=True)
        s3 = torch.topk(p_flat, k=k3, dim=1)[0].mean(dim=1, keepdim=True)

        # MULTI-STATISTIC GLOBAL REPRESENTATION
        k_top = max(1, int(0.10 * N))
        g_mean = feat.mean(dim=1)
        g_max = feat.max(dim=1).values
        g_std = feat.std(dim=1)
        g_top = torch.topk(feat, k=k_top, dim=1)[0].mean(dim=1)

        # DINOv2 CLS token representation
        if cls_token is not None:
            cls_feat = self.cls_proj(cls_token)
        else:
            cls_feat = torch.zeros(B, feat.shape[-1], device=feat.device)

        fusion = torch.cat([g_mean, g_max, g_std, g_top, cls_feat, s1, s2, s3], dim=-1)
        img_logits = self.classifier(fusion)

        return img_logits, patch_logits, feat_grid
