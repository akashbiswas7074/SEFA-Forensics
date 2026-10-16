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

            selected_act = feat[..., indices]
            acts_list.append(selected_act)

            if residual_features is not None and layer_idx in residual_features:
                res = residual_features[layer_idx]
                selected_res = res[..., indices]
                residuals_list.append(selected_res)
            else:
                residuals_list.append(torch.zeros_like(selected_act))

        all_acts = torch.cat(acts_list, dim=-1)
        all_res = torch.cat(residuals_list, dim=-1)
        return all_acts, all_res


class BoundaryGuidedRefiner(nn.Module):
    """
    Lightweight Spatial Refinement Head (<1.5k parameters).
    Sharpens coarse patch-level heatmaps by integrating image gradient boundaries,
    resolving the ViT patch quantization bottleneck and boosting localization mIoU.
    """
    def __init__(self, in_channels: int = 1, hidden_dim: int = 16):
        super().__init__()
        self.refiner = nn.Sequential(
            nn.Conv2d(in_channels + 1, hidden_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden_dim, 1, kernel_size=3, padding=1),
        )

    def forward(self, coarse_mask: torch.Tensor, image: torch.Tensor) -> torch.Tensor:
        """
        coarse_mask: [B, 1, H, W] upsampled coarse probability map
        image: [B, 3, H, W] normalized input image
        Returns:
            refined_mask: [B, 1, H, W] edge-aligned sharp probability map
        """
        # Compute grayscale gradient magnitude for boundary guidance
        gray = 0.2989 * image[:, 0:1] + 0.5870 * image[:, 1:2] + 0.1140 * image[:, 2:3]
        grad_x = torch.abs(gray[:, :, :, 1:] - gray[:, :, :, :-1])
        grad_x = F.pad(grad_x, (0, 1, 0, 0))
        grad_y = torch.abs(gray[:, :, 1:, :] - gray[:, :, :-1, :])
        grad_y = F.pad(grad_y, (0, 0, 0, 1))
        edge_guidance = torch.clamp(grad_x + grad_y, 0.0, 1.0)

        # Concatenate coarse mask with edge guidance
        cat_feat = torch.cat([coarse_mask, edge_guidance], dim=1)
        residual = self.refiner(cat_feat)
        refined_logits = torch.logit(torch.clamp(coarse_mask, 1e-4, 1.0 - 1e-4)) + residual
        return torch.sigmoid(refined_logits)


class AnchorRouter(nn.Module):
    """
    Ultra-Lightweight Multi-Quantile Router (<0.2% total parameters).
    Features Adaptive Multi-Scale Pooling to detect both micro-retouching (<2% pixels)
    and large generative inpainting edits (>50% pixels).
    """
    def __init__(
        self,
        total_anchors: int = 64,
        hidden_dim: int = 64,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.total_anchors = total_anchors
        self.hidden_dim = hidden_dim

        in_dim = total_anchors * 2

        self.gate_mlp = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
        )

        # Patch-level localization projection
        self.patch_head = nn.Linear(hidden_dim // 2, 1)

        # Image-level decision head
        self.image_head = nn.Sequential(
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.GELU(),
            nn.Linear(hidden_dim // 4, 1),
        )

        # Learnable multi-scale pooling combination weights
        self.scale_weights = nn.Parameter(torch.tensor([0.4, 0.35, 0.25]))
        self.log_temp = nn.Parameter(torch.zeros(1))

    def count_trainable_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(
        self,
        anchor_acts: torch.Tensor,
        anchor_residuals: torch.Tensor,
        spatial_hw: Tuple[int, int],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, N, K = anchor_acts.shape
        H_p, W_p = spatial_hw

        # Concatenate anchor activations and stress residuals
        x = torch.cat([anchor_acts, anchor_residuals], dim=-1)  # [B, N, 2K]
        routed_embeds = self.gate_mlp(x)                        # [B, N, hidden_dim // 2]
        patch_logits = self.patch_head(routed_embeds)            # [B, N, 1]

        # Low-res spatial patch grid
        patch_grid = patch_logits.view(B, H_p, W_p, 1).permute(0, 3, 1, 2)
        patch_mask = torch.sigmoid(patch_grid)

        # ADAPTIVE MULTI-QUANTILE POOLING:
        # Scale 1: Micro-retouching (Top 3% of patches)
        k1 = max(1, int(0.03 * N))
        topk1, _ = torch.topk(patch_logits.squeeze(-1), k=k1, dim=1)
        score_micro = topk1.mean(dim=1, keepdim=True)

        # Scale 2: Medium inpainting (Top 15% of patches)
        k2 = max(1, int(0.15 * N))
        topk2, _ = torch.topk(patch_logits.squeeze(-1), k=k2, dim=1)
        score_med = topk2.mean(dim=1, keepdim=True)

        # Scale 3: Macro inpainting (Top 40% of patches)
        k3 = max(1, int(0.40 * N))
        topk3, _ = torch.topk(patch_logits.squeeze(-1), k=k3, dim=1)
        score_macro = topk3.mean(dim=1, keepdim=True)

        sw = F.softmax(self.scale_weights, dim=0)
        multi_scale_patch_score = sw[0] * score_micro + sw[1] * score_med + sw[2] * score_macro

        # Global feature aggregation
        attn_weights = F.softmax(patch_logits.squeeze(-1) / torch.exp(self.log_temp), dim=-1)
        global_embed = torch.bmm(attn_weights.unsqueeze(1), routed_embeds).squeeze(1)
        global_logit = self.image_head(global_embed)

        img_logit = 0.65 * multi_scale_patch_score + 0.35 * global_logit
        return img_logit, patch_mask, routed_embeds
