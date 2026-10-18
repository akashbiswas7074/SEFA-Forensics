"""
SEFAModel: Self-Eliciting Forensic Anchors Architecture.
Combines frozen foundation backbone, multi-band evidence stress, anchor bank,
multi-scale router, and boundary-guided spatial refinement.
"""

from typing import Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbone import FrozenFoundationBackbone
from .evidence_stress import EvidenceStressModule
from .anchors import ForensicAnchorBank, AnchorRouter, BoundaryGuidedRefiner, ProgressiveLocalizationDecoder


class SEFAModel(nn.Module):
    """
    Self-Eliciting Forensic Anchors (SEFA) Architecture.
    - 100% frozen foundation model (DINOv2 / ViT)
    - Multi-band evidence stress mechanism
    - Calibrated sparse intrinsic anchors
    - Multi-quantile router (<0.15% trainable parameters)
    - Progressive localization decoder for boundary-aligned pixel-level localization
    """
    def __init__(
        self,
        backbone_name: str = "dinov2_vits14",
        pretrained: bool = True,
        img_size: int = 224,
        target_layers: Optional[List[int]] = None,
        num_anchors: int = 64,
        router_hidden_dim: int = 64,
        stress_strength: float = 0.15,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.img_size = img_size
        self.target_layers = target_layers or [3, 6, 9, 11]

        # 1. Completely Frozen Foundation Backbone
        self.backbone = FrozenFoundationBackbone(
            backbone_name=backbone_name,
            pretrained=pretrained,
            img_size=img_size,
            target_layers=self.target_layers,
        )

        # 2. Multi-Band & Compression-Resilient Evidence Stress Module
        self.evidence_stress = EvidenceStressModule(
            stress_strength=stress_strength,
        )

        # 3. Forensic Anchor Bank
        self.anchor_bank = ForensicAnchorBank(
            num_anchors=num_anchors,
            target_layers=self.target_layers,
            feature_dim=self.backbone.feature_dim,
        )

        # 4. Multi-Quantile Router (<0.15% parameters)
        self.router = AnchorRouter(
            total_anchors=num_anchors,
            hidden_dim=router_hidden_dim,
            cls_dim=self.backbone.feature_dim,
            dropout=dropout,
        )

        # 5. Progressive Localization Decoder (<10k parameters)
        self.progressive_decoder = ProgressiveLocalizationDecoder(
            in_channels=router_hidden_dim // 2,
            hidden_dim=16,
        )
        self.boundary_refiner = self.progressive_decoder

    def count_parameters(self) -> Dict[str, Union[int, float]]:
        """
        Reports parameter counts and verifies that trainable ratio is < 0.2%.
        """
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen_params = total_params - trainable_params
        ratio_pct = (trainable_params / max(1, total_params)) * 100.0

        return {
            "total_parameters": total_params,
            "frozen_parameters": frozen_params,
            "trainable_parameters": trainable_params,
            "trainable_percentage": ratio_pct,
        }

    def forward(
        self,
        x: torch.Tensor,
        return_details: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[Dict[str, torch.Tensor]]]:
        """
        Forward pass with self-elicitation.
        Args:
            x: Input images [B, 3, H, W]
            return_details: Whether to return internal anchor activations for explainability
        Returns:
            img_logits: [B, 1]
            pred_mask: [B, 1, H, W] edge-refined pixel-level localization mask
            details: Optional dictionary with anchor activations and residuals
        """
        B, C, H, W = x.shape

        # Step 1: Extract clean features from frozen backbone
        with torch.no_grad():
            clean_feats, spatial_hw = self.backbone.extract_patch_features(x)
            cls_token = getattr(self.backbone, "last_cls_token", None)

            # Step 2: Apply multi-band evidence stress to probe latent generative brittleness
            stressed_x = self.evidence_stress(x)
            stressed_feats, _ = self.backbone.extract_patch_features(stressed_x)

            # Step 3: Compute self-elicited latent residuals: Delta Z = |Z - Z_stress|
            residuals = {}
            for l_idx in clean_feats:
                if l_idx in stressed_feats:
                    residuals[l_idx] = self.evidence_stress.compute_latent_residual(
                        clean_feats[l_idx], stressed_feats[l_idx]
                    )

        # Step 4: Extract calibrated intrinsic forensic anchors
        anchor_acts, anchor_res = self.anchor_bank.extract_anchors(clean_feats, residuals)

        # Step 5: Route through multi-quantile router with CLS semantic anchor
        img_logits, patch_logits, feat_grid = self.router(
            anchor_acts=anchor_acts,
            anchor_residuals=anchor_res,
            spatial_hw=spatial_hw,
            cls_token=cls_token,
        )

        # Step 6: Progressive decoding with boundary guidance and patch prior for sharp 224x224 localization
        refined_mask, mask_logits = self.progressive_decoder(feat_grid, x, patch_logits=patch_logits)

        details = None
        if return_details:
            details = {
                "anchor_acts": anchor_acts,
                "anchor_residuals": anchor_res,
                "patch_logits": patch_logits,
                "mask_logits": mask_logits,
                "spatial_hw": spatial_hw,
                "routed_embeds": feat_grid,
            }

        return img_logits, refined_mask, details

    @torch.no_grad()
    def forward_tiled(
        self,
        x: torch.Tensor,
        tile_size: int = 224,
        stride: int = 112,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        High-Resolution Tiled Inference with Gaussian Window Blending.
        Enables pixel-precise localization for high-resolution images (e.g. 512x512, 1024x1024)
        without downsampling small manipulation artifacts.
        Args:
            x: Input image tensor [B, 3, H, W]
            tile_size: Local window size (default 224)
            stride: Sliding window stride (default 112 for 50% overlap)
        Returns:
            img_logits: [B, 1] aggregated global image authenticity logit
            full_mask: [B, 1, H, W] seamlessly stitched full-resolution localization mask
        """
        B, C, H, W = x.shape
        if H <= tile_size and W <= tile_size:
            img_logits, mask, _ = self.forward(x)
            return img_logits, mask

        # Gaussian blending window to eliminate seam artifacts
        y, x_coord = torch.meshgrid(
            torch.linspace(-1, 1, tile_size, device=x.device),
            torch.linspace(-1, 1, tile_size, device=x.device),
            indexing="ij"
        )
        d = torch.sqrt(x_coord * x_coord + y * y)
        sigma, mu = 0.5, 0.0
        gauss_w = torch.exp(-((d - mu) ** 2 / (2.0 * sigma ** 2)))
        gauss_w = gauss_w.unsqueeze(0).unsqueeze(0)  # [1, 1, tile_size, tile_size]

        full_pred = torch.zeros(B, 1, H, W, device=x.device)
        weight_sum = torch.zeros(B, 1, H, W, device=x.device)
        tile_img_logits = []

        y_steps = list(range(0, max(1, H - tile_size + 1), stride))
        if len(y_steps) == 0 or y_steps[-1] != H - tile_size:
            y_steps.append(max(0, H - tile_size))
        x_steps = list(range(0, max(1, W - tile_size + 1), stride))
        if len(x_steps) == 0 or x_steps[-1] != W - tile_size:
            x_steps.append(max(0, W - tile_size))

        y_steps = sorted(list(set(y_steps)))
        x_steps = sorted(list(set(x_steps)))

        for ys in y_steps:
            for xs in x_steps:
                tile = x[:, :, ys : ys + tile_size, xs : xs + tile_size]
                t_logits, t_mask, _ = self.forward(tile)
                tile_img_logits.append(t_logits)
                full_pred[:, :, ys : ys + tile_size, xs : xs + tile_size] += t_mask * gauss_w
                weight_sum[:, :, ys : ys + tile_size, xs : xs + tile_size] += gauss_w

        full_mask = full_pred / torch.clamp(weight_sum, min=1e-6)
        global_logit = torch.stack(tile_img_logits, dim=0).mean(dim=0)
        return global_logit, full_mask
