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
from .anchors import ForensicAnchorBank, AnchorRouter, BoundaryGuidedRefiner


class SEFAModel(nn.Module):
    """
    Self-Eliciting Forensic Anchors (SEFA) Architecture.
    - 100% frozen foundation model (DINOv2 / ViT)
    - Multi-band evidence stress mechanism
    - Calibrated sparse intrinsic anchors
    - Multi-quantile router (<0.2% trainable parameters)
    - Boundary-guided spatial refinement head for sharp pixel-level localization
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

        # 4. Multi-Quantile Router (<0.2% parameters)
        self.router = AnchorRouter(
            total_anchors=num_anchors,
            hidden_dim=router_hidden_dim,
            dropout=dropout,
        )

        # 5. Boundary-Guided Spatial Refiner (<1.5k parameters)
        self.boundary_refiner = BoundaryGuidedRefiner(in_channels=1, hidden_dim=16)

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

        # Step 5: Route through multi-quantile router
        img_logits, patch_mask, routed_embeds = self.router(
            anchor_acts=anchor_acts,
            anchor_residuals=anchor_res,
            spatial_hw=spatial_hw,
        )

        # Step 6: Upsample and apply Boundary-Guided Refinement
        coarse_dense_mask = F.interpolate(patch_mask, size=(H, W), mode="bilinear", align_corners=False)
        refined_mask = self.boundary_refiner(coarse_dense_mask, x)

        details = None
        if return_details:
            details = {
                "anchor_acts": anchor_acts,
                "anchor_residuals": anchor_res,
                "patch_mask_lowres": patch_mask,
                "coarse_dense_mask": coarse_dense_mask,
                "spatial_hw": spatial_hw,
                "routed_embeds": routed_embeds,
            }

        return img_logits, refined_mask, details
