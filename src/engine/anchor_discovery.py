"""
Anchor Discovery Engine for SEFA-Forensics.
Analyzes internal layer responses under evidence stress across a small calibration set
to identify the sparse units with maximal intrinsic forgery sensitivity.
"""

from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..models.sefa_model import SEFAModel


class AnchorDiscoveryEngine:
    """
    Calibrates and uncovers latent forensic anchors within the frozen foundation model.
    Computes Fisher Discriminant Ratio / Welch t-statistic on stress residuals.
    """
    def __init__(
        self,
        model: SEFAModel,
        device: torch.device = torch.device("cpu"),
    ):
        self.model = model
        self.device = device
        self.model.to(self.device)

    def discover_anchors(
        self,
        calibration_loader: DataLoader,
        verbose: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """
        Runs anchor discovery on calibration data pairs.
        Returns:
            calibrated_indices: [num_layers, anchors_per_layer]
            calibrated_weights: [num_layers, anchors_per_layer]
        """
        self.model.eval()
        target_layers = self.model.target_layers
        feature_dim = self.model.backbone.feature_dim
        anchors_per_layer = self.model.anchor_bank.anchors_per_layer

        # Accumulators for real and forged residuals per layer
        real_residuals: Dict[int, List[torch.Tensor]] = {l: [] for l in target_layers}
        forged_residuals: Dict[int, List[torch.Tensor]] = {l: [] for l in target_layers}

        if verbose:
            print(f"[*] Discovering intrinsic forensic anchors across layers {target_layers}...")

        with torch.no_grad():
            for batch in tqdm(calibration_loader, desc="Calibration", disable=not verbose):
                images = batch["image"].to(self.device)
                labels = batch["label"].to(self.device)  # 0: real, 1: forged

                # Clean features
                clean_feats, _ = self.model.backbone.extract_patch_features(images)

                # Stressed features
                stressed_images = self.model.evidence_stress(images)
                stressed_feats, _ = self.model.backbone.extract_patch_features(stressed_images)

                for l_idx in target_layers:
                    if l_idx not in clean_feats or l_idx not in stressed_feats:
                        continue
                    # Compute residual Delta Z = |Z - Z_stress|: [B, N, D]
                    res = self.model.evidence_stress.compute_latent_residual(
                        clean_feats[l_idx], stressed_feats[l_idx]
                    )
                    # Isolated localized forgery response: top 12% peak patches + spatial variance
                    k_p = max(1, int(0.12 * res.shape[1]))
                    top_res = torch.topk(res, k=k_p, dim=1)[0].mean(dim=1)
                    res_sig = 0.7 * top_res + 0.3 * res.std(dim=1)

                    real_mask = (labels == 0)
                    forged_mask = (labels == 1)

                    if real_mask.any():
                        real_residuals[l_idx].append(res_sig[real_mask].cpu())
                    if forged_mask.any():
                        forged_residuals[l_idx].append(res_sig[forged_mask].cpu())

        calibrated_indices_list = []
        calibrated_weights_list = []
        layer_stats = {}

        for idx, l_idx in enumerate(target_layers):
            if not real_residuals[l_idx] or not forged_residuals[l_idx]:
                # Fallback to evenly spaced channels if calibration missing
                chosen_idx = torch.linspace(0, feature_dim - 1, anchors_per_layer).long()
                chosen_weights = torch.ones(anchors_per_layer)
            else:
                real_all = torch.cat(real_residuals[l_idx], dim=0)    # [N_real, D]
                forged_all = torch.cat(forged_residuals[l_idx], dim=0)  # [N_forged, D]

                mu_real = real_all.mean(dim=0)
                var_real = real_all.var(dim=0, unbiased=False) + 1e-6

                mu_forged = forged_all.mean(dim=0)
                var_forged = forged_all.var(dim=0, unbiased=False) + 1e-6

                # Intrinsic Forgery Sensitivity Score (IFSS)
                # Welch's t-statistic / Fisher ratio for channel discrimination
                ifss = torch.abs(mu_forged - mu_real) / torch.sqrt(var_real + var_forged)

                # Select top-k channels with highest sensitivity
                topk_scores, topk_idx = torch.topk(ifss, k=anchors_per_layer)
                chosen_idx = topk_idx.cpu()
                chosen_weights = (topk_scores / (topk_scores.max() + 1e-6)).cpu()

                layer_stats[l_idx] = {
                    "mean_ifss": float(topk_scores.mean()),
                    "max_ifss": float(topk_scores.max()),
                    "top_channels": chosen_idx.tolist()[:5],
                }

            calibrated_indices_list.append(chosen_idx)
            calibrated_weights_list.append(chosen_weights)

        calibrated_indices = torch.stack(calibrated_indices_list).to(self.device)
        calibrated_weights = torch.stack(calibrated_weights_list).to(self.device)

        # Update the model's AnchorBank directly
        self.model.anchor_bank.update_anchors(calibrated_indices, calibrated_weights)

        if verbose:
            print("[+] Intrinsic Forensic Anchors successfully discovered and calibrated:")
            for l_idx, stats in layer_stats.items():
                print(f"  -> Layer {l_idx:2d}: Mean IFSS={stats['mean_ifss']:.3f}, Max={stats['max_ifss']:.3f}, Anchors={stats['top_channels']}")

        return {
            "indices": calibrated_indices,
            "weights": calibrated_weights,
            "stats": layer_stats,
        }
