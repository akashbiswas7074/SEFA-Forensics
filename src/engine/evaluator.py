"""
Comprehensive Evaluator for SEFA-Forensics.
Evaluates both image-level manipulation classification and dense pixel-level localization.
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..models.sefa_model import SEFAModel
from ..utils.metrics import compute_classification_metrics, compute_localization_iou


class SEFAEvaluator:
    """
    Evaluates SEFAModel across test benchmarks.
    """
    def __init__(
        self,
        model: SEFAModel,
        device: torch.device = torch.device("cpu"),
    ):
        self.model = model
        self.device = device
        self.model.to(self.device)

    @torch.no_grad()
    def evaluate_benchmark(
        self,
        dataloader: DataLoader,
        threshold: float = 0.5,
        verbose: bool = True,
    ) -> Dict[str, float]:
        self.model.eval()

        all_y_true = []
        all_y_scores = []
        all_pred_masks = []
        all_gt_masks = []

        if verbose:
            print("[*] Evaluating SEFA-Forensics benchmark...")

        for batch in tqdm(dataloader, desc="Evaluating", disable=not verbose):
            images = batch["image"].to(self.device)
            labels = batch["label"]
            masks = batch["mask"]

            img_logits, pred_masks, _ = self.model(images)
            probs = torch.sigmoid(img_logits).cpu().numpy().flatten()

            all_y_true.extend(labels.numpy().flatten())
            all_y_scores.extend(probs)
            all_pred_masks.append(pred_masks.cpu().numpy().squeeze(1))
            all_gt_masks.append(masks.numpy().squeeze(1))

        all_pred_masks_np = np.concatenate(all_pred_masks, axis=0)
        all_gt_masks_np = np.concatenate(all_gt_masks, axis=0)

        cls_metrics = compute_classification_metrics(all_y_true, all_y_scores, threshold=threshold)
        loc_metrics = compute_localization_iou(all_pred_masks_np, all_gt_masks_np, threshold=threshold)

        results = {**cls_metrics, **loc_metrics}

        if verbose:
            print("[+] Benchmark Evaluation Results:")
            print(f"    Image-Level AUROC : {results['auc']:.4f}")
            print(f"    Image-Level AP    : {results['ap']:.4f}")
            print(f"    Image-Level F1    : {results['f1']:.4f}")
            print(f"    Localization mIoU : {results['localization_iou']:.4f}")
            print(f"    Localization Dice : {results['localization_dice']:.4f}")

        return results
