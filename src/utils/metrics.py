"""
Evaluation Metrics for SEFA-Forensics.
Computes image-level detection metrics (AUROC, AP, F1) and pixel-level localization metrics (mIoU, F1).
"""

from typing import Dict, Tuple, Union
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_fscore_support, roc_curve


def compute_classification_metrics(
    y_true: Union[np.ndarray, list],
    y_scores: Union[np.ndarray, list],
    threshold: float = 0.5,
) -> Dict[str, float]:
    """
    Computes image-level forgery detection metrics.
    """
    y_true = np.array(y_true).astype(int)
    y_scores = np.array(y_scores).astype(float)

    # Handle single class edge case
    if len(np.unique(y_true)) < 2:
        return {
            "auc": 0.5,
            "ap": 0.5,
            "f1": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "accuracy": float(np.mean((y_scores >= threshold) == y_true)),
        }

    auc = roc_auc_score(y_true, y_scores)
    ap = average_precision_score(y_true, y_scores)

    y_pred = (y_scores >= threshold).astype(int)
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
    acc = np.mean(y_pred == y_true)

    return {
        "auc": float(auc),
        "ap": float(ap),
        "f1": float(f1),
        "precision": float(prec),
        "recall": float(rec),
        "accuracy": float(acc),
    }


def compute_localization_iou(
    pred_masks: np.ndarray,
    gt_masks: np.ndarray,
    threshold: float = 0.5,
    smooth: float = 1e-6,
) -> Dict[str, float]:
    """
    Computes pixel-level localization Intersection over Union (IoU) and Dice coefficient.
    pred_masks: [N, H, W] in [0, 1]
    gt_masks: [N, H, W] binary in {0, 1}
    """
    bin_preds = (pred_masks >= threshold).astype(np.float32)
    bin_gts = (gt_masks > 0.5).astype(np.float32)

    intersection = (bin_preds * bin_gts).sum(axis=(-2, -1))
    union = bin_preds.sum(axis=(-2, -1)) + bin_gts.sum(axis=(-2, -1)) - intersection

    # Calculate IoU per sample where ground truth is forged (gt contains at least 1 pixel)
    is_forged = bin_gts.sum(axis=(-2, -1)) > 0

    if np.any(is_forged):
        ious = (intersection[is_forged] + smooth) / (union[is_forged] + smooth)
        mean_iou = float(np.mean(ious))
        dices = (2.0 * intersection[is_forged] + smooth) / (bin_preds[is_forged].sum(axis=(-2, -1)) + bin_gts[is_forged].sum(axis=(-2, -1)) + smooth)
        mean_dice = float(np.mean(dices))
    else:
        mean_iou = 0.0
        mean_dice = 0.0

    return {
        "localization_iou": mean_iou,
        "localization_dice": mean_dice,
    }
