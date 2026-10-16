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


def compute_pixel_f1(
    pred_masks: np.ndarray,
    gt_masks: np.ndarray,
    threshold: float = 0.5,
    smooth: float = 1e-6,
) -> Dict[str, float]:
    """
    Computes global pixel-level precision, recall, and F1-score across all mask pixels.
    pred_masks: [N, H, W] in [0, 1]
    gt_masks: [N, H, W] binary in {0, 1}
    """
    bin_preds = (pred_masks >= threshold).astype(bool).ravel()
    bin_gts = (gt_masks > 0.5).astype(bool).ravel()

    tp = np.logical_and(bin_preds, bin_gts).sum()
    fp = np.logical_and(bin_preds, ~bin_gts).sum()
    fn = np.logical_and(~bin_preds, bin_gts).sum()

    precision = float((tp + smooth) / (tp + fp + smooth))
    recall = float((tp + smooth) / (tp + fn + smooth))
    pixel_f1 = float((2.0 * tp + smooth) / (2.0 * tp + fp + fn + smooth))

    return {
        "pixel_precision": precision,
        "pixel_recall": recall,
        "pixel_f1": pixel_f1,
    }


def compute_boundary_f1(
    pred_masks: np.ndarray,
    gt_masks: np.ndarray,
    threshold: float = 0.5,
    dilation_radius: int = 3,
    smooth: float = 1e-6,
) -> Dict[str, float]:
    """
    Computes Boundary-F1 score with a boundary tolerance buffer epsilon (dilation radius).
    Standard protocol in CVPR/ICCV forensics benchmarks (TruFor, CAT-Net).
    pred_masks: [N, H, W] in [0, 1]
    gt_masks: [N, H, W] in {0, 1}
    """
    from scipy.ndimage import binary_dilation, binary_erosion

    bin_preds = (pred_masks >= threshold).astype(bool)
    bin_gts = (gt_masks > 0.5).astype(bool)

    struct = np.ones((3, 3), dtype=bool)

    f1_scores = []
    precisions = []
    recalls = []

    for i in range(len(bin_gts)):
        gt_i = bin_gts[i]
        pred_i = bin_preds[i]

        # Skip sample if no forgery exists in ground truth
        if not np.any(gt_i):
            continue

        # Extract 1-pixel boundary
        gt_bound = np.logical_xor(gt_i, binary_erosion(gt_i, structure=struct))
        pred_bound = np.logical_xor(pred_i, binary_erosion(pred_i, structure=struct))

        if not np.any(gt_bound) and not np.any(pred_bound):
            f1_scores.append(1.0)
            continue
        if not np.any(gt_bound) or not np.any(pred_bound):
            f1_scores.append(0.0)
            continue

        # Dilate for tolerance
        gt_dil = binary_dilation(gt_bound, structure=struct, iterations=dilation_radius)
        pred_dil = binary_dilation(pred_bound, structure=struct, iterations=dilation_radius)

        # Precision: pred boundary pixels matching within dilated GT boundary
        tp_prec = np.logical_and(pred_bound, gt_dil).sum()
        p = (tp_prec + smooth) / (pred_bound.sum() + smooth)

        # Recall: GT boundary pixels matching within dilated pred boundary
        tp_rec = np.logical_and(gt_bound, pred_dil).sum()
        r = (tp_rec + smooth) / (gt_bound.sum() + smooth)

        f1 = (2.0 * p * r + smooth) / (p + r + smooth)

        precisions.append(p)
        recalls.append(r)
        f1_scores.append(f1)

    mean_f1 = float(np.mean(f1_scores)) if len(f1_scores) > 0 else 0.0
    mean_prec = float(np.mean(precisions)) if len(precisions) > 0 else 0.0
    mean_rec = float(np.mean(recalls)) if len(recalls) > 0 else 0.0

    return {
        "boundary_f1": mean_f1,
        "boundary_precision": mean_prec,
        "boundary_recall": mean_rec,
    }
