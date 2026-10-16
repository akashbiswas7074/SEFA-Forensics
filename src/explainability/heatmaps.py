"""
Forensic Localization Heatmap Generation for SEFA-Forensics.
Produces high-resolution visual overlays and detection bounding boxes.
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from PIL import Image


def generate_forensic_heatmap(
    pred_mask: np.ndarray,
    colormap_name: str = "jet",
) -> np.ndarray:
    """
    Transforms a single-channel probability map [H, W] into an RGB heatmap [H, W, 3].
    """
    mask_clipped = np.clip(pred_mask, 0.0, 1.0)
    cmap = plt.get_cmap(colormap_name)
    rgba = cmap(mask_clipped)
    rgb = (rgba[..., :3] * 255.0).astype(np.uint8)
    return rgb


def visualize_forensic_overlay(
    image: Union[np.ndarray, Image.Image],
    pred_mask: np.ndarray,
    alpha: float = 0.45,
    threshold: float = 0.4,
    colormap_name: str = "jet",
) -> Tuple[np.ndarray, List[Dict[str, int]]]:
    """
    Superimposes the forensic localization heatmap onto the input RGB image,
    and calculates bounding boxes of localized suspicious regions.
    Returns:
        blended_image: [H, W, 3] RGB uint8 image
        detected_boxes: List of bounding boxes [ymin, xmin, ymax, xmax]
    """
    if isinstance(image, Image.Image):
        img_np = np.array(image).astype(np.float32)
    else:
        img_np = image.astype(np.float32)

    h, w = img_np.shape[:2]
    # Resize mask if shapes differ
    if pred_mask.shape != (h, w):
        mask_pil = Image.fromarray((pred_mask * 255).astype(np.uint8))
        mask_pil = mask_pil.resize((w, h), Image.Resampling.BILINEAR)
        pred_mask = np.array(mask_pil).astype(np.float32) / 255.0

    heatmap = generate_forensic_heatmap(pred_mask, colormap_name=colormap_name).astype(np.float32)

    # Highlight regions where probability exceeds threshold
    high_conf = (pred_mask >= threshold)[:, :, np.newaxis]
    blended = np.where(high_conf, (1.0 - alpha) * img_np + alpha * heatmap, img_np)
    blended = np.clip(blended, 0.0, 255.0).astype(np.uint8)

    # Find connected components / bounding boxes of suspicious areas
    boxes = []
    suspicious_pixels = np.argwhere(pred_mask >= threshold)
    if len(suspicious_pixels) > 20:  # Minimal area filter
        ymin, xmin = suspicious_pixels.min(axis=0)
        ymax, xmax = suspicious_pixels.max(axis=0)
        boxes.append({
            "ymin": int(ymin),
            "xmin": int(xmin),
            "ymax": int(ymax),
            "xmax": int(xmax),
            "area_pct": float(len(suspicious_pixels) / (h * w) * 100.0),
        })

    return blended, boxes
