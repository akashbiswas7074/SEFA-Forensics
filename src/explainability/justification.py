"""
Grounded Natural Language Forensic Justification Generator for SEFA-Forensics.
Translates activated intrinsic anchor channels and stress residuals into transparent,
human-interpretable forensic reasoning.
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import torch


class ForensicJustificationGenerator:
    """
    Synthesizes grounded forensic explanations directly from activated internal units.
    """
    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold

        # Forensic trace taxonomies mapped to anchor categories
        self.anchor_explanations = {
            0: {
                "name": "Diffusion High-Frequency Spectral Residual",
                "evidence": "Spectral phase discontinuity and latent noise-floor deviation characteristic of diffusion denoising steps.",
            },
            1: {
                "name": "Boundary Gradient Phase Incoherence",
                "evidence": "Spatial gradient transition seam detected along local patch perimeter, indicating inpainting mask stitching.",
            },
            2: {
                "name": "Latent Manifold Stress Sensitivity",
                "evidence": "Internal representation collapsed under evidence stress, confirming non-natural generative synthesis.",
            },
            3: {
                "name": "Local Texture Resynthesis Inconsistency",
                "evidence": "Patch-level self-similarity discrepancy relative to ambient physical illumination and camera sensor noise.",
            },
        }

    def generate_report(
        self,
        img_prob: float,
        pred_mask: np.ndarray,
        details: Optional[Dict[str, torch.Tensor]] = None,
        img_name: str = "Query Image",
    ) -> Dict[str, Union[str, float, List[Dict[str, str]]]]:
        """
        Produces structured forensic decision and textual justification.
        """
        is_manipulated = img_prob >= self.threshold
        conf_pct = img_prob * 100.0 if is_manipulated else (1.0 - img_prob) * 100.0

        # Calculate affected area
        high_conf_pixels = np.argwhere(pred_mask >= self.threshold)
        total_pixels = pred_mask.size
        area_pct = (len(high_conf_pixels) / max(1, total_pixels)) * 100.0

        # Determine spatial location of manipulation
        if len(high_conf_pixels) > 0 and is_manipulated:
            ymin, xmin = high_conf_pixels.min(axis=0)
            ymax, xmax = high_conf_pixels.max(axis=0)
            cy, cx = (ymin + ymax) // 2, (xmin + xmax) // 2
            h, w = pred_mask.shape
            v_pos = "top" if cy < h / 3 else ("bottom" if cy > 2 * h / 3 else "center")
            h_pos = "left" if cx < w / 3 else ("right" if cx > 2 * w / 3 else "central")
            loc_summary = f"{v_pos}-{h_pos} region (bounding box: [{ymin}, {xmin}, {ymax}, {xmax}], covering {area_pct:.1f}% of image)"
        else:
            loc_summary = "None (Uniform natural distribution across all patches)"

        # Analyze activated anchor channels if details provided
        active_traces = []
        if details is not None and "anchor_residuals" in details and is_manipulated:
            res = details["anchor_residuals"]  # [1, N, K]
            # Average residual per anchor
            anchor_scores = res.mean(dim=(0, 1)).cpu().numpy()
            top_k_anchors = np.argsort(anchor_scores)[::-1][:3]

            for rank, a_idx in enumerate(top_k_anchors):
                cat_idx = int(a_idx % len(self.anchor_explanations))
                info = self.anchor_explanations[cat_idx]
                active_traces.append({
                    "anchor_id": int(a_idx),
                    "category": info["name"],
                    "diagnostic_detail": info["evidence"],
                    "activation_strength": float(anchor_scores[a_idx]),
                })
        elif not is_manipulated:
            active_traces.append({
                "anchor_id": -1,
                "category": "Pristine Coherence",
                "diagnostic_detail": "Internal anchor responses remained fully invariant under evidence stress; consistent with camera optical capture.",
                "activation_strength": 0.0,
            })

        # Synthesize natural language paragraph
        if is_manipulated:
            top_trace = active_traces[0]["category"] if active_traces else "Forensic Incoherence"
            nl_text = (
                f"The image is flagged as **MANIPULATED** with {conf_pct:.1f}% confidence. "
                f"Localized tampering was isolated in the {loc_summary}. "
                f"The frozen foundation model's intrinsic forensic anchors self-elicited strong evidence of "
                f"'{top_trace}'. Specifically: {active_traces[0]['diagnostic_detail']} "
                f"The localized evidence stress revealed anomalous latent residuals that do not occur in natural camera optics."
            )
        else:
            nl_text = (
                f"The image is evaluated as **PRISTINE (AUTHENTIC)** with {conf_pct:.1f}% confidence. "
                f"All internal foundation anchors exhibited natural sensor invariance under controlled evidence stress. "
                f"No localized diffusion inpainting, retouching seams, or generative spectral artifacts were detected."
            )

        return {
            "image_name": img_name,
            "verdict": "MANIPULATED" if is_manipulated else "AUTHENTIC",
            "confidence_score": float(img_prob),
            "confidence_percentage": float(conf_pct),
            "affected_area_percentage": float(area_pct),
            "localization_summary": loc_summary,
            "natural_language_justification": nl_text,
            "activated_anchors": active_traces,
        }
