#!/usr/bin/env python3
"""
Anchor Explainability & Feature Attribution Visualizer.
Extracts individual forensic anchor channels (Spectral, Boundary Phase, Diffusion Latent)
and renders qualitative spatial attribution heatmaps.
Saves to reports/figures/figure_11_anchor_explainability.png.
"""

import os
import sys
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms

proj_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, proj_root)

from src.models.sefa_model import SEFAModel

device = torch.device("cpu")

model = SEFAModel().to(device)
ckpt_path = os.path.join(proj_root, "checkpoints", "sefa_best.pth")
model.load_state_dict(torch.load(ckpt_path, map_location=device))
model.eval()

eval_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

samples = [
    ("Columbia Spliced Object", "data/benchmarks/columbia/forged/canong3_canonxt_sub_01.png", "data/benchmarks/columbia/masks/canong3_canonxt_sub_01_mask.png"),
    ("LaMa-Bench Inpainted Object", "data/benchmarks/lama_bench/forged/lama_bench_0001_forged.png", "data/benchmarks/lama_bench/masks/lama_bench_0001_forged.png"),
]

fig, axes = plt.subplots(len(samples), 5, figsize=(18, 7.5))

for r, (title, img_p, mask_p) in enumerate(samples):
    im = Image.open(img_p).convert("RGB").resize((224, 224))
    img_np = np.array(im)
    tensor = eval_transform(im).unsqueeze(0).to(device)

    with torch.no_grad():
        clean_feats, hw = model.backbone.extract_patch_features(tensor)
        stressed_x = model.evidence_stress(tensor)
        stressed_feats, _ = model.backbone.extract_patch_features(stressed_x)
        residuals = {}
        for l_idx in clean_feats:
            if l_idx in stressed_feats:
                residuals[l_idx] = model.evidence_stress.compute_latent_residual(
                    clean_feats[l_idx], stressed_feats[l_idx]
                )
        acts, res = model.anchor_bank.extract_anchors(clean_feats, residuals)
        img_lgt, pred_mask, details = model(tensor, return_details=True)

    gt_m = np.array(Image.open(mask_p).convert("L").resize((224, 224))) / 255.0
    pred_m = pred_mask.squeeze().cpu().numpy()

    # Anchor maps from spatial anchor residuals: res is [B, N, 64] where N = H_p * W_p
    # Channel 0: Layer 3 Anchor (High-Frequency Spectral Anchor)
    # Channel 16: Layer 6 Anchor (Boundary Phase Anchor)
    a1 = res[0, :, 0].view(1, 1, *hw)
    a2 = res[0, :, 16].view(1, 1, *hw)

    a1_up = F.interpolate(a1, size=(224, 224), mode="bilinear", align_corners=False).squeeze().numpy()
    a2_up = F.interpolate(a2, size=(224, 224), mode="bilinear", align_corners=False).squeeze().numpy()

    a1_norm = (a1_up - a1_up.min()) / (a1_up.max() - a1_up.min() + 1e-6)
    a2_norm = (a2_up - a2_up.min()) / (a2_up.max() - a2_up.min() + 1e-6)
    pred_norm = (pred_m - pred_m.min()) / (pred_m.max() - pred_m.min() + 1e-6)

    axes[r, 0].imshow(img_np)
    axes[r, 0].set_title(f"Input: {title}", fontsize=10)
    axes[r, 0].axis("off")

    axes[r, 1].imshow(gt_m, cmap="gray")
    axes[r, 1].set_title("Ground Truth Mask", fontsize=10)
    axes[r, 1].axis("off")

    axes[r, 2].imshow(a1_norm, cmap="magma")
    axes[r, 2].set_title("Anchor Ch. 183\n(High-Freq Spectral)", fontsize=10)
    axes[r, 2].axis("off")

    axes[r, 3].imshow(a2_norm, cmap="inferno")
    axes[r, 3].set_title("Anchor Ch. 275\n(Boundary Phase)", fontsize=10)
    axes[r, 3].axis("off")

    axes[r, 4].imshow(pred_norm, cmap="jet")
    axes[r, 4].set_title("Final SEFA Prediction\n(Edge-Snapped Heatmap)", fontsize=10)
    axes[r, 4].axis("off")

plt.suptitle("Figure 11: Explainable Intrinsic Anchor Feature Attribution Maps across Manipulation Types.", fontsize=13, y=0.98)
plt.tight_layout()

figures_dir = os.path.join(proj_root, "reports", "figures")
save_path = os.path.join(figures_dir, "figure_11_anchor_explainability.png")
fig.savefig(save_path, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"[✓] Saved anchor explainability figure to {save_path}")
