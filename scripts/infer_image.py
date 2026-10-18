#!/usr/bin/env python3
"""
SEFA-Forensics: Standalone Real-Time Image Inference CLI.
Evaluates any input image, predicts manipulation probability,
computes 224x224 progressive pixel-level localization heatmaps,
and saves the visual overlay.

Usage:
    python scripts/infer_image.py --image path/to/image.png
    python scripts/infer_image.py --image path/to/image.png --output reports/inference/output.png
    python scripts/infer_image.py --dir path/to/folder/
"""

import os
import sys
import argparse
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as T
import matplotlib.pyplot as plt

current_dir = os.path.dirname(os.path.abspath(__file__))
proj_root = os.path.abspath(os.path.join(current_dir, ".."))
if proj_root not in sys.path:
    sys.path.insert(0, proj_root)

from src.models.sefa_model import SEFAModel
from src.explainability.heatmaps import visualize_forensic_overlay


def load_model(ckpt_path: str = "checkpoints/sefa_best.pth", device: torch.device = None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SEFAModel(
        backbone_name="dinov2_vits14",
        pretrained=True,
        img_size=224,
        target_layers=[3, 6, 9, 11],
        num_anchors=64,
        router_hidden_dim=64,
    ).to(device)
    full_ckpt = os.path.join(proj_root, ckpt_path) if not os.path.isabs(ckpt_path) else ckpt_path
    if os.path.exists(full_ckpt):
        model.load_state_dict(torch.load(full_ckpt, map_location=device))
        print(f"[✓] Loaded verified SEFA checkpoint: {full_ckpt}")
    else:
        print(f"[!] Warning: Checkpoint not found at {full_ckpt}, using initial weights.")
    model.eval()
    return model, device


def infer_single_image(
    image_path: str,
    model: torch.nn.Module,
    device: torch.device,
    output_path: str = None,
    tau: float = 0.280,
):
    if not os.path.exists(image_path):
        print(f"[!] Error: Image not found at {image_path}")
        return None

    im = Image.open(image_path).convert("RGB")
    orig_w, orig_h = im.size
    im_resized = im.resize((224, 224))
    img_np = np.array(im_resized)

    transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    tensor = transform(im_resized).unsqueeze(0).to(device)

    with torch.no_grad():
        logit, mask_pred, details = model(tensor, return_details=True)
        prob = torch.sigmoid(logit).item()
        p_map = mask_pred.squeeze().cpu().numpy()

    # Dynamic range normalization
    if prob >= tau:
        mask_np = (p_map - p_map.min()) / (p_map.max() - p_map.min() + 1e-6)
        label = "MANIPULATED"
    else:
        mask_np = (p_map - p_map.min()) / (p_map.max() - p_map.min() + 1e-6) * (prob / tau)
        label = "AUTHENTIC"

    overlay, _ = visualize_forensic_overlay(img_np, mask_np, alpha=0.55, threshold=0.35)

    print("=" * 60)
    print(f" SEFA-Forensics Diagnostic: {os.path.basename(image_path)}")
    print("=" * 60)
    print(f"  Classification:      {label}")
    print(f"  Tampering Prob:      {prob:.4f} (Decision tau* = {tau:.3f})")
    print(f"  Mask Peak Hotspot:   {mask_np.max():.4f} (mean: {mask_np.mean():.4f})")
    print(f"  Resolution:          {orig_w}x{orig_h} (processed at 224x224)")
    print("=" * 60)

    if output_path:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        axes[0].imshow(img_np)
        axes[0].set_title(f"Input Image ({label})", fontsize=10)
        axes[0].axis("off")

        im_heat = axes[1].imshow(mask_np, cmap="jet", vmin=0, vmax=1)
        axes[1].set_title(f"Predicted Heatmap (P={prob:.3f})", fontsize=10)
        axes[1].axis("off")
        plt.colorbar(im_heat, ax=axes[1], fraction=0.046, pad=0.04)

        axes[2].imshow(overlay)
        axes[2].set_title("Forensic Overlay", fontsize=10)
        axes[2].axis("off")

        plt.tight_layout()
        plt.savefig(output_path, dpi=200, bbox_inches="tight")
        plt.close()
        print(f"[✓] Saved diagnostic figure to: {output_path}")

    return {"label": label, "prob": prob, "mask": mask_np}


def main():
    parser = argparse.ArgumentParser(description="SEFA-Forensics Single & Batch Inference CLI")
    parser.add_argument("--image", type=str, help="Path to input image file")
    parser.add_argument("--dir", type=str, help="Path to directory of images to process")
    parser.add_argument("--output", type=str, default=None, help="Path to save output visualization figure")
    parser.add_argument("--tau", type=float, default=0.280, help="Operating threshold for classification (default: 0.280)")
    parser.add_argument("--ckpt", type=str, default="checkpoints/sefa_best.pth", help="Checkpoint path")
    args = parser.parse_args()

    model, device = load_model(args.ckpt)

    if args.image:
        out = args.output or os.path.join(proj_root, "reports", "inference", f"diag_{os.path.splitext(os.path.basename(args.image))[0]}.png")
        infer_single_image(args.image, model, device, output_path=out, tau=args.tau)
    elif args.dir:
        files = [
            os.path.join(args.dir, f) for f in os.listdir(args.dir)
            if f.lower().endswith((".png", ".jpg", ".jpeg", ".tif", ".bmp", ".webp"))
        ]
        print(f"[*] Processing {len(files)} images in directory: {args.dir}")
        for i, f in enumerate(files[:20]):
            out = os.path.join(proj_root, "reports", "inference", f"diag_{i}_{os.path.splitext(os.path.basename(f))[0]}.png")
            infer_single_image(f, model, device, output_path=out, tau=args.tau)
    else:
        # Default test on Columbia Spliced sample
        sample_path = os.path.join(proj_root, "data", "benchmarks", "columbia", "forged", "columbia_0019_forged.png")
        if os.path.exists(sample_path):
            print("[*] No --image argument provided. Testing default benchmark image...")
            out = os.path.join(proj_root, "reports", "inference", "demo_columbia_spliced.png")
            infer_single_image(sample_path, model, device, output_path=out, tau=args.tau)
        else:
            parser.print_help()


if __name__ == "__main__":
    main()
