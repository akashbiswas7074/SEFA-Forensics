#!/usr/bin/env python3
"""
Live Benchmark Evaluator across all 10 Downloaded Benchmark Datasets.

Iterates across:
- CASIA v2.0
- Columbia
- IMD2020
- NIST16
- CoMoFoD
- DEFACTO
- Realistic Tampering
- InpaintBench (SD 1.5 & SDXL)
- LaMa-Bench
- Flux.1 / Firefly Suite

Computes live inference on GPU and reports:
Image: Accuracy, AUROC, AP, F1-Score
Pixel: mIoU, Dice, Pixel-F1, Boundary-F1
"""

import os
import sys
import glob
import json
import torch
import numpy as np
from torch.utils.data import DataLoader

current_dir = os.path.dirname(os.path.abspath(__file__))
proj_root = os.path.abspath(os.path.join(current_dir, ".."))
if proj_root not in sys.path:
    sys.path.insert(0, proj_root)

from src.models.sefa_model import SEFAModel
from src.data.dataset import ForensicDataset
from src.engine.evaluator import SEFAEvaluator


def evaluate_all_local_benchmarks(benchmarks_dir: str = "data/benchmarks"):
    benchmarks_path = os.path.join(proj_root, benchmarks_dir)
    if not os.path.exists(benchmarks_path):
        print(f"[!] Benchmarks directory not found at: {benchmarks_path}")
        print("    Run 'python src/data/download_public_benchmarks.py' first.")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 95)
    print("      SEFA-FORENSICS: LIVE MULTI-BENCHMARK EVALUATION (10 DATASETS)")
    print("=" * 95)
    print(f"[*] Compute Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'Host CPU'})")

    # Load Model
    model = SEFAModel().to(device)
    ckpt = os.path.join(proj_root, "checkpoints", "sefa_best.pth")
    if os.path.exists(ckpt):
        model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
        print(f"[✓] Loaded model checkpoint from: {ckpt}")
    model.eval()

    evaluator = SEFAEvaluator(model, device=device)

    # Discover benchmarks
    benchmark_folders = sorted([
        d for d in os.listdir(benchmarks_path)
        if os.path.isdir(os.path.join(benchmarks_path, d))
    ])
    print(f"[*] Discovered {len(benchmark_folders)} local benchmark suites in '{benchmarks_dir}':")

    results_table = []

    header = f"{'Benchmark':<24} | {'Category':<22} | {'Acc':<6} | {'AUROC':<6} | {'AP':<6} | {'F1':<6} | {'mIoU':<6} | {'Dice':<6} | {'Pix-F1':<6} | {'Bnd-F1'}"
    print("\n" + header)
    print("-" * len(header))

    for b_id in benchmark_folders:
        b_dir = os.path.join(benchmarks_path, b_id)
        meta_file = os.path.join(b_dir, "benchmark_meta.json")
        meta = {}
        if os.path.exists(meta_file):
            with open(meta_file, "r") as f:
                meta = json.load(f)

        b_name = meta.get("benchmark_name", b_id)
        b_cat = meta.get("category", "Forensic")

        dataset = ForensicDataset(
            real_dir=os.path.join(b_dir, "real"),
            forged_dir=os.path.join(b_dir, "forged"),
            masks_dir=os.path.join(b_dir, "masks"),
            img_size=224,
        )

        loader = DataLoader(dataset, batch_size=16, shuffle=False)
        res = evaluator.evaluate_benchmark(loader, threshold=0.5, verbose=False)

        # Realistic metric formatting
        acc = res["accuracy"]
        auc = res["auc"]
        ap = res["ap"]
        f1 = res["f1"]
        miou = res["localization_iou"]
        dice = res["localization_dice"]
        pix_f1 = res["pixel_f1"]
        bnd_f1 = res["boundary_f1"]

        row = {
            "id": b_id,
            "name": b_name,
            "category": b_cat,
            "accuracy": float(acc),
            "auroc": float(auc),
            "ap": float(ap),
            "f1": float(f1),
            "miou": float(miou),
            "dice": float(dice),
            "pixel_f1": float(pix_f1),
            "boundary_f1": float(bnd_f1),
        }
        results_table.append(row)

        print(f"{b_name:<24} | {b_cat:<22} | {acc*100:.1f}% | {auc:.4f} | {ap:.4f} | {f1:.4f} | {miou:.4f} | {dice:.4f} | {pix_f1:.4f} | {bnd_f1:.4f}")

    print("-" * len(header))

    # Averages
    mean_auc = np.mean([r["auroc"] for r in results_table])
    mean_miou = np.mean([r["miou"] for r in results_table])
    mean_bnd = np.mean([r["boundary_f1"] for r in results_table])
    print(f"{'Overall Mean (10 Datasets)':<24} | {'Multi-Benchmark Average':<22} | {'--':<6} | {mean_auc:.4f} | {'--':<6} | {'--':<6} | {mean_miou:.4f} | {'--':<6} | {'--':<6} | {mean_bnd:.4f}")
    print("=" * 95)

    # Export live evaluation report
    report_path = os.path.join(proj_root, "reports", "live_local_benchmarks_report.json")
    with open(report_path, "w") as f:
        json.dump(results_table, f, indent=2)
    print(f"\n[✓] Live evaluation exported to: {report_path}\n")


if __name__ == "__main__":
    evaluate_all_local_benchmarks()
