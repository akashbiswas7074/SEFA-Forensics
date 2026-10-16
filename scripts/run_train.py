#!/usr/bin/env python3
"""
CLI script to train SEFA Anchor Router (<0.2% parameters) with
Strict Grouped Splitting (Zero Identity Leakage) and Robustness Benchmarking.
"""

import os
import sys
import yaml
import argparse
import torch
from torch.utils.data import DataLoader

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.models.sefa_model import SEFAModel
from src.data.dataset import ForensicDataset
from src.engine.anchor_discovery import AnchorDiscoveryEngine
from src.engine.trainer import SEFATrainer
from src.engine.evaluator import SEFAEvaluator


def parse_args():
    parser = argparse.ArgumentParser(description="SEFA Router Training")
    parser.add_argument("--config", type=str, default="configs/default_config.yaml", help="Path to config")
    parser.add_argument("--epochs", type=int, default=5, help="Epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    return parser.parse_args()


def main():
    args = parse_args()
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Training SEFA on device: {device}")

    # Build model with BoundaryGuidedRefiner and Multi-Band Evidence Stress
    model = SEFAModel(
        backbone_name=cfg["model"]["backbone_name"],
        pretrained=cfg["model"]["pretrained"],
        img_size=cfg["model"]["img_size"],
        target_layers=cfg["model"]["target_layers"],
        num_anchors=cfg["model"]["num_anchors"],
        router_hidden_dim=cfg["model"]["router_hidden_dim"],
        stress_strength=cfg["evidence_stress"]["stress_strength"],
    )

    data_dir = cfg["data"]["data_dir"]
    real_dir = os.path.join(data_dir, "real")
    forged_dir = os.path.join(data_dir, "forged")
    masks_dir = os.path.join(data_dir, "masks")

    dataset = ForensicDataset(
        real_dir=real_dir,
        forged_dir=forged_dir,
        masks_dir=masks_dir,
        img_size=cfg["model"]["img_size"],
    )

    if len(dataset) < 4:
        print("[!] Dataset too small. Please run data preparation first.")
        return

    # STRICT GROUP-BASED PARTITIONING (Eliminates Data Leakage)
    train_ds, val_ds, test_ds = dataset.get_grouped_splits(train_ratio=0.70, val_ratio=0.15, seed=42)

    train_loader = DataLoader(train_ds, batch_size=cfg["data"]["batch_size"], shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=cfg["data"]["batch_size"], shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=cfg["data"]["batch_size"], shuffle=False)

    print(f"[*] Strict Grouped Splits (Zero Leakage): Train={len(train_ds)}, Val={len(val_ds)}, Test={len(test_ds)}")

    # Step 1: Discover Anchors on train split
    discovery_engine = AnchorDiscoveryEngine(model=model, device=device)
    discovery_engine.discover_anchors(train_loader, verbose=True)

    # Step 2: Fast lightweight training of router
    trainer = SEFATrainer(
        model=model,
        device=device,
        lr=args.lr,
        bce_weight=cfg["training"]["bce_weight"],
        dice_weight=cfg["training"]["dice_weight"],
    )

    history = trainer.fit(train_loader=train_loader, val_loader=val_loader, epochs=args.epochs, verbose=True)

    # Step 3: Evaluate on held-out test split (Clean Baseline)
    evaluator = SEFAEvaluator(model=model, device=device)
    print("\n--- STANDARD HELD-OUT TEST BENCHMARK ---")
    clean_metrics = evaluator.evaluate_benchmark(test_loader, verbose=True)

    # Step 4: Real-World Robustness Benchmark (Simulated JPEG Q=75)
    print("\n--- REAL-WORLD ROBUSTNESS BENCHMARK (JPEG Compression Q=75) ---")
    jpeg_test_dataset = ForensicDataset(
        real_dir=real_dir,
        forged_dir=forged_dir,
        masks_dir=masks_dir,
        img_size=cfg["model"]["img_size"],
        apply_jpeg_compression=True,
        jpeg_quality=75,
    )
    _, _, jpeg_test_ds = jpeg_test_dataset.get_grouped_splits(train_ratio=0.70, val_ratio=0.15, seed=42)
    jpeg_loader = DataLoader(jpeg_test_ds, batch_size=cfg["data"]["batch_size"], shuffle=False)
    jpeg_metrics = evaluator.evaluate_benchmark(jpeg_loader, verbose=True)

    # Step 5: Sample Explainability & Grounded Diagnostic Report
    print("\n" + "=" * 75)
    print("                SAMPLE EXPLAINABLE FORENSIC INFERENCE")
    print("=" * 75)
    from src.explainability.justification import ForensicJustificationGenerator
    justification_gen = ForensicJustificationGenerator(threshold=0.5)

    test_sample = test_ds[0]
    query_tensor = test_sample["image"].unsqueeze(0).to(device)
    gt_label = int(test_sample["label"].item())

    model.eval()
    with torch.no_grad():
        img_logit, pred_mask, details = model(query_tensor, return_details=True)
        img_prob = float(torch.sigmoid(img_logit).item())
        pred_mask_np = pred_mask.squeeze().cpu().numpy()

    report = justification_gen.generate_report(
        img_prob=img_prob,
        pred_mask=pred_mask_np,
        details=details,
        img_name=os.path.basename(test_sample.get("path", "Query_Sample"))
    )

    print(f"QUERY IMAGE        : {report['image_name']} (Ground Truth: {'FORGED' if gt_label==1 else 'REAL'})")
    print(f"VERDICT            : {report['verdict']}")
    print(f"CONFIDENCE         : {report['confidence_percentage']:.2f}%")
    print(f"LOCALIZATION       : {report['localization_summary']}")
    print("-" * 75)
    print("GROUNDED JUSTIFICATION:")
    print(report['natural_language_justification'])
    print("-" * 75)
    print("ACTIVATED FORENSIC ANCHORS:")
    for a in report['activated_anchors']:
        print(f"  • [Anchor {a['anchor_id']:2d}] {a['category']} (Residual Strength: {a['activation_strength']:.3f})")
        print(f"    Evidence: {a['diagnostic_detail']}")
    print("=" * 75)


if __name__ == "__main__":
    main()
