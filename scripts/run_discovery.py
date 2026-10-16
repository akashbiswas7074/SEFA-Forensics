#!/usr/bin/env python3
"""
CLI script to run Anchor Discovery on calibration pairs.
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


def parse_args():
    parser = argparse.ArgumentParser(description="SEFA Anchor Discovery")
    parser.add_argument("--config", type=str, default="configs/default_config.yaml", help="Path to config file")
    parser.add_argument("--data_dir", type=str, default="data/raw", help="Directory with real/forged pairs")
    parser.add_argument("--num_anchors", type=int, default=64, help="Total anchors to uncover")
    parser.add_argument("--save_path", type=str, default="checkpoints/calibrated_anchors.pt", help="Output file")
    return parser.parse_args()


def main():
    args = parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Running on device: {device}")

    # Build model
    model = SEFAModel(
        backbone_name=cfg["model"]["backbone_name"],
        pretrained=cfg["model"]["pretrained"],
        img_size=cfg["model"]["img_size"],
        target_layers=cfg["model"]["target_layers"],
        num_anchors=args.num_anchors,
        router_hidden_dim=cfg["model"]["router_hidden_dim"],
        stress_strength=cfg["evidence_stress"]["stress_strength"],
    )

    # Calibration dataset
    real_dir = os.path.join(args.data_dir, "real")
    forged_dir = os.path.join(args.data_dir, "forged")
    masks_dir = os.path.join(args.data_dir, "masks")

    dataset = ForensicDataset(
        real_dir=real_dir,
        forged_dir=forged_dir,
        masks_dir=masks_dir,
        img_size=cfg["model"]["img_size"],
    )

    if len(dataset) == 0:
        print("[!] No data found. Run data downloader first or use synthetic generation.")
        return

    loader = DataLoader(dataset, batch_size=cfg["data"]["batch_size"], shuffle=False, num_workers=0)

    # Discovery engine
    engine = AnchorDiscoveryEngine(model=model, device=device)
    results = engine.discover_anchors(loader, verbose=True)

    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    torch.save(results, args.save_path)
    print(f"[+] Saved calibrated anchors to '{args.save_path}'.")


if __name__ == "__main__":
    main()
