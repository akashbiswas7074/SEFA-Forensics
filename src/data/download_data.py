"""
Automated Data Downloader and Benchmark Curator for SEFA-Forensics.
Searches and downloads genuine real vs. inpainting manipulation image pairs
with corresponding ground truth masks from the NTIRE22 Inpainting benchmark (FFHQ & ImageNet).
"""

import os
import sys
import argparse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional
from PIL import Image

BASE_HF_URL = "https://huggingface.co/datasets/afromero/NTIRE22_Inpainting/resolve/main"


def download_file(url: str, dest_path: str, timeout: int = 25) -> bool:
    """Download a file with realistic User-Agent."""
    try:
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36"}
        )
        with urllib.request.urlopen(req, timeout=timeout) as response, open(dest_path, "wb") as out_file:
            out_file.write(response.read())
        return True
    except Exception:
        return False


def download_benchmark_triplet(
    index: int,
    split_name: str,
    real_dir: str,
    forged_dir: str,
    masks_dir: str,
    target_size: int = 224,
) -> bool:
    """
    Downloads a single (real, forged, mask) triplet.
    """
    prefix = f"{index:06d}"
    real_url = f"{BASE_HF_URL}/{split_name}/test_gt/Completion/{prefix}.png"
    forged_url = f"{BASE_HF_URL}/{split_name}/test/Completion/{prefix}.png"
    mask_url = f"{BASE_HF_URL}/{split_name}/test/Completion/{prefix}_mask.png"

    dest_real = os.path.join(real_dir, f"{split_name.lower()}_{prefix}_real.png")
    dest_forged = os.path.join(forged_dir, f"{split_name.lower()}_{prefix}_forged.png")
    dest_mask = os.path.join(masks_dir, f"{split_name.lower()}_{prefix}_forged.png")

    if os.path.exists(dest_real) and os.path.exists(dest_forged) and os.path.exists(dest_mask):
        return True

    t_real = dest_real + ".tmp"
    t_forged = dest_forged + ".tmp"
    t_mask = dest_mask + ".tmp"

    if (download_file(real_url, t_real) and
        download_file(forged_url, t_forged) and
        download_file(mask_url, t_mask)):

        try:
            with Image.open(t_real).convert("RGB") as img_r:
                img_r.resize((target_size, target_size), Image.Resampling.LANCZOS).save(dest_real)
            with Image.open(t_forged).convert("RGB") as img_f:
                img_f.resize((target_size, target_size), Image.Resampling.LANCZOS).save(dest_forged)
            with Image.open(t_mask).convert("L") as img_m:
                img_m.resize((target_size, target_size), Image.Resampling.NEAREST).save(dest_mask)

            for t in [t_real, t_forged, t_mask]:
                if os.path.exists(t):
                    os.remove(t)
            return True
        except Exception:
            return False
    else:
        for t in [t_real, t_forged, t_mask]:
            if os.path.exists(t):
                os.remove(t)
        return False


def setup_dataset(
    output_base_dir: str = "data/raw",
    num_pairs_target: int = 500,
    img_size: int = 224,
    workers: int = 12,
) -> Dict[str, int]:
    """
    Downloads real genuine inpainting benchmarks across FFHQ and ImageNet splits.
    """
    real_dir = os.path.join(output_base_dir, "real")
    forged_dir = os.path.join(output_base_dir, "forged")
    masks_dir = os.path.join(output_base_dir, "masks")

    for d in [real_dir, forged_dir, masks_dir]:
        os.makedirs(d, exist_ok=True)

    print(f"[*] Starting download of {num_pairs_target} benchmark triplets into '{output_base_dir}'...")

    half = num_pairs_target // 2
    tasks = []
    successful = 0

    with ThreadPoolExecutor(max_workers=workers) as executor:
        for i in range(half):
            tasks.append(executor.submit(
                download_benchmark_triplet, i, "FFHQ", real_dir, forged_dir, masks_dir, img_size
            ))
        for i in range(half):
            tasks.append(executor.submit(
                download_benchmark_triplet, i, "ImageNet", real_dir, forged_dir, masks_dir, img_size
            ))

        for idx, future in enumerate(as_completed(tasks)):
            if future.result():
                successful += 1
            if (idx + 1) % 25 == 0 or (idx + 1) == len(tasks):
                print(f"  -> Progress: {idx + 1}/{len(tasks)} processed ({successful} valid triplets saved)")

    summary = {
        "num_real": len(os.listdir(real_dir)),
        "num_forged": len(os.listdir(forged_dir)),
        "num_masks": len(os.listdir(masks_dir)),
    }
    print(f"[+] Complete! Current local benchmark count: {summary}")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download NTIRE22 Inpainting Benchmark")
    parser.add_argument("--num_pairs", type=int, default=500, help="Number of real/forged/mask triplets to download")
    parser.add_argument("--workers", type=int, default=12, help="Parallel download workers")
    parser.add_argument("--data_dir", type=str, default="data/raw", help="Output directory")
    args = parser.parse_args()

    setup_dataset(output_base_dir=args.data_dir, num_pairs_target=args.num_pairs, workers=args.workers)
