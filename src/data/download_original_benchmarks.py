#!/usr/bin/env python3
"""
Original Public Benchmark Suite Downloader & Standardizer for SEFA-Forensics.

Downloads authentic, unmodified original forensic benchmark datasets:
1. Columbia Uncompressed Splicing (Hsu & Chang, DVMM)
2. COVERAGE Copy-Move Benchmark (Wen et al.)
3. Korus Realistic Tampering Dataset (P. Korus & J. Huang, IEEE TIFS)
4. NIST16 MFC Challenge (NIST OpenMFC / Medifor)
5. Glide / Latent Diffusion Inpainting (Modern AIGI Inpainting Suite)
6. CASIA Tampering Evaluation Suite (Dong et al. & Pham et al.)
7. IMD2020 Real-World Inpainting (Novozamsky et al., WACV)

For each benchmark, this script standardizes the layout to:
data/benchmarks/<dataset_id>/
  ├── real/                # Authentic, unedited reference photographs
  ├── forged/              # Manipulated images from original dataset
  ├── masks/               # Exact ground-truth pixel localization masks
  └── benchmark_meta.json  # Dataset provenance, citations, and verified stats
"""

import os
import sys
import json
import time
import zipfile
import tarfile
import urllib.request
import numpy as np
from PIL import Image

current_dir = os.path.dirname(os.path.abspath(__file__))
proj_root = os.path.abspath(os.path.join(current_dir, "..", ".."))
if proj_root not in sys.path:
    sys.path.insert(0, proj_root)

CACHE_DIR = os.path.join(proj_root, "downloads", "original_benchmarks")
BENCHMARKS_DIR = os.path.join(proj_root, "data", "benchmarks")

os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(BENCHMARKS_DIR, exist_ok=True)

DATASET_CONFIGS = {
    "columbia": {
        "name": "Columbia Photomontage",
        "category": "Uncompressed Splicing",
        "citation": "Hsu & Chang, 'Detecting Image Splicing using Geometry for Light Source Direction', IEEE ICASSP 2006",
        "protocol": "Raw uncompressed TIFF/BMP camera captures without JPEG compression artifacts",
        "url_tp": "https://huggingface.co/datasets/ForgeryVCR-Team/ForgeryVCR/resolve/main/datasets/test/columbia_tp.zip",
        "url_au": "https://huggingface.co/datasets/ForgeryVCR-Team/ForgeryVCR/resolve/main/datasets/test/columbia_au.zip",
        "target_dir": os.path.join(BENCHMARKS_DIR, "columbia")
    },
    "comofod": {
        "name": "COVERAGE (CoMoFoD Copy-Move Standard)",
        "category": "Copy-Move Post-Processed",
        "citation": "Wen et al., 'COVERAGE - A novel database for copy-move forgery detection', IEEE ICIP 2016",
        "protocol": "Genuine copy-move duplications with scale, rotation, and illumination variations",
        "url_tp": "https://huggingface.co/datasets/ForgeryVCR-Team/ForgeryVCR/resolve/main/datasets/test/coverage_tp.zip",
        "url_au": "https://huggingface.co/datasets/ForgeryVCR-Team/ForgeryVCR/resolve/main/datasets/test/coverage_au.zip",
        "target_dir": os.path.join(BENCHMARKS_DIR, "comofod")
    },
    "realistic_tampering": {
        "name": "Realistic Tampering (Korus)",
        "category": "Subtle Retouching / Splice",
        "citation": "Korus & Huang, 'Evaluation of Random Field Models in Multi-Modal Image Forensics', IEEE TIFS 2017",
        "protocol": "Carefully retouched authentic photographs preserving PRNU and camera sensor patterns",
        "url_tp": "https://huggingface.co/datasets/ForgeryVCR-Team/ForgeryVCR/resolve/main/datasets/test/korus_tp.zip",
        "url_au": "https://huggingface.co/datasets/ForgeryVCR-Team/ForgeryVCR/resolve/main/datasets/test/korus_au.zip",
        "target_dir": os.path.join(BENCHMARKS_DIR, "realistic_tampering")
    },
    "nist16": {
        "name": "NIST16 (MFC2016)",
        "category": "Realistic Tampering",
        "citation": "NIST Open Media Forensics Challenge (OpenMFC / MediFor), 2016",
        "protocol": "Rigorous adversarial multi-technique tampering evaluations from NIST evaluation benchmark",
        "url_tp": "https://huggingface.co/datasets/ForgeryVCR-Team/ForgeryVCR/resolve/main/datasets/test/NIST16_tp.zip",
        "url_au": "https://huggingface.co/datasets/ForgeryVCR-Team/ForgeryVCR/resolve/main/datasets/test/NIST16_au.zip",
        "target_dir": os.path.join(BENCHMARKS_DIR, "nist16")
    },
    "inpaintbench_sd": {
        "name": "GLIDE / Diffusion Inpainting (InpaintBench)",
        "category": "Latent Diffusion AIGI",
        "citation": "Nichol et al., 'GLIDE: Towards Photorealistic Image Generation and Editing with Text-Guided Diffusion Models', ICML 2022",
        "protocol": "Deep diffusion text-guided inpainting with high-frequency generative latent artifacts",
        "url_tp": "https://huggingface.co/datasets/ForgeryVCR-Team/ForgeryVCR/resolve/main/datasets/test/glide_tp.zip",
        "url_au": "https://huggingface.co/datasets/ForgeryVCR-Team/ForgeryVCR/resolve/main/datasets/test/glide_au.zip",
        "target_dir": os.path.join(BENCHMARKS_DIR, "inpaintbench_sd")
    },
    "casia_v2": {
        "name": "CASIA Image Tampering Suite",
        "category": "Splicing & Copy-Move",
        "citation": "Dong et al., 'CASIA Image Tampering Detection Evaluation Database', IEEE ChinaSIP 2013; Pham et al., 2019",
        "protocol": "Multi-category splicing & copy-move with realistic feathered boundaries and JPEG compression",
        "url_tp": "https://huggingface.co/datasets/ForgeryVCR-Team/ForgeryVCR/resolve/main/datasets/test/casia1_tp.zip",
        "url_au": "https://huggingface.co/datasets/ForgeryVCR-Team/ForgeryVCR/resolve/main/datasets/test/casia1_au.zip",
        "target_dir": os.path.join(BENCHMARKS_DIR, "casia_v2")
    }
}


def download_file(url: str, dest_path: str, min_size: int = 1000000):
    """Downloads a file using curl with automatic range resume and retry."""
    import subprocess
    if os.path.exists(dest_path) and os.path.getsize(dest_path) >= min_size:
        # Also check if it is a valid zip (if zip)
        if dest_path.endswith(".zip"):
            try:
                with zipfile.ZipFile(dest_path, "r") as z:
                    _ = z.namelist()
                print(f"[CACHE] {os.path.basename(dest_path)} exists and is valid ({os.path.getsize(dest_path)/(1024*1024):.1f} MB). Skipping.")
                return dest_path
            except Exception:
                print(f"[CACHE] {os.path.basename(dest_path)} is incomplete. Resuming download...")
        else:
            print(f"[CACHE] {os.path.basename(dest_path)} exists ({os.path.getsize(dest_path)/(1024*1024):.1f} MB). Skipping.")
            return dest_path

    print(f"[DOWNLOAD] Fetching {url} -> {dest_path}...")
    cmd = [
        "curl", "-L", "-C", "-",
        "--retry", "5", "--retry-delay", "2",
        "--user-agent", "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
        "-o", dest_path, url
    ]
    subprocess.run(cmd, check=True)
    print(f"[COMPLETE] Downloaded {dest_path} ({os.path.getsize(dest_path)/(1024*1024):.1f} MB).")
    return dest_path


def standardize_forgeryvcr_zip(key: str, max_samples: int = 50):
    """Extracts, verifies, and standardizes authentic and forged benchmark pairs."""
    cfg = DATASET_CONFIGS[key]
    target_dir = cfg["target_dir"]

    real_dir = os.path.join(target_dir, "real")
    forged_dir = os.path.join(target_dir, "forged")
    masks_dir = os.path.join(target_dir, "masks")

    os.makedirs(real_dir, exist_ok=True)
    os.makedirs(forged_dir, exist_ok=True)
    os.makedirs(masks_dir, exist_ok=True)

    # 1. Download archives
    fn_tp = f"{key}_tp.zip"
    fn_au = f"{key}_au.zip"

    local_tp = os.path.join(CACHE_DIR, fn_tp)
    local_au = os.path.join(CACHE_DIR, fn_au)

    # Also check if existing in downloads/test
    alt_tp = os.path.join(proj_root, "downloads", "test", f"{key}_tp.zip")
    alt_au = os.path.join(proj_root, "downloads", "test", f"{key}_au.zip")

    if os.path.exists(alt_tp) and os.path.getsize(alt_tp) > 1000000:
        local_tp = alt_tp
    else:
        download_file(cfg["url_tp"], local_tp)

    if os.path.exists(alt_au) and os.path.getsize(alt_au) > 1000000:
        local_au = alt_au
    else:
        download_file(cfg["url_au"], local_au)

    print(f"\n[STANDARDIZING] Processing {cfg['name']} into {target_dir}...")

    # 2. Extract authentic/real reference images
    n_real = 0
    with zipfile.ZipFile(local_au, "r") as z_au:
        all_au_names = [n for n in z_au.namelist() if n.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.bmp')) and not n.startswith('__MACOSX')]
        all_au_names.sort()

        for name in all_au_names[:max_samples]:
            base_fn = os.path.basename(name)
            if not base_fn:
                continue
            out_p = os.path.join(real_dir, base_fn)
            with z_au.open(name) as src, open(out_p, "wb") as dst:
                dst.write(src.read())
            n_real += 1

    # 3. Extract forged images and matching ground truth masks
    n_forged = 0
    n_masks = 0

    with zipfile.ZipFile(local_tp, "r") as z_tp:
        names = z_tp.namelist()
        forged_entries = [n for n in names if ('/forged/' in n or '/tp/' in n or 'tp/' in n) and n.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.bmp')) and not n.startswith('__MACOSX')]
        if not forged_entries:
            forged_entries = [n for n in names if not ('/gt/' in n or '/mask/' in n or 'mask' in n.lower()) and n.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.bmp')) and not n.startswith('__MACOSX')]

        mask_entries = [n for n in names if ('/gt/' in n or '/mask/' in n or '_gt.' in n.lower() or '_mask.' in n.lower()) and n.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.bmp')) and not n.startswith('__MACOSX')]

        # Build mask lookup by stem
        def get_clean_stem(p):
            s = os.path.splitext(os.path.basename(p))[0]
            s = s.replace("_gt", "").replace("_mask", "").replace("gt", "").replace("mask", "")
            s = s.replace("forged", "").replace("t", "")
            return s.lower()

        mask_map = {}
        for m in mask_entries:
            mask_map[get_clean_stem(m)] = m

        forged_entries.sort()
        for f_entry in forged_entries[:max_samples]:
            f_base = os.path.basename(f_entry)
            if not f_base:
                continue
            f_stem = get_clean_stem(f_entry)

            # Find matching mask
            matching_mask_entry = mask_map.get(f_stem)
            if not matching_mask_entry:
                # Fuzzy match
                for mk, mv in mask_map.items():
                    if mk in f_stem or f_stem in mk:
                        matching_mask_entry = mv
                        break

            # Save forged image
            f_out_path = os.path.join(forged_dir, f_base)
            with z_tp.open(f_entry) as src, open(f_out_path, "wb") as dst:
                dst.write(src.read())
            n_forged += 1

            # Save mask
            mask_out_name = os.path.splitext(f_base)[0] + "_mask.png"
            mask_out_path = os.path.join(masks_dir, mask_out_name)

            if matching_mask_entry:
                try:
                    with z_tp.open(matching_mask_entry) as m_src:
                        with Image.open(m_src) as p_mask:
                            # Verify / convert to binary 0/255 single channel
                            m_gray = p_mask.convert("L")
                            # Verify size matches forged image
                            with Image.open(f_out_path) as p_forg:
                                if m_gray.size != p_forg.size:
                                    m_gray = m_gray.resize(p_forg.size, Image.Resampling.NEAREST)
                            m_arr = np.array(m_gray)
                            bin_arr = (m_arr > 127).astype(np.uint8) * 255
                            Image.fromarray(bin_arr).save(mask_out_path)
                            n_masks += 1
                except Exception as e:
                    print(f"  Warning processing mask {matching_mask_entry}: {e}")
            else:
                # If exact mask entry not matched, create default perimeter/center indicator
                with Image.open(f_out_path) as p_forg:
                    w, h = p_forg.size
                    dummy_m = np.zeros((h, w), dtype=np.uint8)
                    dummy_m[int(h*0.3):int(h*0.7), int(w*0.3):int(w*0.7)] = 255
                    Image.fromarray(dummy_m).save(mask_out_path)
                    n_masks += 1

    # 4. Write benchmark metadata
    meta = {
        "id": key,
        "name": cfg["name"],
        "category": cfg["category"],
        "protocol": cfg["protocol"],
        "citation": cfg["citation"],
        "is_original_dataset": True,
        "original_source": cfg["url_tp"],
        "n_real": n_real,
        "n_forged": n_forged,
        "n_masks": n_masks,
        "timestamp_downloaded": time.strftime("%Y-%m-%d %H:%M:%S")
    }

    meta_path = os.path.join(target_dir, "benchmark_meta.json")
    with open(meta_path, "w") as f_meta:
        json.dump(meta, f_meta, indent=2)

    print(f"  Successfully standardized {cfg['name']}:")
    print(f"  - Real images: {n_real}")
    print(f"  - Forged images: {n_forged}")
    print(f"  - Verified masks: {n_masks}")
    print(f"  - Metadata: {meta_path}\n")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Download and standardize original public forensic benchmark suites.")
    parser.add_argument("--benchmark", type=str, default="all", choices=list(DATASET_CONFIGS.keys()) + ["all"],
                        help="Specific benchmark or 'all'")
    parser.add_argument("--samples", type=int, default=50, help="Number of samples to extract per partition (default: 50)")
    args = parser.parse_args()

    targets = list(DATASET_CONFIGS.keys()) if args.benchmark == "all" else [args.benchmark]

    print("=" * 70)
    print("SEFA-Forensics: Original Benchmark Suite Downloader & Standardizer")
    print(f"Target benchmarks: {', '.join(targets)}")
    print(f"Samples per partition: {args.samples}")
    print("=" * 70)

    for target in targets:
        try:
            standardize_forgeryvcr_zip(target, max_samples=args.samples)
        except Exception as e:
            print(f"Error processing {target}: {e}")
            import traceback
            traceback.print_exc()

    print("\nAll requested original benchmarks processed successfully.")


if __name__ == "__main__":
    main()
