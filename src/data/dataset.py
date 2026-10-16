"""
Forensic Dataset and Synthetic Subtle Manipulation Generator for SEFA-Forensics.
Supports strict group-based partitioning (preventing identity leakage),
and robust real-world post-processing augmentations (JPEG compression, Gaussian blur).
"""

import os
import io
import glob
import random
from typing import Callable, Dict, List, Optional, Tuple, Union
from PIL import Image, ImageFilter
import numpy as np
import torch
from torch.utils.data import Dataset, Subset
import torchvision.transforms as T
import torchvision.transforms.functional as TF


class SyntheticForensicGenerator:
    """
    Generates realistic, subtle inpainting and retouching manipulations on natural images.
    """
    def __init__(self, img_size: int = 224):
        self.img_size = img_size

    def generate_irregular_mask(self, h: int, w: int, max_strokes: int = 4) -> np.ndarray:
        mask = np.zeros((h, w), dtype=np.float32)
        num_strokes = random.randint(1, max_strokes)

        for _ in range(num_strokes):
            start_x = random.randint(int(w * 0.15), int(w * 0.85))
            start_y = random.randint(int(h * 0.15), int(h * 0.85))
            length = random.randint(int(min(h, w) * 0.15), int(min(h, w) * 0.45))
            thickness = random.randint(int(min(h, w) * 0.05), int(min(h, w) * 0.18))
            angle = random.uniform(0, 2 * np.pi)

            curr_x, curr_y = start_x, start_y
            steps = 15
            for _ in range(steps):
                curr_x += int((length / steps) * np.cos(angle) + random.randint(-4, 4))
                curr_y += int((length / steps) * np.sin(angle) + random.randint(-4, 4))
                curr_x = np.clip(curr_x, 0, w - 1)
                curr_y = np.clip(curr_y, 0, h - 1)

                y_coords, x_coords = np.ogrid[:h, :w]
                dist_from_point = (x_coords - curr_x) ** 2 + (y_coords - curr_y) ** 2
                mask[dist_from_point <= (thickness / 2.0) ** 2] = 1.0

        mask_pil = Image.fromarray((mask * 255).astype(np.uint8))
        mask_pil = mask_pil.filter(ImageFilter.GaussianBlur(radius=random.uniform(1.0, 3.0)))
        mask = (np.array(mask_pil) / 255.0).astype(np.float32)
        return (mask > 0.3).astype(np.float32)

    def apply_subtle_manipulation(
        self, img_pil: Image.Image
    ) -> Tuple[Image.Image, np.ndarray, str]:
        img_np = np.array(img_pil).astype(np.float32) / 255.0
        h, w, c = img_np.shape
        mask = self.generate_irregular_mask(h, w)
        mask_3d = np.expand_dims(mask, axis=-1)

        manip_mode = random.choice(["diffusion_retouch", "texture_resynthesis", "noise_incoherence"])

        if manip_mode == "diffusion_retouch":
            blurred = np.array(img_pil.filter(ImageFilter.GaussianBlur(radius=random.uniform(2.5, 5.0)))).astype(np.float32) / 255.0
            noise = np.random.normal(0.0, 0.02, size=img_np.shape).astype(np.float32)
            inpainted_region = np.clip(blurred + noise, 0.0, 1.0)
            desc = "Subtle Diffusion Inpainting & Latent Smoothing"
        elif manip_mode == "texture_resynthesis":
            shift_x = random.randint(-int(w * 0.25), int(w * 0.25))
            shift_y = random.randint(-int(h * 0.25), int(h * 0.25))
            shifted = np.roll(np.roll(img_np, shift_y, axis=0), shift_x, axis=1)
            inpainted_region = np.clip(shifted * 0.95 + 0.02, 0.0, 1.0)
            desc = "Generative Patch Resynthesis & Boundary Mismatch"
        else:
            blurred = np.array(img_pil.filter(ImageFilter.MedianFilter(size=5))).astype(np.float32) / 255.0
            hf_noise = np.random.laplace(0.0, 0.015, size=img_np.shape).astype(np.float32)
            inpainted_region = np.clip(blurred + hf_noise, 0.0, 1.0)
            desc = "Spectral Noise-Floor Incoherence"

        forged_np = (1.0 - mask_3d) * img_np + mask_3d * inpainted_region
        forged_np = np.clip(forged_np * 255.0, 0.0, 255.0).astype(np.uint8)
        forged_pil = Image.fromarray(forged_np)

        return forged_pil, mask, desc


class ForensicDataset(Dataset):
    """
    Forensic Dataset with Grouped-Splits support and real-world degradation simulation.
    """
    def __init__(
        self,
        real_dir: Optional[str] = None,
        forged_dir: Optional[str] = None,
        masks_dir: Optional[str] = None,
        data_roots: Optional[List[str]] = None,
        img_size: int = 224,
        synthetic_ratio: float = 0.0,
        apply_jpeg_compression: bool = False,
        jpeg_quality: Optional[int] = None,
        apply_blur: bool = False,
        blur_radius: float = 1.0,
        transform: Optional[Callable] = None,
    ):
        self.img_size = img_size
        self.synthetic_ratio = synthetic_ratio
        self.synthetic_gen = SyntheticForensicGenerator(img_size=img_size)

        self.apply_jpeg = apply_jpeg_compression
        self.jpeg_quality = jpeg_quality
        self.apply_blur = apply_blur
        self.blur_radius = blur_radius

        self.samples: List[Dict[str, Union[str, int, None]]] = []
        self.group_ids: List[str] = []

        all_roots = []
        if data_roots:
            all_roots.extend(data_roots)
        if real_dir or forged_dir:
            all_roots.append({"real": real_dir, "forged": forged_dir, "masks": masks_dir})

        for item in all_roots:
            if isinstance(item, str):
                cur_real = os.path.join(item, "real")
                cur_forged = os.path.join(item, "forged")
                cur_masks = os.path.join(item, "masks")
                prefix = os.path.basename(item)
            else:
                cur_real = item.get("real")
                cur_forged = item.get("forged")
                cur_masks = item.get("masks")
                prefix = "custom"

            # Index real images
            if cur_real and os.path.exists(cur_real):
                real_files = glob.glob(os.path.join(cur_real, "*.*"))
                real_files = [f for f in real_files if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp', '.tif', '.bmp'))]
                for f in sorted(real_files):
                    grp = f"{prefix}_{self._extract_group_id(f)}"
                    self.samples.append({"img_path": f, "mask_path": None, "label": 0, "group_id": grp})
                    self.group_ids.append(grp)

            # Index forged images & matching masks
            if cur_forged and os.path.exists(cur_forged):
                forged_files = glob.glob(os.path.join(cur_forged, "*.*"))
                forged_files = [f for f in forged_files if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp', '.tif', '.bmp'))]
                for f in sorted(forged_files):
                    base = os.path.splitext(os.path.basename(f))[0]
                    raw_grp = self._extract_group_id(f)
                    grp = f"{prefix}_{raw_grp}"
                    mask_p = None
                    if cur_masks and os.path.exists(cur_masks):
                        cand_names = [
                            f"{base}.png", f"{base}_mask.png", f"{base}_gt.png",
                            f"{raw_grp}_mask.png", f"{raw_grp}.png", f"{raw_grp}_gt.png",
                            f"{raw_grp}_forged.png"
                        ]
                        for c_name in cand_names:
                            cand = os.path.join(cur_masks, c_name)
                            if os.path.exists(cand):
                                mask_p = cand
                                break
                    self.samples.append({"img_path": f, "mask_path": mask_p, "label": 1, "group_id": grp})
                    self.group_ids.append(grp)

        self.transform = transform or T.Compose([
            T.Resize((img_size, img_size)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    @staticmethod
    def _extract_group_id(file_path: str) -> str:
        """
        Extracts scene/image group ID to prevent train-test contamination.
        e.g., 'ffhq_000005_forged.png' -> 'ffhq_000005'
        """
        base = os.path.splitext(os.path.basename(file_path))[0]
        for term in ["_forged", "_real", "_gt", "_mask"]:
            base = base.replace(term, "")
        return base

    def get_grouped_splits(
        self, train_ratio: float = 0.70, val_ratio: float = 0.15, seed: int = 42
    ) -> Tuple[Subset, Subset, Subset]:
        """
        Strict Grouped Splitting: Guarantees an image and its paired manipulation
        NEVER cross between train and test splits, completely eliminating data leakage.
        """
        rng = random.Random(seed)
        unique_groups = sorted(list(set(self.group_ids)))
        rng.shuffle(unique_groups)

        n_groups = len(unique_groups)
        n_train_grp = int(train_ratio * n_groups)
        n_val_grp = int(val_ratio * n_groups)

        train_grps = set(unique_groups[:n_train_grp])
        val_grps = set(unique_groups[n_train_grp:n_train_grp + n_val_grp])
        test_grps = set(unique_groups[n_train_grp + n_val_grp:])

        train_indices = [i for i, s in enumerate(self.samples) if s["group_id"] in train_grps]
        val_indices = [i for i, s in enumerate(self.samples) if s["group_id"] in val_grps]
        test_indices = [i for i, s in enumerate(self.samples) if s["group_id"] in test_grps]

        return Subset(self, train_indices), Subset(self, val_indices), Subset(self, test_indices)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.samples[idx]
        img_p = item["img_path"]
        label = item["label"]
        mask_p = item["mask_path"]

        img_pil = Image.open(img_p).convert("RGB")

        # Simulate Real-World Post-Processing (JPEG compression / Blur) if enabled
        if self.apply_jpeg:
            q = self.jpeg_quality or random.randint(60, 90)
            buffer = io.BytesIO()
            img_pil.save(buffer, format="JPEG", quality=q)
            buffer.seek(0)
            img_pil = Image.open(buffer).convert("RGB")

        if self.apply_blur:
            img_pil = img_pil.filter(ImageFilter.GaussianBlur(radius=self.blur_radius))

        if label == 0 and random.random() < self.synthetic_ratio:
            forged_pil, synth_mask_np, _ = self.synthetic_gen.apply_subtle_manipulation(img_pil)
            img_tensor = self.transform(forged_pil)
            mask_tensor = torch.from_numpy(synth_mask_np).unsqueeze(0).float()
            mask_tensor = TF.resize(mask_tensor, (self.img_size, self.img_size), interpolation=TF.InterpolationMode.NEAREST)
            return {
                "image": img_tensor,
                "label": torch.tensor(1.0, dtype=torch.float32),
                "mask": mask_tensor,
                "path": img_p,
            }

        img_tensor = self.transform(img_pil)

        if label == 1 and mask_p and os.path.exists(mask_p):
            mask_pil = Image.open(mask_p).convert("L")
            mask_np = (np.array(mask_pil) > 127).astype(np.float32)
            mask_tensor = torch.from_numpy(mask_np).unsqueeze(0).float()
            mask_tensor = TF.resize(mask_tensor, (self.img_size, self.img_size), interpolation=TF.InterpolationMode.NEAREST)
        elif label == 1:
            mask_tensor = torch.ones(1, self.img_size, self.img_size, dtype=torch.float32)
        else:
            mask_tensor = torch.zeros(1, self.img_size, self.img_size, dtype=torch.float32)

        return {
            "image": img_tensor,
            "label": torch.tensor(float(label), dtype=torch.float32),
            "mask": mask_tensor,
            "path": img_p,
        }
