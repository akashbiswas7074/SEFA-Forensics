"""
Lightweight Trainer for SEFA-Forensics.
Trains only the router weights (<0.2% total parameters) with joint image-level and pixel-level loss.
"""

import os
from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..models.sefa_model import SEFAModel


class DiceLoss(nn.Module):
    """Dice loss for spatial manipulation localization."""
    def __init__(self, smooth: float = 1e-5):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # pred: [B, 1, H, W] in [0, 1]
        # target: [B, 1, H, W] in [0, 1]
        pred_flat = pred.contiguous().view(-1)
        target_flat = target.contiguous().view(-1)
        intersection = (pred_flat * target_flat).sum()
        dice = (2.0 * intersection + self.smooth) / (pred_flat.sum() + target_flat.sum() + self.smooth)
        return 1.0 - dice


class SEFATrainer:
    """
    Ultra-fast trainer for SEFA-Forensics.
    """
    def __init__(
        self,
        model: SEFAModel,
        device: torch.device = torch.device("cpu"),
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        bce_weight: float = 1.0,
        dice_weight: float = 0.5,
        checkpoint_dir: str = "checkpoints",
    ):
        self.model = model
        self.device = device
        self.model.to(self.device)

        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)

        # Train ONLY router and learnable aggregation weights
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=weight_decay)
        self.dice_loss_fn = DiceLoss()
        self.bce_img_fn = nn.BCEWithLogitsLoss()
        self.bce_mask_fn = nn.BCELoss()

    def train_epoch(self, dataloader: DataLoader) -> Dict[str, float]:
        self.model.train()
        # Strictly ensure backbone stays in eval mode
        self.model.backbone.eval()

        total_loss = 0.0
        total_img_loss = 0.0
        total_loc_loss = 0.0
        num_batches = 0

        for batch in dataloader:
            images = batch["image"].to(self.device)
            labels = batch["label"].to(self.device).unsqueeze(-1)  # [B, 1]
            masks = batch["mask"].to(self.device)                  # [B, 1, H, W]

            self.optimizer.zero_grad()

            img_logits, pred_masks, _ = self.model(images)

            # Image-level forgery classification loss
            loss_img = self.bce_img_fn(img_logits, labels)

            # Pixel-level localization loss
            loss_dice = self.dice_loss_fn(pred_masks, masks)
            loss_mask_bce = self.bce_mask_fn(pred_masks, masks)
            loss_loc = loss_dice + 0.5 * loss_mask_bce

            # Total composite loss
            loss = self.bce_weight * loss_img + self.dice_weight * loss_loc

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.router.parameters(), max_norm=1.0)
            self.optimizer.step()

            total_loss += loss.item()
            total_img_loss += loss_img.item()
            total_loc_loss += loss_loc.item()
            num_batches += 1

        return {
            "loss": total_loss / max(1, num_batches),
            "loss_img": total_img_loss / max(1, num_batches),
            "loss_loc": total_loc_loss / max(1, num_batches),
        }

    @torch.no_grad()
    def evaluate(self, dataloader: DataLoader) -> Dict[str, float]:
        self.model.eval()
        total_loss = 0.0
        all_labels = []
        all_preds = []
        num_batches = 0

        for batch in dataloader:
            images = batch["image"].to(self.device)
            labels = batch["label"].to(self.device).unsqueeze(-1)
            masks = batch["mask"].to(self.device)

            img_logits, pred_masks, _ = self.model(images)

            loss_img = self.bce_img_fn(img_logits, labels)
            loss_loc = self.dice_loss_fn(pred_masks, masks)
            loss = self.bce_weight * loss_img + self.dice_weight * loss_loc

            total_loss += loss.item()
            probs = torch.sigmoid(img_logits)

            all_labels.extend(labels.cpu().numpy().flatten())
            all_preds.extend(probs.cpu().numpy().flatten())
            num_batches += 1

        val_loss = total_loss / max(1, num_batches)
        return {
            "val_loss": val_loss,
            "all_labels": all_labels,
            "all_preds": all_preds,
        }

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        epochs: int = 10,
        verbose: bool = True,
    ) -> List[Dict[str, float]]:
        history = []
        best_val_loss = float("inf")

        if verbose:
            pinfo = self.model.count_parameters()
            print(f"[*] Starting SEFA Router Training ({epochs} epochs)...")
            print(f"    Trainable parameters: {pinfo['trainable_parameters']:,} ({pinfo['trainable_percentage']:.4f}% of model)")

        for ep in range(1, epochs + 1):
            train_metrics = self.train_epoch(train_loader)
            log_str = f"Epoch {ep:02d}/{epochs:02d} | Train Loss: {train_metrics['loss']:.4f} (Img: {train_metrics['loss_img']:.4f}, Loc: {train_metrics['loss_loc']:.4f})"

            if val_loader:
                val_metrics = self.evaluate(val_loader)
                log_str += f" | Val Loss: {val_metrics['val_loss']:.4f}"
                if val_metrics["val_loss"] < best_val_loss:
                    best_val_loss = val_metrics["val_loss"]
                    torch.save(self.model.state_dict(), os.path.join(self.checkpoint_dir, "sefa_best.pth"))
                metrics = {**train_metrics, "val_loss": val_metrics["val_loss"]}
            else:
                metrics = train_metrics

            if verbose:
                print(log_str)
            history.append(metrics)

        # Save final checkpoint
        torch.save(self.model.state_dict(), os.path.join(self.checkpoint_dir, "sefa_final.pth"))
        return history
