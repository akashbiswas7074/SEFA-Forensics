"""
Lightweight Trainer for SEFA-Forensics.
Trains only the router weights (<0.2% total parameters) with joint image-level and pixel-level loss.
"""

import os
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..models.sefa_model import SEFAModel


class DiceLoss(nn.Module):
    """Per-sample Dice loss for spatial manipulation localization."""
    def __init__(self, smooth: float = 1e-5):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        K = pred.shape[0]
        if K == 0:
            return torch.tensor(0.0, device=pred.device)
        pred_flat = pred.view(K, -1)
        target_flat = target.view(K, -1)
        intersection = (pred_flat * target_flat).sum(dim=1)
        cardinality = pred_flat.sum(dim=1) + target_flat.sum(dim=1)
        dice = (2.0 * intersection + self.smooth) / (cardinality + self.smooth)
        return (1.0 - dice).mean()


class FocalTverskyLoss(nn.Module):
    """
    Focal Tversky Loss for highly imbalanced spatial manipulation localization.
    alpha=0.7 heavily penalizes false-positive background noise (which occupies ~86% of pixels).
    beta=0.3 handles sparse forged regions (~14% of pixels).
    gamma=1.33 focuses gradient on hard boundary and edge ambiguities.
    """
    def __init__(self, alpha: float = 0.7, beta: float = 0.3, gamma: float = 1.33, smooth: float = 1e-5):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.smooth = smooth

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        K = pred.shape[0]
        if K == 0:
            return torch.tensor(0.0, device=pred.device)
        pred_flat = pred.view(K, -1)
        target_flat = target.view(K, -1)
        tp = (pred_flat * target_flat).sum(dim=1)
        fp = (pred_flat * (1.0 - target_flat)).sum(dim=1)
        fn = ((1.0 - pred_flat) * target_flat).sum(dim=1)
        tversky = (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)
        focal_tversky = torch.pow(torch.clamp(1.0 - tversky, min=0.0), 1.0 / self.gamma)
        return focal_tversky.mean()


class SEFATrainer:
    """
    Ultra-fast trainer for SEFA-Forensics with balanced localization loss.
    """
    def __init__(
        self,
        model: SEFAModel,
        device: torch.device = torch.device("cpu"),
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        bce_weight: float = 1.0,
        dice_weight: float = 1.0,
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
        self.dice_loss_fn = FocalTverskyLoss(alpha=0.7, beta=0.3, gamma=1.33)
        self.bce_img_fn = nn.BCEWithLogitsLoss()

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
            is_fake = (labels == 1.0).squeeze(-1)
            is_real = ~is_fake

            if is_fake.any():
                fake_preds = pred_masks[is_fake]
                fake_targets = masks[is_fake]
                loss_dice = self.dice_loss_fn(fake_preds, fake_targets)
                # Positive-weighted BCE on forged regions to counteract background imbalance
                weights = fake_targets * 4.0 + 1.0
                loss_mask_bce = F.binary_cross_entropy(fake_preds, fake_targets, weight=weights)
            else:
                loss_dice = torch.tensor(0.0, device=self.device)
                loss_mask_bce = torch.tensor(0.0, device=self.device)

            if is_real.any():
                # Regularize authentic images to stay clean (low false positive activations)
                loss_clean = pred_masks[is_real].mean()
            else:
                loss_clean = torch.tensor(0.0, device=self.device)

            loss_loc = loss_dice + 0.4 * loss_mask_bce + 0.4 * loss_clean

            # Total composite loss
            loss = self.bce_weight * loss_img + self.dice_weight * loss_loc

            loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in self.model.parameters() if p.requires_grad], max_norm=1.0)
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
            is_fake = (labels == 1.0).squeeze(-1)
            is_real = ~is_fake

            if is_fake.any():
                fake_preds = pred_masks[is_fake]
                fake_targets = masks[is_fake]
                loss_dice = self.dice_loss_fn(fake_preds, fake_targets)
                weights = fake_targets * 4.0 + 1.0
                loss_mask_bce = F.binary_cross_entropy(fake_preds, fake_targets, weight=weights)
            else:
                loss_dice = torch.tensor(0.0, device=self.device)
                loss_mask_bce = torch.tensor(0.0, device=self.device)

            if is_real.any():
                loss_clean = pred_masks[is_real].mean()
            else:
                loss_clean = torch.tensor(0.0, device=self.device)

            loss_loc = loss_dice + 0.4 * loss_mask_bce + 0.4 * loss_clean
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

    def calibrate_threshold(self, val_loader: DataLoader) -> float:
        """Finds optimal operating threshold tau on validation set maximizing F1."""
        val_res = self.evaluate(val_loader)
        y_true = np.array(val_res["all_labels"])
        y_scores = np.array(val_res["all_preds"])
        best_f1, best_tau = 0.0, 0.5
        for tau in np.linspace(0.1, 0.9, 81):
            y_pred = (y_scores >= tau).astype(int)
            tp = ((y_pred == 1) & (y_true == 1)).sum()
            fp = ((y_pred == 1) & (y_true == 0)).sum()
            fn = ((y_pred == 0) & (y_true == 1)).sum()
            f1 = (2 * tp + 1e-6) / (2 * tp + fp + fn + 1e-6)
            if f1 > best_f1:
                best_f1 = f1
                best_tau = tau
        return float(best_tau)

    @torch.no_grad()
    def calibrate_localization_threshold(self, val_loader: DataLoader, metric: str = "iou") -> float:
        """
        Calibrates optimal localization threshold tau_loc* on validation set
        to maximize mIoU or Dice score against foreground/background pixel imbalance.
        """
        self.model.eval()
        preds, targets = [], []
        for batch in val_loader:
            imgs = batch["image"].to(self.device)
            lbls = batch["label"].to(self.device)
            msks = batch["mask"].to(self.device)
            _, pred_masks, _ = self.model(imgs)
            is_f = (lbls == 1.0)
            if is_f.any():
                preds.extend(pred_masks[is_f].squeeze(1).cpu().numpy())
                targets.extend(msks[is_f].squeeze(1).cpu().numpy())

        if len(preds) == 0:
            return 0.5

        best_score, best_tau = 0.0, 0.50
        for tau in np.linspace(0.20, 0.85, 66):
            scores = []
            for p, g in zip(preds, targets):
                if g.sum() > 0:
                    pb = (p >= tau).astype(float)
                    inter = (pb * g).sum()
                    if metric == "dice":
                        score = (2.0 * inter + 1e-6) / (pb.sum() + g.sum() + 1e-6)
                    else:
                        union = pb.sum() + g.sum() - inter
                        score = (inter + 1e-6) / (union + 1e-6)
                    scores.append(score)
            mean_s = np.mean(scores) if len(scores) > 0 else 0.0
            if mean_s > best_score:
                best_score = mean_s
                best_tau = tau
        return float(best_tau)

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        epochs: int = 10,
        verbose: bool = True,
        cache_backbone: bool = True,
    ) -> List[Dict[str, float]]:
        history = []
        best_val_loss = float("inf")

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=epochs, eta_min=1e-5
        )

        if verbose:
            pinfo = self.model.count_parameters()
            print(f"[*] Starting SEFA Router Training ({epochs} epochs)...")
            print(f"    Trainable parameters: {pinfo['trainable_parameters']:,} ({pinfo['trainable_percentage']:.4f}% of model)")

        if cache_backbone:
            if verbose:
                print(f"[*] Pre-extracting frozen backbone features for fast zero-overhead training...")

            def _extract_split(loader):
                items = []
                self.model.eval()
                with torch.no_grad():
                    for b in loader:
                        imgs = b["image"].to(self.device)
                        lbls = b["label"].to(self.device).unsqueeze(-1)
                        msks = b["mask"].to(self.device)

                        clean_feats, spatial_hw = self.model.backbone.extract_patch_features(imgs)
                        cls_token = getattr(self.model.backbone, "last_cls_token", None)
                        stressed_x = self.model.evidence_stress(imgs)
                        stressed_feats, _ = self.model.backbone.extract_patch_features(stressed_x)
                        residuals = {}
                        for l_idx in clean_feats:
                            if l_idx in stressed_feats:
                                residuals[l_idx] = self.model.evidence_stress.compute_latent_residual(
                                    clean_feats[l_idx], stressed_feats[l_idx]
                                )
                        acts, res = self.model.anchor_bank.extract_anchors(clean_feats, residuals)
                        items.append({
                            "acts": acts.cpu(),
                            "res": res.cpu(),
                            "hw": spatial_hw,
                            "cls": cls_token.cpu() if cls_token is not None else None,
                            "img": imgs.cpu(),
                            "mask": msks.cpu(),
                            "label": lbls.cpu(),
                        })
                return items

            train_cached = _extract_split(train_loader)
            val_cached = _extract_split(val_loader) if val_loader else None

            for ep in range(1, epochs + 1):
                self.model.train()
                self.model.backbone.eval()
                tot_loss, tot_img, tot_loc, nb = 0.0, 0.0, 0.0, 0
                perm = torch.randperm(len(train_cached))
                for p_idx in perm:
                    it = train_cached[p_idx]
                    self.optimizer.zero_grad()
                    acts = it["acts"].to(self.device)
                    res = it["res"].to(self.device)
                    cls_t = it["cls"].to(self.device) if it["cls"] is not None else None
                    imgs = it["img"].to(self.device)
                    msks = it["mask"].to(self.device)
                    lbls = it["label"].to(self.device)

                    img_logits, patch_logits, feat_grid = self.model.router(
                        anchor_acts=acts,
                        anchor_residuals=res,
                        spatial_hw=it["hw"],
                        cls_token=cls_t,
                    )
                    pred_masks, _ = self.model.progressive_decoder(feat_grid, imgs, patch_logits=patch_logits)

                    loss_img = self.bce_img_fn(img_logits, lbls)
                    is_fake = (lbls == 1.0).squeeze(-1)
                    is_real = ~is_fake

                    if is_fake.any():
                        fake_preds = pred_masks[is_fake]
                        fake_targets = msks[is_fake]
                        loss_dice = self.dice_loss_fn(fake_preds, fake_targets)
                        weights = fake_targets * 4.0 + 1.0
                        loss_mask_bce = F.binary_cross_entropy(fake_preds, fake_targets, weight=weights)
                        # Multi-scale patch supervision
                        target_patch = F.interpolate(fake_targets, size=it["hw"], mode="area")
                        loss_patch_bce = F.binary_cross_entropy_with_logits(patch_logits[is_fake], target_patch)
                    else:
                        loss_dice = torch.tensor(0.0, device=self.device)
                        loss_mask_bce = torch.tensor(0.0, device=self.device)
                        loss_patch_bce = torch.tensor(0.0, device=self.device)

                    if is_real.any():
                        loss_clean = pred_masks[is_real].mean() + torch.sigmoid(patch_logits[is_real]).mean()
                    else:
                        loss_clean = torch.tensor(0.0, device=self.device)

                    loss_loc = loss_dice + 0.4 * loss_mask_bce + 0.3 * loss_patch_bce + 0.4 * loss_clean
                    loss = self.bce_weight * loss_img + self.dice_weight * loss_loc
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_([p for p in self.model.parameters() if p.requires_grad], max_norm=1.0)
                    self.optimizer.step()

                    tot_loss += loss.item()
                    tot_img += loss_img.item()
                    tot_loc += loss_loc.item()
                    nb += 1

                train_metrics = {
                    "loss": tot_loss / max(1, nb),
                    "loss_img": tot_img / max(1, nb),
                    "loss_loc": tot_loc / max(1, nb),
                }
                scheduler.step()
                log_str = f"Epoch {ep:02d}/{epochs:02d} | Train Loss: {train_metrics['loss']:.4f} (Img: {train_metrics['loss_img']:.4f}, Loc: {train_metrics['loss_loc']:.4f})"

                if val_cached:
                    self.model.eval()
                    v_tot_img, v_tot_dice, v_nb, v_fake_nb = 0.0, 0.0, 0, 0
                    v_y_true, v_y_scores = [], []
                    with torch.no_grad():
                        for vit in val_cached:
                            acts = vit["acts"].to(self.device)
                            res = vit["res"].to(self.device)
                            cls_t = vit["cls"].to(self.device) if vit["cls"] is not None else None
                            imgs = vit["img"].to(self.device)
                            msks = vit["mask"].to(self.device)
                            lbls = vit["label"].to(self.device)

                            ilg, plg, fg = self.model.router(acts, res, vit["hw"], cls_t)
                            pmk, _ = self.model.progressive_decoder(fg, imgs, patch_logits=plg)

                            l_img = self.bce_img_fn(ilg, lbls)
                            v_tot_img += l_img.item()
                            v_nb += 1

                            probs = torch.sigmoid(ilg).cpu().numpy().flatten()
                            v_y_scores.extend(probs)
                            v_y_true.extend(lbls.cpu().numpy().flatten())

                            is_f = (lbls == 1.0).squeeze(-1)
                            if is_f.any():
                                ld = self.dice_loss_fn(pmk[is_f], msks[is_f])
                                v_tot_dice += ld.item()
                                v_fake_nb += 1

                    val_img_loss = v_tot_img / max(1, v_nb)
                    val_dice_loss = (v_tot_dice / max(1, v_fake_nb)) if v_fake_nb > 0 else 0.0
                    val_loss = val_img_loss + 0.6 * val_dice_loss
                    v_y_true, v_y_scores = np.array(v_y_true), np.array(v_y_scores)
                    val_acc = np.mean((v_y_scores >= 0.5) == v_y_true)
                    log_str += f" | Val Loss: {val_loss:.4f} (Img: {val_img_loss:.4f}, Dice: {val_dice_loss:.4f}, Acc: {val_acc*100:.1f}%)"
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        torch.save(self.model.state_dict(), os.path.join(self.checkpoint_dir, "sefa_best.pth"))
                    metrics = {**train_metrics, "val_loss": val_loss, "val_img_loss": val_img_loss, "val_dice_loss": val_dice_loss, "val_acc": val_acc}
                else:
                    metrics = train_metrics

                if verbose:
                    print(log_str, flush=True)
                history.append(metrics)

        else:
            for ep in range(1, epochs + 1):
                train_metrics = self.train_epoch(train_loader)
                scheduler.step()
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
                    print(log_str, flush=True)
                history.append(metrics)

        # Load best weights before returning
        best_path = os.path.join(self.checkpoint_dir, "sefa_best.pth")
        if os.path.exists(best_path):
            self.model.load_state_dict(torch.load(best_path, map_location=self.device))
        torch.save(self.model.state_dict(), os.path.join(self.checkpoint_dir, "sefa_final.pth"))
        return history
