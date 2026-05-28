"""
MobileNetV2 + CBAM Skin Cancer Classifier
==========================================
Works in both Google Colab (Drive-mounted) and local environments.
Dataset: HAM10000 / ISIC 2018  —  7-class dermoscopic lesion classification

Usage
-----
  # Local
  python train_mobilenetv2_cbam.py --data_dir /path/to/processed --env local

  # Colab (auto-mounts Drive)
  python train_mobilenetv2_cbam.py --env colab

  # Eval only (loads best checkpoint)
  python train_mobilenetv2_cbam.py --eval_only --checkpoint runs/<timestamp>/best_model.pth
"""

# ── stdlib ──────────────────────────────────────────────────────────────────
import argparse
import os
import sys
import time
import json
import copy
import random
from datetime import datetime
from pathlib import Path

# ── third-party ─────────────────────────────────────────────────────────────
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from torchvision.models import MobileNet_V2_Weights
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, balanced_accuracy_score
)
import matplotlib
matplotlib.use("Agg")                       # non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  0.  REPRODUCIBILITY                                                     ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  1.  ENVIRONMENT DETECTION & PATH SETUP                                  ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def setup_environment(env: str, data_dir: str | None) -> Path:
    """
    Returns the path to the `processed/` directory that contains
    train/ val/ test/ sub-folders.
    """
    if env == "colab":
        try:
            from google.colab import drive          # type: ignore
            drive.mount("/content/drive", force_remount=False)
            print("[ENV] Google Colab — Drive mounted.")
        except ImportError:
            print("[WARN] google.colab not available; treating as local.")
        base = Path("/content/drive/MyDrive/Skin-Cancer-Dataset/processed")
    else:
        if data_dir is None:
            raise ValueError("--data_dir is required when --env local")
        base = Path(data_dir)

    if not base.exists():
        raise FileNotFoundError(f"Dataset root not found: {base}")

    for split in ("train", "val", "test"):
        if not (base / split).exists():
            raise FileNotFoundError(f"Missing split folder: {base / split}")

    print(f"[ENV] Dataset root : {base}")
    return base


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  2.  DATA LOADING & AUGMENTATION                                         ║
# ╚══════════════════════════════════════════════════════════════════════════╝

IMG_SIZE   = 224          # MobileNetV2 default
MEAN       = [0.485, 0.456, 0.406]
STD        = [0.229, 0.224, 0.225]

# ── class label map (matches dataset folder names) ──────────────────────────
CLASS_NAMES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
FULL_NAMES  = {
    "akiec": "Actinic keratosis",
    "bcc"  : "Basal cell carcinoma",
    "bkl"  : "Benign keratosis",
    "df"   : "Dermatofibroma",
    "mel"  : "Melanoma",
    "nv"   : "Melanocytic nevus",
    "vasc" : "Vascular lesion",
}


def get_transforms(split: str) -> transforms.Compose:
    if split == "train":
        return transforms.Compose([
            transforms.RandomResizedCrop(IMG_SIZE, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2,
                                   saturation=0.2, hue=0.05),
            transforms.RandomRotation(20),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
        ])
    else:
        return transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(IMG_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
        ])


def build_dataloaders(data_root: Path, batch_size: int,
                      num_workers: int) -> dict[str, DataLoader]:
    loaders = {}
    for split in ("train", "val", "test"):
        dataset = datasets.ImageFolder(
            root=str(data_root / split),
            transform=get_transforms(split),
        )
        shuffle = (split == "train")
        loaders[split] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=(num_workers > 0),
        )
        print(f"  [{split:5s}] {len(dataset):>6,} images  |  "
              f"{len(loaders[split]):>4,} batches")

    # sanity-check class ordering
    actual = loaders["train"].dataset.classes
    assert actual == CLASS_NAMES, (
        f"Folder classes {actual} != expected {CLASS_NAMES}.\n"
        "Ensure the dataset folder names match exactly."
    )
    return loaders


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  3.  CBAM MODULE                                                         ║
# ╚══════════════════════════════════════════════════════════════════════════╝

class ChannelAttention(nn.Module):
    """Squeeze-and-excitation style channel attention."""

    def __init__(self, in_channels: int, reduction: int = 16) -> None:
        super().__init__()
        mid = max(in_channels // reduction, 8)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, mid, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, in_channels, 1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        return x * self.sigmoid(avg_out + max_out)


class SpatialAttention(nn.Module):
    """Spatial attention gate."""

    def __init__(self, kernel_size: int = 7) -> None:
        super().__init__()
        pad = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=pad, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = x.mean(dim=1, keepdim=True)
        max_out = x.max(dim=1, keepdim=True).values
        attn    = torch.cat([avg_out, max_out], dim=1)
        return x * self.sigmoid(self.conv(attn))


class CBAM(nn.Module):
    """Convolutional Block Attention Module (Woo et al., 2018)."""

    def __init__(self, in_channels: int,
                 reduction: int = 16,
                 spatial_kernel: int = 7) -> None:
        super().__init__()
        self.channel  = ChannelAttention(in_channels, reduction)
        self.spatial  = SpatialAttention(spatial_kernel)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.channel(x)
        x = self.spatial(x)
        return x


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  4.  MODEL — MobileNetV2 + CBAM                                          ║
# ╚══════════════════════════════════════════════════════════════════════════╝

class MobileNetV2CBAM(nn.Module):
    """
    MobileNetV2 backbone with CBAM injected after the last inverted-residual
    block (before the classifier) and a custom 7-class head.
    """

    def __init__(self, num_classes: int = 7,
                 dropout: float = 0.4,
                 cbam_reduction: int = 16) -> None:
        super().__init__()

        # ── backbone ────────────────────────────────────────────────────────
        backbone = models.mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1)

        # features[0..18] = MobileNetV2 feature extractor
        self.features = backbone.features          # output: (B, 1280, 7, 7)

        # ── CBAM after last conv block ───────────────────────────────────────
        self.cbam = CBAM(in_channels=1280, reduction=cbam_reduction)

        # ── classifier head ─────────────────────────────────────────────────
        self.pool      = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(1280, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout / 2),
            nn.Linear(512, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)           # (B, 1280, H, W)
        x = self.cbam(x)               # attention
        x = self.pool(x)               # (B, 1280, 1, 1)
        x = torch.flatten(x, 1)        # (B, 1280)
        return self.classifier(x)      # (B, num_classes)

    def freeze_backbone(self) -> None:
        """Freeze all backbone layers for warm-up phase."""
        for p in self.features.parameters():
            p.requires_grad = False

    def unfreeze_backbone(self) -> None:
        """Unfreeze all layers for full fine-tuning."""
        for p in self.features.parameters():
            p.requires_grad = True


def count_params(model: nn.Module) -> tuple[int, int]:
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  5.  TRAINING UTILITIES                                                  ║
# ╚══════════════════════════════════════════════════════════════════════════╝

class AverageMeter:
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.val = self.avg = self.sum = self.count = 0.0

    def update(self, val: float, n: int = 1) -> None:
        self.val   = val
        self.sum  += val * n
        self.count += n
        self.avg   = self.sum / self.count


def accuracy(output: torch.Tensor, target: torch.Tensor) -> float:
    with torch.no_grad():
        pred = output.argmax(dim=1)
        return (pred == target).float().mean().item() * 100.0


def train_one_epoch(model, loader, criterion, optimizer,
                    device, scaler) -> tuple[float, float]:
    model.train()
    loss_m = AverageMeter()
    acc_m  = AverageMeter()

    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)

        with torch.amp.autocast(device_type=device.type,
                                enabled=(device.type == "cuda")):
            logits = model(imgs)
            loss   = criterion(logits, labels)

        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        loss_m.update(loss.item(), imgs.size(0))
        acc_m.update(accuracy(logits, labels), imgs.size(0))

    return loss_m.avg, acc_m.avg


@torch.inference_mode()
def evaluate(model, loader, criterion, device) -> tuple[float, float]:
    model.eval()
    loss_m = AverageMeter()
    acc_m  = AverageMeter()

    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        logits = model(imgs)
        loss   = criterion(logits, labels)
        loss_m.update(loss.item(), imgs.size(0))
        acc_m.update(accuracy(logits, labels), imgs.size(0))

    return loss_m.avg, acc_m.avg


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  6.  FULL EVALUATION (test set)                                          ║
# ╚══════════════════════════════════════════════════════════════════════════╝

@torch.inference_mode()
def full_evaluation(model, loader, device,
                    save_dir: Path) -> dict:
    """
    Computes per-class metrics, confusion matrix, ROC-AUC,
    saves plots, and returns a metrics dict.
    """
    model.eval()
    all_preds   = []
    all_labels  = []
    all_probs   = []

    for imgs, labels in loader:
        imgs = imgs.to(device)
        logits = model(imgs)
        probs  = torch.softmax(logits, dim=1).cpu().numpy()
        preds  = logits.argmax(dim=1).cpu().numpy()
        all_probs.append(probs)
        all_preds.append(preds)
        all_labels.append(labels.numpy())

    all_probs  = np.concatenate(all_probs,  axis=0)
    all_preds  = np.concatenate(all_preds,  axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    # ── classification report ────────────────────────────────────────────────
    report = classification_report(
        all_labels, all_preds,
        target_names=CLASS_NAMES,
        output_dict=True,
    )
    print("\n" + "─" * 60)
    print(classification_report(all_labels, all_preds,
                                target_names=CLASS_NAMES))

    # ── balanced accuracy ────────────────────────────────────────────────────
    bal_acc = balanced_accuracy_score(all_labels, all_preds) * 100.0
    print(f"Balanced Accuracy : {bal_acc:.2f}%")

    # ── macro ROC-AUC ────────────────────────────────────────────────────────
    try:
        auc = roc_auc_score(all_labels, all_probs,
                            multi_class="ovr", average="macro")
        print(f"Macro ROC-AUC     : {auc:.4f}")
    except ValueError:
        auc = float("nan")
        print("[WARN] ROC-AUC computation failed (only one class in batch?).")

    # ── confusion matrix ─────────────────────────────────────────────────────
    cm = confusion_matrix(all_labels, all_preds)
    _plot_confusion_matrix(cm, CLASS_NAMES, save_dir / "confusion_matrix.png")

    # ── per-class accuracy bar chart ─────────────────────────────────────────
    per_class_acc = cm.diagonal() / cm.sum(axis=1) * 100.0
    _plot_per_class_accuracy(per_class_acc, CLASS_NAMES,
                             save_dir / "per_class_accuracy.png")

    metrics = {
        "balanced_accuracy": bal_acc,
        "roc_auc_macro"    : auc,
        "per_class"        : {
            cls: {
                "precision": report[cls]["precision"],
                "recall"   : report[cls]["recall"],
                "f1"       : report[cls]["f1-score"],
                "support"  : report[cls]["support"],
                "accuracy" : float(per_class_acc[i]),
            }
            for i, cls in enumerate(CLASS_NAMES)
        },
        "macro_avg" : report["macro avg"],
        "weighted_avg": report["weighted avg"],
    }
    return metrics


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  7.  PLOTTING HELPERS                                                    ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def _plot_confusion_matrix(cm: np.ndarray,
                           class_names: list[str],
                           save_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 7))
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    sns.heatmap(
        cm_norm, annot=True, fmt=".2f", cmap="Blues",
        xticklabels=class_names, yticklabels=class_names,
        linewidths=0.5, ax=ax,
    )
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("True",      fontsize=12)
    ax.set_title("Normalised Confusion Matrix — MobileNetV2+CBAM", fontsize=13)
    plt.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"[PLOT] Confusion matrix saved → {save_path}")


def _plot_per_class_accuracy(acc: np.ndarray,
                             class_names: list[str],
                             save_path: Path) -> None:
    full = [FULL_NAMES[c] for c in class_names]
    colours = ["#e74c3c" if a < 60 else "#f39c12" if a < 80 else "#27ae60"
               for a in acc]
    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.barh(full, acc, color=colours, edgecolor="white")
    ax.bar_label(bars, fmt="%.1f%%", padding=4, fontsize=9)
    ax.set_xlim(0, 110)
    ax.set_xlabel("Accuracy (%)", fontsize=11)
    ax.set_title("Per-class Accuracy — MobileNetV2+CBAM", fontsize=13)
    ax.axvline(80, linestyle="--", color="grey", linewidth=0.8, alpha=0.6)
    plt.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"[PLOT] Per-class accuracy saved → {save_path}")


def plot_training_curves(history: dict, save_dir: Path) -> None:
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # loss
    axes[0].plot(epochs, history["train_loss"], label="Train", marker="o", ms=3)
    axes[0].plot(epochs, history["val_loss"],   label="Val",   marker="o", ms=3)
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Cross-Entropy Loss")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # accuracy
    axes[1].plot(epochs, history["train_acc"], label="Train", marker="o", ms=3)
    axes[1].plot(epochs, history["val_acc"],   label="Val",   marker="o", ms=3)
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy (%)")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    fig.suptitle("MobileNetV2 + CBAM — Training Curves", fontsize=14)
    plt.tight_layout()
    path = save_dir / "training_curves.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[PLOT] Training curves saved → {path}")


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  8.  MAIN TRAINING LOOP                                                  ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def train(cfg: argparse.Namespace) -> None:
    seed_everything(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[DEVICE] {device}")

    # ── output dir ──────────────────────────────────────────────────────────
    run_id  = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(cfg.output_dir) / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[OUT]  {out_dir}")

    # ── data ────────────────────────────────────────────────────────────────
    data_root = setup_environment(cfg.env, cfg.data_dir)
    loaders   = build_dataloaders(data_root, cfg.batch_size, cfg.num_workers)

    # ── model ────────────────────────────────────────────────────────────────
    model = MobileNetV2CBAM(
        num_classes=7,
        dropout=cfg.dropout,
        cbam_reduction=cfg.cbam_reduction,
    ).to(device)

    total, trainable = count_params(model)
    print(f"[MODEL] Total params    : {total:,}")
    print(f"[MODEL] Trainable params: {trainable:,}")

    # ── loss — label smoothing helps with oversampled minority classes ───────
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    # ── two-phase training ───────────────────────────────────────────────────
    # Phase 1: backbone frozen, only head + CBAM trained  (warm-up)
    # Phase 2: full fine-tuning with lower LR
    model.freeze_backbone()

    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=cfg.lr_head,
        weight_decay=cfg.weight_decay,
    )
    scaler = torch.amp.GradScaler(enabled=(device.type == "cuda"))

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.warmup_epochs, eta_min=1e-6
    )

    history: dict[str, list] = {
        "train_loss": [], "val_loss": [],
        "train_acc" : [], "val_acc" : [],
        "lr"        : [],
    }

    best_val_acc  = 0.0
    best_ckpt     = out_dir / "best_model.pth"
    patience_ctr  = 0

    total_epochs = cfg.warmup_epochs + cfg.finetune_epochs

    for epoch in range(1, total_epochs + 1):
        t0 = time.time()

        # switch phases
        if epoch == cfg.warmup_epochs + 1:
            print(f"\n{'─'*60}")
            print(f"[PHASE 2] Unfreezing backbone — full fine-tune begins.")
            model.unfreeze_backbone()
            optimizer = optim.AdamW(
                model.parameters(),
                lr=cfg.lr_full,
                weight_decay=cfg.weight_decay,
            )
            scaler = torch.amp.GradScaler(enabled=(device.type == "cuda"))
            scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer, T_0=cfg.finetune_epochs, T_mult=1, eta_min=1e-7
            )

        train_loss, train_acc = train_one_epoch(
            model, loaders["train"], criterion, optimizer, device, scaler
        )
        val_loss, val_acc = evaluate(
            model, loaders["val"], criterion, device
        )
        scheduler.step()

        elapsed = time.time() - t0
        current_lr = optimizer.param_groups[0]["lr"]

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)
        history["lr"].append(current_lr)

        phase_tag = "W" if epoch <= cfg.warmup_epochs else "F"
        print(
            f"[{phase_tag}] Epoch {epoch:>3}/{total_epochs}  "
            f"| TrLoss {train_loss:.4f}  TrAcc {train_acc:5.2f}%  "
            f"| ValLoss {val_loss:.4f}  ValAcc {val_acc:5.2f}%  "
            f"| LR {current_lr:.2e}  | {elapsed:.0f}s"
        )

        # checkpoint
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                "epoch"     : epoch,
                "model_state": model.state_dict(),
                "val_acc"   : val_acc,
                "cfg"       : vars(cfg),
            }, best_ckpt)
            print(f"  ✔  New best val acc: {best_val_acc:.2f}% — saved.")
            patience_ctr = 0
        else:
            patience_ctr += 1
            if patience_ctr >= cfg.patience:
                print(f"[EARLY STOP] No improvement for {cfg.patience} epochs.")
                break

    # ── save last checkpoint & history ──────────────────────────────────────
    torch.save(model.state_dict(), out_dir / "last_model.pth")
    with open(out_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)
    plot_training_curves(history, out_dir)

    # ── final evaluation on test set ─────────────────────────────────────────
    print(f"\n{'═'*60}")
    print("Loading best checkpoint for test-set evaluation …")
    ckpt = torch.load(best_ckpt, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    model.to(device)

    metrics = full_evaluation(model, loaders["test"], device, out_dir)
    with open(out_dir / "test_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\n[DONE] All outputs saved to: {out_dir}")
    print(f"       Best val acc : {best_val_acc:.2f}%")
    print(f"       Balanced acc : {metrics['balanced_accuracy']:.2f}%")
    print(f"       ROC-AUC      : {metrics['roc_auc_macro']:.4f}")


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  9.  EVAL-ONLY MODE                                                      ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def eval_only(cfg: argparse.Namespace) -> None:
    seed_everything(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if cfg.checkpoint is None:
        raise ValueError("--checkpoint is required for --eval_only mode.")

    ckpt_path = Path(cfg.checkpoint)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    out_dir = ckpt_path.parent / "eval_results"
    out_dir.mkdir(exist_ok=True)

    data_root = setup_environment(cfg.env, cfg.data_dir)
    loaders   = build_dataloaders(data_root, cfg.batch_size, cfg.num_workers)

    model = MobileNetV2CBAM(num_classes=7).to(device)
    ckpt  = torch.load(ckpt_path, map_location=device, weights_only=False)
    state = ckpt["model_state"] if "model_state" in ckpt else ckpt
    model.load_state_dict(state)

    print(f"[EVAL] Checkpoint  : {ckpt_path}")
    print(f"[EVAL] Output dir  : {out_dir}")

    metrics = full_evaluation(model, loaders["test"], device, out_dir)
    with open(out_dir / "test_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\n[DONE] Results saved to: {out_dir}")


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║ 10.  CLI                                                                 ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="MobileNetV2 + CBAM — Skin Cancer Classifier",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # environment
    p.add_argument("--env",      choices=["local", "colab"], default="local",
                   help="Runtime environment.")
    p.add_argument("--data_dir", type=str, default=None,
                   help="Path to processed/ folder (local env only).")

    # mode
    p.add_argument("--eval_only",  action="store_true",
                   help="Skip training; evaluate a saved checkpoint.")
    p.add_argument("--checkpoint", type=str, default=None,
                   help="Path to .pth checkpoint (required for --eval_only).")

    # training hyper-params
    p.add_argument("--warmup_epochs",   type=int,   default=5)
    p.add_argument("--finetune_epochs", type=int,   default=30)
    p.add_argument("--batch_size",      type=int,   default=32)
    p.add_argument("--lr_head",         type=float, default=1e-3,
                   help="LR for head+CBAM during warm-up phase.")
    p.add_argument("--lr_full",         type=float, default=2e-4,
                   help="LR for full fine-tuning phase.")
    p.add_argument("--weight_decay",    type=float, default=1e-4)
    p.add_argument("--dropout",         type=float, default=0.4)
    p.add_argument("--cbam_reduction",  type=int,   default=16,
                   help="Channel reduction ratio for CBAM.")
    p.add_argument("--patience",        type=int,   default=10,
                   help="Early stopping patience (epochs).")

    # misc
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--seed",        type=int, default=42)
    p.add_argument("--output_dir",  type=str, default="runs",
                   help="Root directory for run outputs.")

    return p.parse_args()


# ── entry point ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    args = parse_args()

    # Colab-friendly: reduce workers if multiprocessing causes issues
    if args.env == "colab" and args.num_workers > 2:
        args.num_workers = 2

    if args.eval_only:
        eval_only(args)
    else:
        train(args)
