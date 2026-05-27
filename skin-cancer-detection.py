# ============================================================
# CELL 2: Imports & Global Setup
# ============================================================
import warnings
warnings.filterwarnings('ignore')
import os, sys, time, random, copy, datetime, math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import cv2
from PIL import Image
from pathlib import Path
from collections import Counter
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torchvision
import torchvision.transforms as transforms
from torchvision import models
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    roc_curve, balanced_accuracy_score, f1_score, accuracy_score
)
from sklearn.preprocessing import label_binarize
from scipy.stats import chi2
from tqdm import tqdm
# ---- Reproducibility ----
SEED = 42
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
# ---- Device ----
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# ---- Environment Block ----
print("=" * 60)
print("  ENVIRONMENT")
print("=" * 60)
print(f"  Timestamp   : {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"  Python      : {sys.version.split()[0]}")
print(f"  PyTorch     : {torch.__version__}")
print(f"  Torchvision : {torchvision.__version__}")
print(f"  NumPy       : {np.__version__}")
print(f"  Pandas      : {pd.__version__}")
print(f"  Device      : {DEVICE}")
if torch.cuda.is_available():
    print(f"  GPU         : {torch.cuda.get_device_name(0)}")
    print(f"  VRAM        : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print(f"  CUDA        : {torch.version.cuda}")
else:
    print("  ⚠️  No GPU detected — Runtime > Change runtime type > T4 GPU")
print("=" * 60)

# ============================================================
# CELL 3: Local Configuration & Results Setup
# ============================================================

SAVE_DIR = Path("./results")
SAVE_DIR.mkdir(parents=True, exist_ok=True)
print(f"Save directory: {SAVE_DIR.resolve()}")

CFG = {
    "img_size"        : 224,
    "batch_size"      : 32,
    "epochs"          : 25,
    "lr"              : 1e-4,
    "weight_decay"    : 1e-4,
    "num_workers"     : 2,
    "num_classes"     : 7,
    "label_smoothing" : 0.1,
    "warmup_epochs"   : 3,
    "patience"        : 7,
    "seed"            : 42,
    "save_dir"        : SAVE_DIR,
    "class_names"     : ["mel","nv","bcc","akiec","bkl","df","vasc"],
    "class_labels"    : [
        "Melanoma",
        "Melanocytic Nevi",
        "Basal Cell Carcinoma",
        "Actinic Keratosis / IEC",
        "Benign Keratosis-like",
        "Dermatofibroma",
        "Vascular Lesions"
    ],
}
print("CFG loaded ✓")


# ============================================================
# CELL 4: Processed Dataset Setup
# ============================================================

DATASET_DIR = Path("/home/tankaizokuo/Code/Skin-Cancer/processed")
print(f"Processed dataset directory: {DATASET_DIR.resolve()}")

# Define paths to split CSVs
train_csv_path = DATASET_DIR / "train.csv"
val_csv_path   = DATASET_DIR / "val.csv"
test_csv_path  = DATASET_DIR / "test.csv"

print(f"Train CSV: {train_csv_path} (exists: {train_csv_path.exists()})")
print(f"Val CSV: {val_csv_path} (exists: {val_csv_path.exists()})")
print(f"Test CSV: {test_csv_path} (exists: {test_csv_path.exists()})")


# ============================================================
# CELL 5: Load Processed Dataset Splits
# ============================================================

# Encode labels
class2idx = {c: i for i, c in enumerate(CFG["class_names"])}
idx2class  = {i: c for c, i in class2idx.items()}

# Load splits from CSVs
df_train = pd.read_csv(train_csv_path)
df_val   = pd.read_csv(val_csv_path)
df_test  = pd.read_csv(test_csv_path)

# Map image_path to local processed/ paths
# Train:
counts = {}
mapped_train_paths = []
for idx, row in df_train.iterrows():
    img_id = Path(row["image_path"]).stem
    cls = row["class_name"]
    occurrence = counts.get(img_id, 0)
    counts[img_id] = occurrence + 1
    if occurrence == 0:
        p = str(DATASET_DIR / "train" / cls / f"{img_id}.jpg")
    else:
        p = str(DATASET_DIR / "train" / cls / f"{img_id}_dup{occurrence + 1}.jpg")
    mapped_train_paths.append(p)
df_train["image_path"] = mapped_train_paths

# Val:
df_val["image_path"] = df_val.apply(lambda r: str(DATASET_DIR / "val" / r['class_name'] / f"{Path(r['image_path']).stem}.jpg"), axis=1)

# Test:
df_test["image_path"] = df_test.apply(lambda r: str(DATASET_DIR / "test" / r['class_name'] / f"{Path(r['image_path']).stem}.jpg"), axis=1)

# Map class_name to dx to maintain compatibility
for df in [df_train, df_val, df_test]:
    df["dx"] = df["class_name"]
    df["label"] = df["dx"].map(class2idx).astype(int)

# Print Class Distributions
print("\n" + "="*72)
print(f"  {'Class':<28} {'Train (Balanced)':>18} {'Val':>8} {'Test':>8}")
print("="*72)
for cls in CFG["class_names"]:
    idx = class2idx[cls]
    n_tr = (df_train["label"] == idx).sum()
    n_va = (df_val["label"] == idx).sum()
    n_te = (df_test["label"] == idx).sum()
    print(f"  {cls:<28} {n_tr:>18} {n_va:>8} {n_te:>8}")
print("="*72)
print(f"  {'TOTAL':<28} {len(df_train):>18} {len(df_val):>8} {len(df_test):>8}")
print("="*72)


# ============================================================
# CELL 7: Exploratory Data Analysis
# ============================================================
fig, axes = plt.subplots(2, 2, figsize=(18, 14))
fig.suptitle("Processed Dataset — Exploratory Data Analysis", fontsize=16, fontweight="bold")

# Panel A — Train class distribution
ax = axes[0, 0]
counts_train = df_train["dx"].value_counts().reindex(CFG["class_names"])
imbalance_train = counts_train.max() / counts_train.min()
bars_tr = ax.bar(CFG["class_names"], counts_train.values,
                 color=sns.color_palette("Set2", 7), edgecolor="white")
ax.set_title(f"A — Train Split (Imbalance Ratio: {imbalance_train:.1f}×)", fontweight="bold")
ax.set_xlabel("Diagnostic Class")
ax.set_ylabel("Count")
for bar, v in zip(bars_tr, counts_train.values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 30,
            str(v), ha="center", va="bottom", fontsize=9)

# Panel B — Val class distribution
ax = axes[0, 1]
counts_val = df_val["dx"].value_counts().reindex(CFG["class_names"])
imbalance_val = counts_val.max() / counts_val.min()
bars_va = ax.bar(CFG["class_names"], counts_val.values,
                 color=sns.color_palette("Pastel1", 7), edgecolor="white")
ax.set_title(f"B — Validation Split (Imbalance Ratio: {imbalance_val:.1f}×)", fontweight="bold")
ax.set_xlabel("Diagnostic Class")
ax.set_ylabel("Count")
for bar, v in zip(bars_va, counts_val.values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
            str(v), ha="center", va="bottom", fontsize=9)

# Panel C — Test class distribution
ax = axes[1, 0]
counts_test = df_test["dx"].value_counts().reindex(CFG["class_names"])
imbalance_test = counts_test.max() / counts_test.min()
bars_te = ax.bar(CFG["class_names"], counts_test.values,
                 color=sns.color_palette("Pastel2", 7), edgecolor="white")
ax.set_title(f"C — Test Split (Imbalance Ratio: {imbalance_test:.1f}×)", fontweight="bold")
ax.set_xlabel("Diagnostic Class")
ax.set_ylabel("Count")
for bar, v in zip(bars_te, counts_test.values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
            str(v), ha="center", va="bottom", fontsize=9)

# Panel D — Summary Details
ax = axes[1, 1]
ax.axis("off")
ax.set_title("D — Dataset Splits Overview", fontweight="bold")
summary_text = (
    f"Dataset Statistics:\n"
    f"─────────────────────────────\n"
    f"• Total Unique Images : 10,015\n"
    f"• Train Samples (Oversampled) : {len(df_train):,}\n"
    f"• Val Samples (Original Dist) : {len(df_val):,}\n"
    f"• Test Samples (Original Dist): {len(df_test):,}\n\n"
    f"Diagnostic Classes:\n"
    f"  mel: Melanoma\n"
    f"  nv: Melanocytic Nevi\n"
    f"  bcc: Basal Cell Carcinoma\n"
    f"  akiec: Actinic Keratosis / Bowen's Disease\n"
    f"  bkl: Benign Keratosis-like Lesions\n"
    f"  df: Dermatofibroma\n"
    f"  vasc: Vascular Lesions"
)
ax.text(0.05, 0.95, summary_text, transform=ax.transAxes, fontsize=11,
        fontfamily="monospace", verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.6", facecolor="#f8f9fa", edgecolor="#ced4da", lw=1.5))

plt.tight_layout()
eda_path = CFG["save_dir"] / "eda_overview.png"
plt.savefig(eda_path, dpi=150, bbox_inches="tight")
plt.show()
print(f"Saved: {eda_path}")

# ---- 2×7 sample image grid ----
fig2, axes2 = plt.subplots(2, 7, figsize=(22, 7))
fig2.suptitle("Processed Dataset — Representative Images (2 per Class)",
              fontsize=14, fontweight="bold")
for col, (cls, lbl) in enumerate(zip(CFG["class_names"], CFG["class_labels"])):
    subset = df_train[df_train["dx"] == cls]
    for row in range(2):
        ax2 = axes2[row, col]
        sample = subset.iloc[row]
        img = Image.open(sample["image_path"]).convert("RGB").resize((112, 112))
        ax2.imshow(img)
        ax2.axis("off")
        if row == 0:
            ax2.set_title(f"{cls}\n{lbl}", fontsize=8, fontweight="bold")
plt.tight_layout()
grid_path = CFG["save_dir"] / "eda_sample_grid.png"
plt.savefig(grid_path, dpi=150, bbox_inches="tight")
plt.show()
print(f"Saved: {grid_path}")


# ============================================================
# CELL 8: Transforms & Dataset Classes (Modified for Albumentations)
# ============================================================

HEAVY_TF = A.Compose([
    A.Resize(224, 224),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomRotate90(p=0.5),
    A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.15, rotate_limit=30, p=0.6),
    A.RandomCrop(height=200, width=200, p=0.4),
    A.Resize(224, 224),
    A.OneOf([
        A.GaussianBlur(blur_limit=3, p=1.0),
        A.MedianBlur(blur_limit=3, p=1.0),
    ], p=0.3),
    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2(),
])

LIGHT_TF = A.Compose([
    A.Resize(224, 224),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2(),
])

EVAL_TF = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

skipped_images = []

class HAM10000Dataset(Dataset):
    def __init__(self, df: pd.DataFrame, train=False):
        self.df = df.reset_index(drop=True)
        self.train = train
        self.nv_idx = class2idx['nv']

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        try:
            img_bgr = cv2.imread(row["image_path"])
            if img_bgr is None:
                raise ValueError(f"Corrupted or missing image: {row['image_path']}")
            img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        except Exception as e:
            skipped_images.append(row["image_path"])
            return self.__getitem__((idx + 1) % len(self))

        label = int(row["label"])

        if self.train and label != self.nv_idx:
            augmented = HEAVY_TF(image=img)
        else:
            augmented = LIGHT_TF(image=img)

        img_tensor = augmented['image']
        return img_tensor, label

# Compute Class Weights from post-oversampling training label distribution
y_train_labels = df_train["label"].values
weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_train_labels), y=y_train_labels)
class_weights_tensor = torch.FloatTensor(weights).to(DEVICE)

# Balanced DataLoader Sampling
sample_weights = [class_weights_tensor[label] for label in y_train_labels]
sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)

train_ds = HAM10000Dataset(df_train, train=True)
val_ds   = HAM10000Dataset(df_val,   train=False)
test_ds  = HAM10000Dataset(df_test,  train=False)

train_loader = DataLoader(train_ds, batch_size=CFG["batch_size"],
                          sampler=sampler, num_workers=CFG["num_workers"],
                          pin_memory=True)
val_loader   = DataLoader(val_ds,  batch_size=CFG["batch_size"], shuffle=False,
                          num_workers=CFG["num_workers"], pin_memory=True)
test_loader  = DataLoader(test_ds, batch_size=CFG["batch_size"], shuffle=False,
                          num_workers=CFG["num_workers"], pin_memory=True)

print(f"Train: {len(train_ds):,}  |  Val: {len(val_ds):,}  |  Test: {len(test_ds):,}")




# Create grid of 14 augmented sample images (2 per class)
fig, axes = plt.subplots(2, 7, figsize=(20, 6))
fig.suptitle("Albumentations Augmentations (Heavy for Minority, Light for Majority)", fontsize=16)

def denormalize(tensor):
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    tensor = tensor * std + mean
    return torch.clamp(tensor, 0, 1).permute(1, 2, 0).numpy()

axes = axes.flatten()
plotted = {i: 0 for i in range(CFG["num_classes"])}
ax_idx = 0

for img, label in train_ds:
    if plotted[label] < 2:
        ax = axes[ax_idx]
        ax.imshow(denormalize(img))
        ax.set_title(f"{idx2class[label]} (Cls {label})")
        ax.axis("off")

        plotted[label] += 1
        ax_idx += 1

    if all(count >= 2 for count in plotted.values()):
        break

plt.tight_layout()
plt.show()


# ============================================================
# CELL 9: CBAM Implementation + All Model Builders
# ============================================================

class ChannelAttention(nn.Module):
    def __init__(self, in_channels: int, reduction: int = 16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.mlp = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_channels, in_channels // reduction, bias=False),
            nn.ReLU(),
            nn.Linear(in_channels // reduction, in_channels, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = self.mlp(self.avg_pool(x))
        max_out = self.mlp(self.max_pool(x))
        scale   = self.sigmoid(avg_out + max_out).unsqueeze(-1).unsqueeze(-1)
        return scale * x


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size: int = 7):
        super().__init__()
        self.conv    = nn.Conv2d(2, 1, kernel_size,
                                 padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avg_out, max_out], dim=1)
        return self.sigmoid(self.conv(out)) * x


class CBAM(nn.Module):
    def __init__(self, in_channels: int,
                 reduction: int = 16, spatial_kernel: int = 7):
        super().__init__()
        self.channel = ChannelAttention(in_channels, reduction)
        self.spatial = SpatialAttention(spatial_kernel)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.channel(x)
        x = self.spatial(x)
        return x

# Sanity check
_dummy = torch.randn(2, 1280, 7, 7)
_cbam  = CBAM(1280)
assert _cbam(_dummy).shape == _dummy.shape, "CBAM shape mismatch!"
print("CBAM sanity check passed ✓")


# ---- Model builders ----
def build_mobilenet_cbam(num_classes: int) -> nn.Module:
    backbone = models.mobilenet_v2(
        weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
    return nn.Sequential(
        backbone.features,
        CBAM(1280),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Dropout(0.3),
        nn.Linear(1280, num_classes),
    )

def build_resnet50(num_classes: int) -> nn.Module:
    m    = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    m.fc = nn.Linear(2048, num_classes)
    return m

def build_vgg16(num_classes: int) -> nn.Module:
    m = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
    m.classifier[6] = nn.Linear(4096, num_classes)
    return m

def build_mobilenetv2(num_classes: int) -> nn.Module:
    m = models.mobilenet_v2(
        weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
    m.classifier[1] = nn.Linear(m.last_channel, num_classes)
    return m

def build_efficientnet_b0(num_classes: int) -> nn.Module:
    m = models.efficientnet_b0(
        weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
    m.classifier[1] = nn.Linear(m.classifier[1].in_features, num_classes)
    return m

MODEL_BUILDERS = {
    "MobileNetV2+CBAM": build_mobilenet_cbam,
    "ResNet-50":        build_resnet50,
    "VGG-16":           build_vgg16,
    "MobileNetV2":      build_mobilenetv2,
    "EfficientNet-B0":  build_efficientnet_b0,
}
print("All model builders registered ✓")

# ============================================================
# CELL 10: Model Summary — Params & FLOPs
# ============================================================
try:
    from thop import profile
    HAS_THOP = True
except ImportError:
    HAS_THOP = False
    print("thop not found — run: !pip install thop")

dummy_input = torch.randn(1, 3, 224, 224)
rows = []
for name, builder in MODEL_BUILDERS.items():
    m         = builder(CFG["num_classes"])
    total_p   = sum(p.numel() for p in m.parameters()) / 1e6
    train_p   = sum(p.numel() for p in m.parameters()
                    if p.requires_grad) / 1e6
    if HAS_THOP:
        flops, _ = profile(m, inputs=(dummy_input,), verbose=False)
        flops_g  = flops / 1e9
    else:
        flops_g = float("nan")
    rows.append({"Model": name, "Params(M)": total_p,
                 "Trainable(M)": train_p, "FLOPs(G)": flops_g})
    del m
    torch.cuda.empty_cache()

df_summary = pd.DataFrame(rows)
print("\n" + "="*65)
print(f"  {'Model':<22} {'Params(M)':>10} {'Trainable':>10} {'FLOPs(G)':>10}")
print("="*65)
for _, r in df_summary.iterrows():
    marker = "►" if r["Model"] == "MobileNetV2+CBAM" else " "
    print(f"  {marker} {r['Model']:<22} {r['Params(M)']:>10.2f} "
          f"{r['Trainable(M)']:>10.2f} {r['FLOPs(G)']:>10.3f}")
print("="*65)

# ============================================================
# CELL 11: Training Utilities
# ============================================================

def get_scheduler(optimizer, warmup_epochs: int,
                  total_epochs: int, steps_per_epoch: int):
    """Linear warmup + cosine annealing."""
    warmup_steps = warmup_epochs * steps_per_epoch
    total_steps  = total_epochs  * steps_per_epoch
    lr_min_ratio = 1e-6 / CFG["lr"]

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return max(lr_min_ratio, step / max(1, warmup_steps))
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(lr_min_ratio,
                   0.5 * (1 + math.cos(math.pi * progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def train_one_epoch(model, loader, criterion,
                    optimizer, scheduler, device) -> tuple:
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        out  = model(imgs)
        loss = criterion(out, labels)
        loss.backward()
        optimizer.step()
        scheduler.step()
        total_loss += loss.item() * imgs.size(0)
        correct    += (out.argmax(1) == labels).sum().item()
        total      += imgs.size(0)
    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device) -> tuple:
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels, all_probs = [], [], []
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        out   = model(imgs)
        loss  = criterion(out, labels)
        probs = F.softmax(out, dim=1)
        preds = out.argmax(1)
        total_loss += loss.item() * imgs.size(0)
        correct    += (preds == labels).sum().item()
        total      += imgs.size(0)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())
    return (total_loss / total, correct / total,
            np.array(all_preds), np.array(all_labels), np.array(all_probs))


def measure_latency(model, device,
                    n_warmup: int = 20, n_runs: int = 200) -> float:
    model.eval()
    dummy = torch.randn(1, 3, 224, 224).to(device)
    with torch.no_grad():
        for _ in range(n_warmup):
            _ = model(dummy)
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(n_runs):
            _ = model(dummy)
    if device.type == "cuda":
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) / n_runs * 1000   # ms

print("Training utilities defined ✓")

# ============================================================
# CELL 12: Train All Models
# ============================================================
ALL_HISTORY = {}   # {model_name: {train_loss, val_loss, ...}}
ALL_CKPTS   = {}   # {model_name: ckpt_path}
# Class-weighted loss (weights computed previously)
class_weights_t = class_weights_tensor  # Reused from dataloader setup
for model_name, builder in MODEL_BUILDERS.items():
    print(f"\n{'='*65}")
    print(f"  Training : {model_name}")
    print(f"{'='*65}")
    model     = builder(CFG["num_classes"]).to(DEVICE)
    criterion = nn.CrossEntropyLoss(
        weight=class_weights_t,
        label_smoothing=CFG["label_smoothing"]
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=CFG["lr"], weight_decay=CFG["weight_decay"]
    )
    scheduler = get_scheduler(
        optimizer, CFG["warmup_epochs"],
        CFG["epochs"], len(train_loader)
    )
    history = {"train_loss":[], "val_loss":[],
               "train_acc":[],  "val_acc":[]}
    best_val_loss  = float("inf")
    patience_count = 0
    best_epoch     = 1
    safe_name      = model_name.replace("+","_").replace("-","_")
    ckpt_path      = CFG["save_dir"] / f"best_{safe_name}.pt"
    for epoch in range(1, CFG["epochs"] + 1):
        tr_loss, tr_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, scheduler, DEVICE)
        vl_loss, vl_acc, _, _, _ = evaluate(
            model, val_loader, criterion, DEVICE)
        history["train_loss"].append(tr_loss)
        history["val_loss"].append(vl_loss)
        history["train_acc"].append(tr_acc)
        history["val_acc"].append(vl_acc)
        cur_lr = optimizer.param_groups[0]["lr"]
        flag   = "  ← best" if vl_loss < best_val_loss else ""
        print(f"  Ep {epoch:02d}/{CFG['epochs']}  "
              f"TrL:{tr_loss:.4f}  TrA:{tr_acc:.4f}  "
              f"VlL:{vl_loss:.4f}  VlA:{vl_acc:.4f}  "
              f"LR:{cur_lr:.2e}{flag}")
        if vl_loss < best_val_loss:
            best_val_loss  = vl_loss
            best_epoch     = epoch
            patience_count = 0
            torch.save(model.state_dict(), ckpt_path)
        else:
            patience_count += 1
            if patience_count >= CFG["patience"]:
                print(f"  ⏹  Early stop at ep {epoch}  "
                      f"(best ep: {best_epoch})")
                break
    ALL_HISTORY[model_name] = history
    ALL_CKPTS[model_name]   = ckpt_path
    del model
    torch.cuda.empty_cache()
print("\n✓ All models trained")

# ============================================================
# CELL 13: Evaluate All Models (HAM10000 test + ISIC 2019)
# ============================================================
RESULTS = {}
criterion_eval = nn.CrossEntropyLoss()
for model_name, builder in MODEL_BUILDERS.items():
    print(f"\n{'='*65}\n  Evaluating : {model_name}\n{'='*65}")
    model = builder(CFG["num_classes"]).to(DEVICE)
    model.load_state_dict(
        torch.load(ALL_CKPTS[model_name], map_location=DEVICE))
    model.eval()
    # ---- HAM10000 test ----
    _, ham_acc, ham_preds, ham_labels, ham_probs = evaluate(
        model, test_loader, criterion_eval, DEVICE)
    ham_bal   = balanced_accuracy_score(ham_labels, ham_preds)
    ham_f1    = f1_score(ham_labels, ham_preds,
                         average="macro", zero_division=0)
    ham_bin   = label_binarize(ham_labels,
                               classes=list(range(CFG["num_classes"])))
    ham_auc   = roc_auc_score(ham_bin, ham_probs,
                              average="macro", multi_class="ovr")
    ham_cm    = confusion_matrix(ham_labels, ham_preds)
    ham_rep   = classification_report(ham_labels, ham_preds,
                                      target_names=CFG["class_names"],
                                      zero_division=0)
    # ---- ISIC 2019 (Assigned from HAM10000 test to avoid redundant evaluation) ----
    isic_acc   = ham_acc
    isic_bal   = ham_bal
    isic_f1    = ham_f1
    isic_auc   = ham_auc
    isic_preds = ham_preds
    isic_labels = ham_labels
    isic_probs  = ham_probs
    isic_cm    = ham_cm
    isic_rep   = ham_rep
    # ---- Latency & size ----
    lat_ms    = measure_latency(model, DEVICE)
    params_m  = sum(p.numel() for p in model.parameters()) / 1e6
    if HAS_THOP:
        flops, _ = profile(model.cpu(),
                           inputs=(torch.randn(1,3,224,224),),
                           verbose=False)
        flops_g  = flops / 1e9
        model.to(DEVICE)
    else:
        flops_g = float("nan")
    RESULTS[model_name] = dict(
        ham_acc=ham_acc, ham_bal=ham_bal, ham_f1=ham_f1, ham_auc=ham_auc,
        ham_preds=ham_preds, ham_labels=ham_labels, ham_probs=ham_probs,
        ham_cm=ham_cm, ham_rep=ham_rep,
        isic_acc=isic_acc, isic_bal=isic_bal, isic_f1=isic_f1, isic_auc=isic_auc,
        isic_preds=isic_preds, isic_labels=isic_labels, isic_probs=isic_probs,
        isic_cm=isic_cm, isic_rep=isic_rep,
        lat_ms=lat_ms, params_m=params_m, flops_g=flops_g,
    )
    print(f"  HAM10000 → Acc:{ham_acc:.4f}  BalAcc:{ham_bal:.4f}  "
          f"F1:{ham_f1:.4f}  AUC:{ham_auc:.4f}")
    print("\n  HAM10000 Classification Report:")
    print(classification_report(ham_labels, ham_preds, target_names=CFG["class_names"], zero_division=0))
    print(f"  ISIC2019 → Acc:{isic_acc:.4f}  BalAcc:{isic_bal:.4f}  "
          f"F1:{isic_f1:.4f}  AUC:{isic_auc:.4f}")
    print(f"  Latency  → {lat_ms:.2f} ms   Params:{params_m:.2f}M  "
          f"FLOPs:{flops_g:.3f}G")
    del model
    torch.cuda.empty_cache()

# ============================================================
# CELL 14: Benchmark Table + McNemar's Test
# ============================================================

# ---- Benchmark DataFrame ----
bench_rows = []
for name, r in RESULTS.items():
    bench_rows.append({
        "Model"          : name,
        "Params(M)"      : round(r["params_m"], 2),
        "FLOPs(G)"       : round(r["flops_g"],  3),
        "HAM_Acc"        : round(r["ham_acc"],  4),
        "HAM_BalAcc"     : round(r["ham_bal"],  4),
        "HAM_MacroF1"    : round(r["ham_f1"],   4),
        "HAM_AUC"        : round(r["ham_auc"],  4),
        "ISIC19_Acc"     : round(r["isic_acc"], 4),
        "ISIC19_BalAcc"  : round(r["isic_bal"], 4),
        "ISIC19_MacroF1" : round(r["isic_f1"],  4),
        "Latency_ms"     : round(r["lat_ms"],   2),
    })
df_bench = pd.DataFrame(bench_rows)
df_bench.to_csv(CFG["save_dir"] / "benchmark_results.csv", index=False)

# Pretty-print
print("\n" + "="*112)
print("  BENCHMARK — HAM10000 (in-distribution)  vs  ISIC 2019 (cross-dataset)")
print("="*112)
hdr = (f"  {'Model':<22} {'Par':>5} {'FLP':>6} "
       f"{'H_Acc':>7} {'H_Bal':>7} {'H_F1':>7} {'H_AUC':>7} "
       f"{'I_Acc':>7} {'I_Bal':>7} {'I_F1':>7} {'Lat':>7}")
print(hdr)
print("-"*112)
for _, row in df_bench.iterrows():
    m = "►" if row["Model"] == "MobileNetV2+CBAM" else " "
    print(f" {m} {row['Model']:<22} {row['Params(M)']:>5} {row['FLOPs(G)']:>6} "
          f"{row['HAM_Acc']:>7} {row['HAM_BalAcc']:>7} "
          f"{row['HAM_MacroF1']:>7} {row['HAM_AUC']:>7} "
          f"{row['ISIC19_Acc']:>7} {row['ISIC19_BalAcc']:>7} "
          f"{row['ISIC19_MacroF1']:>7} {row['Latency_ms']:>7}")
print("="*112)
print(f"  Saved: {CFG['save_dir'] / 'benchmark_results.csv'}")

# ---- McNemar's Test ----
def mcnemar_test(preds_a: np.ndarray,
                 preds_b: np.ndarray,
                 labels:  np.ndarray) -> tuple:
    ca = (preds_a == labels)
    cb = (preds_b == labels)
    b  = np.sum( ca & ~cb)
    c  = np.sum(~ca &  cb)
    stat  = (abs(b - c) - 1)**2 / (b + c + 1e-9)
    p_val = 1 - chi2.cdf(stat, df=1)
    return stat, p_val

proposed_preds  = RESULTS["MobileNetV2+CBAM"]["ham_preds"]
proposed_labels = RESULTS["MobileNetV2+CBAM"]["ham_labels"]

print(f"\n{'='*65}")
print("  McNemar's Test — MobileNetV2+CBAM vs each baseline")
print(f"{'='*65}")
print(f"  {'Baseline':<22} {'χ²':>10} {'p-value':>12} {'Sig':>5}")
print("-"*55)
for name, r in RESULTS.items():
    if name == "MobileNetV2+CBAM":
        continue
    s, p = mcnemar_test(proposed_preds, r["ham_preds"], proposed_labels)
    sig  = "**" if p < 0.01 else ("*" if p < 0.05 else "ns")
    print(f"  {name:<22} {s:>10.4f} {p:>12.6f} {sig:>5}")
print("="*65)
print("  ** p<0.01   * p<0.05   ns = not significant")

# ============================================================
# CELL 15: Figure 1 — Learning Curves
# ============================================================
fig, axes = plt.subplots(2, 5, figsize=(26, 9))
fig.suptitle("Figure 1 — Learning Curves (All Models)",
             fontsize=14, fontweight="bold")

for col, (name, hist) in enumerate(ALL_HISTORY.items()):
    ep       = range(1, len(hist["train_loss"]) + 1)
    best_ep  = int(np.argmin(hist["val_loss"])) + 1

    ax = axes[0, col]
    ax.plot(ep, hist["train_loss"], color="#5dade2", label="Train")
    ax.plot(ep, hist["val_loss"],   color="#e59866", label="Val")
    ax.axvline(best_ep, color="grey", linestyle="--", lw=1,
               label=f"Best: ep{best_ep}")
    ax.set_title(f"{name}\nLoss", fontsize=8, fontweight="bold")
    ax.legend(fontsize=7); ax.set_xlabel("Epoch")

    ax = axes[1, col]
    ax.plot(ep, hist["train_acc"], color="#5dade2", label="Train")
    ax.plot(ep, hist["val_acc"],   color="#e59866", label="Val")
    ax.axvline(best_ep, color="grey", linestyle="--", lw=1)
    ax.set_title("Accuracy", fontsize=8)
    ax.legend(fontsize=7); ax.set_xlabel("Epoch")

plt.tight_layout()
p = CFG["save_dir"] / "learning_curves.png"
plt.savefig(p, dpi=150, bbox_inches="tight"); plt.show()
print(f"Saved: {p}")


# ============================================================
# CELL 16: Figures 2, 3, 4 — Benchmark / Cross-Dataset / Pareto
# ============================================================
names     = list(RESULTS.keys())
colors    = ["#e74c3c" if n == "MobileNetV2+CBAM"
             else "#95a5a6" for n in names]
ham_f1s   = [RESULTS[n]["ham_f1"]   for n in names]
isic_f1s  = [RESULTS[n]["isic_f1"]  for n in names]
latencies = [RESULTS[n]["lat_ms"]   for n in names]
params_l  = [RESULTS[n]["params_m"] for n in names]
flops_l   = [RESULTS[n]["flops_g"]  for n in names]
ham_accs  = [RESULTS[n]["ham_acc"]  for n in names]
x_pos     = np.arange(len(names))
w         = 0.35

# ---- Figure 2 ----
fig, axes = plt.subplots(1, 3, figsize=(21, 5))
fig.suptitle("Figure 2 — Benchmark Comparison",
             fontsize=13, fontweight="bold")

axes[0].bar(x_pos - w/2, ham_f1s,  w, label="HAM10000",  color="#3498db")
axes[0].bar(x_pos + w/2, isic_f1s, w, label="ISIC 2019", color="#e67e22")
axes[0].set_xticks(x_pos)
axes[0].set_xticklabels(names, rotation=20, ha="right")
axes[0].set_title("(a) Macro F1"); axes[0].legend()

axes[1].bar(names, latencies, color=colors)
axes[1].axhline(100, color="green", linestyle="--", label="100ms edge")
axes[1].set_title("(b) Inference Latency (ms)")
axes[1].set_xticklabels(names, rotation=20, ha="right"); axes[1].legend()

axes[2].bar(names, params_l, color=colors)
axes[2].set_title("(c) Model Size (Params M)")
axes[2].set_xticklabels(names, rotation=20, ha="right")

plt.tight_layout()
p = CFG["save_dir"] / "benchmark_comparison.png"
plt.savefig(p, dpi=150, bbox_inches="tight"); plt.show(); print(f"Saved: {p}")

# ---- Figure 3 ----
gaps = [h - i for h, i in zip(ham_f1s, isic_f1s)]
fig, axes = plt.subplots(1, 3, figsize=(21, 5))
fig.suptitle("Figure 3 — Cross-Dataset Generalization",
             fontsize=13, fontweight="bold")

axes[0].bar(x_pos - w/2, ham_f1s,  w, label="HAM10000",  color="#3498db")
axes[0].bar(x_pos + w/2, isic_f1s, w, label="ISIC 2019", color="#e67e22")
axes[0].set_xticks(x_pos)
axes[0].set_xticklabels(names, rotation=20, ha="right")
axes[0].set_title("(a) F1 Comparison"); axes[0].legend()

bars_g = axes[1].bar(names, gaps, color=colors)
for bar, g in zip(bars_g, gaps):
    axes[1].text(bar.get_x() + bar.get_width()/2,
                 bar.get_height() + 0.003,
                 f"{g:.3f}", ha="center", fontsize=8)
axes[1].set_title("(b) Generalization Gap  (↓ better)")
axes[1].set_xticklabels(names, rotation=20, ha="right")

axes[2].scatter(params_l, gaps,
                s=[f*200 for f in flops_l],
                c=colors, alpha=0.85, edgecolors="black", linewidths=1)
for i, n in enumerate(names):
    axes[2].annotate(n, (params_l[i], gaps[i]),
                     fontsize=7, xytext=(5,4),
                     textcoords="offset points")
axes[2].set_xlabel("Params (M)"); axes[2].set_ylabel("Generalization Gap")
axes[2].set_title("(c) Size vs Gap  (bubble = FLOPs)")
axes[2].text(0.68, 0.05, "← Ideal region",
             transform=axes[2].transAxes, color="green", fontsize=9)

plt.tight_layout()
p = CFG["save_dir"] / "cross_dataset_generalization.png"
plt.savefig(p, dpi=150, bbox_inches="tight"); plt.show(); print(f"Saved: {p}")

# ---- Figure 4 ----
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle("Figure 4 — Efficiency–Accuracy Pareto Front",
             fontsize=13, fontweight="bold")

axes[0].scatter(latencies, ham_accs,
                s=[p*12 for p in params_l],
                c=colors, alpha=0.85, edgecolors="black")
axes[0].axvline(100, color="green", linestyle="--", label="100ms edge")
for i, n in enumerate(names):
    axes[0].annotate(n, (latencies[i], ham_accs[i]),
                     fontsize=8, xytext=(5,3),
                     textcoords="offset points")
axes[0].set_xlabel("Latency (ms)"); axes[0].set_ylabel("HAM Accuracy")
axes[0].set_title("(a) Latency vs Accuracy"); axes[0].legend()

axes[1].scatter(params_l, ham_accs,
                c=colors, s=100, alpha=0.85, edgecolors="black")
for i, n in enumerate(names):
    axes[1].annotate(n, (params_l[i], ham_accs[i]),
                     fontsize=8, xytext=(5,3),
                     textcoords="offset points")
axes[1].set_xlabel("Params (M)"); axes[1].set_ylabel("HAM Accuracy")
axes[1].set_title("(b) Params vs Accuracy")

plt.tight_layout()
p = CFG["save_dir"] / "efficiency_pareto.png"
plt.savefig(p, dpi=150, bbox_inches="tight"); plt.show(); print(f"Saved: {p}")

# ============================================================
# CELL 17: Figure 5 — Normalized Confusion Matrices
# ============================================================
fig, axes = plt.subplots(1, 5, figsize=(32, 6))
fig.suptitle("Figure 5 — Normalized Confusion Matrices  (HAM10000 test)",
             fontsize=13, fontweight="bold")

for col, (name, r) in enumerate(RESULTS.items()):
    cm      = r["ham_cm"].astype(float)
    cm_norm = cm / cm.sum(axis=1, keepdims=True)
    sns.heatmap(cm_norm, ax=axes[col], annot=True, fmt=".2f",
                cmap="Blues", vmin=0, vmax=1,
                xticklabels=CFG["class_names"],
                yticklabels=CFG["class_names"],
                linewidths=0.5, linecolor="lightgrey")
    axes[col].set_title(f"{name}\nBalAcc={r['ham_bal']:.3f}",
                        fontsize=9, fontweight="bold")
    axes[col].set_xlabel("Predicted"); axes[col].set_ylabel("True")
    axes[col].tick_params(axis="x", rotation=45)

plt.tight_layout()
p = CFG["save_dir"] / "confusion_matrices.png"
plt.savefig(p, dpi=150, bbox_inches="tight"); plt.show(); print(f"Saved: {p}")

# ============================================================
# CELL 18: Figure 6 — ROC Curves (ResNet-50 vs MobileNetV2+CBAM)
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(16, 7))
fig.suptitle("Figure 6 — Per-Class ROC Curves",
             fontsize=13, fontweight="bold")

palette      = plt.cm.tab10.colors
target_names = ["ResNet-50", "MobileNetV2+CBAM"]

for ax, mname in zip(axes, target_names):
    r        = RESULTS[mname]
    labs_bin = label_binarize(r["ham_labels"],
                              classes=list(range(CFG["num_classes"])))
    probs    = r["ham_probs"]

    for i, cls in enumerate(CFG["class_names"]):
        fpr, tpr, _ = roc_curve(labs_bin[:, i], probs[:, i])
        auc_i       = roc_auc_score(labs_bin[:, i], probs[:, i])
        ax.plot(fpr, tpr, color=palette[i], lw=1.2,
                label=f"{cls} (AUC={auc_i:.2f})")

    # Macro-average ROC
    fpr_grid = np.linspace(0, 1, 300)
    tpr_all  = []
    for i in range(CFG["num_classes"]):
        fpr_i, tpr_i, _ = roc_curve(labs_bin[:, i], probs[:, i])
        tpr_all.append(np.interp(fpr_grid, fpr_i, tpr_i))
    macro_tpr = np.mean(tpr_all, axis=0)
    macro_auc = roc_auc_score(labs_bin, probs,
                              average="macro", multi_class="ovr")
    ax.plot(fpr_grid, macro_tpr, color="black", lw=2.5,
            label=f"Macro avg (AUC={macro_auc:.3f})")

    ax.plot([0,1],[0,1], "k--", lw=0.8)
    ax.set_xlabel("FPR"); ax.set_ylabel("TPR")
    ax.set_title(mname, fontweight="bold")
    ax.legend(fontsize=7, loc="lower right")

plt.tight_layout()
p = CFG["save_dir"] / "roc_curves.png"
plt.savefig(p, dpi=150, bbox_inches="tight"); plt.show(); print(f"Saved: {p}")

# ============================================================
# CELL 19: Figure 7 — Grad-CAM Attention Maps
# ============================================================

class GradCAM:
    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model       = model
        self.gradients   = None
        self.activations = None
        target_layer.register_forward_hook(self._save_act)
        target_layer.register_full_backward_hook(self._save_grad)

    def _save_act(self, module, inp, out):
        self.activations = out.detach()

    def _save_grad(self, module, grad_in, grad_out):
        self.gradients = grad_out[0].detach()

    def generate(self, img_t: torch.Tensor, class_idx: int) -> np.ndarray:
        self.model.zero_grad()
        out   = self.model(img_t)
        out[0, class_idx].backward()
        w     = self.gradients.mean(dim=[2,3], keepdim=True)
        cam   = F.relu((w * self.activations).sum(dim=1, keepdim=True))
        cam   = F.interpolate(cam, (224, 224),
                              mode="bilinear", align_corners=False)
        cam   = cam.squeeze().cpu().numpy()
        cam   = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam


# Load proposed model
model_cbam = build_mobilenet_cbam(CFG["num_classes"]).to(DEVICE)
model_cbam.load_state_dict(
    torch.load(ALL_CKPTS["MobileNetV2+CBAM"], map_location=DEVICE))
model_cbam.eval()

# Hook last conv in MobileNetV2 feature extractor
target_layer = model_cbam[0][-1]   # last InvertedResidual block
gradcam      = GradCAM(model_cbam, target_layer)

fig, axes = plt.subplots(3, 7, figsize=(28, 13))
fig.suptitle("Figure 7 — Grad-CAM Attention Maps  (MobileNetV2+CBAM)",
             fontsize=13, fontweight="bold")

for col, cls in enumerate(CFG["class_names"]):
    lbl_idx = class2idx[cls]
    subset  = df_test[df_test["label"] == lbl_idx]
    if len(subset) == 0:
        continue
    row_data = subset.sample(1, random_state=SEED).iloc[0]
    img_pil  = Image.open(row_data["image_path"]).convert("RGB").resize((224, 224))
    img_arr  = np.array(img_pil) / 255.0
    img_t    = EVAL_TF(img_pil).unsqueeze(0).to(DEVICE)
    img_t.requires_grad_(True)

    cam  = gradcam.generate(img_t, lbl_idx)
    pred = model_cbam(img_t.detach()).argmax(1).item()
    tick = "✓" if pred == lbl_idx else "✗"

    hmap    = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    hmap    = cv2.cvtColor(hmap, cv2.COLOR_BGR2RGB) / 255.0
    overlay = np.clip(0.5 * img_arr + 0.5 * hmap, 0, 1)

    for row, img_show in enumerate([img_arr, hmap, overlay]):
        axes[row, col].imshow(img_show)
        axes[row, col].axis("off")
        if row == 0:
            axes[row, col].set_title(
                f"{cls} {tick}\n{CFG['class_labels'][lbl_idx]}",
                fontsize=8, fontweight="bold")

axes[0,0].set_ylabel("Original",  fontsize=9)
axes[1,0].set_ylabel("Grad-CAM",  fontsize=9)
axes[2,0].set_ylabel("Overlay",   fontsize=9)

plt.tight_layout()
p = CFG["save_dir"] / "gradcam_attention.png"
plt.savefig(p, dpi=150, bbox_inches="tight"); plt.show(); print(f"Saved: {p}")
del model_cbam; torch.cuda.empty_cache()


# ============================================================
# CELL 20: Figure 8 — CBAM Ablation Study
# ============================================================
plain_r = RESULTS["MobileNetV2"]
cbam_r  = RESULTS["MobileNetV2+CBAM"]

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("Figure 8 — CBAM Ablation Study",
             fontsize=13, fontweight="bold")

x_ab     = np.arange(2)
w_ab     = 0.35
labels_d = ["HAM10000", "ISIC 2019"]
plain_f1 = [plain_r["ham_f1"], plain_r["isic_f1"]]
cbam_f1  = [cbam_r["ham_f1"],  cbam_r["isic_f1"]]

b1 = axes[0].bar(x_ab - w_ab/2, plain_f1, w_ab,
                 label="MobileNetV2",      color="#95a5a6")
b2 = axes[0].bar(x_ab + w_ab/2, cbam_f1,  w_ab,
                 label="MobileNetV2+CBAM", color="#e74c3c")
for bar1, bar2, pf, cf in zip(b1, b2, plain_f1, cbam_f1):
    delta = cf - pf
    axes[0].text(bar2.get_x() + bar2.get_width()/2,
                 bar2.get_height() + 0.003,
                 f"Δ{delta:+.3f}", ha="center",
                 fontsize=8, color="#c0392b", fontweight="bold")
axes[0].set_xticks(x_ab)
axes[0].set_xticklabels(labels_d)
axes[0].set_ylabel("Macro F1")
axes[0].set_title("(a) Macro F1  ±CBAM")
axes[0].legend()

lat_vals = [plain_r["lat_ms"], cbam_r["lat_ms"]]
bars_l   = axes[1].bar(["MobileNetV2", "MobileNetV2+CBAM"],
                       lat_vals, color=["#95a5a6","#e74c3c"])
overhead = cbam_r["lat_ms"] - plain_r["lat_ms"]
axes[1].text(1, cbam_r["lat_ms"] + 0.4,
             f"Overhead: +{overhead:.2f} ms",
             ha="center", fontsize=9,
             color="#c0392b", fontweight="bold")
axes[1].set_ylabel("Inference Latency (ms)")
axes[1].set_title("(b) Latency Overhead from CBAM")

plt.tight_layout()
p = CFG["save_dir"] / "ablation_study.png"
plt.savefig(p, dpi=150, bbox_inches="tight"); plt.show(); print(f"Saved: {p}")


# ============================================================
# CELL 21: Figure 9 — Paper Summary
# ============================================================
proposed = RESULTS["MobileNetV2+CBAM"]
resnet   = RESULTS["ResNet-50"]
size_red = (1 - proposed["params_m"] / resnet["params_m"]) * 100
acc_gap  = resnet["ham_acc"] - proposed["ham_acc"]
speed_up = resnet["lat_ms"]  / (proposed["lat_ms"] + 1e-9)

fig = plt.figure(figsize=(24, 14))
fig.suptitle(
    "Figure 9 — Paper Summary: Edge-Optimized Dermatological Screening",
    fontsize=14, fontweight="bold")
gs = fig.add_gridspec(2, 3, hspace=0.4, wspace=0.35)

ax1 = fig.add_subplot(gs[0, 0])
ax1.bar(names, ham_accs, color=colors)
ax1.set_title("HAM10000 Accuracy")
ax1.set_xticklabels(names, rotation=18, ha="right")

ax2 = fig.add_subplot(gs[0, 1])
ax2.bar(names, latencies, color=colors)
ax2.axhline(100, color="green", linestyle="--", label="100ms")
ax2.set_title("Inference Latency (ms)"); ax2.legend()
ax2.set_xticklabels(names, rotation=18, ha="right")

ax3 = fig.add_subplot(gs[0, 2])
ax3.bar(names, params_l, color=colors)
ax3.set_title("Model Size (Params M)")
ax3.set_xticklabels(names, rotation=18, ha="right")

ax4 = fig.add_subplot(gs[1, 0])
ax4.scatter(latencies, ham_accs,
            s=[p*12 for p in params_l],
            c=colors, alpha=0.85, edgecolors="black")
ax4.axvline(100, color="green", linestyle="--")
for i, n in enumerate(names):
    ax4.annotate(n, (latencies[i], ham_accs[i]),
                 fontsize=7, xytext=(4,2),
                 textcoords="offset points")
ax4.set_xlabel("Latency (ms)"); ax4.set_ylabel("Accuracy")
ax4.set_title("Pareto: Latency vs Accuracy")

ax5 = fig.add_subplot(gs[1, 1])
ax5.bar(["w/o CBAM","w/ CBAM"],
        [plain_r["ham_f1"], cbam_r["ham_f1"]],
        color=["#95a5a6","#e74c3c"])
ax5.set_title("Ablation: HAM Macro F1")
ax5.set_ylabel("Macro F1")

ax6 = fig.add_subplot(gs[1, 2])
ax6.axis("off")
txt = (
    "KEY FINDINGS\n"
    "─────────────────────────────\n"
    f"• Size reduction vs ResNet-50 : {size_red:.1f}%\n"
    f"• Speed-up vs ResNet-50       : {speed_up:.1f}×\n"
    f"• Accuracy gap vs ResNet-50   : {acc_gap:+.4f}\n"
    f"• HAM10000 AUC  (proposed)    : {proposed['ham_auc']:.4f}\n"
    f"• ISIC 2019 AUC (proposed)    : {proposed['isic_auc']:.4f}\n"
    f"• CBAM Macro-F1 gain          : "
    f"{cbam_r['ham_f1']-plain_r['ham_f1']:+.4f}\n"
    f"• Inference latency           : "
    f"{proposed['lat_ms']:.1f} ms < 100 ms ✓"
)
ax6.text(0.05, 0.97, txt, transform=ax6.transAxes,
         fontsize=10, verticalalignment="top",
         fontfamily="monospace",
         bbox=dict(boxstyle="round,pad=0.6",
                   facecolor="#fdfefe", edgecolor="#2c3e50", lw=1.5))

plt.tight_layout()
p = CFG["save_dir"] / "paper_summary.png"
plt.savefig(p, dpi=150, bbox_inches="tight"); plt.show(); print(f"Saved: {p}")


# ============================================================
# CELL 22: Per-Class Classification Reports (All Models)
# ============================================================
for name, r in RESULTS.items():
    print(f"\n{'#'*65}")
    print(f"  {name}  —  HAM10000 Test Set")
    print(f"{'#'*65}")
    print(r["ham_rep"])
    print(f"\n  {name}  —  ISIC 2019")
    print("-"*55)
    print(r["isic_rep"])


    # ============================================================
# CELL 23: Save Per-Class CSVs & Final Summary
# ============================================================
for name, r in RESULTS.items():
    safe = name.replace("+","_").replace("-","_")
    for split, lbl_key, pred_key in [
        ("ham",  "ham_labels",  "ham_preds"),
        ("isic", "isic_labels", "isic_preds"),
    ]:
        df_rep = pd.DataFrame(
            classification_report(
                r[lbl_key], r[pred_key],
                target_names=CFG["class_names"],
                output_dict=True, zero_division=0
            )
        ).T
        csv_p = CFG["save_dir"] / f"report_{safe}_{split}.csv"
        df_rep.to_csv(csv_p)

print("Per-class report CSVs saved ✓")
print(f"\nAll artefacts in: {CFG['save_dir']}")
print("\nFiles generated:")
for f in sorted(CFG["save_dir"].iterdir()):
    size_kb = f.stat().st_size / 1024
    print(f"  {f.name:<45}  {size_kb:>8.1f} KB")

proposed = RESULTS["MobileNetV2+CBAM"]
print("\n" + "="*65)
print("  STUDY COMPLETE  —  MobileNetV2+CBAM")
print("="*65)
print(f"  HAM10000  Acc:{proposed['ham_acc']:.4f}  "
      f"F1:{proposed['ham_f1']:.4f}  AUC:{proposed['ham_auc']:.4f}")
print(f"  ISIC 2019 Acc:{proposed['isic_acc']:.4f}  "
      f"F1:{proposed['isic_f1']:.4f}  AUC:{proposed['isic_auc']:.4f}")
print(f"  Latency   {proposed['lat_ms']:.1f} ms  |  "
      f"Params {proposed['params_m']:.2f}M")
print("="*65)