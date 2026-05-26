import os, sys, random, shutil, logging
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.utils import resample
from sklearn.model_selection import train_test_split
from collections import Counter

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
log = logging.getLogger(__name__)

BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
DATASETS_DIR = os.path.join(BASE_DIR, "datasets")

HAM_DIR      = os.path.join(DATASETS_DIR, "kmader/skin-cancer-mnist-ham10000/versions/2")
HAM_META     = os.path.join(HAM_DIR, "HAM10000_metadata.csv")
HAM_IMG_DIRS = [
    os.path.join(HAM_DIR, "HAM10000_images_part_1"),
    os.path.join(HAM_DIR, "HAM10000_images_part_2"),
    os.path.join(HAM_DIR, "ham10000_images_part_1"),
    os.path.join(HAM_DIR, "ham10000_images_part_2"),
    HAM_DIR,  # fallback: flat directory
]

ISIC_DIR     = os.path.join(DATASETS_DIR, "trantoanthang/isic-2018/versions/5")
ISIC_LABEL   = os.path.join(ISIC_DIR, "ISIC2018_Task3_Training_GroundTruth.csv")
ISIC_IMG_DIRS = [
    os.path.join(ISIC_DIR, "data/images/train"),
    os.path.join(ISIC_DIR, "data/images/val"),
    os.path.join(ISIC_DIR, "data/images/test"),
    os.path.join(ISIC_DIR, "data/images"),
    ISIC_DIR,
]

OUTPUT_DIR   = os.path.join(BASE_DIR, "processed")
TRAIN_DIR    = os.path.join(OUTPUT_DIR, "train")
VAL_DIR      = os.path.join(OUTPUT_DIR, "val")
TEST_DIR     = os.path.join(OUTPUT_DIR, "test")

LABEL_MAP = {'mel':0, 'nv':1, 'bcc':2, 'akiec':3, 'bkl':4, 'df':5, 'vasc':6}
CLASS_NAMES = ['mel', 'nv', 'bcc', 'akiec', 'bkl', 'df', 'vasc']

def find_image(image_id: str, search_dirs: list) -> str | None:
    """Search multiple directories for image_id with common extensions."""
    for d in search_dirs:
        for ext in ('.jpg', '.jpeg', '.png', '.JPG'):
            p = os.path.join(d, image_id + ext)
            if os.path.exists(p):
                return p
    return None

def load_ham10000() -> pd.DataFrame:
    log.info("Loading HAM10000...")
    meta = pd.read_csv(HAM_META)
    meta['image_path'] = meta['image_id'].apply(lambda x: find_image(x, HAM_IMG_DIRS))
    missing = meta['image_path'].isna().sum()
    log.warning(f"  HAM10000: {missing} images not found, dropping.")
    meta = meta.dropna(subset=['image_path'])
    meta['label'] = meta['dx'].str.lower().map(LABEL_MAP)
    meta = meta.dropna(subset=['label'])
    meta['label'] = meta['label'].astype(int)
    meta['dataset'] = 'HAM10000'
    log.info(f"  Loaded {len(meta)} HAM10000 samples.")
    return meta[['image_path', 'label', 'dataset']]

def load_isic2018() -> pd.DataFrame:
    log.info("Loading ISIC 2018...")
    gt = pd.read_csv(ISIC_LABEL)
    one_hot_cols = ['MEL', 'NV', 'BCC', 'AKIEC', 'BKL', 'DF', 'VASC']
    gt['dx'] = gt[one_hot_cols].idxmax(axis=1).str.lower()
    gt['image_path'] = gt['image'].apply(lambda x: find_image(x, ISIC_IMG_DIRS))
    missing = gt['image_path'].isna().sum()
    log.warning(f"  ISIC 2018: {missing} images not found, dropping.")
    gt = gt.dropna(subset=['image_path'])
    gt['label'] = gt['dx'].map(LABEL_MAP)
    gt = gt.dropna(subset=['label'])
    gt['label'] = gt['label'].astype(int)
    gt['dataset'] = 'ISIC2018'
    log.info(f"  Loaded {len(gt)} ISIC 2018 samples.")
    return gt[['image_path', 'label', 'dataset']]

def validate_images(df: pd.DataFrame) -> pd.DataFrame:
    log.info("Validating images (corrupt check)...")
    valid_mask = []
    for _, row in df.iterrows():
        try:
            with Image.open(row['image_path']) as img:
                img.verify()
            valid_mask.append(True)
        except Exception as e:
            log.warning(f"  Corrupt/unreadable: {row['image_path']} — {e}")
            valid_mask.append(False)
    n_dropped = sum(1 for v in valid_mask if not v)
    log.info(f"  Dropped {n_dropped} corrupt images.")
    return df[valid_mask].reset_index(drop=True)

def print_distribution(df: pd.DataFrame, title: str):
    total = len(df)
    print(f"\n{'─'*45}")
    print(f"  {title}  (total: {total})")
    print(f"{'─'*45}")
    counts = df['label'].value_counts().sort_index()
    for idx, cnt in counts.items():
        bar = '█' * int(cnt / total * 40)
        print(f"  {CLASS_NAMES[idx]:>6}  {cnt:>5}  ({cnt/total*100:5.1f}%)  {bar}")
    print(f"{'─'*45}\n")

def split_data(df: pd.DataFrame):
    train, temp = train_test_split(df, test_size=0.2, stratify=df['label'], random_state=SEED)
    val, test   = train_test_split(temp, test_size=0.5, stratify=temp['label'], random_state=SEED)
    log.info(f"Split → Train: {len(train)} | Val: {len(val)} | Test: {len(test)}")
    return train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True)

def oversample(train_df: pd.DataFrame) -> pd.DataFrame:
    log.info("Oversampling minority classes...")
    majority_count = train_df['label'].value_counts().max()
    parts = []
    for lbl in range(len(CLASS_NAMES)):
        subset = train_df[train_df['label'] == lbl]
        if len(subset) == 0:
            log.warning(f"  Class '{CLASS_NAMES[lbl]}' has 0 samples — skipping.")
            continue
        if len(subset) < majority_count:
            subset = resample(subset, replace=True, n_samples=majority_count, random_state=SEED)
            log.info(f"  {CLASS_NAMES[lbl]:>6}: upsampled to {majority_count}")
        parts.append(subset)
    balanced = pd.concat(parts).sample(frac=1, random_state=SEED).reset_index(drop=True)
    return balanced

def export_split(df: pd.DataFrame, split_dir: str, split_name: str):
    log.info(f"Exporting {split_name} split to {split_dir}...")
    counters = Counter()
    for _, row in df.iterrows():
        cls_name  = CLASS_NAMES[row['label']]
        cls_dir   = os.path.join(split_dir, cls_name)
        os.makedirs(cls_dir, exist_ok=True)

        src      = row['image_path']
        basename = os.path.splitext(os.path.basename(src))[0]
        ext      = os.path.splitext(src)[1]
        counters[basename] += 1
        count    = counters[basename]
        suffix   = f"_dup{count}" if count > 1 else ""
        dst      = os.path.join(cls_dir, f"{basename}{suffix}{ext}")

        shutil.copy2(src, dst)
    log.info(f"  ✅ {split_name}: {len(df)} files exported.")

def save_csv(df: pd.DataFrame, path: str):
    df.to_csv(path, index=False)
    log.info(f"  CSV saved → {path}")

if __name__ == "__main__":
    # 1. Load
    ham  = load_ham10000()
    isic = load_isic2018()
    df   = pd.concat([ham, isic], ignore_index=True)
    df   = df.drop_duplicates(subset=['image_path']).reset_index(drop=True)
    log.info(f"Combined dataset: {len(df)} total samples after deduplication.")

    # 2. Validate
    df = validate_images(df)

    # 3. Stats before
    print_distribution(df, "FULL DATASET — before balancing")

    # 4. Split (on unbalanced data)
    train_df, val_df, test_df = split_data(df)

    # 5. Oversample train only
    print_distribution(train_df, "TRAIN SPLIT — before oversampling")
    train_balanced = oversample(train_df)
    print_distribution(train_balanced, "TRAIN SPLIT — after oversampling")

    # 6. Export images
    if os.path.exists(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR)
        log.info("Cleared existing output directory.")
    export_split(train_balanced, TRAIN_DIR, "train")
    export_split(val_df,         VAL_DIR,   "val")
    export_split(test_df,        TEST_DIR,  "test")

    # 7. Save CSVs
    train_balanced['class_name'] = train_balanced['label'].map(dict(enumerate(CLASS_NAMES)))
    val_df['class_name']         = val_df['label'].map(dict(enumerate(CLASS_NAMES)))
    test_df['class_name']        = test_df['label'].map(dict(enumerate(CLASS_NAMES)))
    save_csv(train_balanced, os.path.join(OUTPUT_DIR, "train.csv"))
    save_csv(val_df,         os.path.join(OUTPUT_DIR, "val.csv"))
    save_csv(test_df,        os.path.join(OUTPUT_DIR, "test.csv"))

    log.info("✅ fix_dataset.py complete  Output ready in /processed/")
