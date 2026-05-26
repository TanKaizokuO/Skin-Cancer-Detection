# Skin Cancer Detection - Dataset Preparation Pipeline

A Jupyter notebook-based pipeline designed to preprocess, balance, and split skin lesion images from the HAM10000 and ISIC 2018 datasets, preparing them for deep learning model training.

## Table of Contents
- [Project Architecture](#project-architecture)
- [Dataset Preprocessing Pipeline](#dataset-preprocessing-pipeline)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Kaggle Credentials Setup](#kaggle-credentials-setup)
- [How to Run](#how-to-run)
- [Processed Dataset Output](#processed-dataset-output)
- [Citing & References](#citing--references)

---

## Project Architecture

```
Skin-Cancer/
├── datasets/
│   ├── kmader/skin-cancer-mnist-ham10000/
│   │   └── versions/2/                         ← HAM10000 raw source
│   └── trantoanthang/isic-2018/
│       └── versions/5/                         ← ISIC 2018 raw source
├── processed/                                  ← Training-ready outputs (gitignored)
│   ├── train/                                  ← Balanced training split (37,548 images)
│   ├── val/                                    ← Validation split (1,001 images)
│   └── test/                                   ← Test split (1,002 images)
├── dataset.ipynb                               ← Pipeline execution notebook
├── DATASET.md                                  ← Detailed dataset specifications
├── .gitignore
└── README.md                                   ← This file
```

---

## Dataset Preprocessing Pipeline

The pipeline handles common issues in medical image datasets, particularly class imbalance and dataset overlap:

1. **Alignment & Mapping**: Unifies metadata columns and maps image IDs to local file paths.
2. **Deduplication**: Automatically drops overlapping images between HAM10000 and ISIC 2018 classification sources.
3. **Integrity Validation**: Runs a Pillow corruption check on every image.
4. **Stratified Split**: Splits data into train (80%), val (10%), and test (10%) splits while preserving class ratios.
5. **Oversampling (Train-Only)**: Resolves severe class imbalance by upsampling the minority classes (with replacement) to match the count of the majority class (`nv`, 5,364 images), generating a balanced training set of **37,548 images**.

---

## Getting Started

### Prerequisites
- Python 3.10+
- [uv](https://github.com/astral-sh/uv) (recommended fast Python package installer and runner)

### Installation
Clone the repository and install dependencies:
```bash
git clone https://github.com/TanKaizokuO/Skin-Cancer-Detection.git
cd Skin-Cancer
uv venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

### Kaggle Credentials Setup
The script uses the Kaggle API to download the missing Ground Truth files.
1. Place your `kaggle.json` key in the project root folder.
2. The pipeline will automatically copy it to `~/.kaggle/kaggle.json` and adjust permissions.

---

## How to Run

Open and execute the Jupyter Notebook cells inside:
[dataset.ipynb](./dataset.ipynb)

You can run it using VS Code's built-in notebook editor, JupyterLab, or Jupyter Notebook. Upon running all cells, the data distribution and class balance will be shown, and the final training-ready outputs will be written to the `processed/` directory.

---

## Processed Dataset Output

The generated dataset resides in `processed/` and contains subfolders for each of the 7 diagnostic classes:
- **mel** (Melanoma)
- **nv** (Melanocytic nevus)
- **bcc** (Basal cell carcinoma)
- **akiec** (Actinic keratosis / Bowen's disease)
- **bkl** (Benign keratosis)
- **df** (Dermatofibroma)
- **vasc** (Vascular lesion)

For detailed split statistics, class distributions, and directory layout, see [DATASET.md](./DATASET.md).

---

## Citing & References
* Tschandl, P., Rosendahl, C. & Kittler, H. The HAM10000 dataset, a large collection of multi-source dermatoscopic images of common pigmented skin lesions. *Sci Data* 5, 180161 (2018). doi:10.1038/sdata.2018.161
* Codella, N., Gutman, D., Celebi, M. E., Helba, B., Marchetti, M. A., Dusza, S., A. Halpern, A., et al. "Skin Lesion Analysis Towards Melanoma Detection: A Challenge at the International Symposium on Biomedical Imaging (ISBI) 2018, Hosted by the International Skin Imaging Collaboration (ISIC)." arXiv:1803.08417 (2018).
