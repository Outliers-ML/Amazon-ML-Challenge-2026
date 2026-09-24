# 🚀 Amazon ML Challenge 2026 — End-to-End Solution Framework

[![Python 3.14](https://img.shields.io/badge/Python-3.14-blue.svg)](https://www.python.org/)
[![PyTorch 2.14](https://img.shields.io/badge/PyTorch-2.14_CUDA_13-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An enterprise-grade, modular, and competitive machine learning framework designed for the **Amazon ML Challenge 2026** (72-hour Hackathon). Engineered for maximum throughput, leak-free validation, multimodal modeling (Text, Vision, Tabular), and seamless ensemble stacking.

---

## 📌 Competition Overview

- **Host:** Amazon & Unstop
- **Format:** 72-Hour ML Hackathon (25 Sep – 27 Sep 2026)
- **Goal:** Build predictive solutions on real-world Amazon catalog/customer datasets
- **Deliverables:**
  1. Validated leaderboard prediction file (`.csv` / `.zip`)
  2. 1–2 page solution methodology document
  3. Clean, modular, reproducible code repository

---

## 📂 Repository Architecture

```text
Amazon/
├── configs/                          # Experiment & model configurations (YAML)
│   ├── base_config.yaml              # Global default settings & hyperparameters
│   └── model_configs/
│       ├── lgbm_baseline.yaml        # LightGBM GBDT experiment configuration
│       ├── catboost_baseline.yaml    # CatBoost GPU experiment configuration
│       └── xgboost_baseline.yaml     # XGBoost Hist experiment configuration
│
├── data/                             # Dataset storage (Ignored by Git)
│   ├── raw/                          # Original competition train.csv, test.csv, images
│   ├── processed/                    # Cleaned, imputed, and transformed features
│   └── interim/                      # Temporary preprocessed shards
│
├── notebooks/                        # Interactive analysis and experimentation
│   └── 01_exploratory_data_analysis.ipynb # Full EDA, target distribution & sanity checks
│
├── src/                              # Core modular ML package
│   ├── __init__.py
│   ├── config.py                     # Dataclasses & YAML configuration loader
│   ├── data/
│   │   ├── dataset.py                # PyTorch Tabular & Multimodal Datasets
│   │   ├── loader.py                 # Auto-detecting CSV/Parquet data loader
│   │   └── preprocessor.py           # Text cleaning, numeric scaling, label encoding
│   ├── features/
│   │   ├── tabular_features.py       # Out-of-fold target & frequency encodings
│   │   ├── text_features.py          # TF-IDF + TruncatedSVD & lexical statistics
│   │   └── image_features.py         # Timm / ResNet pretrained image embedding extractor
│   ├── models/
│   │   ├── ensemble.py               # Rank averaging, SciPy SLSQP blending & stacking
│   │   ├── nn_models.py              # Residual Tabular MLP & Multimodal Late-Fusion Net
│   │   └── tree_models.py            # Unified LightGBM, XGBoost, CatBoost interfaces
│   ├── pipeline/
│   │   ├── trainer.py                # N-Fold Cross-Validation engine with OOF tracking
│   │   └── validator.py              # Stratified, Group, and standard KFold splitters
│   ├── evaluation/
│   │   └── metrics.py                # F1 (Macro/Micro), Accuracy, RMSE, MAE, MAPE, SMAPE
│   └── utils/
│       ├── logger.py                 # Colored console + rotating file logger
│       ├── seed.py                   # Deterministic seeding for Python, NumPy, PyTorch
│       └── submission.py             # Format verification and zip packager
│
├── scripts/                          # Reproducible CLI pipelines
│   ├── 00_download_data.sh           # Helper script to unpack raw competition data
│   ├── 01_train.py                   # Model training and cross-validation entry point
│   ├── 02_evaluate.py                # Out-of-fold evaluation and feature importance CLI
│   └── 03_predict_submission.py      # Test inference and submission generator
│
├── checkpoints/                      # Serialized model weights & fold artifacts
├── submissions/                      # Generated submission files (.csv and .zip)
├── logs/                             # Execution and training logs
├── tests/                            # PyTest unit and smoke test suite
│   └── test_smoke.py
├── requirements.txt                  # Locked Python package dependencies
├── pyproject.toml                    # Editable build and pytest configuration
└── .gitignore                        # Strict rules protecting data, weights, and caches
```

---

## ⚡ Environment & Hardware Setup

The codebase is configured for the **`outliers`** Conda environment on Linux with high-performance GPU acceleration (**NVIDIA A100 80GB VRAM**, CUDA 13.0).

### Activate Conda Environment
```bash
conda activate outliers
```

### Install Core Packages (Already Configured)
```bash
pip install -r requirements.txt
pip install -e .
```

### Jupyter Kernel
The environment is registered as a Jupyter kernel:
```bash
python -m ipykernel install --user --name outliers --display-name "Python (outliers)"
```

---

## 🏁 Workflow & Quick Start

### 1. Place Raw Competition Data
Once the dataset is released, place or unpack it into `data/raw/`:
```bash
# Using the helper script:
bash scripts/00_download_data.sh path/to/dataset.zip

# Or manually copy:
cp path/to/train.csv data/raw/train.csv
cp path/to/test.csv data/raw/test.csv
```

### 2. Configure Experiment
Edit column names and parameters in `configs/base_config.yaml` or create a new YAML in `configs/model_configs/`:
```yaml
experiment_name: "lgbm_v1"
data:
  train_file: "train.csv"
  test_file: "test.csv"
  target_col: "price"       # Set your target column name
  id_col: "sample_id"       # Set your unique identifier
  text_cols: ["title", "description"]
  numerical_cols: ["item_weight"]
  categorical_cols: ["category_id"]
```

### 3. Run Training with Cross-Validation
Train an N-Fold ensemble with early stopping, out-of-fold scoring, and automatic checkpointing:
```bash
python scripts/01_train.py --config configs/model_configs/lgbm_baseline.yaml
```

### 4. Evaluate Models & Feature Importance
```bash
python scripts/02_evaluate.py --config configs/model_configs/lgbm_baseline.yaml
```

### 5. Generate Validated Submission
Computes fold-averaged predictions on test data, verifies formatting (zero NaNs, matching IDs), and exports both `.csv` and `.zip` into `submissions/`:
```bash
python scripts/03_predict_submission.py --config configs/model_configs/lgbm_baseline.yaml
```

---

## 🧪 Running Smoke & Unit Tests
Run the test suite to verify pipeline integrity:
```bash
pytest tests/
```

---

## 🔄 Git Version Control Workflow

The local Git repository is initialized on branch `main`.

### Check Git Status
```bash
git status
```

### Commit Changes
```bash
git add .
git commit -m "feat: setup competitive ML architecture for Amazon ML Challenge 2026"
```

### Adding Remote Later
When you create a GitHub / GitLab / AWS CodeCommit repository:
```bash
git remote add origin <YOUR_REMOTE_URL>
git branch -M main
git push -u origin main
```

---

## 🛡️ Coding & Documentation Guidelines

- **Docstrings:** All classes, functions, and modules strictly follow **Google Python Style** docstrings with full type annotations.
- **Reproducibility:** Universal deterministic seeding via `src.utils.seed.seed_everything(seed=42)`.
- **Validation Guard:** Out-of-fold target encodings and CV splitting are isolated to prevent data leakage between folds.
