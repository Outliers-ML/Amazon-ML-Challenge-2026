#!/usr/bin/env python3
"""CLI Inference & Submission Generation Script for Amazon ML Challenge 2026.

Loads trained fold models, generates test predictions via ensemble averaging,
validates format integrity, and packages output CSV and ZIP into `submissions/`.

Usage:
    python scripts/03_predict_submission.py --config configs/model_configs/lgbm_baseline.yaml
"""

import argparse
from pathlib import Path
import sys
import joblib
import numpy as np
import pandas as pd

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import CompetitionConfig
from src.data.loader import DataLoaderManager
from src.data.preprocessor import DataPreprocessor
from src.utils.logger import get_logger
from src.utils.submission import create_submission


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate competition submission.")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/base_config.yaml",
        help="Path to YAML experiment configuration.",
    )
    parser.add_argument(
        "--sample-sub",
        type=str,
        default=None,
        help="Optional path to sample_submission.csv for strict format validation.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = CompetitionConfig.from_yaml(args.config)
    logger = get_logger("PredictCLI")

    ckpt_dir = Path(config.paths.checkpoints_dir) / config.experiment_name
    model_paths = sorted(list(ckpt_dir.glob("model_fold_*.joblib")))
    if not model_paths:
        logger.error(f"No checkpoint models found in: {ckpt_dir}")
        return

    logger.info(f"Loading {len(model_paths)} fold models from {ckpt_dir}")
    models = [joblib.load(p) for p in model_paths]

    # Load test data
    data_mgr = DataLoaderManager(config)
    _, test_df = data_mgr.get_raw_data()

    id_col = config.data.id_col
    target_col = config.data.target_col
    ids = test_df[id_col].values

    feature_cols = [c for c in test_df.columns if c not in [target_col, id_col]]
    X_test = test_df[feature_cols]

    # Preprocess
    preprocessor = DataPreprocessor(
        numerical_cols=config.data.numerical_cols,
        categorical_cols=config.data.categorical_cols,
        text_cols=config.data.text_cols,
    )
    X_test_proc = preprocessor.fit_transform(X_test)

    # Average fold predictions
    logger.info("Computing multi-fold ensemble predictions...")
    fold_preds = [m.predict(X_test_proc) for m in models]
    final_preds = np.mean(fold_preds, axis=0)

    # Save and validate submission
    sub_path = create_submission(
        ids=ids,
        predictions=final_preds,
        output_dir=config.paths.submissions_dir,
        experiment_name=config.experiment_name,
        id_col=id_col,
        prediction_col=target_col,
        sample_submission_path=args.sample_sub,
        create_zip=True,
    )
    logger.info(f"Submission pipeline completed successfully! Output: {sub_path}")


if __name__ == "__main__":
    main()
