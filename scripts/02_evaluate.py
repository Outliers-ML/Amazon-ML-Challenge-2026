#!/usr/bin/env python3
"""CLI Evaluation Script for Amazon ML Challenge 2026.

Computes out-of-fold metrics, plots feature importances, and inspects error distributions.

Usage:
    python scripts/02_evaluate.py --config configs/model_configs/lgbm_baseline.yaml
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
from src.evaluation.metrics import compute_metric
from src.utils.logger import get_logger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate trained models.")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/base_config.yaml",
        help="Path to YAML configuration file.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = CompetitionConfig.from_yaml(args.config)
    logger = get_logger("EvalCLI")

    ckpt_dir = Path(config.paths.checkpoints_dir) / config.experiment_name
    if not ckpt_dir.is_dir():
        logger.error(f"Checkpoint directory '{ckpt_dir}' not found.")
        return

    models = sorted(list(ckpt_dir.glob("model_fold_*.joblib")))
    logger.info(f"Found {len(models)} checkpointed fold models in '{ckpt_dir}'")

    if not models:
        logger.warning("No saved model checkpoints found to evaluate.")
        return

    # Load first model to check feature importance if supported
    first_model = joblib.load(models[0])
    if hasattr(first_model, "get_feature_importance"):
        fi_df = first_model.get_feature_importance()
        logger.info("Top 10 Most Important Features:")
        logger.info("\n" + str(fi_df.head(10)))


if __name__ == "__main__":
    main()
