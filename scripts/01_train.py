#!/usr/bin/env python3
"""CLI Training Script for Amazon ML Challenge 2026.

Usage:
    python scripts/01_train.py --config configs/model_configs/lgbm_baseline.yaml
"""

import argparse
from pathlib import Path
import sys
import numpy as np

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import CompetitionConfig
from src.data.loader import DataLoaderManager
from src.data.preprocessor import DataPreprocessor
from src.models.tree_models import CatBoostModel, LightGBMModel, XGBoostModel
from src.pipeline.trainer import CrossValidationTrainer
from src.utils.logger import get_logger
from src.utils.seed import seed_everything


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train model with Cross-Validation.")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/base_config.yaml",
        help="Path to YAML experiment configuration file.",
    )
    return parser.parse_args()


def get_model_factory(model_type: str, params: dict, is_classification: bool):
    """Instantiate model factory for cross validation trainer."""
    m_type = model_type.lower()
    if m_type == "lightgbm":
        return lambda: LightGBMModel(params=params, is_classification=is_classification)
    elif m_type == "xgboost":
        return lambda: XGBoostModel(params=params, is_classification=is_classification)
    elif m_type == "catboost":
        return lambda: CatBoostModel(params=params, is_classification=is_classification)
    else:
        raise ValueError(f"Unsupported model_type: '{model_type}'")


def main() -> None:
    """Main training routine."""
    args = parse_args()
    config = CompetitionConfig.from_yaml(args.config)

    logger = get_logger(
        "TrainCLI",
        log_file=config.paths.logs_dir / f"train_{config.experiment_name}.log",
    )
    logger.info(f"Loaded config from: {args.config}")
    logger.info(f"Experiment Name: {config.experiment_name}")

    # Set seeds
    seed_everything(config.train.seed)

    # Load raw data
    data_mgr = DataLoaderManager(config)
    try:
        train_df, test_df = data_mgr.get_raw_data()
    except FileNotFoundError as e:
        logger.warning(
            f"Dataset not found yet: {e}. Place your competition data in 'data/raw/'"
        )
        return

    target_col = config.data.target_col
    id_col = config.data.id_col

    feature_cols = [c for c in train_df.columns if c not in [target_col, id_col]]
    X = train_df[feature_cols]
    y = train_df[target_col]

    # Preprocessing
    preprocessor = DataPreprocessor(
        numerical_cols=config.data.numerical_cols,
        categorical_cols=config.data.categorical_cols,
        text_cols=config.data.text_cols,
    )
    X_processed = preprocessor.fit_transform(X)

    is_classification = not np.issubdtype(y.dtype, np.floating)
    model_factory = get_model_factory(
        model_type=config.model.model_type,
        params=config.model.params,
        is_classification=is_classification,
    )

    trainer = CrossValidationTrainer(
        config=config,
        model_factory=model_factory,
        is_classification=is_classification,
    )

    oof_preds, oof_score = trainer.train_cv(X=X_processed, y=y)
    logger.info(f"Training completed successfully! OOF Score: {oof_score:.5f}")


if __name__ == "__main__":
    main()
