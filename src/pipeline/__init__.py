"""Pipeline execution package for cross-validation and end-to-end training."""

from src.pipeline.trainer import CrossValidationTrainer
from src.pipeline.validator import SplitStrategy, create_cv_splits

__all__ = ["CrossValidationTrainer", "SplitStrategy", "create_cv_splits"]
