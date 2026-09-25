"""Pipeline execution package for cross-validation and end-to-end training."""

from src.pipeline.er_trainer import EREnsembleTrainer, ERModelTrainer, compute_macro_f05, optimize_f05_threshold
from src.pipeline.trainer import CrossValidationTrainer
from src.pipeline.validator import SplitStrategy, create_cv_splits

__all__ = [
    "CrossValidationTrainer",
    "EREnsembleTrainer",
    "ERModelTrainer",
    "SplitStrategy",
    "compute_macro_f05",
    "create_cv_splits",
    "optimize_f05_threshold",
]
