"""Cross-validation split strategies for competition datasets.

Supports StratifiedKFold, GroupKFold, and standard KFold with seed determinism.
"""

from enum import Enum
from typing import Generator, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, KFold, StratifiedKFold

from src.utils.logger import get_logger

logger = get_logger("Validator")


class SplitStrategy(str, Enum):
    """Enumeration of supported split strategies."""
    STRATIFIED = "stratified"
    GROUP = "group"
    STANDARD = "standard"


def create_cv_splits(
    df: pd.DataFrame,
    target_col: Optional[str] = None,
    group_col: Optional[str] = None,
    n_splits: int = 5,
    stratified: bool = True,
    seed: int = 42,
) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
    """Generate cross-validation train/validation index pairs.

    Args:
        df: Input DataFrame.
        target_col: Target column name (required if stratified=True).
        group_col: Group column name (if using GroupKFold).
        n_splits: Number of folds.
        stratified: Whether to stratify by target distribution.
        seed: Random seed for shuffling.

    Yields:
        Tuples of (train_indices, val_indices).
    """
    if group_col and group_col in df.columns:
        logger.info(f"Using GroupKFold split on '{group_col}' with {n_splits} splits.")
        gkf = GroupKFold(n_splits=n_splits)
        groups = df[group_col].values
        for train_idx, val_idx in gkf.split(df, groups=groups):
            yield train_idx, val_idx
    elif stratified and target_col and target_col in df.columns:
        logger.info(f"Using StratifiedKFold split on '{target_col}' with {n_splits} splits (seed={seed}).")
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        y = df[target_col].values
        for train_idx, val_idx in skf.split(df, y):
            yield train_idx, val_idx
    else:
        logger.info(f"Using standard KFold with {n_splits} splits (seed={seed}).")
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
        for train_idx, val_idx in kf.split(df):
            yield train_idx, val_idx
