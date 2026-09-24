"""Tabular feature engineering module.

Includes out-of-fold target encoding, frequency encoding, numerical aggregation,
and cross-feature interaction generation to boost tabular modeling performance.
"""

from typing import Dict, List, Optional
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from src.utils.logger import get_logger

logger = get_logger("TabularFeatures")


class TargetEncoder:
    """Out-of-Fold Target Encoder to prevent data leakage during cross-validation."""

    def __init__(self, categorical_cols: List[str], smoothing: float = 10.0, n_splits: int = 5, seed: int = 42) -> None:
        """Initialize TargetEncoder.

        Args:
            categorical_cols: List of categorical columns to encode.
            smoothing: Smoothing weight towards global prior mean.
            n_splits: Number of folds used for out-of-fold computation.
            seed: Random seed for fold splitting.
        """
        self.categorical_cols = categorical_cols
        self.smoothing = smoothing
        self.n_splits = n_splits
        self.seed = seed
        self.global_mean: float = 0.0
        self.encodings: Dict[str, pd.Series] = {}

    def fit_transform(self, df: pd.DataFrame, target: pd.Series) -> pd.DataFrame:
        """Compute out-of-fold target encodings on training data.

        Args:
            df: Training DataFrame containing categorical columns.
            target: Target series.

        Returns:
            DataFrame with out-of-fold target encoded features.
        """
        self.global_mean = float(target.mean())
        encoded_df = pd.DataFrame(index=df.index)
        kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=self.seed)

        for col in self.categorical_cols:
            if col not in df.columns:
                continue

            oof_col = pd.Series(index=df.index, dtype=np.float32)

            # Compute smoothed target encoding on training folds
            for tr_idx, val_idx in kf.split(df):
                tr_data, val_data = df.iloc[tr_idx], df.iloc[val_idx]
                tr_target = target.iloc[tr_idx]

                stats = tr_target.groupby(tr_data[col], observed=False).agg(["count", "mean"])
                counts = stats["count"]
                means = stats["mean"]
                smoothed = (counts * means + self.smoothing * self.global_mean) / (counts + self.smoothing)

                oof_col.iloc[val_idx] = val_data[col].map(smoothed).fillna(self.global_mean)

            encoded_df[f"{col}_te"] = oof_col

            # Compute full training encodings for test inference
            full_stats = target.groupby(df[col], observed=False).agg(["count", "mean"])
            full_counts = full_stats["count"]
            full_means = full_stats["mean"]
            self.encodings[col] = (
                (full_counts * full_means + self.smoothing * self.global_mean) / (full_counts + self.smoothing)
            )

        return encoded_df

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply learned target encodings to unseen (test) data.

        Args:
            df: Test DataFrame.

        Returns:
            DataFrame of target encoded columns.
        """
        encoded_df = pd.DataFrame(index=df.index)
        for col in self.categorical_cols:
            if col in self.encodings and col in df.columns:
                encoded_df[f"{col}_te"] = df[col].map(self.encodings[col]).fillna(self.global_mean)
        return encoded_df


def generate_tabular_features(
    df: pd.DataFrame,
    numerical_cols: Optional[List[str]] = None,
    categorical_cols: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Generate frequency encodings and basic numerical aggregations.

    Args:
        df: Input DataFrame.
        numerical_cols: List of numeric column names.
        categorical_cols: List of categorical column names.

    Returns:
        Augmented DataFrame copy containing new engineered features.
    """
    out = df.copy()

    # 1. Frequency Encodings
    if categorical_cols:
        for col in categorical_cols:
            if col in out.columns:
                freq = out[col].value_counts(normalize=True)
                out[f"{col}_freq"] = out[col].map(freq).fillna(0.0)

    # 2. Numerical summary aggregations across numeric features
    if numerical_cols and len(numerical_cols) >= 2:
        present_nums = [c for c in numerical_cols if c in out.columns]
        if len(present_nums) >= 2:
            out["num_sum"] = out[present_nums].sum(axis=1)
            out["num_mean"] = out[present_nums].mean(axis=1)
            out["num_std"] = out[present_nums].std(axis=1).fillna(0.0)
            out["num_skew"] = out[present_nums].skew(axis=1).fillna(0.0)

    return out
