"""Data preprocessing and feature cleaning module.

Includes textual normalization, HTML/special characters stripping,
numerical imputation, scaling, and categorical encoding.
"""

import html
import re
from typing import List, Optional
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

from src.utils.logger import get_logger

logger = get_logger("Preprocessor")


class TextPreprocessor:
    """Cleans, normalizes, and sanitizes unstructured product text data."""

    def __init__(self, lower: bool = True, strip_html: bool = True, clean_whitespace: bool = True) -> None:
        """Initialize TextPreprocessor options.

        Args:
            lower: Whether to lowercase strings.
            strip_html: Whether to unescape and remove HTML markup tags.
            clean_whitespace: Whether to collapse multiple whitespaces.
        """
        self.lower = lower
        self.strip_html = strip_html
        self.clean_whitespace = clean_whitespace
        self._html_tag_re = re.compile(r"<[^>]+>")
        self._whitespace_re = re.compile(r"\s+")

    def clean_text(self, text: Optional[str]) -> str:
        """Sanitize a single text string.

        Args:
            text: Raw input text.

        Returns:
            Normalized clean string.
        """
        if pd.isna(text) or text is None:
            return ""

        text = str(text)

        if self.strip_html:
            text = html.unescape(text)
            text = self._html_tag_re.sub(" ", text)

        if self.lower:
            text = text.lower()

        if self.clean_whitespace:
            text = self._whitespace_re.sub(" ", text).strip()

        return text

    def transform_series(self, series: pd.Series) -> pd.Series:
        """Apply text normalization to an entire pandas Series."""
        return series.apply(self.clean_text)


class DataPreprocessor:
    """Handles structured tabular feature transformation, imputation, and encoding."""

    def __init__(
        self,
        numerical_cols: Optional[List[str]] = None,
        categorical_cols: Optional[List[str]] = None,
        text_cols: Optional[List[str]] = None,
        scale_numeric: bool = False,
    ) -> None:
        """Initialize DataPreprocessor.

        Args:
            numerical_cols: List of continuous numerical columns.
            categorical_cols: List of categorical columns.
            text_cols: List of text columns.
            scale_numeric: Whether to standardize numeric columns with StandardScaler.
        """
        self.numerical_cols = numerical_cols or []
        self.categorical_cols = categorical_cols or []
        self.text_cols = text_cols or []
        self.scale_numeric = scale_numeric

        self.text_cleaner = TextPreprocessor()
        self.scaler = StandardScaler() if scale_numeric else None
        self.categorical_encoders = {}
        self.num_medians = {}

    def fit(self, df: pd.DataFrame) -> "DataPreprocessor":
        """Compute training statistics (medians, categories, scalers).

        Args:
            df: Training DataFrame.

        Returns:
            Self.
        """
        # Fit numerical medians
        for col in self.numerical_cols:
            if col in df.columns:
                self.num_medians[col] = df[col].median()

        # Fit numerical scaler
        if self.scale_numeric and self.numerical_cols:
            present_num = [c for c in self.numerical_cols if c in df.columns]
            if present_num:
                imputed = df[present_num].fillna(self.num_medians)
                self.scaler.fit(imputed)

        # Fit categorical encoders
        for col in self.categorical_cols:
            if col in df.columns:
                le = LabelEncoder()
                # Map unknown / unseen to -1
                vals = df[col].astype(str).fillna("__MISSING__")
                le.fit(vals)
                self.categorical_encoders[col] = le

        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform dataframe using learned parameters.

        Args:
            df: DataFrame to transform.

        Returns:
            Transformed DataFrame copy.
        """
        out = df.copy()

        # 1. Clean text features
        for col in self.text_cols:
            if col in out.columns:
                out[col] = self.text_cleaner.transform_series(out[col])

        # 2. Impute and scale numerical
        for col in self.numerical_cols:
            if col in out.columns:
                median_val = self.num_medians.get(col, 0.0)
                out[col] = out[col].fillna(median_val)

        if self.scale_numeric and self.numerical_cols:
            present_num = [c for c in self.numerical_cols if c in out.columns]
            if present_num:
                out[present_num] = self.scaler.transform(out[present_num])

        # 3. Encode categoricals
        for col in self.categorical_cols:
            if col in out.columns and col in self.categorical_encoders:
                le = self.categorical_encoders[col]
                vals = out[col].astype(str).fillna("__MISSING__")
                # Map unseen categories gracefully
                mapping = {label: idx for idx, label in enumerate(le.classes_)}
                out[col] = vals.map(mapping).fillna(-1).astype(int)

        return out

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fit on DataFrame and transform it."""
        return self.fit(df).transform(df)
