"""Text feature extraction module.

Extracts dense statistical signals (lengths, uppercase ratio, digit ratio, punctuation)
and vectorized representations (TF-IDF with TruncatedSVD components) suitable for both
tree-based algorithms and neural architectures.
"""

import string
from typing import List, Optional, Sequence
import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer

from src.utils.logger import get_logger

logger = get_logger("TextFeatures")


def compute_text_statistics(series: pd.Series, prefix: str = "txt") -> pd.DataFrame:
    """Compute dense structural and lexical metrics from a series of strings.

    Metrics include character length, word count, mean word length, uppercase ratio,
    digit ratio, and punctuation frequency.

    Args:
        series: Pandas series of text records.
        prefix: Column prefix for the generated features.

    Returns:
        DataFrame containing generated statistical feature columns.
    """
    s = series.fillna("").astype(str)

    stats = pd.DataFrame(index=series.index)
    stats[f"{prefix}_char_len"] = s.str.len()
    stats[f"{prefix}_word_cnt"] = s.apply(lambda x: len(x.split()))
    stats[f"{prefix}_mean_word_len"] = stats[f"{prefix}_char_len"] / (stats[f"{prefix}_word_cnt"] + 1e-5)
    stats[f"{prefix}_upper_ratio"] = s.apply(
        lambda x: sum(1 for c in x if c.isupper()) / (len(x) + 1e-5)
    )
    stats[f"{prefix}_digit_ratio"] = s.apply(
        lambda x: sum(1 for c in x if c.isdigit()) / (len(x) + 1e-5)
    )
    punc_set = set(string.punctuation)
    stats[f"{prefix}_punc_cnt"] = s.apply(lambda x: sum(1 for c in x if c in punc_set))

    return stats


class TextFeatureExtractor:
    """TF-IDF vectorization with optional TruncatedSVD dimensionality reduction."""

    def __init__(
        self,
        max_features: int = 5000,
        ngram_range: tuple = (1, 2),
        n_components: Optional[int] = 50,
        random_state: int = 42,
    ) -> None:
        """Initialize TextFeatureExtractor.

        Args:
            max_features: Maximum vocabulary size for TfidfVectorizer.
            ngram_range: Minimum and maximum ngram boundaries.
            n_components: If specified, fit TruncatedSVD to produce dense embeddings of this size.
            random_state: Seed for TruncatedSVD.
        """
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            sublinear_tf=True,
            strip_accents="unicode",
        )
        self.n_components = n_components
        self.svd = (
            TruncatedSVD(n_components=n_components, random_state=random_state)
            if n_components
            else None
        )

    def fit(self, texts: Sequence) -> "TextFeatureExtractor":
        """Fit TF-IDF and SVD on a text sequence."""
        logger.info("Fitting TF-IDF vectorizer...")
        tfidf_mat = self.vectorizer.fit_transform(texts)
        if self.svd:
            logger.info(f"Fitting TruncatedSVD ({self.n_components} components)...")
            self.svd.fit(tfidf_mat)
        return self

    def transform(self, texts: Sequence, prefix: str = "tfidf_svd") -> np.ndarray:
        """Transform text sequence into feature array."""
        tfidf_mat = self.vectorizer.transform(texts)
        if self.svd:
            return self.svd.transform(tfidf_mat)
        return tfidf_mat.toarray()

    def fit_transform(self, texts: Sequence, prefix: str = "tfidf_svd") -> np.ndarray:
        """Fit and transform in one step."""
        return self.fit(texts).transform(texts, prefix=prefix)
