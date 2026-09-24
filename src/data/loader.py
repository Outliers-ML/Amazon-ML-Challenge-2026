"""Dataset loading and validation management.

Handles loading of competition training and test sets in various formats (CSV, parquet),
sanity checks schema, verifies data integrity, and extracts partitions.
"""

from pathlib import Path
from typing import Optional, Tuple
import pandas as pd

from src.config import CompetitionConfig
from src.utils.logger import get_logger

logger = get_logger("DataLoader")


def load_competition_data(
    train_path: Path | str,
    test_path: Path | str,
    id_col: Optional[str] = None,
    target_col: Optional[str] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load train and test datasets from disk with format auto-detection.

    Supports .csv, .parquet, and compressed .csv.gz / .zip formats.

    Args:
        train_path: Filepath to training data file.
        test_path: Filepath to test data file.
        id_col: Optional column name for ID to check presence.
        target_col: Optional column name for target to verify presence in train.

    Returns:
        Tuple of (train_df, test_df).

    Raises:
        FileNotFoundError: If either path does not exist.
        ValueError: If mandatory ID or target columns are absent.
    """
    tr_p = Path(train_path)
    te_p = Path(test_path)

    if not tr_p.is_file():
        raise FileNotFoundError(f"Train dataset file not found at: {tr_p.resolve()}")
    if not te_p.is_file():
        raise FileNotFoundError(f"Test dataset file not found at: {te_p.resolve()}")

    logger.info(f"Loading training data from: {tr_p.name}")
    if tr_p.suffix == ".parquet":
        train_df = pd.read_parquet(tr_p)
    else:
        train_df = pd.read_csv(tr_p)

    logger.info(f"Loading test data from: {te_p.name}")
    if te_p.suffix == ".parquet":
        test_df = pd.read_parquet(te_p)
    else:
        test_df = pd.read_csv(te_p)

    logger.info(f"Train shape: {train_df.shape} | Test shape: {test_df.shape}")

    # Integrity validations
    if id_col:
        if id_col not in train_df.columns:
            raise ValueError(f"id_col '{id_col}' missing from training dataframe.")
        if id_col not in test_df.columns:
            raise ValueError(f"id_col '{id_col}' missing from test dataframe.")

    if target_col:
        if target_col not in train_df.columns:
            raise ValueError(f"target_col '{target_col}' missing from training dataframe.")

    return train_df, test_df


class DataLoaderManager:
    """Manages reading and access to raw and processed datasets based on configuration."""

    def __init__(self, config: CompetitionConfig) -> None:
        """Initialize DataLoaderManager with an experiment configuration.

        Args:
            config: CompetitionConfig instance.
        """
        self.config = config
        self.raw_dir = config.paths.raw_data_dir
        self.processed_dir = config.paths.processed_data_dir

    def get_raw_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load raw training and test data according to the config filenames."""
        train_file = self.raw_dir / self.config.data.train_file
        test_file = self.raw_dir / self.config.data.test_file
        return load_competition_data(
            train_path=train_file,
            test_path=test_file,
            id_col=self.config.data.id_col,
            target_col=self.config.data.target_col,
        )
