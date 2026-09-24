"""Submission validation and artifact generation module.

Provides utilities to generate, validate, and package submission files for the
Amazon ML Challenge leaderboard, verifying column presence, row count,
data types, and absence of NaNs/nulls.
"""

from datetime import datetime
from pathlib import Path
from typing import Optional, Sequence
import zipfile
import numpy as np
import pandas as pd

from src.utils.logger import get_logger

logger = get_logger("SubmissionUtil")


def validate_submission(
    submission_df: pd.DataFrame,
    expected_sample_df: Optional[pd.DataFrame] = None,
    expected_rows: Optional[int] = None,
    id_col: str = "id",
    prediction_col: str = "prediction",
) -> bool:
    """Validate a submission dataframe against expected structure and constraints.

    Args:
        submission_df: DataFrame containing the model predictions.
        expected_sample_df: Optional sample submission DataFrame to match columns and IDs with.
        expected_rows: Optional integer specifying required number of rows.
        id_col: Name of ID column in submission.
        prediction_col: Name of prediction target column.

    Returns:
        True if all validation checks pass.

    Raises:
        ValueError: If row count mismatch, missing columns, duplicate IDs, or NaNs detected.
    """
    # 1. Column existence check
    if id_col not in submission_df.columns:
        raise ValueError(f"Missing required identifier column: '{id_col}'")
    if prediction_col not in submission_df.columns:
        raise ValueError(f"Missing required prediction column: '{prediction_col}'")

    # 2. Row count check
    if expected_rows is not None and len(submission_df) != expected_rows:
        raise ValueError(
            f"Row count mismatch! Expected {expected_rows} rows, but got {len(submission_df)}"
        )

    # 3. Duplicate ID check
    if submission_df[id_col].duplicated().any():
        n_dups = submission_df[id_col].duplicated().sum()
        raise ValueError(f"Found {n_dups} duplicate IDs in submission!")

    # 4. Null / NaN / Inf checks
    if submission_df[prediction_col].isnull().any():
        n_nans = submission_df[prediction_col].isnull().sum()
        raise ValueError(f"Prediction column contains {n_nans} NaN or Null values!")

    if np.issubdtype(submission_df[prediction_col].dtype, np.number):
        if np.isinf(submission_df[prediction_col]).any():
            raise ValueError("Prediction column contains Infinite values!")

    # 5. Exact ID match check against sample submission
    if expected_sample_df is not None:
        if len(submission_df) != len(expected_sample_df):
            raise ValueError(
                f"Submission length ({len(submission_df)}) differs from sample submission ({len(expected_sample_df)})"
            )
        if not (submission_df[id_col].values == expected_sample_df[id_col].values).all():
            raise ValueError("Submission IDs do not exactly match sample submission ID ordering!")

    logger.info(
        f"Validation passed: {len(submission_df)} rows, valid columns [{id_col}, {prediction_col}], zero NaNs."
    )
    return True


def create_submission(
    ids: Sequence,
    predictions: Sequence,
    output_dir: Path | str = "submissions",
    experiment_name: str = "model",
    id_col: str = "id",
    prediction_col: str = "prediction",
    sample_submission_path: Optional[Path | str] = None,
    create_zip: bool = True,
) -> Path:
    """Format, validate, and write a submission CSV and optional zip archive.

    Args:
        ids: List or array of test IDs.
        predictions: List or array of corresponding predictions.
        output_dir: Directory where the submission CSV will be saved.
        experiment_name: Prefix used to name the output file.
        id_col: Column header for the ID column.
        prediction_col: Column header for the predictions.
        sample_submission_path: Optional path to sample submission CSV for structural comparison.
        create_zip: Whether to package the CSV into a .zip file (commonly required by hackathons).

    Returns:
        Path to the generated submission CSV file.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    submission_df = pd.DataFrame({id_col: ids, prediction_col: predictions})

    expected_sample_df = None
    if sample_submission_path:
        sample_path = Path(sample_submission_path)
        if sample_path.is_file():
            expected_sample_df = pd.read_csv(sample_path)

    validate_submission(
        submission_df=submission_df,
        expected_sample_df=expected_sample_df,
        id_col=id_col,
        prediction_col=prediction_col,
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_filename = f"sub_{experiment_name}_{timestamp}.csv"
    csv_path = out_dir / csv_filename

    submission_df.to_csv(csv_path, index=False)
    logger.info(f"Submission CSV saved to: {csv_path.resolve()}")

    if create_zip:
        zip_path = out_dir / f"sub_{experiment_name}_{timestamp}.zip"
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
            zipf.write(csv_path, arcname=csv_filename)
        logger.info(f"Compressed submission ZIP saved to: {zip_path.resolve()}")

    return csv_path
