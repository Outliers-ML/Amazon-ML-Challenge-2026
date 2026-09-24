"""Smoke tests verifying integrity of pipeline components."""

import numpy as np
import pandas as pd
import pytest

from src.config import CompetitionConfig, PathConfig
from src.data.preprocessor import DataPreprocessor, TextPreprocessor
from src.evaluation.metrics import compute_metric, smape
from src.models.ensemble import WeightedBlender, rank_average
from src.models.nn_models import TabularMLP
from src.models.tree_models import LightGBMModel
from src.utils.seed import seed_everything
from src.utils.submission import validate_submission


def test_seed_everything():
    seed_everything(123)
    val1 = np.random.rand()
    seed_everything(123)
    val2 = np.random.rand()
    assert val1 == val2


def test_text_preprocessor():
    cleaner = TextPreprocessor(lower=True, strip_html=True)
    raw = "<p>Amazon <b>Echo</b> Dot &amp; Smart Speaker!   </p>"
    cleaned = cleaner.clean_text(raw)
    assert cleaned == "amazon echo dot & smart speaker!"


def test_metrics():
    y_true = np.array([10.0, 20.0, 30.0])
    y_pred = np.array([10.0, 20.0, 30.0])
    assert smape(y_true, y_pred) == 0.0
    assert compute_metric("rmse", y_true, y_pred) == 0.0


def test_tabular_preprocessor_and_model():
    # Synthetic dataset
    df = pd.DataFrame({
        "num1": [1.0, 2.0, 3.0, np.nan, 5.0] * 10,
        "cat1": ["A", "B", "A", "C", "B"] * 10,
        "txt1": ["apple phone", "samsung galaxy", "google pixel", "oneplus", "nothing"] * 10,
    })
    y = np.array([0, 1, 0, 1, 0] * 10)

    preprocessor = DataPreprocessor(
        numerical_cols=["num1"],
        categorical_cols=["cat1"],
        text_cols=["txt1"],
        scale_numeric=True,
    )
    X_proc = preprocessor.fit_transform(df)
    assert X_proc["num1"].isna().sum() == 0

    # Fit LightGBM
    lgb_model = LightGBMModel(params={"n_estimators": 10, "verbosity": -1}, is_classification=True)
    lgb_model.fit(X_proc[["num1", "cat1"]], y)
    preds = lgb_model.predict(X_proc[["num1", "cat1"]])
    assert len(preds) == len(y)


def test_submission_validator():
    sub_df = pd.DataFrame({"id": [1, 2, 3], "prediction": [0.1, 0.5, 0.9]})
    assert validate_submission(sub_df, expected_rows=3, id_col="id", prediction_col="prediction")


def test_nn_model():
    import torch
    model = TabularMLP(input_dim=10, output_dim=1)
    dummy_input = torch.randn(4, 10)
    out = model(dummy_input)
    assert out.shape == (4, 1)
