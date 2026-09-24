import numpy as np
import pytest
from src.pipeline.er_trainer import ERModelTrainer, compute_macro_f05, optimize_f05_threshold


def test_macro_f05_calculation():
    # Ground truth: S1-1 matches [S2-A, S2-B], S1-2 matches [], S1-3 matches [S3-C]
    gt = {
        "S1-1": {"S2-A", "S2-B"},
        "S1-2": set(),  # singleton
        "S1-3": {"S3-C"},
    }
    # Prediction: S1-1 -> [S2-A, S2-B], S1-2 -> [], S1-3 -> [S3-C, S3-WRONG]
    preds = {
        "S1-1": ["S2-A", "S2-B"],
        "S1-2": [],
        "S1-3": ["S3-C", "S3-WRONG"],
    }
    score = compute_macro_f05(gt, preds)
    assert 0.0 < score <= 1.0
    # S1-1: P=1, R=1 -> F0.5=1.0
    # S1-2: Singleton correctly empty -> 1.0
    # S1-3: P=0.5, R=1.0 -> F0.5 = (1.25 * 0.5 * 1.0) / (0.25 * 0.5 + 1.0) = 0.625 / 1.125 = 0.5555...
    # Macro avg: (1.0 + 1.0 + 0.5555) / 3 = 0.8518...
    expected = (1.0 + 1.0 + (1.25 * 0.5 / 1.125)) / 3.0
    assert abs(score - expected) < 1e-4


def test_macro_f05_edge_cases():
    # Empty ground truth map
    assert compute_macro_f05({}, {}) == 0.0

    # Singleton predicted non-empty -> 0.0
    assert compute_macro_f05({"S1-1": set()}, {"S1-1": ["S2-A"]}) == 0.0

    # Non-singleton with empty prediction -> 0.0
    assert compute_macro_f05({"S1-1": {"S2-A"}}, {"S1-1": []}) == 0.0

    # Non-singleton missing from prediction map -> 0.0
    assert compute_macro_f05({"S1-1": {"S2-A"}}, {}) == 0.0

    # Zero overlap -> 0.0
    assert compute_macro_f05({"S1-1": {"S2-A"}}, {"S1-1": ["S2-B"]}) == 0.0


def test_threshold_optimizer():
    s1_ids = ["S1-1", "S1-1", "S1-2", "S1-2"]
    cand_ids = ["S2-1", "S2-2", "S2-3", "S2-4"]
    probs = np.array([0.9, 0.4, 0.3, 0.2])
    gt = {"S1-1": {"S2-1"}, "S1-2": set()}

    best_tau, best_score = optimize_f05_threshold(s1_ids, cand_ids, probs, gt)
    assert 0.5 <= best_tau <= 0.95
    assert best_score >= 0.99  # Threshold ~0.5-0.8 gives 1.0 on both entities


def test_er_model_trainer_fit_and_predict():
    np.random.seed(42)
    N = 100
    X = np.random.randn(N, 22).astype(np.float32)
    # Signal: feature 0 strongly correlated with y
    y = (X[:, 0] > 0.0).astype(np.int32)
    s1_groups = [f"S1-{i // 4}" for i in range(N)]

    trainer = ERModelTrainer(n_splits=3, seed=42)
    model = trainer.train(X, y, s1_groups)
    probs = model.predict_proba(X)[:, 1]
    assert len(probs) == N
    assert np.all((probs >= 0.0) & (probs <= 1.0))
