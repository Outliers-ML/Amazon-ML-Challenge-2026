from pathlib import Path
import pytest
from src.utils.experiment_tracker import ExperimentTracker


def test_tracker_lifecycle_and_ledger(tmp_path: Path):
    ledger_path = tmp_path / "experiments" / "runs.json"
    tracker = ExperimentTracker(ledger_path=ledger_path, use_mlflow=False)

    run = tracker.start_run(
        run_name="Test-LGBM-Run",
        params={
            "model_type": "LightGBM",
            "sample_train_s1": 50000,
            "max_candidates": 25,
            "learning_rate": 0.05,
        },
    )
    assert run.run_id is not None
    assert tracker.current_run is not None

    tracker.log_metrics({
        "macro_f05": 0.875,
        "precision": 0.910,
        "recall": 0.780,
        "best_tau": 0.75,
    })

    tracker.log_feature_importances({
        "name_jaro_winkler": 420.0,
        "addr_postal_code_status": 310.0,
        "name_clean_exact_match": 290.0,
    })

    tracker.log_threshold_curve(
        thresholds=[0.50, 0.60, 0.70, 0.75, 0.80, 0.90],
        scores=[0.81, 0.84, 0.86, 0.875, 0.85, 0.79],
    )

    tracker.log_partition_summary({
        "US": {"total": 663106, "matched": 226497, "singletons": 436609},
        "France": {"total": 259452, "matched": 107566, "singletons": 151886},
        "India": {"total": 809986, "matched": 254818, "singletons": 555168},
    })

    completed_run = tracker.end_run()

    assert completed_run is not None
    assert tracker.current_run is None
    assert ledger_path.exists()

    # Retrieve and verify
    retrieved = tracker.get_run(completed_run["run_id"])
    assert retrieved is not None
    assert retrieved["run_name"] == "Test-LGBM-Run"
    assert retrieved["metrics"]["macro_f05"] == 0.875
    assert retrieved["params"]["sample_train_s1"] == 50000
    assert "name_jaro_winkler" in retrieved["feature_importances"]
    assert len(retrieved["threshold_curve"]["thresholds"]) == 6
    assert retrieved["partition_summary"]["France"]["matched"] == 107566


def test_tracker_leaderboard_sorting(tmp_path: Path):
    ledger_path = tmp_path / "experiments" / "runs.json"
    tracker = ExperimentTracker(ledger_path=ledger_path, use_mlflow=False)

    tracker.start_run(run_name="Run-Low", params={"max_candidates": 10})
    tracker.log_metrics({"macro_f05": 0.720, "best_tau": 0.60})
    tracker.end_run()

    tracker.start_run(run_name="Run-High", params={"max_candidates": 25})
    tracker.log_metrics({"macro_f05": 0.890, "best_tau": 0.75})
    tracker.end_run()

    lb = tracker.get_leaderboard()
    assert len(lb) == 2
    assert lb.iloc[0]["run_name"] == "Run-High"
    assert lb.iloc[0]["macro_f05"] == 0.890
    assert lb.iloc[1]["run_name"] == "Run-Low"


def test_tracker_backfill_and_persistence(tmp_path: Path):
    ledger_path = tmp_path / "runs.json"
    tracker = ExperimentTracker(ledger_path=ledger_path, use_mlflow=False)

    tracker.record_completed_run(
        run_name="Baseline-Historical",
        params={"model_type": "LightGBM", "sample_train_s1": 100000},
        metrics={"macro_f05": 0.852, "best_tau": 0.70},
        partition_summary={"US": {"matched": 200000}},
    )

    runs = tracker.list_runs()
    assert len(runs) == 1
    assert runs[0]["run_name"] == "Baseline-Historical"
