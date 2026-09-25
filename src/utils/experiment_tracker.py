"""Experiment Tracker and Run Registry for Entity Resolution.

Tracks hyperparameter configurations, validation Macro F_0.5 metrics,
precision/recall curves, feature importances, and geographic partition statistics.
Persists runs locally to a structured JSON ledger (experiments/runs.json)
and optionally logs to MLflow.
"""

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import json
from pathlib import Path
import time
from typing import Any, Dict, List, Optional
import uuid

import pandas as pd


@dataclass
class ExperimentRun:
    """Represents a single experiment execution."""

    run_id: str
    run_name: str
    timestamp: str
    params: Dict[str, Any] = field(default_factory=dict)
    metrics: Dict[str, float] = field(default_factory=dict)
    feature_importances: Dict[str, float] = field(default_factory=dict)
    threshold_curve: Dict[str, List[float]] = field(default_factory=dict)
    partition_summary: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    duration_seconds: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert run record to JSON-serializable dictionary."""
        return asdict(self)


class ExperimentTracker:
    """Manages experiment logging, persistence, and leaderboard generation."""

    def __init__(
        self,
        ledger_path: Optional[Path | str] = None,
        use_mlflow: bool = True,
        experiment_name: str = "business-entity-resolution",
    ) -> None:
        if ledger_path is None:
            self.ledger_path = Path("experiments") / "runs.json"
        else:
            self.ledger_path = Path(ledger_path)

        self.ledger_path.parent.mkdir(parents=True, exist_ok=True)
        self.use_mlflow = use_mlflow
        self.experiment_name = experiment_name
        self.current_run: Optional[ExperimentRun] = None
        self._start_time: Optional[float] = None
        self._mlflow_run = None

        if self.use_mlflow:
            self._init_mlflow()

    def _init_mlflow(self) -> None:
        """Attempt to initialize MLflow without breaking if unavailable."""
        try:
            import mlflow
            mlflow.set_experiment(self.experiment_name)
        except Exception:
            # Fall back gracefully to local ledger only
            self.use_mlflow = False

    def start_run(
        self,
        run_name: str,
        params: Optional[Dict[str, Any]] = None,
    ) -> ExperimentRun:
        """Start a new experiment tracking session."""
        run_id = f"run-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"
        timestamp = datetime.now(timezone.utc).isoformat()

        self._start_time = time.time()
        self.current_run = ExperimentRun(
            run_id=run_id,
            run_name=run_name,
            timestamp=timestamp,
            params=dict(params or {}),
        )

        if self.use_mlflow:
            try:
                import mlflow
                self._mlflow_run = mlflow.start_run(run_name=run_name)
                if params:
                    mlflow.log_params(params)
            except Exception:
                self._mlflow_run = None

        return self.current_run

    def log_metrics(self, metrics: Dict[str, float]) -> None:
        """Record quantitative evaluation metrics for the active run."""
        if self.current_run is None:
            raise RuntimeError("No active experiment run. Call start_run() first.")
        self.current_run.metrics.update(metrics)

        if self.use_mlflow and self._mlflow_run:
            try:
                import mlflow
                mlflow.log_metrics(metrics)
            except Exception:
                pass

    def log_feature_importances(self, importances: Dict[str, float]) -> None:
        """Record model feature importance weights."""
        if self.current_run is None:
            raise RuntimeError("No active experiment run. Call start_run() first.")
        self.current_run.feature_importances.update(importances)

    def log_threshold_curve(
        self,
        thresholds: List[float],
        scores: List[float],
    ) -> None:
        """Record candidate decision thresholds and resulting Macro F_0.5 scores."""
        if self.current_run is None:
            raise RuntimeError("No active experiment run. Call start_run() first.")
        self.current_run.threshold_curve = {
            "thresholds": [float(t) for t in thresholds],
            "scores": [float(s) for s in scores],
        }

    def log_partition_summary(self, summary: Dict[str, Dict[str, Any]]) -> None:
        """Record per-country prediction and matching partition breakdown."""
        if self.current_run is None:
            raise RuntimeError("No active experiment run. Call start_run() first.")
        self.current_run.partition_summary.update(summary)

    def end_run(self) -> Optional[Dict[str, Any]]:
        """End the active run, persist to ledger, and return the run dictionary."""
        if self.current_run is None:
            return None

        if self._start_time is not None:
            self.current_run.duration_seconds = round(time.time() - self._start_time, 2)

        run_dict = self.current_run.to_dict()
        self._save_run_to_ledger(run_dict)

        if self.use_mlflow and self._mlflow_run:
            try:
                import mlflow
                mlflow.end_run()
            except Exception:
                pass
            self._mlflow_run = None

        self.current_run = None
        self._start_time = None
        return run_dict

    def record_completed_run(
        self,
        run_name: str,
        params: Optional[Dict[str, Any]] = None,
        metrics: Optional[Dict[str, float]] = None,
        feature_importances: Optional[Dict[str, float]] = None,
        threshold_curve: Optional[Dict[str, List[float]]] = None,
        partition_summary: Optional[Dict[str, Dict[str, Any]]] = None,
        duration_seconds: float = 0.0,
        run_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Directly record an already completed or historical experiment run."""
        rid = run_id or f"run-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"
        run_obj = ExperimentRun(
            run_id=rid,
            run_name=run_name,
            timestamp=datetime.now(timezone.utc).isoformat(),
            params=dict(params or {}),
            metrics=dict(metrics or {}),
            feature_importances=dict(feature_importances or {}),
            threshold_curve=dict(threshold_curve or {}),
            partition_summary=dict(partition_summary or {}),
            duration_seconds=duration_seconds,
        )
        run_dict = run_obj.to_dict()
        self._save_run_to_ledger(run_dict)
        return run_dict

    def _save_run_to_ledger(self, run_dict: Dict[str, Any]) -> None:
        """Atomically append or update a run entry in the local JSON ledger."""
        runs = self.list_runs()
        # Update existing run if id matches, else append
        updated = False
        for idx, r in enumerate(runs):
            if r.get("run_id") == run_dict["run_id"]:
                runs[idx] = run_dict
                updated = True
                break
        if not updated:
            runs.append(run_dict)

        temp_path = self.ledger_path.with_suffix(".tmp")
        with open(temp_path, "w", encoding="utf-8") as f:
            json.dump(runs, f, indent=2)
        temp_path.replace(self.ledger_path)

    def list_runs(self) -> List[Dict[str, Any]]:
        """Return all recorded experiment runs from the ledger."""
        if not self.ledger_path.exists():
            return []
        try:
            with open(self.ledger_path, "r", encoding="utf-8") as f:
                content = f.read().strip()
                if not content:
                    return []
                data = json.loads(content)
                return data if isinstance(data, list) else []
        except Exception:
            return []

    def get_run(self, run_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve run metadata by run_id."""
        for r in self.list_runs():
            if r.get("run_id") == run_id:
                return r
        return None

    def get_leaderboard(self) -> pd.DataFrame:
        """Return a structured leaderboard DataFrame of all runs."""
        runs = self.list_runs()
        if not runs:
            return pd.DataFrame()

        rows = []
        for r in runs:
            m = r.get("metrics", {})
            p = r.get("params", {})
            row = {
                "run_id": r.get("run_id"),
                "run_name": r.get("run_name"),
                "timestamp": r.get("timestamp"),
                "macro_f05": m.get("macro_f05", 0.0),
                "best_tau": m.get("best_tau"),
                "precision": m.get("precision"),
                "recall": m.get("recall"),
                "sample_train_s1": p.get("sample_train_s1"),
                "max_candidates": p.get("max_candidates"),
                "duration_s": r.get("duration_seconds", 0.0),
            }
            rows.append(row)

        df = pd.DataFrame(rows)
        if "macro_f05" in df.columns:
            df = df.sort_values(by="macro_f05", ascending=False).reset_index(drop=True)
        return df
