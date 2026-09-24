"""Cross-Validation Training and Orchestration Engine.

Coordinates fold splitting, training of individual fold models, tracking out-of-fold
predictions, evaluation scoring, model serialization, and multi-fold test inference.
"""

from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple
import joblib
import numpy as np
import pandas as pd

from src.config import CompetitionConfig
from src.evaluation.metrics import compute_metric
from src.pipeline.validator import create_cv_splits
from src.utils.logger import get_logger

logger = get_logger("Trainer")


class CrossValidationTrainer:
    """Manages full N-fold cross validation workflow."""

    def __init__(
        self,
        config: CompetitionConfig,
        model_factory: Callable[[], Any],
        is_classification: bool = True,
    ) -> None:
        """Initialize CrossValidationTrainer.

        Args:
            config: Master experiment configuration.
            model_factory: Callable that creates a new fresh model instance.
            is_classification: Whether the problem is classification or regression.
        """
        self.config = config
        self.model_factory = model_factory
        self.is_classification = is_classification
        self.models: List[Any] = []
        self.oof_predictions: Optional[np.ndarray] = None
        self.fold_scores: List[float] = []

    def train_cv(
        self,
        X: pd.DataFrame,
        y: pd.Series | np.ndarray,
        group_series: Optional[pd.Series] = None,
    ) -> Tuple[np.ndarray, float]:
        """Execute N-Fold cross validation.

        Args:
            X: Feature matrix DataFrame.
            y: Target Series or array.
            group_series: Optional grouping series for GroupKFold.

        Returns:
            Tuple of (oof_predictions_array, overall_oof_score).
        """
        n_samples = len(X)
        n_splits = self.config.train.n_splits
        metric_name = self.config.train.metric
        save_dir = Path(self.config.paths.checkpoints_dir) / self.config.experiment_name
        save_dir.mkdir(parents=True, exist_ok=True)

        y_arr = np.asarray(y)
        oof = np.zeros(n_samples, dtype=np.float32)
        self.models = []
        self.fold_scores = []

        # Prepare splitting df
        split_df = pd.DataFrame({"target": y_arr}, index=X.index)
        group_col = None
        if group_series is not None:
            split_df["group"] = group_series.values
            group_col = "group"

        splits = create_cv_splits(
            df=split_df,
            target_col="target",
            group_col=group_col,
            n_splits=n_splits,
            stratified=self.config.train.stratified,
            seed=self.config.train.seed,
        )

        logger.info(
            f"Starting {n_splits}-fold CV for '{self.config.experiment_name}' using metric '{metric_name}'"
        )

        for fold, (train_idx, val_idx) in enumerate(splits):
            X_tr, y_tr = X.iloc[train_idx], y_arr[train_idx]
            X_va, y_val = X.iloc[val_idx], y_arr[val_idx]

            model = self.model_factory()
            model.fit(
                X_train=X_tr,
                y_train=y_tr,
                X_val=X_va,
                y_val=y_val,
                early_stopping_rounds=self.config.model.early_stopping_rounds,
                verbose=False,
            )

            val_preds = model.predict(X_va)
            oof[val_idx] = val_preds

            fold_score = compute_metric(metric_name, y_val, val_preds)
            self.fold_scores.append(fold_score)
            self.models.append(model)

            # Checkpoint model
            ckpt_path = save_dir / f"model_fold_{fold}.joblib"
            joblib.dump(model, ckpt_path)
            logger.info(f"Fold {fold + 1}/{n_splits} - {metric_name}: {fold_score:.5f}")

        overall_score = compute_metric(metric_name, y_arr, oof)
        self.oof_predictions = oof

        logger.info(f"--- Finished CV Training ---")
        logger.info(f"Mean Fold Score: {np.mean(self.fold_scores):.5f} +/- {np.std(self.fold_scores):.5f}")
        logger.info(f"Overall OOF {metric_name}: {overall_score:.5f}")

        return oof, overall_score

    def predict_test(self, X_test: pd.DataFrame) -> np.ndarray:
        """Generate ensembled test predictions by averaging across all fold models.

        Args:
            X_test: Test features DataFrame.

        Returns:
            1D array of ensembled test predictions.
        """
        if not self.models:
            raise ValueError("No models trained. Run train_cv first or load saved models.")

        fold_preds = []
        for model in self.models:
            preds = model.predict(X_test)
            fold_preds.append(preds)

        # Average predictions across folds
        return np.mean(fold_preds, axis=0)
