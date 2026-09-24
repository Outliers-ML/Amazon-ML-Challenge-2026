"""Unified wrappers for Gradient Boosted Decision Tree models.

Provides standardized classes for LightGBM, XGBoost, and CatBoost with uniform
interfaces for fitting, predicting, early stopping, and feature importances.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd

from src.utils.logger import get_logger

logger = get_logger("TreeModels")


class BaseTreeModel(ABC):
    """Abstract base class establishing the common interface for tree models."""

    def __init__(self, params: Optional[Dict[str, Any]] = None, is_classification: bool = True) -> None:
        self.params = params or {}
        self.is_classification = is_classification
        self.model = None
        self.feature_names: List[str] = []

    @abstractmethod
    def fit(
        self,
        X_train: pd.DataFrame | np.ndarray,
        y_train: pd.Series | np.ndarray,
        X_val: Optional[pd.DataFrame | np.ndarray] = None,
        y_val: Optional[pd.Series | np.ndarray] = None,
        early_stopping_rounds: int = 50,
        verbose: bool = False,
    ) -> "BaseTreeModel":
        """Fit model to training partition with optional validation set for early stopping."""
        pass

    @abstractmethod
    def predict(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Generate final predictions."""
        pass

    @abstractmethod
    def predict_proba(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Generate class probability distributions for classification."""
        pass

    @abstractmethod
    def get_feature_importance(self) -> pd.DataFrame:
        """Return DataFrame ranking features by importance."""
        pass


class LightGBMModel(BaseTreeModel):
    """Wrapper around LightGBM Regressor / Classifier."""

    def fit(
        self,
        X_train: pd.DataFrame | np.ndarray,
        y_train: pd.Series | np.ndarray,
        X_val: Optional[pd.DataFrame | np.ndarray] = None,
        y_val: Optional[pd.Series | np.ndarray] = None,
        early_stopping_rounds: int = 50,
        verbose: bool = False,
    ) -> "LightGBMModel":
        import lightgbm as lgb

        if isinstance(X_train, pd.DataFrame):
            self.feature_names = list(X_train.columns)

        default_params = {
            "random_state": 42,
            "n_estimators": 1500,
            "learning_rate": 0.05,
            "verbosity": -1,
        }
        default_params.update(self.params)

        if self.is_classification:
            self.model = lgb.LGBMClassifier(**default_params)
        else:
            self.model = lgb.LGBMRegressor(**default_params)

        callbacks = []
        if early_stopping_rounds and X_val is not None:
            callbacks.append(lgb.early_stopping(stopping_rounds=early_stopping_rounds, verbose=verbose))
        if verbose:
            callbacks.append(lgb.log_evaluation(period=100))

        eval_set = [(X_val, y_val)] if (X_val is not None and y_val is not None) else None

        self.model.fit(
            X_train,
            y_train,
            eval_set=eval_set,
            callbacks=callbacks if callbacks else None,
        )
        return self

    def predict(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    def predict_proba(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        if not self.is_classification:
            raise ValueError("predict_proba is only available for classification models.")
        return self.model.predict_proba(X)

    def get_feature_importance(self) -> pd.DataFrame:
        importances = self.model.feature_importances_
        names = self.feature_names or [f"f_{i}" for i in range(len(importances))]
        df = pd.DataFrame({"feature": names, "importance": importances})
        return df.sort_values(by="importance", ascending=False).reset_index(drop=True)


class XGBoostModel(BaseTreeModel):
    """Wrapper around XGBoost Regressor / Classifier."""

    def fit(
        self,
        X_train: pd.DataFrame | np.ndarray,
        y_train: pd.Series | np.ndarray,
        X_val: Optional[pd.DataFrame | np.ndarray] = None,
        y_val: Optional[pd.Series | np.ndarray] = None,
        early_stopping_rounds: int = 50,
        verbose: bool = False,
    ) -> "XGBoostModel":
        import xgboost as xgb

        if isinstance(X_train, pd.DataFrame):
            self.feature_names = list(X_train.columns)

        default_params = {
            "random_state": 42,
            "n_estimators": 1500,
            "learning_rate": 0.05,
            "tree_method": "hist",
        }
        default_params.update(self.params)

        if early_stopping_rounds and X_val is not None:
            default_params["early_stopping_rounds"] = early_stopping_rounds

        if self.is_classification:
            self.model = xgb.XGBClassifier(**default_params)
        else:
            self.model = xgb.XGBRegressor(**default_params)

        eval_set = [(X_val, y_val)] if (X_val is not None and y_val is not None) else None

        self.model.fit(
            X_train,
            y_train,
            eval_set=eval_set,
            verbose=verbose,
        )
        return self

    def predict(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    def predict_proba(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        if not self.is_classification:
            raise ValueError("predict_proba is only available for classification models.")
        return self.model.predict_proba(X)

    def get_feature_importance(self) -> pd.DataFrame:
        importances = self.model.feature_importances_
        names = self.feature_names or [f"f_{i}" for i in range(len(importances))]
        df = pd.DataFrame({"feature": names, "importance": importances})
        return df.sort_values(by="importance", ascending=False).reset_index(drop=True)


class CatBoostModel(BaseTreeModel):
    """Wrapper around CatBoost Regressor / Classifier."""

    def fit(
        self,
        X_train: pd.DataFrame | np.ndarray,
        y_train: pd.Series | np.ndarray,
        X_val: Optional[pd.DataFrame | np.ndarray] = None,
        y_val: Optional[pd.Series | np.ndarray] = None,
        early_stopping_rounds: int = 50,
        verbose: bool = False,
    ) -> "CatBoostModel":
        import catboost as cb

        if isinstance(X_train, pd.DataFrame):
            self.feature_names = list(X_train.columns)

        default_params = {
            "random_seed": 42,
            "iterations": 1500,
            "learning_rate": 0.05,
            "verbose": 100 if verbose else 0,
        }
        default_params.update(self.params)

        if early_stopping_rounds and X_val is not None:
            default_params["early_stopping_rounds"] = early_stopping_rounds

        if self.is_classification:
            self.model = cb.CatBoostClassifier(**default_params)
        else:
            self.model = cb.CatBoostRegressor(**default_params)

        eval_set = (X_val, y_val) if (X_val is not None and y_val is not None) else None

        self.model.fit(X_train, y_train, eval_set=eval_set, verbose=verbose)
        return self

    def predict(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    def predict_proba(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        if not self.is_classification:
            raise ValueError("predict_proba is only available for classification models.")
        return self.model.predict_proba(X)

    def get_feature_importance(self) -> pd.DataFrame:
        importances = self.model.get_feature_importance()
        names = self.feature_names or [f"f_{i}" for i in range(len(importances))]
        df = pd.DataFrame({"feature": names, "importance": importances})
        return df.sort_values(by="importance", ascending=False).reset_index(drop=True)
