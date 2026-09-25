"""Model training and Macro F_0.5 decision threshold calibration for Entity Resolution."""

import math
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Union
import catboost as cb
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
import xgboost as xgb


def filter_candidates_for_cross_encoder(
    candidates: Union[Sequence[Dict[str, Any]], pd.DataFrame],
    cutoff: float = 0.12,
    max_keep: int = 8,
    prob_key: str = "p_gbdt",
    s1_key: str = "s1_id",
) -> Union[List[Dict[str, Any]], pd.DataFrame]:
    """Filter candidate items where probability >= cutoff and retain at most max_keep.

    Supports both Sequence[Dict[str, Any]] and pd.DataFrame inputs.
    """
    if isinstance(candidates, pd.DataFrame):
        return filter_candidates_dataframe(
            candidates,
            cutoff=cutoff,
            max_keep=max_keep,
            prob_col=prob_key,
            s1_col=s1_key,
        )

    if candidates is None or len(candidates) == 0:
        return []
    filtered = []
    for c in candidates:
        if c is None or not isinstance(c, dict):
            continue
        val = c.get(prob_key)
        if val is None:
            continue
        try:
            score = float(val)
        except (ValueError, TypeError):
            continue
        if math.isnan(score):
            continue
        if score >= cutoff:
            filtered.append(c)

    if not filtered:
        return []

    # Sort descending by probability with stable preservation of ties
    filtered.sort(key=lambda item: float(item.get(prob_key, 0.0)), reverse=True)
    return filtered[:max_keep]


def filter_candidates_dataframe(
    df: pd.DataFrame,
    cutoff: float = 0.12,
    max_keep: int = 8,
    prob_col: str = "p_gbdt",
    s1_col: str = "s1_id",
) -> pd.DataFrame:
    """Filter DataFrame of candidates where prob_col >= cutoff and retain at most max_keep per s1_col."""
    if df is None or df.empty:
        return pd.DataFrame(columns=df.columns if df is not None else None)

    if prob_col not in df.columns:
        return pd.DataFrame(columns=df.columns)

    prob_series = pd.to_numeric(df[prob_col], errors="coerce").fillna(0.0)
    filtered = df[prob_series >= cutoff].copy()
    if filtered.empty:
        return filtered

    # Sort descending by prob_col
    filtered = filtered.sort_values(by=prob_col, ascending=False, kind="mergesort")

    if s1_col in filtered.columns:
        return filtered.groupby(s1_col, as_index=False, group_keys=False).head(max_keep)
    return filtered.head(max_keep)


def _is_nan(val: Any) -> bool:
    """Check if value is None, unparseable, or NaN (handles float, np.float32, np.float64, etc.)."""
    if val is None:
        return True
    try:
        return bool(math.isnan(float(val)))
    except (ValueError, TypeError):
        return True


def compute_meta_probability(
    p_ce: Optional[float],
    p_xgb: float,
    p_cat: float,
    w_ce: float = 0.50,
    w_xgb: float = 0.25,
    w_cat: float = 0.25,
) -> float:
    """Compute meta probability blending Cross-Encoder and GBDT model predictions.

    If p_ce is present:
        total_w = w_ce + w_xgb + w_cat
        p_final = (w_ce / total_w) * p_ce + (w_xgb / total_w) * p_xgb + (w_cat / total_w) * p_cat
    If p_ce is None/NaN:
        gbdt_total_w = w_xgb + w_cat
        p_final = (w_xgb / gbdt_total_w) * p_xgb + (w_cat / gbdt_total_w) * p_cat
    """
    has_ce = not _is_nan(p_ce)

    if has_ce:
        total_w = float(w_ce + w_xgb + w_cat)
        if total_w > 0:
            p_final = (w_ce / total_w) * float(p_ce) + (w_xgb / total_w) * float(p_xgb) + (w_cat / total_w) * float(p_cat)
        else:
            p_final = (float(p_ce) + float(p_xgb) + float(p_cat)) / 3.0
    else:
        gbdt_total_w = float(w_xgb + w_cat)
        if gbdt_total_w > 0:
            p_final = (w_xgb / gbdt_total_w) * float(p_xgb) + (w_cat / gbdt_total_w) * float(p_cat)
        else:
            p_final = (float(p_xgb) + float(p_cat)) / 2.0

    return float(np.clip(p_final, 0.0, 1.0))


def compute_meta_probability_vectorized(
    p_ce_arr: Sequence[Optional[float]],
    p_xgb_arr: np.ndarray,
    p_cat_arr: np.ndarray,
    w_ce: float = 0.50,
    w_xgb: float = 0.25,
    w_cat: float = 0.25,
) -> np.ndarray:
    """Vectorized meta probability blending Cross-Encoder and GBDT model predictions."""
    n = len(p_ce_arr)
    if n == 0:
        return np.array([], dtype=np.float32)

    p_xgb = np.asarray(p_xgb_arr, dtype=np.float32)
    p_cat = np.asarray(p_cat_arr, dtype=np.float32)

    clean_ce_list = []
    for x in p_ce_arr:
        if _is_nan(x):
            clean_ce_list.append(np.nan)
        else:
            clean_ce_list.append(float(x))
    p_ce_clean = np.array(clean_ce_list, dtype=np.float32)

    has_ce = ~np.isnan(p_ce_clean)
    p_final = np.zeros(n, dtype=np.float32)

    if np.any(has_ce):
        total_w = float(w_ce + w_xgb + w_cat)
        if total_w > 0:
            p_final[has_ce] = (
                (w_ce / total_w) * p_ce_clean[has_ce]
                + (w_xgb / total_w) * p_xgb[has_ce]
                + (w_cat / total_w) * p_cat[has_ce]
            )
        else:
            p_final[has_ce] = (p_ce_clean[has_ce] + p_xgb[has_ce] + p_cat[has_ce]) / 3.0

    no_ce = ~has_ce
    if np.any(no_ce):
        gbdt_total_w = float(w_xgb + w_cat)
        if gbdt_total_w > 0:
            p_final[no_ce] = (
                (w_xgb / gbdt_total_w) * p_xgb[no_ce]
                + (w_cat / gbdt_total_w) * p_cat[no_ce]
            )
        else:
            p_final[no_ce] = 0.5 * p_xgb[no_ce] + 0.5 * p_cat[no_ce]

    return np.clip(p_final, 0.0, 1.0)


def mine_hard_negatives(
    blocker_candidates: Dict[str, Sequence[str]],
    ground_truth: Dict[str, Sequence[str]],
    sample_size: int = 500000,
    seed: int = 42,
    max_negatives_per_entity: int = 10,
) -> List[Tuple[str, str, int]]:
    """Mine ground truth positives and blocker false positive hard negatives.

    Samples up to sample_size S1 entities deterministically using np.random.RandomState(seed).
    For each entity:
      - Ground truth matches in blocker candidates (and missed ground truth matches): label = 1.
      - Candidates retrieved by blocker that are NOT in ground truth: label = 0, capped at max_negatives_per_entity.
    Returns list of (s1_id, cand_id, label).
    """
    if not blocker_candidates and not ground_truth:
        return []
    if sample_size <= 0:
        return []

    all_s1 = sorted(set(ground_truth.keys()) | set(blocker_candidates.keys()))
    if not all_s1:
        return []

    if len(all_s1) > sample_size:
        rng = np.random.RandomState(seed)
        sampled_indices = rng.choice(len(all_s1), size=sample_size, replace=False)
        sampled_s1 = [all_s1[i] for i in sorted(sampled_indices)]
    else:
        sampled_s1 = all_s1

    mined_pairs: List[Tuple[str, str, int]] = []

    for s1_id in sampled_s1:
        gt_targets = ground_truth.get(s1_id, [])
        if isinstance(gt_targets, str):
            gt_targets = [gt_targets]
        gt_set = set(str(t).strip() for t in gt_targets if str(t).strip())

        cands = blocker_candidates.get(s1_id, [])
        if isinstance(cands, str):
            cands = [cands]
        cand_list = [str(c).strip() for c in cands if str(c).strip()]

        seen_pairs: Set[Tuple[str, str]] = set()

        # 1. Ground truth matches in blocker candidates: label = 1
        for cand_id in cand_list:
            if cand_id in gt_set:
                pair = (s1_id, cand_id)
                if pair not in seen_pairs:
                    seen_pairs.add(pair)
                    mined_pairs.append((s1_id, cand_id, 1))

        # 2. Missed ground truth matches (not retrieved by blocker): label = 1
        for true_id in sorted(gt_set):
            pair = (s1_id, true_id)
            if pair not in seen_pairs:
                seen_pairs.add(pair)
                mined_pairs.append((s1_id, true_id, 1))

        # 3. Blocker candidates NOT in ground truth (hard negatives): label = 0
        neg_count = 0
        if max_negatives_per_entity > 0:
            for cand_id in cand_list:
                if cand_id not in gt_set:
                    pair = (s1_id, cand_id)
                    if pair not in seen_pairs:
                        seen_pairs.add(pair)
                        mined_pairs.append((s1_id, cand_id, 0))
                        neg_count += 1
                        if neg_count >= max_negatives_per_entity:
                            break

    return mined_pairs


def compute_macro_f05(gt_map: Dict[str, Any], pred_map: Dict[str, Sequence[str]]) -> float:
    """Compute official competition Macro F_0.5 score."""
    f05_scores = []
    for s1_id, true_targets in gt_map.items():
        true_targets = set(true_targets) if not isinstance(true_targets, set) else true_targets
        predicted = set(pred_map.get(s1_id, []))

        # Singleton logic
        if len(true_targets) == 0:
            f05_scores.append(1.0 if len(predicted) == 0 else 0.0)
            continue

        if len(predicted) == 0:
            f05_scores.append(0.0)
            continue

        tp = len(true_targets & predicted)
        precision = tp / len(predicted)
        recall = tp / len(true_targets)

        if precision == 0 and recall == 0:
            f05_scores.append(0.0)
        else:
            # F_0.5 = 1.25 * P * R / (0.25 * P + R)
            score = (1.25 * precision * recall) / (0.25 * precision + recall)
            f05_scores.append(score)

    return float(np.mean(f05_scores)) if f05_scores else 0.0


def optimize_f05_threshold(
    s1_ids: Sequence[str],
    cand_ids: Sequence[str],
    probs: np.ndarray,
    gt_map: Dict[str, Any],
    tau_steps: int = 25,
    min_tau: float = 0.50,
    max_tau: float = 0.95,
    return_curve: bool = False,
) -> Tuple[float, float] | Tuple[float, float, List[float], List[float]]:
    """Find threshold tau* in [min_tau, max_tau] maximizing macro F_0.5."""
    if not (len(s1_ids) == len(cand_ids) == len(probs)):
        raise ValueError(
            f"Mismatched input lengths: len(s1_ids)={len(s1_ids)}, "
            f"len(cand_ids)={len(cand_ids)}, len(probs)={len(probs)}"
        )

    thresholds = [float(t) for t in np.linspace(min_tau, max_tau, tau_steps)]
    scores = []
    for tau in thresholds:
        pred_map: Dict[str, List[str]] = {s1: [] for s1 in gt_map.keys()}
        mask = probs >= tau
        for s1, cand, keep in zip(s1_ids, cand_ids, mask):
            if keep and s1 in pred_map:
                pred_map[s1].append(cand)

        score = compute_macro_f05(gt_map, pred_map)
        scores.append(score)

    max_score = max(scores) if scores else 0.0
    plateau_taus = [tau for tau, s in zip(thresholds, scores) if max_score - s <= 1e-5]
    best_tau = float(np.median(plateau_taus)) if plateau_taus else float(thresholds[0])
    best_score = float(max_score)

    if return_curve:
        return best_tau, best_score, thresholds, scores
    return best_tau, best_score


class ERModelTrainer:
    def __init__(self, n_splits: int = 5, seed: int = 42, use_gpu: bool = False) -> None:
        self.n_splits = n_splits
        self.seed = seed
        self.use_gpu = use_gpu
        self.tr_indices_: Optional[np.ndarray] = None
        self.val_indices_: Optional[np.ndarray] = None
        lgb_params: Dict[str, Any] = {
            "n_estimators": 1000,
            "learning_rate": 0.05,
            "num_leaves": 31,
            "max_depth": 6,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "random_state": seed,
            "n_jobs": -1,
            "verbose": -1,
        }
        if use_gpu:
            lgb_params["device"] = "gpu"
        self.model = lgb.LGBMClassifier(**lgb_params)

    def train(self, X: np.ndarray, y: np.ndarray, s1_groups: Sequence[str]) -> lgb.LGBMClassifier:
        X_arr = np.asarray(X)
        y_arr = np.asarray(y)
        n_groups = len(set(s1_groups))
        effective_splits = min(self.n_splits, n_groups)

        if effective_splits >= 2:
            gkf = GroupKFold(n_splits=effective_splits)
            splits = list(gkf.split(X_arr, y_arr, groups=s1_groups))
            tr_idx, val_idx = splits[0]
            for t_idx, v_idx in splits:
                if len(np.unique(y_arr[t_idx])) >= 2:
                    tr_idx, val_idx = t_idx, v_idx
                    if set(y_arr[v_idx]).issubset(set(y_arr[t_idx])):
                        break
            self.tr_indices_ = tr_idx
            self.val_indices_ = val_idx
        else:
            self.tr_indices_ = np.arange(len(X_arr))
            self.val_indices_ = np.arange(len(X_arr))
            tr_idx = self.tr_indices_
            val_idx = self.val_indices_

        can_eval = (
            effective_splits >= 2
            and len(np.unique(y_arr[tr_idx])) >= 2
            and len(np.unique(y_arr[val_idx])) >= 2
            and set(y_arr[val_idx]).issubset(set(y_arr[tr_idx]))
            and len(val_idx) > 0
        )
        if can_eval:
            self.model.fit(
                X_arr[tr_idx],
                y_arr[tr_idx],
                eval_set=[(X_arr[val_idx], y_arr[val_idx])],
                callbacks=[lgb.early_stopping(50, verbose=False)],
            )
        elif len(np.unique(y_arr[tr_idx])) >= 2:
            self.model.fit(X_arr[tr_idx], y_arr[tr_idx])
        else:
            self.model.fit(X_arr, y_arr)
        return self.model

    def predict_val_proba(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Return validation probabilities and validation indices on held-out entities."""
        if self.val_indices_ is None:
            raise RuntimeError("Model has not been trained yet.")
        X_arr = np.asarray(X)
        X_val = X_arr[self.val_indices_]
        probs_val = self.model.predict_proba(X_val)[:, 1]
        return probs_val, self.val_indices_


class EREnsembleTrainer:
    """Multi-model ensemble trainer combining LightGBM, CatBoost, and XGBoost."""

    def __init__(
        self,
        models: Optional[List[str]] = None,
        weights: Optional[List[float]] = None,
        n_splits: int = 5,
        seed: int = 42,
        use_gpu: bool = False,
    ) -> None:
        self.model_names = models or ["lgbm", "catboost", "xgboost"]
        self.n_splits = n_splits
        self.seed = seed
        self.use_gpu = use_gpu
        self.tr_indices_: Optional[np.ndarray] = None
        self.val_indices_: Optional[np.ndarray] = None
        self.models_: Dict[str, Any] = {}

        if weights is not None:
            total_w = sum(weights)
            self.weights = [w / total_w for w in weights]
        else:
            self.weights = [1.0 / len(self.model_names)] * len(self.model_names)

    def train(self, X: np.ndarray, y: np.ndarray, s1_groups: Sequence[str]) -> "EREnsembleTrainer":
        X_arr = np.asarray(X, dtype=np.float32)
        y_arr = np.asarray(y, dtype=np.int32)
        n_groups = len(set(s1_groups))
        effective_splits = min(self.n_splits, n_groups)

        if effective_splits >= 2:
            gkf = GroupKFold(n_splits=effective_splits)
            splits = list(gkf.split(X_arr, y_arr, groups=s1_groups))
            tr_idx, val_idx = splits[0]
            for t_idx, v_idx in splits:
                if len(np.unique(y_arr[t_idx])) >= 2:
                    tr_idx, val_idx = t_idx, v_idx
                    if set(y_arr[v_idx]).issubset(set(y_arr[t_idx])):
                        break
            self.tr_indices_ = tr_idx
            self.val_indices_ = val_idx
        else:
            self.tr_indices_ = np.arange(len(X_arr))
            self.val_indices_ = np.arange(len(X_arr))
            tr_idx = self.tr_indices_
            val_idx = self.val_indices_

        can_eval = (
            effective_splits >= 2
            and len(np.unique(y_arr[tr_idx])) >= 2
            and len(np.unique(y_arr[val_idx])) >= 2
            and set(y_arr[val_idx]).issubset(set(y_arr[tr_idx]))
            and len(val_idx) > 0
        )

        for name in self.model_names:
            name_lower = name.lower()
            if name_lower == "lgbm":
                lgb_params: Dict[str, Any] = {
                    "n_estimators": 1000,
                    "learning_rate": 0.05,
                    "num_leaves": 31,
                    "max_depth": 6,
                    "subsample": 0.8,
                    "colsample_bytree": 0.8,
                    "random_state": self.seed,
                    "n_jobs": -1,
                    "verbose": -1,
                }
                if self.use_gpu:
                    lgb_params["device"] = "gpu"
                m = lgb.LGBMClassifier(**lgb_params)
                if can_eval:
                    m.fit(
                        X_arr[tr_idx],
                        y_arr[tr_idx],
                        eval_set=[(X_arr[val_idx], y_arr[val_idx])],
                        callbacks=[lgb.early_stopping(50, verbose=False)],
                    )
                else:
                    m.fit(X_arr[tr_idx], y_arr[tr_idx])
                self.models_[name_lower] = m

            elif name_lower == "catboost":
                cb_params: Dict[str, Any] = {
                    "iterations": 1000,
                    "learning_rate": 0.05,
                    "depth": 6,
                    "random_seed": self.seed,
                    "thread_count": -1,
                    "verbose": 0,
                }
                if self.use_gpu:
                    cb_params["task_type"] = "GPU"
                m = cb.CatBoostClassifier(**cb_params)
                if can_eval:
                    m.fit(
                        X_arr[tr_idx],
                        y_arr[tr_idx],
                        eval_set=(X_arr[val_idx], y_arr[val_idx]),
                        early_stopping_rounds=50,
                        verbose=False,
                    )
                else:
                    m.fit(X_arr[tr_idx], y_arr[tr_idx], verbose=False)
                self.models_[name_lower] = m

            elif name_lower == "xgboost":
                xgb_params: Dict[str, Any] = {
                    "n_estimators": 1000,
                    "learning_rate": 0.05,
                    "max_depth": 6,
                    "subsample": 0.8,
                    "colsample_bytree": 0.8,
                    "random_state": self.seed,
                    "n_jobs": -1,
                    "eval_metric": "logloss",
                    "tree_method": "hist",
                }
                if self.use_gpu:
                    xgb_params["device"] = "cuda"
                if can_eval:
                    xgb_params["early_stopping_rounds"] = 50
                m = xgb.XGBClassifier(**xgb_params)
                if can_eval:
                    m.fit(
                        X_arr[tr_idx],
                        y_arr[tr_idx],
                        eval_set=[(X_arr[val_idx], y_arr[val_idx])],
                        verbose=False,
                    )
                else:
                    m.fit(X_arr[tr_idx], y_arr[tr_idx], verbose=False)
                self.models_[name_lower] = m

        return self

    def predict_components(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        """Predict positive class probabilities for each individual ensemble member."""
        X_arr = np.asarray(X, dtype=np.float32)
        preds: Dict[str, np.ndarray] = {}
        for name in self.model_names:
            model = self.models_.get(name.lower())
            if model is not None:
                preds[name.lower()] = model.predict_proba(X_arr)[:, 1]
        return preds

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Blend probability predictions across all trained ensemble members."""
        X_arr = np.asarray(X, dtype=np.float32)
        total_p = np.zeros(len(X_arr), dtype=np.float64)

        for name, weight in zip(self.model_names, self.weights):
            model = self.models_.get(name.lower())
            if model is not None:
                p = model.predict_proba(X_arr)[:, 1]
                total_p += weight * p

        p_pos = np.clip(total_p, 0.0, 1.0)
        return np.column_stack([1.0 - p_pos, p_pos])

    def predict_val_proba(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Return blended validation probabilities on held-out entities."""
        if self.val_indices_ is None:
            raise RuntimeError("Ensemble has not been trained yet.")
        X_arr = np.asarray(X, dtype=np.float32)
        X_val = X_arr[self.val_indices_]
        probs_val = self.predict_proba(X_val)[:, 1]
        return probs_val, self.val_indices_

    @property
    def feature_importances_(self) -> np.ndarray:
        """Weighted average feature importances across all ensemble members."""
        if not self.models_:
            return np.array([])
        importances = []
        for name, weight in zip(self.model_names, self.weights):
            model = self.models_.get(name.lower())
            if model is not None and hasattr(model, "feature_importances_"):
                fi = np.array(model.feature_importances_, dtype=np.float64)
                fi_norm = fi / (np.sum(fi) + 1e-9)
                importances.append(weight * fi_norm)
        if importances:
            return np.sum(importances, axis=0)
        return np.array([])
