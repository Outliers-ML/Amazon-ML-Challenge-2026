"""Model training and Macro F_0.5 decision threshold calibration for Entity Resolution."""

from typing import Any, Dict, List, Optional, Sequence, Set, Tuple
import lightgbm as lgb
import numpy as np
from sklearn.model_selection import GroupKFold


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
) -> Tuple[float, float]:
    """Find threshold tau* in [min_tau, max_tau] maximizing macro F_0.5."""
    if not (len(s1_ids) == len(cand_ids) == len(probs)):
        raise ValueError(
            f"Mismatched input lengths: len(s1_ids)={len(s1_ids)}, "
            f"len(cand_ids)={len(cand_ids)}, len(probs)={len(probs)}"
        )

    thresholds = np.linspace(min_tau, max_tau, tau_steps)
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

    return best_tau, best_score


class ERModelTrainer:
    def __init__(self, n_splits: int = 5, seed: int = 42) -> None:
        self.n_splits = n_splits
        self.seed = seed
        self.tr_indices_: Optional[np.ndarray] = None
        self.val_indices_: Optional[np.ndarray] = None
        self.model = lgb.LGBMClassifier(
            n_estimators=1000,
            learning_rate=0.05,
            num_leaves=31,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=seed,
            n_jobs=-1,
            verbose=-1,
        )

    def train(self, X: np.ndarray, y: np.ndarray, s1_groups: Sequence[str]) -> lgb.LGBMClassifier:
        X_arr = np.asarray(X)
        y_arr = np.asarray(y)
        gkf = GroupKFold(n_splits=self.n_splits)
        splits = list(gkf.split(X_arr, y_arr, groups=s1_groups))
        tr_idx, val_idx = splits[0]
        for t_idx, v_idx in splits:
            if len(np.unique(y_arr[t_idx])) >= 2:
                tr_idx, val_idx = t_idx, v_idx
                if set(y_arr[v_idx]).issubset(set(y_arr[t_idx])):
                    break
        self.tr_indices_ = tr_idx
        self.val_indices_ = val_idx

        can_eval = (
            len(np.unique(y_arr[tr_idx])) >= 2
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
