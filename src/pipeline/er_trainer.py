"""Model training and Macro F_0.5 decision threshold calibration for Entity Resolution."""

from typing import Dict, List, Sequence, Set, Tuple
import lightgbm as lgb
import numpy as np
from sklearn.model_selection import GroupKFold


def compute_macro_f05(gt_map: Dict[str, Set[str]], pred_map: Dict[str, List[str]]) -> float:
    """Compute official competition Macro F_0.5 score."""
    f05_scores = []
    for s1_id, true_targets in gt_map.items():
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
            score = (1.25 * precision * recall) / (0.25 * precision + recall + 1e-9)
            f05_scores.append(score)

    return float(np.mean(f05_scores)) if f05_scores else 0.0


def optimize_f05_threshold(
    s1_ids: Sequence[str],
    cand_ids: Sequence[str],
    probs: np.ndarray,
    gt_map: Dict[str, Set[str]],
    tau_steps: int = 25,
) -> Tuple[float, float]:
    """Find threshold tau* in [0.50, 0.95] maximizing macro F_0.5."""
    best_tau = 0.75
    best_score = -1.0

    thresholds = np.linspace(0.50, 0.95, tau_steps)
    for tau in thresholds:
        pred_map: Dict[str, List[str]] = {s1: [] for s1 in gt_map.keys()}
        mask = probs >= tau
        for s1, cand, keep in zip(s1_ids, cand_ids, mask):
            if keep and s1 in pred_map:
                pred_map[s1].append(cand)

        score = compute_macro_f05(gt_map, pred_map)
        if score > best_score:
            best_score = score
            best_tau = float(tau)

    return best_tau, best_score


class ERModelTrainer:
    def __init__(self, n_splits: int = 5, seed: int = 42) -> None:
        self.n_splits = n_splits
        self.seed = seed
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
        tr_idx, val_idx = next(gkf.split(X_arr, y_arr, groups=s1_groups))
        self.model.fit(
            X_arr[tr_idx],
            y_arr[tr_idx],
            eval_set=[(X_arr[val_idx], y_arr[val_idx])],
            callbacks=[lgb.early_stopping(50, verbose=False)],
        )
        return self.model
