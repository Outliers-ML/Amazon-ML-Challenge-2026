#!/usr/bin/env python3
"""End-to-End Business Entity Resolution Pipeline Orchestrator.

Orchestrates multi-key candidate blocking, pairwise feature extraction,
LightGBM classifier training, macro F_0.5 decision threshold calibration,
partition-by-partition test set inference, and automated submission packaging.
"""

import argparse
from pathlib import Path
import subprocess
import sys
from typing import Dict, List, Optional, Set

# Ensure project root or package directory is in sys.path
SCRIPT_DIR = Path(__file__).resolve().parent
if (SCRIPT_DIR / "src").is_dir():
    if str(SCRIPT_DIR) not in sys.path:
        sys.path.insert(0, str(SCRIPT_DIR))
    PROJECT_ROOT = SCRIPT_DIR
elif (SCRIPT_DIR.parent / "src").is_dir():
    if str(SCRIPT_DIR.parent) not in sys.path:
        sys.path.insert(0, str(SCRIPT_DIR.parent))
    PROJECT_ROOT = SCRIPT_DIR.parent
else:
    PROJECT_ROOT = SCRIPT_DIR.parent
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from src.data.blocking import MultiKeyBlocker
from src.features.pairwise_features import PairwiseFeatureExtractor
from src.pipeline.er_trainer import ERModelTrainer, optimize_f05_threshold
from src.utils.experiment_tracker import ExperimentTracker


def write_matching_results(
    matching_map: Dict[str, List[str]],
    output_path: Path | str,
    append: bool = False,
    write_header: bool = True,
) -> None:
    """Write matching results TSV matching competition schema.

    Supports streaming append mode across partition chunks.
    """
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if append else "w"
    with open(out, mode, encoding="utf-8") as f:
        if write_header:
            f.write("source1_entity_id\tmatched_entity_ids\n")
        for s1_id in sorted(matching_map.keys()):
            matched_str = ",".join(matching_map[s1_id])
            f.write(f"{s1_id}\t{matched_str}\n")


def load_ground_truth(gt_path: Path) -> Dict[str, Set[str]]:
    """Parse train_ground_truth.tsv into a mapping of S1 entity -> set of matched target IDs."""
    gt_df = pd.read_csv(gt_path, sep="\t", dtype=str).fillna("")
    gt_map: Dict[str, Set[str]] = {}
    for _, row in gt_df.iterrows():
        s1 = str(row["source1_entity_id"]).strip()
        m_str = str(row.get("matched_entity_ids", "")).strip()
        if not m_str or m_str == "nan":
            gt_map[s1] = set()
        else:
            gt_map[s1] = {x.strip() for x in m_str.split(",") if x.strip()}
    return gt_map


def main():
    parser = argparse.ArgumentParser(
        description="Run end-to-end Entity Resolution pipeline (train -> calibrate -> infer -> pack)."
    )
    parser.add_argument(
        "--train-dir",
        default="student_resource/dataset/train",
        help="Directory with train_source1/2/3.tsv and train_ground_truth.tsv",
    )
    parser.add_argument(
        "--test-dir",
        default="student_resource/dataset/test",
        help="Directory with test_source1/2/3.tsv",
    )
    parser.add_argument(
        "--output-dir",
        default="output",
        help="Directory for generated TSVs (default: output)",
    )
    parser.add_argument(
        "--sample-train-s1",
        type=int,
        default=50000,
        help="Number of S1 training records to sample (-1 for full dataset, default: 50000)",
    )
    parser.add_argument(
        "--max-candidates",
        type=int,
        default=25,
        help="Maximum candidates per S1 entity during blocking (default: 25)",
    )
    parser.add_argument(
        "--max-postings",
        type=int,
        default=500,
        help="Maximum postings list cutoff for inverted index (default: 500)",
    )
    parser.add_argument(
        "--n-splits",
        type=int,
        default=5,
        help="Number of GroupKFold validation splits (default: 5)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--tau",
        type=float,
        default=None,
        help="Manual probability threshold override (default: auto-calibrated)",
    )
    parser.add_argument(
        "--infer-chunk-size",
        type=int,
        default=50000,
        help="Batch size for feature extraction and model prediction on test pairs (default: 50000)",
    )
    parser.add_argument(
        "--zip-name",
        default="outliers_submission.zip",
        help="Output submission zip name (default: outliers_submission.zip)",
    )
    parser.add_argument(
        "--skip-pack",
        action="store_true",
        help="Skip calling pack_submission.py at the end of the run",
    )
    parser.add_argument(
        "--experiment-name",
        default="Baseline-LGBM",
        help="Experiment run name for tracking and dashboard (default: Baseline-LGBM)",
    )
    parser.add_argument(
        "--no-track",
        action="store_true",
        help="Disable logging experiment to tracker/dashboard ledger",
    )

    args = parser.parse_args()

    # Initialize experiment tracker
    tracker = None
    if not args.no_track:
        ledger_path = PROJECT_ROOT / "experiments" / "runs.json"
        tracker = ExperimentTracker(ledger_path=ledger_path, use_mlflow=False)
        tracker.start_run(
            run_name=args.experiment_name,
            params={
                "model_type": "LightGBM",
                "sample_train_s1": args.sample_train_s1,
                "max_candidates": args.max_candidates,
                "max_postings": args.max_postings,
                "n_splits": args.n_splits,
                "seed": args.seed,
                "infer_chunk_size": args.infer_chunk_size,
            },
        )

    train_dir = Path(args.train_dir)
    test_dir = Path(args.test_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    matching_file = output_dir / "matching_results.tsv"
    candidate_file = output_dir / "candidate_pairs.tsv"

    # Reset output files
    if matching_file.exists():
        matching_file.unlink()
    if candidate_file.exists():
        candidate_file.unlink()

    blocker = MultiKeyBlocker(max_candidates=args.max_candidates, max_postings=args.max_postings)
    extractor = PairwiseFeatureExtractor()

    # =========================================================================
    # Stage 1: Load training data, run blocking, train model, calibrate tau*
    # =========================================================================
    print("=" * 60)
    print("Stage 1: Training & Threshold Calibration")
    print("=" * 60)

    tr_s1_path = train_dir / "train_source1.tsv"
    tr_s2_path = train_dir / "train_source2.tsv"
    tr_s3_path = train_dir / "train_source3.tsv"
    tr_gt_path = train_dir / "train_ground_truth.tsv"

    if not tr_s1_path.exists() or not tr_gt_path.exists():
        raise FileNotFoundError(f"Training files missing in {train_dir}")

    print(f"Loading training data from {train_dir}...")
    train_s1 = pd.read_csv(tr_s1_path, sep="\t", dtype=str).fillna("")
    train_s2 = pd.read_csv(tr_s2_path, sep="\t", dtype=str).fillna("") if tr_s2_path.exists() else pd.DataFrame(columns=["entity_id", "business_name", "business_address", "country"])
    train_s3 = pd.read_csv(tr_s3_path, sep="\t", dtype=str).fillna("") if tr_s3_path.exists() else pd.DataFrame(columns=["entity_id", "business_name", "business_address", "country"])
    gt_map = load_ground_truth(tr_gt_path)

    # Sample train_s1 if requested
    if args.sample_train_s1 > 0 and len(train_s1) > args.sample_train_s1:
        print(f"Sampling {args.sample_train_s1} records from {len(train_s1)} training S1 entities...")
        if "country" in train_s1.columns:
            sampled_dfs = []
            for country, grp in train_s1.groupby(train_s1["country"].fillna("UNKNOWN")):
                n_c = int(round(args.sample_train_s1 * len(grp) / len(train_s1)))
                n_c = max(1, min(len(grp), n_c))
                sampled_dfs.append(grp.sample(n=n_c, random_state=args.seed))
            train_s1 = pd.concat(sampled_dfs, ignore_index=True)
            if len(train_s1) > args.sample_train_s1:
                train_s1 = train_s1.sample(n=args.sample_train_s1, random_state=args.seed).reset_index(drop=True)
        else:
            train_s1 = train_s1.sample(n=args.sample_train_s1, random_state=args.seed).reset_index(drop=True)

    print(f"Effective training S1 entities: {len(train_s1)}")

    # Blocking per country partition on training set
    train_cands: Dict[str, List[str]] = {}
    train_countries = train_s1["country"].fillna("UNKNOWN").unique()
    for country in train_countries:
        s1_p = train_s1[train_s1["country"].fillna("UNKNOWN") == country]
        s2_p = train_s2[train_s2["country"].fillna("UNKNOWN") == country]
        s3_p = train_s3[train_s3["country"].fillna("UNKNOWN") == country]
        cands_p = blocker.block_country_partition(s1_p, s2_p, s3_p)
        train_cands.update(cands_p)

    # Fast row lookup dictionaries
    s1_dict = {row["entity_id"]: row for row in train_s1.to_dict(orient="records")}
    target_dict = {row["entity_id"]: row for row in train_s2.to_dict(orient="records")}
    target_dict.update({row["entity_id"]: row for row in train_s3.to_dict(orient="records")})

    # Assemble training pairs (s1, target)
    pair_s1_rows = []
    pair_target_rows = []
    pair_s1_ids = []
    pair_cand_ids = []
    pair_ranks = []
    pair_scores = []
    pair_is_blocker = []
    y_list = []

    for s1_id in train_s1["entity_id"]:
        s1_row = s1_dict.get(s1_id)
        if not s1_row:
            continue
        cands = train_cands.get(s1_id, [])
        true_matches = gt_map.get(s1_id, set())

        for rank_idx, cand_id in enumerate(cands, start=1):
            t_row = target_dict.get(cand_id, {"entity_id": cand_id, "business_name": "", "business_address": ""})
            pair_s1_rows.append(s1_row)
            pair_target_rows.append(t_row)
            pair_s1_ids.append(s1_id)
            pair_cand_ids.append(cand_id)
            pair_ranks.append(rank_idx)
            pair_scores.append(1.0 / rank_idx)
            pair_is_blocker.append(True)
            y_list.append(1 if cand_id in true_matches else 0)

        # Ensure missed ground truth matches are also included as positive training pairs
        seen_cands = set(cands)
        for tm in true_matches:
            if tm not in seen_cands and tm in target_dict:
                t_row = target_dict[tm]
                pair_s1_rows.append(s1_row)
                pair_target_rows.append(t_row)
                pair_s1_ids.append(s1_id)
                pair_cand_ids.append(tm)
                pair_ranks.append(len(cands) + 1)
                pair_scores.append(0.0)
                pair_is_blocker.append(False)
                y_list.append(1)

    print(f"Total candidate pairs assembled: {len(pair_s1_rows)} (positives: {sum(y_list)})")

    # Feature extraction & Model training
    if len(pair_s1_rows) > 0 and len(np.unique(y_list)) > 1:
        X_train = extractor.extract_pairs_matrix(
            pair_s1_rows, pair_target_rows, ranks=pair_ranks, scores=pair_scores
        )
        y_train = np.array(y_list, dtype=np.int32)
        s1_groups = pair_s1_ids

        n_groups = len(set(s1_groups))
        n_splits = min(args.n_splits, max(2, n_groups))
        trainer = ERModelTrainer(n_splits=n_splits, seed=args.seed)
        model = trainer.train(X_train, y_train, s1_groups)

        probs_val, val_idx = trainer.predict_val_proba(X_train)

        # Filter validation pairs to only those generated by the blocker to prevent distribution shift
        val_blocker_mask = np.array([pair_is_blocker[i] for i in val_idx], dtype=bool)
        val_idx_blocker = val_idx[val_blocker_mask]
        probs_val_blocker = probs_val[val_blocker_mask]

        val_s1 = [s1_groups[i] for i in val_idx_blocker]
        val_cands = [pair_cand_ids[i] for i in val_idx_blocker]

        # Genuine validation entities: all S1 entities in the validation fold
        val_entities = set(s1_groups[i] for i in val_idx)
        val_gt_map = {s1: gt_map.get(s1, set()) for s1 in val_entities}

        best_tau, best_score, tau_curve, score_curve = optimize_f05_threshold(
            val_s1, val_cands, probs_val_blocker, val_gt_map, return_curve=True
        )
        print(f"[+] Optimal threshold tau* = {best_tau:.4f} with validation Macro F_0.5 = {best_score:.4f}")

        if tracker:
            tracker.log_threshold_curve(tau_curve, score_curve)
            if hasattr(model, "feature_importances_"):
                fi = dict(zip(extractor.FEATURE_NAMES, [float(x) for x in model.feature_importances_]))
                tracker.log_feature_importances(fi)
    else:
        print("[!] Warning: Insufficient class diversity in training pairs. Using fallback threshold tau = 0.50")
        best_tau = 0.50
        best_score = 0.0
        model = None

    if args.tau is not None:
        best_tau = args.tau
        print(f"[!] Overriding threshold with user-specified tau = {best_tau:.4f}")

    # =========================================================================
    # Stage 2: Dynamic Test Set Processing (Partition by Country)
    # =========================================================================
    print("=" * 60)
    print("Stage 2: Dynamic Test Set Inference per Country Partition")
    print("=" * 60)

    te_s1_path = test_dir / "test_source1.tsv"
    te_s2_path = test_dir / "test_source2.tsv"
    te_s3_path = test_dir / "test_source3.tsv"

    if not te_s1_path.exists():
        raise FileNotFoundError(f"Test source1 file missing: {te_s1_path}")

    print(f"Loading test set from {test_dir}...")
    test_s1 = pd.read_csv(te_s1_path, sep="\t", dtype=str).fillna("")
    test_s2 = pd.read_csv(te_s2_path, sep="\t", dtype=str).fillna("") if te_s2_path.exists() else pd.DataFrame(columns=["entity_id", "business_name", "business_address", "country"])
    test_s3 = pd.read_csv(te_s3_path, sep="\t", dtype=str).fillna("") if te_s3_path.exists() else pd.DataFrame(columns=["entity_id", "business_name", "business_address", "country"])

    print(f"Test entities — S1: {len(test_s1)}, S2: {len(test_s2)}, S3: {len(test_s3)}")

    unique_test_countries = test_s1["country"].fillna("UNKNOWN").unique()
    print(f"Dynamic test partitions identified ({len(unique_test_countries)}): {list(unique_test_countries)}")

    partition_summaries: Dict[str, Dict[str, int]] = {}

    for p_idx, country in enumerate(unique_test_countries):
        is_first = (p_idx == 0)
        s1_part = test_s1[test_s1["country"].fillna("UNKNOWN") == country]
        s2_part = test_s2[test_s2["country"].fillna("UNKNOWN") == country]
        s3_part = test_s3[test_s3["country"].fillna("UNKNOWN") == country]

        print(f"\nProcessing partition [{p_idx + 1}/{len(unique_test_countries)}] — Country: '{country}' "
              f"(S1: {len(s1_part)}, S2: {len(s2_part)}, S3: {len(s3_part)})...")

        # 1. Blocking
        part_cands = blocker.block_country_partition(s1_part, s2_part, s3_part)
        # Ensure all S1 entities in partition have an entry
        for s1_id in s1_part["entity_id"]:
            if s1_id not in part_cands:
                part_cands[s1_id] = []

        # Stream write candidate pairs
        MultiKeyBlocker.write_candidate_pairs(
            part_cands,
            candidate_file,
            append=(not is_first),
            write_header=is_first,
        )

        # 2. Pairwise Feature Extraction & Inference
        s1_part_dict = {row["entity_id"]: row for row in s1_part.to_dict(orient="records")}
        target_part_dict = {row["entity_id"]: row for row in s2_part.to_dict(orient="records")}
        target_part_dict.update({row["entity_id"]: row for row in s3_part.to_dict(orient="records")})

        test_pair_s1 = []
        test_pair_target = []
        test_pair_s1_ids = []
        test_pair_cand_ids = []
        test_pair_ranks = []
        test_pair_scores = []

        for s1_id in s1_part["entity_id"]:
            s1_row = s1_part_dict[s1_id]
            cand_list = part_cands.get(s1_id, [])
            for r_idx, c_id in enumerate(cand_list, start=1):
                t_row = target_part_dict.get(c_id, {"entity_id": c_id, "business_name": "", "business_address": ""})
                test_pair_s1.append(s1_row)
                test_pair_target.append(t_row)
                test_pair_s1_ids.append(s1_id)
                test_pair_cand_ids.append(c_id)
                test_pair_ranks.append(r_idx)
                test_pair_scores.append(1.0 / r_idx)

        # Match decisions
        part_matches: Dict[str, List[str]] = {s1_id: [] for s1_id in s1_part["entity_id"]}

        n_pairs = len(test_pair_s1)
        if n_pairs > 0 and model is not None:
            chunk_size = args.infer_chunk_size
            for start_idx in range(0, n_pairs, chunk_size):
                end_idx = min(start_idx + chunk_size, n_pairs)
                chunk_s1 = test_pair_s1[start_idx:end_idx]
                chunk_target = test_pair_target[start_idx:end_idx]
                chunk_ranks = test_pair_ranks[start_idx:end_idx]
                chunk_scores = test_pair_scores[start_idx:end_idx]

                X_chunk = extractor.extract_pairs_matrix(
                    chunk_s1, chunk_target, ranks=chunk_ranks, scores=chunk_scores
                )
                chunk_probs = model.predict_proba(X_chunk)[:, 1]

                for i in range(start_idx, end_idx):
                    p = chunk_probs[i - start_idx]
                    if p >= best_tau:
                        s1_id = test_pair_s1_ids[i]
                        c_id = test_pair_cand_ids[i]
                        part_matches[s1_id].append(c_id)

        # Stream write matching results
        write_matching_results(
            part_matches,
            matching_file,
            append=(not is_first),
            write_header=is_first,
        )

        n_non_empty = sum(1 for m in part_matches.values() if m)
        partition_summaries[country] = {
            "total": len(part_matches),
            "matched": n_non_empty,
            "singletons": len(part_matches) - n_non_empty,
        }
        print(f"  Partition '{country}' completed: {len(part_matches)} S1 rows written "
              f"({n_non_empty} matched, {len(part_matches) - n_non_empty} singletons).")

    print("\n[+] Dynamic test set inference completed.")
    print(f"  matching_results: {matching_file}")
    print(f"  candidate_pairs:  {candidate_file}")

    if tracker:
        tot_matched = sum(p["matched"] for p in partition_summaries.values())
        tot_singletons = sum(p["singletons"] for p in partition_summaries.values())
        tracker.log_metrics({
            "macro_f05": best_score,
            "best_tau": best_tau,
            "train_pairs": len(pair_s1_rows),
            "train_positives": sum(y_list),
            "total_test_s1": len(test_s1),
            "total_test_matched": tot_matched,
            "total_test_singletons": tot_singletons,
        })
        tracker.log_partition_summary(partition_summaries)
        completed_run = tracker.end_run()
        if completed_run:
            print(f"[+] Run '{args.experiment_name}' recorded in experiment tracker (ID: {completed_run['run_id']})")

    # =========================================================================
    # Stage 3: Packaging & Validation Gate
    # =========================================================================
    if not args.skip_pack:
        print("\n" + "=" * 60)
        print("Stage 3: Validation Gate & Submission Packaging")
        print("=" * 60)

        pack_script = Path(__file__).resolve().parent / "pack_submission.py"
        if not pack_script.is_file():
            pack_script = Path(__file__).resolve().parent.parent / "scripts" / "pack_submission.py"
        if pack_script.is_file():
            pack_cmd = [
                sys.executable,
                str(pack_script),
                "--matching", str(matching_file),
                "--candidate", str(candidate_file),
                "--test-dir", str(test_dir),
                "--output-zip", str(args.zip_name),
            ]
            res = subprocess.run(pack_cmd)
            if res.returncode != 0:
                print(f"[-] Error: pack_submission failed with exit code {res.returncode}", file=sys.stderr)
                sys.exit(res.returncode)
        else:
            print("[!] Note: pack_submission.py not found; skipping automatic packaging step.")

    print("\n[✔] Entity Resolution pipeline finished successfully.")


if __name__ == "__main__":
    main()
