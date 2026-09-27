#!/usr/bin/env python3
"""Production Model Training Script for Amazon ML Challenge 2026.

Ingests student_resource/dataset/train/, blocks candidate pairs using MultiTierBlocker,
mines ~2M hard negative pairs from 500k S1 entities, fine-tunes BGE-reranker-v2-m3 with
BinaryFocalLoss on NVIDIA A100, trains the 32-feature GPU XGBoost + CatBoost ensemble
with 5-fold GroupKFold, and saves serialized model weights to models/.
"""

import argparse
import os
from pathlib import Path
import pickle
import sys
import time
from typing import Dict, List, Set, Tuple

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import torch

from src.data.blocking import MultiTierBlocker
from src.features.pairwise_features import PairwiseFeatureExtractor
from src.models.cross_encoder import BinaryFocalLoss, CrossEncoderReranker, format_pair_text
from src.pipeline.er_trainer import EREnsembleTrainer, mine_hard_negatives, optimize_f05_threshold


def load_ground_truth(gt_path: Path) -> Dict[str, Set[str]]:
    """Parse train_ground_truth.tsv into S1 -> set of matched target IDs."""
    gt_df = pd.read_csv(gt_path, sep="\t", dtype=str).fillna("")
    s1_vals = gt_df["source1_entity_id"].astype(str).str.strip().tolist()
    m_vals = gt_df["matched_entity_ids"].astype(str).str.strip().tolist() if "matched_entity_ids" in gt_df.columns else [""] * len(s1_vals)
    gt_map: Dict[str, Set[str]] = {}
    for s1, m_str in zip(s1_vals, m_vals):
        if not m_str or m_str.lower() in ("nan", "none", "null"):
            gt_map[s1] = set()
        else:
            gt_map[s1] = {x.strip() for x in m_str.split(",") if x.strip()}
    return gt_map


def main():
    parser = argparse.ArgumentParser(description="Train production ER models on NVIDIA A100.")
    parser.add_argument(
        "--train-dir",
        default="student_resource/dataset/train",
        help="Directory with train_source1/2/3.tsv and train_ground_truth.tsv",
    )
    parser.add_argument(
        "--output-models-dir",
        default="models",
        help="Directory to save trained model weights (default: models)",
    )
    parser.add_argument(
        "--sample-s1",
        type=int,
        default=500000,
        help="Number of S1 training records to sample (default: 500000)",
    )
    parser.add_argument(
        "--max-negatives-per-entity",
        type=int,
        default=4,
        help="Max hard negatives per S1 entity from blocker (default: 4)",
    )
    parser.add_argument(
        "--max-targets-per-partition",
        type=int,
        default=-1,
        help="Max target records per partition to block against during training (-1 for full, default: -1)",
    )
    parser.add_argument(
        "--ce-sample-pairs",
        type=int,
        default=100000,
        help="Number of mined pairs to use for Cross-Encoder fine-tuning (default: 100000)",
    )
    parser.add_argument(
        "--ce-epochs",
        type=int,
        default=1,
        help="Cross-Encoder training epochs (default: 1)",
    )
    parser.add_argument(
        "--ce-batch-size",
        type=int,
        default=64,
        help="Cross-Encoder batch size on A100 (default: 64)",
    )
    parser.add_argument(
        "--ce-lr",
        type=float,
        default=2e-5,
        help="Cross-Encoder learning rate (default: 2e-5)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    parser.add_argument(
        "--skip-cross-encoder",
        action="store_true",
        help="Skip fine-tuning Cross-Encoder and train GBDT only",
    )

    args = parser.parse_args()

    train_dir = Path(args.train_dir)
    output_dir = Path(args.output_models_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    models_saved_dir = PROJECT_ROOT / "models_saved"
    models_saved_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print(" Amazon ML Challenge 2026: Production Model Training Pipeline ")
    print("=" * 70)
    print(f"Device: {'CUDA (' + torch.cuda.get_device_name(0) + ')' if torch.cuda.is_available() else 'CPU'}")
    print(f"S1 training entities target: {args.sample_s1:,}")
    print(f"Max hard negatives per entity: {args.max_negatives_per_entity}")
    print(f"Output directory: {output_dir}")

    t0 = time.time()
    cached_npz = output_dir / "training_data.npz"
    cached_pkl = output_dir / "training_data.pkl"

    if cached_npz.exists():
        print(f"\n[+] Found cached training data at {cached_npz}! Loading to skip Steps 1-4 & feature extraction...")
        t_load = time.time()
        cdata = np.load(cached_npz, allow_pickle=True)
        X_train = cdata["X_train"]
        y_train = cdata["y_train"]
        s1_groups = cdata["s1_groups"].tolist()
        cand_ids_mined = cdata["cand_ids_mined"].tolist()
        tr_gt_path = train_dir / "train_ground_truth.tsv"
        gt_map = load_ground_truth(tr_gt_path)
        print(f"  Loaded X_train shape: {X_train.shape}, y_train: {y_train.shape} in {time.time()-t_load:.1f}s")
    elif cached_pkl.exists():
        print(f"\n[+] Found cached training data at {cached_pkl}! Loading to skip Steps 1-4 & feature extraction...")
        t_load = time.time()
        with open(cached_pkl, "rb") as f:
            cdata = pickle.load(f)
        X_train = cdata["X_train"]
        y_train = cdata["y_train"]
        s1_groups = cdata["s1_groups"]
        cand_ids_mined = cdata["cand_ids_mined"]
        tr_gt_path = train_dir / "train_ground_truth.tsv"
        gt_map = load_ground_truth(tr_gt_path)
        print(f"  Loaded X_train shape: {X_train.shape}, y_train: {y_train.shape} in {time.time()-t_load:.1f}s")
    else:
        # 1. Load Data
        print("\n[Step 1/5] Ingesting training datasets...")
        tr_s1_path = train_dir / "train_source1.tsv"
        tr_s2_path = train_dir / "train_source2.tsv"
        tr_s3_path = train_dir / "train_source3.tsv"
        tr_gt_path = train_dir / "train_ground_truth.tsv"

        train_s1 = pd.read_csv(tr_s1_path, sep="\t", dtype=str).fillna("")
        train_s2 = pd.read_csv(tr_s2_path, sep="\t", dtype=str).fillna("")
        train_s3 = pd.read_csv(tr_s3_path, sep="\t", dtype=str).fillna("")
        gt_map = load_ground_truth(tr_gt_path)

        print(f"  Loaded raw: S1={len(train_s1):,}, S2={len(train_s2):,}, S3={len(train_s3):,}, GT={len(gt_map):,}")

        # Stratified S1 sampling
        if 0 < args.sample_s1 < len(train_s1):
            print(f"  Sampling {args.sample_s1:,} entities stratified by country...")
            sampled_dfs = []
            for country, grp in train_s1.groupby(train_s1["country"].fillna("UNKNOWN")):
                n_c = int(round(args.sample_s1 * len(grp) / len(train_s1)))
                n_c = max(1, min(len(grp), n_c))
                sampled_dfs.append(grp.sample(n=n_c, random_state=args.seed))
            train_s1 = pd.concat(sampled_dfs, ignore_index=True)
            if len(train_s1) > args.sample_s1:
                train_s1 = train_s1.sample(n=args.sample_s1, random_state=args.seed).reset_index(drop=True)
            print(f"  Sampled {len(train_s1):,} S1 entities in {time.time()-t0:.1f}s")

        # 2. Multi-Tier Candidate Blocking
        t_block = time.time()
        print("\n[Step 2/5] Running MultiTierBlocker across country partitions...")
        blocker = MultiTierBlocker(max_candidates=35, bucket_ceiling=250)
        train_cands: Dict[str, List[str]] = {}

        train_countries = train_s1["country"].fillna("UNKNOWN").unique()
        for country in train_countries:
            s1_p = train_s1[train_s1["country"].fillna("UNKNOWN") == country]
            s2_p = train_s2[train_s2["country"].fillna("UNKNOWN") == country]
            s3_p = train_s3[train_s3["country"].fillna("UNKNOWN") == country]

            if args.max_targets_per_partition > 0:
                gt_targets_for_p: Set[str] = set()
                for eid in s1_p["entity_id"]:
                    gt_targets_for_p.update(gt_map.get(eid, set()))

                half = args.max_targets_per_partition // 2
                s2_gt = s2_p[s2_p["entity_id"].isin(gt_targets_for_p)]
                s2_other = s2_p[~s2_p["entity_id"].isin(gt_targets_for_p)]
                if len(s2_other) > half:
                    s2_other = s2_other.sample(n=half, random_state=args.seed)
                s2_p = pd.concat([s2_gt, s2_other], ignore_index=True)

                s3_gt = s3_p[s3_p["entity_id"].isin(gt_targets_for_p)]
                s3_other = s3_p[~s3_p["entity_id"].isin(gt_targets_for_p)]
                if len(s3_other) > half:
                    s3_other = s3_other.sample(n=half, random_state=args.seed)
                s3_p = pd.concat([s3_gt, s3_other], ignore_index=True)

            print(f"  Blocking partition '{country}' (S1={len(s1_p):,}, S2={len(s2_p):,}, S3={len(s3_p):,})...", flush=True)
            t_p = time.time()
            cands_p = blocker.block_country_partition(s1_p, s2_p, s3_p)
            train_cands.update(cands_p)
            print(f"    -> Done in {time.time()-t_p:.1f}s. Generated candidates for {len(cands_p):,} entities.", flush=True)

        print(f"  Total blocking completed in {time.time()-t_block:.1f}s")

        # 3. Hard Negative Mining
        t_mine = time.time()
        print("\n[Step 3/5] Mining hard negative pairs and ground truth positives...")
        mined_tuples = mine_hard_negatives(
            blocker_candidates=train_cands,
            ground_truth=gt_map,
            sample_size=len(train_s1),
            seed=args.seed,
            max_negatives_per_entity=args.max_negatives_per_entity,
        )

        s1_ids_mined = [t[0] for t in mined_tuples]
        cand_ids_mined = [t[1] for t in mined_tuples]
        y_mined = [t[2] for t in mined_tuples]
        n_pos = sum(y_mined)
        n_neg = len(y_mined) - n_pos

        print(f"  Total pairs mined: {len(mined_tuples):,} (Positives: {n_pos:,}, Hard Negatives: {n_neg:,}, Ratio: {n_neg/max(1,n_pos):.2f}:1)")
        print(f"  Mining completed in {time.time()-t_mine:.1f}s")

        # Index referenced records for fast lookup
        print("  Indexing candidate record attributes...")
        s1_dict = {row["entity_id"]: row for row in train_s1.to_dict(orient="records")}
        needed_cand_ids = set(cand_ids_mined)
        s2_needed = train_s2[train_s2["entity_id"].isin(needed_cand_ids)]
        s3_needed = train_s3[train_s3["entity_id"].isin(needed_cand_ids)]
        cand_dict = {row["entity_id"]: row for row in s2_needed.to_dict(orient="records")}
        cand_dict.update({row["entity_id"]: row for row in s3_needed.to_dict(orient="records")})
        print(f"  Target record dictionary ready ({len(cand_dict):,} unique candidates).")

        # 4. Fine-Tune Cross-Encoder
        ce_model_path = output_dir / "cross_encoder"
        if not args.skip_cross_encoder:
            print("\n[Step 4/5] Fine-tuning BAAI/bge-reranker-v2-m3 with Focal Loss on A100...")
            t_ce = time.time()
            # Select balanced subset for cross-encoder
            rng = np.random.RandomState(args.seed)
            pos_indices = [i for i, y in enumerate(y_mined) if y == 1]
            neg_indices = [i for i, y in enumerate(y_mined) if y == 0]

            target_ce_pairs = min(args.ce_sample_pairs, len(mined_tuples))
            n_ce_pos = min(len(pos_indices), target_ce_pairs // 2)
            n_ce_neg = min(len(neg_indices), target_ce_pairs - n_ce_pos)

            chosen_pos = rng.choice(pos_indices, size=n_ce_pos, replace=False).tolist()
            chosen_neg = rng.choice(neg_indices, size=n_ce_neg, replace=False).tolist()
            ce_indices = chosen_pos + chosen_neg
            rng.shuffle(ce_indices)

            ce_train_pairs = []
            ce_labels = []
            for idx in ce_indices:
                s1_id = s1_ids_mined[idx]
                cand_id = cand_ids_mined[idx]
                label = y_mined[idx]

                s1_row = s1_dict.get(s1_id, {"business_name": "", "business_address": "", "country": ""})
                c_row = cand_dict.get(cand_id, {"business_name": "", "business_address": "", "country": ""})

                pair_seq = format_pair_text(
                    s1_name=s1_row.get("business_name", ""),
                    s1_addr=s1_row.get("business_address", ""),
                    s1_country=s1_row.get("country", ""),
                    cand_name=c_row.get("business_name", ""),
                    cand_addr=c_row.get("business_address", ""),
                    cand_country=c_row.get("country", ""),
                )
                ce_train_pairs.append(pair_seq)
                ce_labels.append(label)

            print(f"  Training Cross-Encoder on {len(ce_train_pairs):,} text pairs (batch_size={args.ce_batch_size}, lr={args.ce_lr})...")
            reranker = CrossEncoderReranker(
                model_name="BAAI/bge-reranker-v2-m3",
                device="cuda" if torch.cuda.is_available() else "cpu",
                max_length=128,
            )
            reranker.fit(
                train_pairs=ce_train_pairs,
                labels=ce_labels,
                epochs=args.ce_epochs,
                lr=args.ce_lr,
                batch_size=args.ce_batch_size,
            )
            reranker.save(str(ce_model_path))
            print(f"  [+] Cross-Encoder fine-tuned and saved to {ce_model_path} in {time.time()-t_ce:.1f}s")
        else:
            print("\n[Step 4/5] Skipping Cross-Encoder fine-tuning (--skip-cross-encoder).")

        # 5. Extract 32 Features
        pair_s1_rows = []
        pair_target_rows = []
        pair_ranks = []
        pair_scores = []

        # Map candidate rank from blocker
        for s1_id, cand_id in zip(s1_ids_mined, cand_ids_mined):
            s1_row = s1_dict.get(s1_id, {"entity_id": s1_id, "business_name": "", "business_address": "", "country": ""})
            c_row = cand_dict.get(cand_id, {"entity_id": cand_id, "business_name": "", "business_address": "", "country": ""})
            cands_for_s1 = train_cands.get(s1_id, [])
            if cand_id in cands_for_s1:
                rank = cands_for_s1.index(cand_id) + 1
                score = 1.0 / rank
            else:
                rank = len(cands_for_s1) + 1
                score = 0.0

            pair_s1_rows.append(s1_row)
            pair_target_rows.append(c_row)
            pair_ranks.append(rank)
            pair_scores.append(score)

        extractor = PairwiseFeatureExtractor()
        print(f"\n[Step 5/5] Extracting 32 tabular features for {len(pair_s1_rows):,} pairs...")
        X_train = extractor.extract_pairs_matrix(
            pair_s1_rows, pair_target_rows, ranks=pair_ranks, scores=pair_scores
        )
        y_train = np.array(y_mined, dtype=np.int32)
        s1_groups = s1_ids_mined

        # Cache features and metadata for instant restarts
        np.savez_compressed(
            cached_npz,
            X_train=X_train,
            y_train=y_train,
            s1_groups=np.array(s1_groups),
            cand_ids_mined=np.array(cand_ids_mined),
        )
        print(f"  [+] Cached training feature matrix and metadata to {cached_npz}")

    # 5b. Train GPU GBDT Ensemble across 5 folds
    t_gbdt = time.time()
    print("\nTraining GPU GBDT Ensemble (XGBoost GPU + CatBoost GPU) across 5 folds...")
    print(f"  Feature matrix shape: {X_train.shape}, memory: {X_train.nbytes / (1024*1024):.1f} MB")

    trainer = EREnsembleTrainer(
        models=["xgboost", "catboost"],
        weights=[0.50, 0.50],
        n_splits=5,
        seed=args.seed,
        use_gpu=torch.cuda.is_available(),
    )
    trainer.train(X_train, y_train, s1_groups)

    # Threshold calibration
    probs_val, val_idx = trainer.predict_val_proba(X_train)
    val_s1 = [s1_groups[i] for i in val_idx]
    val_cands = [cand_ids_mined[i] for i in val_idx]
    val_entities = set(val_s1)
    val_gt_map = {s1: gt_map.get(s1, set()) for s1 in val_entities}

    best_tau, best_score, tau_curve, score_curve = optimize_f05_threshold(
        val_s1, val_cands, probs_val, val_gt_map, return_curve=True
    )
    print(f"\n[+] GBDT Ensemble Training Complete!")
    print(f"  Optimal Decision Threshold tau* = {best_tau:.4f}")
    print(f"  Validation Macro F_0.5 = {best_score:.4f}")

    # Save models
    ensemble_save_path = output_dir / "ensemble_model.pkl"
    models_saved_path = models_saved_dir / "ensemble_model.pkl"

    payload = {
        "model": trainer,
        "best_tau": best_tau,
        "best_score": best_score,
        "feature_names": PairwiseFeatureExtractor.FEATURE_NAMES,
        "weights": trainer.weights,
        "models": trainer.model_names,
    }

    with open(ensemble_save_path, "wb") as f:
        pickle.dump(payload, f)
    with open(models_saved_path, "wb") as f:
        pickle.dump(payload, f)

    print(f"  [+] Saved ensemble model to {ensemble_save_path} and {models_saved_path}")
    print(f"\n[✔] Production Training Finished Successfully in {time.time()-t0:.1f}s!")


if __name__ == "__main__":
    main()
