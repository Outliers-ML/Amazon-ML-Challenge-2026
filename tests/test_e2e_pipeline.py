import subprocess
import sys
import zipfile
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from src.data.blocking import MultiKeyBlocker
from src.features.pairwise_features import PairwiseFeatureExtractor
from src.pipeline.er_trainer import ERModelTrainer, optimize_f05_threshold


def test_submission_packager_dry_run(tmp_path):
    out_dir = tmp_path / "output"
    out_dir.mkdir(parents=True)
    (out_dir / "matching_results.tsv").write_text("source1_entity_id\tmatched_entity_ids\nS1-01\tS2-02\n", encoding="utf-8")
    (out_dir / "candidate_pairs.tsv").write_text("source1_entity_id\tcandidate_entity_ids\nS1-01\tS2-02\n", encoding="utf-8")

    res = subprocess.run([
        sys.executable,
        "scripts/pack_submission.py",
        "--matching", str(out_dir / "matching_results.tsv"),
        "--candidate", str(out_dir / "candidate_pairs.tsv"),
        "--skip-validation",
        "--dry-run"
    ], capture_output=True, text=True)
    assert res.returncode == 0, f"stdout:\n{res.stdout}\nstderr:\n{res.stderr}"
    assert "DRY-RUN" in res.stdout


def test_e2e_toy_dataset_flow(tmp_path):
    # Synthetic small datasets
    s1 = pd.DataFrame({
        "entity_id": ["S1-01", "S1-02"],
        "business_name": ["Acme Tools", "Solo Corp"],
        "business_address": ["123 Main St 10001", "55 Park Ave 10002"],
        "country": ["US", "US"]
    })
    s2 = pd.DataFrame({
        "entity_id": ["S2-11"],
        "business_name": ["Acme Tools Inc"],
        "business_address": ["123 Main St 10001"],
        "country": ["US"]
    })
    s3 = pd.DataFrame({
        "entity_id": ["S3-22"],
        "business_name": ["Acme Hardware"],
        "business_address": ["123 Main St"],
        "country": ["US"]
    })
    gt_map = {"S1-01": {"S2-11", "S3-22"}, "S1-02": set()}

    blocker = MultiKeyBlocker(max_candidates=10)
    cands = blocker.block_country_partition(s1, s2, s3)
    assert "S2-11" in cands["S1-01"]
    assert len(cands["S1-02"]) == 0

    extractor = PairwiseFeatureExtractor()
    s1_rows = [s1.iloc[0].to_dict(), s1.iloc[0].to_dict()]
    target_rows = [s2.iloc[0].to_dict(), s3.iloc[0].to_dict()]
    X = extractor.extract_pairs_matrix(s1_rows, target_rows, ranks=[1, 2], scores=[10.0, 5.0])
    y = np.array([1, 1], dtype=np.int32)
    assert X.shape == (2, 22)

    # Threshold optimization
    probs = np.array([0.95, 0.85])
    best_tau, best_score = optimize_f05_threshold(["S1-01", "S1-01"], ["S2-11", "S3-22"], probs, gt_map)
    assert best_score == 1.0


def test_pack_submission_creates_valid_zip(tmp_path):
    # Setup mock output directory and test dataset
    out_dir = tmp_path / "output"
    out_dir.mkdir(parents=True)
    test_dir = tmp_path / "test_dir"
    test_dir.mkdir(parents=True)

    # Test source files
    test_s1 = pd.DataFrame({
        "entity_id": ["S1-01", "S1-02"],
        "business_name": ["Acme Tools", "Solo Corp"],
        "business_address": ["123 Main St 10001", "55 Park Ave 10002"],
        "country": ["US", "US"]
    })
    test_s2 = pd.DataFrame({
        "entity_id": ["S2-11"],
        "business_name": ["Acme Tools Inc"],
        "business_address": ["123 Main St 10001"],
        "country": ["US"]
    })
    test_s3 = pd.DataFrame({
        "entity_id": ["S3-22"],
        "business_name": ["Acme Hardware"],
        "business_address": ["123 Main St"],
        "country": ["US"]
    })
    test_s1.to_csv(test_dir / "test_source1.tsv", sep="\t", index=False)
    test_s2.to_csv(test_dir / "test_source2.tsv", sep="\t", index=False)
    test_s3.to_csv(test_dir / "test_source3.tsv", sep="\t", index=False)

    # Output files
    matching_file = out_dir / "matching_results.tsv"
    matching_file.write_text(
        "source1_entity_id\tmatched_entity_ids\n"
        "S1-01\tS2-11\n"
        "S1-02\t\n",
        encoding="utf-8"
    )
    candidate_file = out_dir / "candidate_pairs.tsv"
    candidate_file.write_text(
        "source1_entity_id\tcandidate_entity_ids\n"
        "S1-01\tS2-11,S3-22\n"
        "S1-02\t\n",
        encoding="utf-8"
    )

    zip_path = tmp_path / "test_submission.zip"
    res = subprocess.run([
        "/dist_home/suryansh/miniforge3/envs/outliers/bin/python",
        "scripts/pack_submission.py",
        "--matching", str(matching_file),
        "--candidate", str(candidate_file),
        "--test-dir", str(test_dir),
        "--output-zip", str(zip_path),
    ], capture_output=True, text=True)

    assert res.returncode == 0, f"pack_submission stdout:\n{res.stdout}\nstderr:\n{res.stderr}"
    assert zip_path.exists()

    with zipfile.ZipFile(zip_path, "r") as zf:
        namelist = zf.namelist()
        assert "output/matching_results.tsv" in namelist
        assert "output/candidate_pairs.tsv" in namelist
        assert "code/business_entity_resolution/README.md" in namelist
        assert "code/business_entity_resolution/requirements.txt" in namelist
        assert "Documentation_template.md" in namelist
        # Verify src files exist inside code/business_entity_resolution/src/
        assert any(n.startswith("code/business_entity_resolution/src/") for n in namelist)


def test_run_entity_resolution_e2e(tmp_path):
    # Create complete synthetic train and test environments
    train_dir = tmp_path / "train"
    train_dir.mkdir(parents=True)
    test_dir = tmp_path / "test"
    test_dir.mkdir(parents=True)
    out_dir = tmp_path / "output"

    # Train files
    tr_s1 = pd.DataFrame({
        "entity_id": [f"S1-{i:02d}" for i in range(1, 7)],
        "business_name": [
            "Acme Tools Corp", "Acme Hardware Store", "Beta Logistics Inc",
            "Gamma Services", "Delta Cafe", "Epsilon Books"
        ],
        "business_address": [
            "123 Main St, New York, NY 10001", "123 Main Street, NY 10001", "456 Market St, SF, CA 94105",
            "789 Broadway, NY 10003", "101 5th Ave, NY 10003", "202 Elm St, Austin, TX 78701"
        ],
        "country": ["US", "US", "US", "US", "US", "US"]
    })
    tr_s2 = pd.DataFrame({
        "entity_id": ["S2-01", "S2-03", "S2-05"],
        "business_name": ["Acme Tools", "Beta Logistics", "Delta Coffee Cafe"],
        "business_address": ["123 Main St 10001", "456 Market St 94105", "101 5th Ave"],
        "country": ["US", "US", "US"]
    })
    tr_s3 = pd.DataFrame({
        "entity_id": ["S3-01", "S3-04"],
        "business_name": ["Acme Tools Corporation", "Gamma Services LLC"],
        "business_address": ["123 Main Street 10001", "789 Broadway"],
        "country": ["US", "US"]
    })
    tr_gt = pd.DataFrame({
        "source1_entity_id": [f"S1-{i:02d}" for i in range(1, 7)],
        "matched_entity_ids": [
            "S2-01,S3-01",
            "",
            "S2-03",
            "S3-04",
            "S2-05",
            ""
        ]
    })
    tr_s1.to_csv(train_dir / "train_source1.tsv", sep="\t", index=False)
    tr_s2.to_csv(train_dir / "train_source2.tsv", sep="\t", index=False)
    tr_s3.to_csv(train_dir / "train_source3.tsv", sep="\t", index=False)
    tr_gt.to_csv(train_dir / "train_ground_truth.tsv", sep="\t", index=False)

    # Test files with multiple countries including France
    te_s1 = pd.DataFrame({
        "entity_id": ["S1-T1", "S1-T2", "S1-T3"],
        "business_name": ["Acme Tools", "Beta Logistics", "Boulangerie Parisienne"],
        "business_address": ["123 Main St 10001", "456 Market St 94105", "10 Rue de la Paix, 75002 Paris"],
        "country": ["US", "US", "France"]
    })
    te_s2 = pd.DataFrame({
        "entity_id": ["S2-T1", "S2-T3"],
        "business_name": ["Acme Tools Inc", "Boulangerie Parisienne SARL"],
        "business_address": ["123 Main St 10001", "10 Rue de la Paix 75002 Paris"],
        "country": ["US", "France"]
    })
    te_s3 = pd.DataFrame({
        "entity_id": ["S3-T2"],
        "business_name": ["Beta Logistics Co"],
        "business_address": ["456 Market St"],
        "country": ["US"]
    })
    te_s1.to_csv(test_dir / "test_source1.tsv", sep="\t", index=False)
    te_s2.to_csv(test_dir / "test_source2.tsv", sep="\t", index=False)
    te_s3.to_csv(test_dir / "test_source3.tsv", sep="\t", index=False)

    zip_file = tmp_path / "submission_out.zip"
    res = subprocess.run([
        "/dist_home/suryansh/miniforge3/envs/outliers/bin/python",
        "scripts/run_entity_resolution.py",
        "--train-dir", str(train_dir),
        "--test-dir", str(test_dir),
        "--output-dir", str(out_dir),
        "--zip-name", str(zip_file),
        "--sample-train-s1", "10",
        "--n-splits", "2",
    ], capture_output=True, text=True)

    assert res.returncode == 0, f"run_entity_resolution failed:\nstdout:\n{res.stdout}\nstderr:\n{res.stderr}"

    # Verify matching_results and candidate_pairs
    assert (out_dir / "matching_results.tsv").exists()
    assert (out_dir / "candidate_pairs.tsv").exists()
    assert zip_file.exists()

    # Verify that France entity is present in matching_results and candidate_pairs
    match_lines = (out_dir / "matching_results.tsv").read_text().splitlines()
    cand_lines = (out_dir / "candidate_pairs.tsv").read_text().splitlines()
    assert any(line.startswith("S1-T3\t") for line in match_lines)
    assert any(line.startswith("S1-T3\t") for line in cand_lines)
