#!/usr/bin/env python3
"""Submission Packager and Official Validator Gate for ML Challenge 2026.

Validates output/matching_results.tsv and output/candidate_pairs.tsv against
the official submission validator rules, and packages outliers_submission.zip
matching competition structure.
"""

import argparse
from pathlib import Path
import subprocess
import sys
from typing import Dict, List, Optional
import zipfile

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


REPRODUCTION_README = """# Business Entity Resolution Pipeline

## Overview
This package contains the self-contained, reproducible pipeline for the ML Challenge 2026 Business Entity Resolution task. It implements multi-key candidate blocking, pairwise feature extraction (lexical, token, fuzzy, and positional similarity), GBDT classification (LightGBM), decision threshold calibration on Macro F_0.5, and partition-wise test set inference.

## Environment Setup
Ensure Python 3.10+ is installed with the required dependencies:
```bash
pip install -r requirements.txt
```

## Running the End-to-End Pipeline
To reproduce the matching results and candidate pairs:
```bash
python run_entity_resolution.py --train-dir student_resource/dataset/train --test-dir student_resource/dataset/test --output-dir output
```

### Key Command-Line Options
- `--train-dir`: Directory containing `train_source1.tsv`, `train_source2.tsv`, `train_source3.tsv`, and `train_ground_truth.tsv` (default: `student_resource/dataset/train`).
- `--test-dir`: Directory containing `test_source1.tsv`, `test_source2.tsv`, and `test_source3.tsv` (default: `student_resource/dataset/test`).
- `--output-dir`: Output directory for generated TSV files (default: `output`).
- `--sample-train-s1`: Number of S1 training records to sample (-1 for full dataset, default: `50000`).
- `--max-candidates`: Maximum candidates per S1 entity during blocking (default: `25`).
- `--max-postings`: Maximum postings list cutoff for inverted index (default: `500`).
- `--n-splits`: Number of GroupKFold cross-validation splits (default: `5`).
- `--seed`: Random seed for reproducibility (default: `42`).
- `--tau`: Manual probability threshold override (default: auto-calibrated via F_0.5).
- `--infer-chunk-size`: Batch size for feature extraction and inference on candidate pairs (default: `50000`).
- `--skip-pack`: Skip automatic zip packaging at the end of execution.

## Output Files
- `output/matching_results.tsv`: Final predicted matches per Source 1 entity (scored on leaderboard).
- `output/candidate_pairs.tsv`: Candidate pairs proposed by the blocking engine before scoring.
"""


def find_file(candidate_paths):
    """Return the first existing path among candidate paths, or None."""
    for p in candidate_paths:
        if p and Path(p).is_file():
            return Path(p)
    return None


def find_dir(candidate_paths):
    """Return the first existing directory among candidate paths, or None."""
    for p in candidate_paths:
        if p and Path(p).is_dir():
            return Path(p)
    return None


def resolve_input_path(p: str | Path | None) -> Optional[Path]:
    """Resolve path prioritizing direct presence, then relative to PROJECT_ROOT."""
    if p is None:
        return None
    path = Path(p)
    if path.is_file() or path.is_dir() or path.is_absolute():
        return path
    if (PROJECT_ROOT / path).exists():
        return PROJECT_ROOT / path
    return path


def run_validator(
    validator_path: Path,
    matching_path: Path,
    candidate_path: Path,
    test_dir: Path,
    check_ids: bool = False,
) -> bool:
    """Invoke student_resource/utils/validate_submission.py and return True if successful."""
    cmd = [
        sys.executable,
        str(validator_path),
        "--matching", str(matching_path),
        "--candidate", str(candidate_path),
        "--test-dir", str(test_dir),
    ]
    if check_ids:
        cmd.append("--check-ids")
    print(f"Running submission validator: {' '.join(cmd)}")
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        print(f"[-] Validation FAILED (exit code {res.returncode}):", file=sys.stderr)
        if res.stdout:
            print(res.stdout, file=sys.stderr)
        if res.stderr:
            print(res.stderr, file=sys.stderr)
        return False

    print("[+] Validation PASSED:")
    if res.stdout:
        print(res.stdout.strip())
    return True


def package_submission_zip(
    matching_path: Path,
    candidate_path: Path,
    src_dir: Path,
    entrypoint_path: Optional[Path] = None,
    readme_path: Optional[Path] = None,
    requirements_path: Optional[Path] = None,
    doc_template_path: Optional[Path] = None,
    output_zip: Path = Path("outliers_submission_v3.zip"),
    mirror_dual_structure: bool = True,
) -> Path:
    """Create compliant competition submission zip file with optional dual-hierarchy mirroring."""
    output_zip = Path(output_zip)
    output_zip.parent.mkdir(parents=True, exist_ok=True)
    print(f"Creating submission archive: {output_zip}")

    prefixes = ["code/business_entity_resolution/"]
    if mirror_dual_structure:
        prefixes.append("business_entity_resolution/code/")

    with zipfile.ZipFile(output_zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        # 1. Output files
        zf.write(matching_path, arcname="output/matching_results.tsv")
        zf.write(candidate_path, arcname="output/candidate_pairs.tsv")

        # 2. Documentation template
        if doc_template_path and Path(doc_template_path).is_file():
            zf.write(doc_template_path, arcname="Documentation_template.md")
        else:
            zf.writestr(
                "Documentation_template.md",
                "# Business Entity Resolution Documentation\n\nMethodology details.\n"
            )

        # 3. Dedicated reproduction README
        for prefix in prefixes:
            if readme_path and Path(readme_path).is_file():
                zf.write(readme_path, arcname=f"{prefix}README.md")
            else:
                zf.writestr(
                    f"{prefix}README.md",
                    REPRODUCTION_README
                )

        # 4. Entrypoint script (run_entity_resolution.py)
        for prefix in prefixes:
            if entrypoint_path and Path(entrypoint_path).is_file():
                zf.write(entrypoint_path, arcname=f"{prefix}run_entity_resolution.py")
            elif (PROJECT_ROOT / "scripts" / "run_entity_resolution.py").is_file():
                zf.write(
                    PROJECT_ROOT / "scripts" / "run_entity_resolution.py",
                    arcname=f"{prefix}run_entity_resolution.py"
                )
            else:
                print(f"[!] Warning: Entrypoint script not found; omitting from {prefix}.")

        # 5. Requirements
        for prefix in prefixes:
            if requirements_path and Path(requirements_path).is_file():
                zf.write(requirements_path, arcname=f"{prefix}requirements.txt")
            elif (PROJECT_ROOT / "requirements.txt").is_file():
                zf.write(
                    PROJECT_ROOT / "requirements.txt",
                    arcname=f"{prefix}requirements.txt"
                )
            else:
                zf.writestr(
                    f"{prefix}requirements.txt",
                    "numpy\npandas\nlightgbm\nrapidfuzz\nscikit-learn\n"
                )

        # 6. Source code directory
        if src_dir and Path(src_dir).is_dir():
            src_path = Path(src_dir)
            if (src_path / "src").is_dir():
                effective_src = src_path / "src"
            else:
                effective_src = src_path

            for f in sorted(effective_src.rglob("*")):
                if f.is_file():
                    rel_p = f.relative_to(effective_src)
                    # Exclude pycache, egg-info, pytest, hidden files
                    if any(
                        part.startswith(".")
                        or part == "__pycache__"
                        or part.endswith(".egg-info")
                        or part.endswith(".dist-info")
                        or part.endswith(".pyc")
                        or part.endswith(".pyo")
                        for part in rel_p.parts
                    ):
                        continue
                    for prefix in prefixes:
                        arcname = f"{prefix}src/{rel_p.as_posix()}"
                        zf.write(f, arcname=arcname)

    print(f"[+] Successfully generated {output_zip} ({output_zip.stat().st_size} bytes)")
    return output_zip


def create_submission_archive(
    output_dir: str | Path,
    code_dir: str | Path,
    doc_file: Optional[str | Path],
    archive_path: str | Path,
    entrypoint_path: Optional[str | Path] = None,
    readme_path: Optional[str | Path] = None,
    requirements_path: Optional[str | Path] = None,
    mirror_dual_structure: bool = True,
) -> Path:
    """Programmatic API to build competition submission zip file."""
    out_dir = Path(output_dir)
    if out_dir.is_file():
        matching_path = out_dir
        candidate_path = out_dir.parent / "candidate_pairs.tsv"
    else:
        matching_path = out_dir / "matching_results.tsv"
        if not matching_path.is_file() and (out_dir / "output" / "matching_results.tsv").is_file():
            matching_path = out_dir / "output" / "matching_results.tsv"
        candidate_path = out_dir / "candidate_pairs.tsv"
        if not candidate_path.is_file() and (out_dir / "output" / "candidate_pairs.tsv").is_file():
            candidate_path = out_dir / "output" / "candidate_pairs.tsv"

    if not matching_path.is_file():
        raise FileNotFoundError(f"Matching results file not found at: {matching_path}")
    if not candidate_path.is_file():
        raise FileNotFoundError(f"Candidate pairs file not found at: {candidate_path}")

    return package_submission_zip(
        matching_path=matching_path,
        candidate_path=candidate_path,
        src_dir=Path(code_dir),
        entrypoint_path=Path(entrypoint_path) if entrypoint_path else None,
        readme_path=Path(readme_path) if readme_path else None,
        requirements_path=Path(requirements_path) if requirements_path else None,
        doc_template_path=Path(doc_file) if doc_file else None,
        output_zip=Path(archive_path),
        mirror_dual_structure=mirror_dual_structure,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Validate and package official submission zip for ML Challenge 2026."
    )
    parser.add_argument(
        "--matching", "-m",
        default=str(PROJECT_ROOT / "output" / "matching_results.tsv"),
        help="Path to matching_results.tsv (default: output/matching_results.tsv)",
    )
    parser.add_argument(
        "--candidate", "-c",
        default=str(PROJECT_ROOT / "output" / "candidate_pairs.tsv"),
        help="Path to candidate_pairs.tsv (default: output/candidate_pairs.tsv)",
    )
    parser.add_argument(
        "--test-dir", "-t",
        default=None,
        help="Directory containing test_source1.tsv for validation.",
    )
    parser.add_argument(
        "--validator-path",
        default=None,
        help="Path to student_resource/utils/validate_submission.py",
    )
    parser.add_argument(
        "--output-zip", "-o",
        default=str(PROJECT_ROOT / "outliers_submission_v3.zip"),
        help="Target submission zip path (default: outliers_submission_v3.zip)",
    )
    parser.add_argument(
        "--src-dir",
        default=str(PROJECT_ROOT / "src"),
        help="Path to source code directory (default: src)",
    )
    parser.add_argument(
        "--entrypoint",
        default=str(PROJECT_ROOT / "scripts" / "run_entity_resolution.py"),
        help="Path to run_entity_resolution.py (default: scripts/run_entity_resolution.py)",
    )
    parser.add_argument(
        "--readme",
        default=None,
        help="Path to README.md",
    )
    parser.add_argument(
        "--requirements",
        default=str(PROJECT_ROOT / "requirements.txt"),
        help="Path to requirements.txt (default: requirements.txt)",
    )
    parser.add_argument(
        "--doc-template",
        default=None,
        help="Path to Documentation_template.md",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Perform pre-flight checks without creating zip.",
    )
    parser.add_argument(
        "--skip-validation",
        action="store_true",
        help="Skip executing the submission validator gate.",
    )
    parser.add_argument(
        "--check-ids",
        action="store_true",
        help="Run validator with --check-ids flag enabled.",
    )
    parser.add_argument(
        "--no-mirror",
        action="store_true",
        help="Do not mirror hierarchy into business_entity_resolution/code/.",
    )

    args = parser.parse_args()

    matching_path = resolve_input_path(args.matching)
    candidate_path = resolve_input_path(args.candidate)
    src_dir = resolve_input_path(args.src_dir)
    entrypoint_path = resolve_input_path(args.entrypoint)
    readme_path = resolve_input_path(args.readme)
    requirements_path = resolve_input_path(args.requirements)
    output_zip = Path(args.output_zip)

    # Resolve validator path
    validator_path = find_file([
        args.validator_path,
        str(PROJECT_ROOT / "student_resource" / "utils" / "validate_submission.py"),
        str(PROJECT_ROOT / "utils" / "validate_submission.py"),
        "student_resource/utils/validate_submission.py",
        "../student_resource/utils/validate_submission.py",
        "utils/validate_submission.py",
    ])

    # Resolve documentation template path
    doc_template_path = find_file([
        args.doc_template,
        str(PROJECT_ROOT / "student_resource" / "Documentation_template.md"),
        str(PROJECT_ROOT / "Documentation_template.md"),
        str(PROJECT_ROOT / "docs" / "Documentation_template.md"),
        "student_resource/Documentation_template.md",
        "../student_resource/Documentation_template.md",
        "Documentation_template.md",
        "docs/Documentation_template.md",
    ])

    # Resolve test directory
    test_dir = find_dir([
        args.test_dir,
        str(PROJECT_ROOT / "student_resource" / "dataset" / "test"),
        str(PROJECT_ROOT / "dataset" / "test"),
        str(PROJECT_ROOT / "data" / "raw" / "dataset" / "test"),
        str(PROJECT_ROOT / "data" / "test"),
        "student_resource/dataset/test",
        "dataset/test",
        "data/raw/dataset/test",
        "data/test",
    ])

    print("=== Submissions Packager Pre-Flight Check ===")
    print(f"Matching file: {matching_path} (exists: {matching_path.is_file()})")
    print(f"Candidate file: {candidate_path} (exists: {candidate_path.is_file()})")
    print(f"Entrypoint script: {entrypoint_path} (exists: {entrypoint_path.is_file() if entrypoint_path else False})")
    print(f"Test directory: {test_dir}")
    print(f"Validator script: {validator_path}")
    print(f"Doc template: {doc_template_path}")
    print(f"Dry run: {args.dry_run}")

    # Enforce mandatory file existence before any dry-run exit
    if not matching_path.is_file():
        print(f"[-] Error: Matching results file not found: {matching_path}", file=sys.stderr)
        sys.exit(1)

    if not candidate_path.is_file():
        print(f"[-] Error: Candidate pairs file not found: {candidate_path}", file=sys.stderr)
        sys.exit(1)

    # Validate output files if validator and test directory exist
    if not args.skip_validation:
        if validator_path and test_dir:
            passed = run_validator(
                validator_path,
                matching_path,
                candidate_path,
                test_dir,
                check_ids=args.check_ids,
            )
            if not passed:
                print("[-] Validation check failed. Aborting packaging.", file=sys.stderr)
                sys.exit(1)
        else:
            print("[!] Warning: Validator script or test directory not found; skipping official validation.")

    if args.dry_run:
        print("[DRY-RUN] Pre-flight check and validation completed successfully.")
        sys.exit(0)

    # Package zip
    package_submission_zip(
        matching_path=matching_path,
        candidate_path=candidate_path,
        src_dir=src_dir,
        entrypoint_path=entrypoint_path,
        readme_path=readme_path,
        requirements_path=requirements_path,
        doc_template_path=doc_template_path,
        output_zip=output_zip,
        mirror_dual_structure=not args.no_mirror,
    )


if __name__ == "__main__":
    main()
