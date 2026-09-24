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
import zipfile

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


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


def run_validator(
    validator_path: Path,
    matching_path: Path,
    candidate_path: Path,
    test_dir: Path,
) -> bool:
    """Invoke student_resource/utils/validate_submission.py and return True if successful."""
    cmd = [
        sys.executable,
        str(validator_path),
        "--matching", str(matching_path),
        "--candidate", str(candidate_path),
        "--test-dir", str(test_dir),
    ]
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
    readme_path: Path,
    requirements_path: Path,
    doc_template_path: Path,
    output_zip: Path,
) -> None:
    """Create compliant competition submission zip file."""
    output_zip.parent.mkdir(parents=True, exist_ok=True)
    print(f"Creating submission archive: {output_zip}")

    with zipfile.ZipFile(output_zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        # 1. Output files
        zf.write(matching_path, arcname="output/matching_results.tsv")
        zf.write(candidate_path, arcname="output/candidate_pairs.tsv")

        # 2. Documentation template
        if doc_template_path and doc_template_path.is_file():
            zf.write(doc_template_path, arcname="Documentation_template.md")
        else:
            zf.writestr(
                "Documentation_template.md",
                "# Business Entity Resolution Documentation\n\nMethodology details.\n"
            )

        # 3. Code documentation & requirements
        if readme_path and readme_path.is_file():
            zf.write(readme_path, arcname="code/business_entity_resolution/README.md")
        else:
            zf.writestr(
                "code/business_entity_resolution/README.md",
                "# Business Entity Resolution Pipeline\n\nInstructions.\n"
            )

        if requirements_path and requirements_path.is_file():
            zf.write(requirements_path, arcname="code/business_entity_resolution/requirements.txt")
        else:
            zf.writestr(
                "code/business_entity_resolution/requirements.txt",
                "numpy\npandas\nlightgbm\nrapidfuzz\nscikit-learn\n"
            )

        # 4. Source code directory
        if src_dir and src_dir.is_dir():
            for f in sorted(src_dir.rglob("*")):
                if f.is_file():
                    # Exclude pycache, egg-info, pytest, hidden files
                    if any(
                        part.startswith(".")
                        or part == "__pycache__"
                        or part.endswith(".egg-info")
                        or part.endswith(".pyc")
                        for part in f.parts
                    ):
                        continue
                    rel_p = f.relative_to(src_dir)
                    arcname = f"code/business_entity_resolution/src/{rel_p.as_posix()}"
                    zf.write(f, arcname=arcname)

    print(f"[+] Successfully generated {output_zip} ({output_zip.stat().st_size} bytes)")


def main():
    parser = argparse.ArgumentParser(
        description="Validate and package official submission zip for ML Challenge 2026."
    )
    parser.add_argument(
        "--matching", "-m",
        default="output/matching_results.tsv",
        help="Path to matching_results.tsv (default: output/matching_results.tsv)",
    )
    parser.add_argument(
        "--candidate", "-c",
        default="output/candidate_pairs.tsv",
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
        default="outliers_submission.zip",
        help="Target submission zip path (default: outliers_submission.zip)",
    )
    parser.add_argument(
        "--src-dir",
        default="src",
        help="Path to source code directory (default: src)",
    )
    parser.add_argument(
        "--readme",
        default="README.md",
        help="Path to README.md (default: README.md)",
    )
    parser.add_argument(
        "--requirements",
        default="requirements.txt",
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

    args = parser.parse_args()

    matching_path = Path(args.matching)
    candidate_path = Path(args.candidate)
    src_dir = Path(args.src_dir)
    readme_path = Path(args.readme)
    requirements_path = Path(args.requirements)
    output_zip = Path(args.output_zip)

    # Resolve validator path
    validator_path = find_file([
        args.validator_path,
        "student_resource/utils/validate_submission.py",
        "../student_resource/utils/validate_submission.py",
        "utils/validate_submission.py",
    ])

    # Resolve documentation template path
    doc_template_path = find_file([
        args.doc_template,
        "student_resource/Documentation_template.md",
        "../student_resource/Documentation_template.md",
        "Documentation_template.md",
        "docs/Documentation_template.md",
    ])

    # Resolve test directory
    test_dir = find_dir([
        args.test_dir,
        "student_resource/dataset/test",
        "dataset/test",
        "data/raw/dataset/test",
        "data/test",
    ])

    print("=== Submissions Packager Pre-Flight Check ===")
    print(f"Matching file: {matching_path} (exists: {matching_path.is_file()})")
    print(f"Candidate file: {candidate_path} (exists: {candidate_path.is_file()})")
    print(f"Test directory: {test_dir}")
    print(f"Validator script: {validator_path}")
    print(f"Doc template: {doc_template_path}")
    print(f"Dry run: {args.dry_run}")

    if args.dry_run:
        print("[DRY-RUN] Pre-flight check completed.")
        if matching_path.is_file() and candidate_path.is_file() and validator_path and test_dir and not args.skip_validation:
            valid = run_validator(validator_path, matching_path, candidate_path, test_dir)
            if not valid:
                print("[-] Dry run validation failed.", file=sys.stderr)
                sys.exit(1)
            print("[+] Dry run validation succeeded.")
        sys.exit(0)

    # Enforce file existence
    if not matching_path.is_file():
        print(f"[-] Error: Matching results file not found: {matching_path}", file=sys.stderr)
        sys.exit(1)

    if not candidate_path.is_file():
        print(f"[-] Error: Candidate pairs file not found: {candidate_path}", file=sys.stderr)
        sys.exit(1)

    # Validate output files if validator and test directory exist
    if not args.skip_validation:
        if validator_path and test_dir:
            passed = run_validator(validator_path, matching_path, candidate_path, test_dir)
            if not passed:
                print("[-] Validation check failed. Aborting packaging.", file=sys.stderr)
                sys.exit(1)
        else:
            print("[!] Warning: Validator script or test directory not found; skipping official validation.")

    # Package zip
    package_submission_zip(
        matching_path=matching_path,
        candidate_path=candidate_path,
        src_dir=src_dir,
        readme_path=readme_path,
        requirements_path=requirements_path,
        doc_template_path=doc_template_path,
        output_zip=output_zip,
    )


if __name__ == "__main__":
    main()
