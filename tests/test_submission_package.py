"""Tests for submission packaging and pipeline orchestration (Task 7)."""

import os
from pathlib import Path
import subprocess
import sys
import zipfile
import pytest

from scripts.pack_submission import create_submission_archive, package_submission_zip

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def test_zip_contains_dual_directory_structure(tmp_path):
    """Verify create_submission_archive generates a zip containing both primary

    and mirrored code hierarchies, output files, and documentation.
    """
    out_dir = tmp_path / "output"
    out_dir.mkdir(parents=True)
    matching_tsv = out_dir / "matching_results.tsv"
    matching_tsv.write_text("source1_entity_id\tmatched_entity_ids\nS1-01\tS2-01\n", encoding="utf-8")
    candidate_tsv = out_dir / "candidate_pairs.tsv"
    candidate_tsv.write_text("source1_entity_id\tcandidate_entity_ids\nS1-01\tS2-01\n", encoding="utf-8")

    doc_file = tmp_path / "Documentation_template.md"
    doc_file.write_text("# Test Docs\nCustom doc content.", encoding="utf-8")

    src_dir = tmp_path / "src"
    src_dir.mkdir(parents=True)
    (src_dir / "model.py").write_text("# model code", encoding="utf-8")
    subpkg = src_dir / "utils"
    subpkg.mkdir(parents=True)
    (subpkg / "helper.py").write_text("# helper code", encoding="utf-8")

    # Unwanted files that should be filtered out
    cache_dir = src_dir / "__pycache__"
    cache_dir.mkdir()
    (cache_dir / "model.cpython-310.pyc").write_bytes(b"byte-code")
    (src_dir / "stray.pyc").write_bytes(b"byte-code")
    egg_dir = src_dir / "package.egg-info"
    egg_dir.mkdir()
    (egg_dir / "PKG-INFO").write_text("pkg info", encoding="utf-8")
    (src_dir / ".hidden_file").write_text("hidden", encoding="utf-8")

    entrypoint = tmp_path / "run_entity_resolution.py"
    entrypoint.write_text("# entrypoint", encoding="utf-8")
    readme = tmp_path / "README.md"
    readme.write_text("# Reproduction README", encoding="utf-8")
    reqs = tmp_path / "requirements.txt"
    reqs.write_text("torch\npandas\n", encoding="utf-8")

    zip_dual = tmp_path / "test_dual.zip"

    # Test dual structure enabled
    res_path = create_submission_archive(
        output_dir=out_dir,
        code_dir=src_dir,
        doc_file=doc_file,
        archive_path=zip_dual,
        entrypoint_path=entrypoint,
        readme_path=readme,
        requirements_path=reqs,
        mirror_dual_structure=True,
    )
    assert Path(res_path).is_file()

    with zipfile.ZipFile(zip_dual, "r") as zf:
        namelist = set(zf.namelist())

        # 1. Output files
        assert "output/matching_results.tsv" in namelist
        assert "output/candidate_pairs.tsv" in namelist

        # 2. Documentation template
        assert "Documentation_template.md" in namelist
        doc_content = zf.read("Documentation_template.md").decode("utf-8")
        assert "Custom doc content." in doc_content

        # 3. Primary hierarchy (code/business_entity_resolution/...)
        assert "code/business_entity_resolution/README.md" in namelist
        assert "code/business_entity_resolution/run_entity_resolution.py" in namelist
        assert "code/business_entity_resolution/requirements.txt" in namelist
        assert "code/business_entity_resolution/src/model.py" in namelist
        assert "code/business_entity_resolution/src/utils/helper.py" in namelist

        # 4. Mirrored hierarchy (business_entity_resolution/code/...)
        assert "business_entity_resolution/code/README.md" in namelist
        assert "business_entity_resolution/code/run_entity_resolution.py" in namelist
        assert "business_entity_resolution/code/requirements.txt" in namelist
        assert "business_entity_resolution/code/src/model.py" in namelist
        assert "business_entity_resolution/code/src/utils/helper.py" in namelist

        # 5. Excluded unwanted files
        for name in namelist:
            assert "__pycache__" not in name
            assert not name.endswith(".pyc")
            assert ".egg-info" not in name
            # No hidden file names
            parts = Path(name).parts
            assert not any(part.startswith(".") for part in parts)

    # Test mirror_dual_structure=False
    zip_single = tmp_path / "test_single.zip"
    create_submission_archive(
        output_dir=out_dir,
        code_dir=src_dir,
        doc_file=None,
        archive_path=zip_single,
        entrypoint_path=entrypoint,
        readme_path=readme,
        requirements_path=reqs,
        mirror_dual_structure=False,
    )
    with zipfile.ZipFile(zip_single, "r") as zf:
        namelist_single = set(zf.namelist())
        assert "code/business_entity_resolution/README.md" in namelist_single
        assert "business_entity_resolution/code/README.md" not in namelist_single
        # Fallback doc content
        assert "Documentation_template.md" in namelist_single
        assert "Business Entity Resolution" in zf.read("Documentation_template.md").decode("utf-8")


def test_run_pipeline_script_syntax_and_permissions():
    """Verify run_pipeline.sh exists, is executable, has valid bash shebang,

    and passes syntax verification.
    """
    script_path = PROJECT_ROOT / "run_pipeline.sh"
    assert script_path.is_file(), f"run_pipeline.sh not found at {script_path}"
    assert os.access(script_path, os.X_OK), "run_pipeline.sh must be executable"

    content = script_path.read_text(encoding="utf-8")
    lines = content.splitlines()
    assert len(lines) > 0
    assert lines[0].startswith("#!"), "run_pipeline.sh must have a shebang"
    assert "bash" in lines[0], "run_pipeline.sh must use bash shebang"
    assert "set -euo pipefail" in content, "run_pipeline.sh must set -euo pipefail"
    assert "run_entity_resolution.py" in content, "run_pipeline.sh must call run_entity_resolution.py"
    assert "--check-ids" in content, "run_pipeline.sh must pass --check-ids to validator"
    assert "outliers_submission_v3.zip" in content, "run_pipeline.sh must package outliers_submission_v3.zip"

    # Check syntax with bash -n
    res = subprocess.run(["bash", "-n", str(script_path)], capture_output=True, text=True)
    assert res.returncode == 0, f"bash -n failed with syntax error:\n{res.stderr}"


def test_pack_submission_cli(tmp_path):
    """Test CLI execution of pack_submission.py for both dry-run and archive packaging."""
    out_dir = tmp_path / "output"
    out_dir.mkdir(parents=True)
    matching_tsv = out_dir / "matching_results.tsv"
    matching_tsv.write_text("source1_entity_id\tmatched_entity_ids\nS1-01\tS2-01\n", encoding="utf-8")
    candidate_tsv = out_dir / "candidate_pairs.tsv"
    candidate_tsv.write_text("source1_entity_id\tcandidate_entity_ids\nS1-01\tS2-01\n", encoding="utf-8")

    test_dir = tmp_path / "dataset" / "test"
    test_dir.mkdir(parents=True)
    (test_dir / "test_source1.tsv").write_text("entity_id\tbusiness_name\tbusiness_address\tcountry\nS1-01\tAcme\tRoad\tUS\n", encoding="utf-8")

    doc_file = tmp_path / "Documentation_template.md"
    doc_file.write_text("# Doc Template\n", encoding="utf-8")

    zip_out = tmp_path / "cli_out.zip"

    # Test dry run
    cmd_dry = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "pack_submission.py"),
        "--matching", str(matching_tsv),
        "--candidate", str(candidate_tsv),
        "--test-dir", str(test_dir),
        "--doc-template", str(doc_file),
        "--output-zip", str(zip_out),
        "--dry-run",
    ]
    res_dry = subprocess.run(cmd_dry, capture_output=True, text=True)
    assert res_dry.returncode == 0, f"Dry-run failed: {res_dry.stderr}\n{res_dry.stdout}"
    assert not zip_out.exists(), "Dry-run should not create zip file"

    # Test full pack CLI
    cmd_pack = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "pack_submission.py"),
        "--matching", str(matching_tsv),
        "--candidate", str(candidate_tsv),
        "--test-dir", str(test_dir),
        "--doc-template", str(doc_file),
        "--output-zip", str(zip_out),
    ]
    res_pack = subprocess.run(cmd_pack, capture_output=True, text=True)
    assert res_pack.returncode == 0, f"Pack failed: {res_pack.stderr}\n{res_pack.stdout}"
    assert zip_out.is_file(), "Zip archive should have been created"

    with zipfile.ZipFile(zip_out, "r") as zf:
        names = zf.namelist()
        assert "output/matching_results.tsv" in names
        assert "output/candidate_pairs.tsv" in names
        assert "Documentation_template.md" in names
        assert "code/business_entity_resolution/README.md" in names
        assert "business_entity_resolution/code/README.md" in names


def test_create_submission_archive_missing_files_and_direct_file(tmp_path):
    """Test FileNotFoundError on missing files, out_dir as direct file path, and artifact exclusions (.dist-info, .pyo)."""
    src_dir = tmp_path / "src"
    src_dir.mkdir(parents=True)
    (src_dir / "app.py").write_text("# app", encoding="utf-8")
    (src_dir / "cached.pyo").write_bytes(b"pyo")
    dist_info = src_dir / "demo-1.0.dist-info"
    dist_info.mkdir(parents=True)
    (dist_info / "METADATA").write_text("metadata", encoding="utf-8")

    archive_zip = tmp_path / "test_hardening.zip"

    # 1. Missing output files should raise FileNotFoundError
    empty_out = tmp_path / "empty_output"
    empty_out.mkdir()
    with pytest.raises(FileNotFoundError, match="Matching results file not found"):
        create_submission_archive(
            output_dir=empty_out,
            code_dir=src_dir,
            doc_file=None,
            archive_path=archive_zip,
        )

    # Missing candidate file should also raise FileNotFoundError
    matching_only = tmp_path / "matching_only"
    matching_only.mkdir()
    (matching_only / "matching_results.tsv").write_text("source1_entity_id\tmatched_entity_ids\n", encoding="utf-8")
    with pytest.raises(FileNotFoundError, match="Candidate pairs file not found"):
        create_submission_archive(
            output_dir=matching_only,
            code_dir=src_dir,
            doc_file=None,
            archive_path=archive_zip,
        )

    # 2. When output_dir is a direct file path to matching_results.tsv
    matching_file = matching_only / "matching_results.tsv"
    candidate_file = matching_only / "candidate_pairs.tsv"
    candidate_file.write_text("source1_entity_id\tcandidate_entity_ids\n", encoding="utf-8")

    res_path = create_submission_archive(
        output_dir=matching_file,  # direct file path
        code_dir=src_dir,
        doc_file=None,
        archive_path=archive_zip,
    )
    assert Path(res_path).is_file()

    with zipfile.ZipFile(archive_zip, "r") as zf:
        namelist = set(zf.namelist())
        assert "output/matching_results.tsv" in namelist
        assert "output/candidate_pairs.tsv" in namelist
        # Verify .pyo and .dist-info are excluded
        for name in namelist:
            assert not name.endswith(".pyo")
            assert ".dist-info" not in name

