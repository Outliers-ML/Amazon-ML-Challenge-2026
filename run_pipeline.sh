#!/usr/bin/env bash
set -euo pipefail

# ML Challenge 2026: End-to-End Business Entity Resolution Pipeline Orchestrator

# Auto-detect script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Auto-detect Python executable
if [ -x "/dist_home/suryansh/miniforge3/envs/outliers/bin/python" ]; then
    PYTHON_BIN="/dist_home/suryansh/miniforge3/envs/outliers/bin/python"
elif command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="$(command -v python3)"
else
    PYTHON_BIN="python"
fi

echo "=================================================="
echo "ML Challenge 2026: End-to-End Pipeline Orchestration"
echo "Python interpreter: $PYTHON_BIN"
echo "Script directory:   $SCRIPT_DIR"
echo "=================================================="

# Step 1: Run Entity Resolution Pipeline
# If arguments are passed, forward "$@"; otherwise default to dataset paths
if [ "$#" -eq 0 ]; then
    RUN_ARGS=(
        "--train-dir" "$SCRIPT_DIR/student_resource/dataset/train"
        "--test-dir" "$SCRIPT_DIR/student_resource/dataset/test"
        "--output-dir" "$SCRIPT_DIR/output"
    )
else
    RUN_ARGS=("$@")
fi

echo ""
echo ">>> [Step 1/3] Running scripts/run_entity_resolution.py..."
"$PYTHON_BIN" "$SCRIPT_DIR/scripts/run_entity_resolution.py" "${RUN_ARGS[@]}"

# Step 2: Run Official Submission Validator Gate with --check-ids
echo ""
echo ">>> [Step 2/3] Running student_resource/utils/validate_submission.py --check-ids..."
VALIDATOR_SCRIPT="$SCRIPT_DIR/student_resource/utils/validate_submission.py"
MATCHING_FILE="$SCRIPT_DIR/output/matching_results.tsv"
CANDIDATE_FILE="$SCRIPT_DIR/output/candidate_pairs.tsv"
TEST_DIR="$SCRIPT_DIR/student_resource/dataset/test"

if [ -f "$VALIDATOR_SCRIPT" ]; then
    "$PYTHON_BIN" "$VALIDATOR_SCRIPT" \
        --matching "$MATCHING_FILE" \
        --candidate "$CANDIDATE_FILE" \
        --test-dir "$TEST_DIR" \
        --check-ids
else
    echo "[!] Warning: $VALIDATOR_SCRIPT not found; skipping Step 2 validation gate."
fi

# Step 3: Run Submission Packager with dual structure
echo ""
echo ">>> [Step 3/3] Running scripts/pack_submission.py --output-zip outliers_submission_v3.zip..."
PACK_SCRIPT="$SCRIPT_DIR/scripts/pack_submission.py"
"$PYTHON_BIN" "$PACK_SCRIPT" \
    --matching "$MATCHING_FILE" \
    --candidate "$CANDIDATE_FILE" \
    --test-dir "$TEST_DIR" \
    --output-zip "$SCRIPT_DIR/outliers_submission_v3.zip"

echo ""
echo "=================================================="
echo "[✔] Pipeline execution, validation, and packaging completed successfully!"
echo "Archive produced: $SCRIPT_DIR/outliers_submission_v3.zip"
echo "=================================================="
