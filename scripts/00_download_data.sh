#!/usr/bin/env bash
# ==============================================================================
# Amazon ML Challenge 2026 - Data Helper Script
# Place or extract raw competition files into data/raw/
# ==============================================================================

set -euo pipefail

TARGET_DIR="data/raw"
mkdir -p "${TARGET_DIR}"

echo "=========================================="
echo "Amazon ML Challenge 2026 - Data Directory"
echo "Target directory: ${TARGET_DIR}"
echo "=========================================="

if [ $# -eq 0 ]; then
    echo "Usage:"
    echo "  bash scripts/00_download_data.sh <path_to_zip_or_csv>"
    echo ""
    echo "Current contents of ${TARGET_DIR}:"
    ls -lh "${TARGET_DIR}"
    exit 0
fi

INPUT_PATH="$1"

if [[ "${INPUT_PATH}" == *.zip ]]; then
    echo "Extracting ${INPUT_PATH} into ${TARGET_DIR}..."
    unzip -q -o "${INPUT_PATH}" -d "${TARGET_DIR}"
    echo "Extracted successfully."
elif [[ -f "${INPUT_PATH}" ]]; then
    echo "Copying ${INPUT_PATH} into ${TARGET_DIR}..."
    cp "${INPUT_PATH}" "${TARGET_DIR}/"
    echo "Copied successfully."
else
    echo "Error: ${INPUT_PATH} does not exist."
    exit 1
fi

echo "Current files in ${TARGET_DIR}:"
ls -lh "${TARGET_DIR}"
