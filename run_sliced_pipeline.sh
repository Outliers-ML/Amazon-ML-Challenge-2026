#!/usr/bin/env bash
set -e

PYTHON="/dist_home/suryansh/miniforge3/envs/outliers/bin/python"
MODEL="models/ensemble_model.pkl"

echo "=== Starting Sliced India Inference (809,986 entities) ==="
# Slices of 100k
for start in 0 100000 200000 300000 400000 500000 600000 700000 800000; do
    end=$((start + 100000))
    if [ $end -gt 809986 ]; then
        end=809986
    fi
    echo "[+] Running India slice: ${start}:${end}..."
    $PYTHON -u scripts/run_entity_resolution.py \
        --countries India \
        --s1-slice "${start}:${end}" \
        --append-output \
        --load-model "$MODEL" \
        --use-cross-encoder \
        --use-gpu-gbdt \
        --device cuda
done

echo "=== Starting Sliced US Inference (663,106 entities) ==="
for start in 0 100000 200000 300000 400000 500000 600000; do
    end=$((start + 100000))
    if [ $end -gt 663106 ]; then
        end=663106
    fi
    echo "[+] Running US slice: ${start}:${end}..."
    $PYTHON -u scripts/run_entity_resolution.py \
        --countries US \
        --s1-slice "${start}:${end}" \
        --append-output \
        --load-model "$MODEL" \
        --use-cross-encoder \
        --use-gpu-gbdt \
        --device cuda
done

echo "=== Verifying Full Output File Counts ==="
wc -l output/matching_results.tsv output/candidate_pairs.tsv

echo "=== Running Official Submission Validator ==="
$PYTHON student_resource/utils/validate_submission.py \
    --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv \
    --test-dir student_resource/dataset/test \
    --check-ids

echo "=== Assembling Final Submission Package ==="
$PYTHON scripts/pack_submission.py \
    --output-zip outliers_submission_v3.zip \
    --test-dir student_resource/dataset/test

echo "[✔] All partitions finished, validated, and packaged successfully!"
