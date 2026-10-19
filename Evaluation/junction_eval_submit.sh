#!/bin/bash
# Submit junction detection evaluation as a SLURM chain.
# Jobs run in series: nnunet inference → sam inference → metrics computation.
# An inference stage is skipped if its model list in
# Evaluation/compute_metrics_config.py (NNUNET_EVALUATIONS / SAM_MODELS_RUNS)
# is empty.
#
# Usage (from repo root):
#   bash Evaluation/junction_eval_submit.sh [--dataset-dir DIR] [--pred-dir DIR]
#
# All arguments are forwarded to every stage, e.g. to evaluate on the synthetic
# junction detection dataset instead of JUNCTION_DETECTION_DATASET_DIR:
#   bash Evaluation/junction_eval_submit.sh --dataset-dir /path/to/junction_detection_synthetic

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
JOB_SCRIPT="${SCRIPT_DIR}/junction_eval_job.sh"

count_models() {
    (cd "$REPO_ROOT" && python3 -c \
        "from Evaluation.compute_metrics_config import $1; print(len($1))")
}
N_NNUNET=$(count_models NNUNET_EVALUATIONS)
N_SAM=$(count_models SAM_MODELS_RUNS)

echo "Submitting junction evaluation pipeline..."
echo "Job script: ${JOB_SCRIPT}"
echo "Forwarded args: $*"
echo "Models: ${N_NNUNET} nnUNet, ${N_SAM} SAM"
echo ""

if (( N_NNUNET == 0 && N_SAM == 0 )); then
    echo "Error: no models configured in Evaluation/compute_metrics_config.py"
    exit 1
fi

PREV_JOB=""

# --- Stage 1: nnUNet inference ---
if (( N_NNUNET > 0 )); then
    PREV_JOB=$(sbatch --parsable \
        --job-name="junc-infer-nnunet" \
        --export=ALL,STAGE=nnunet \
        "$JOB_SCRIPT" "$@")
    echo "Stage 1 (nnunet inference): job ${PREV_JOB}"
else
    echo "Stage 1 (nnunet inference): skipped (NNUNET_EVALUATIONS is empty)"
fi

# --- Stage 2: SAM inference (waits for nnUNet) ---
if (( N_SAM > 0 )); then
    PREV_JOB=$(sbatch --parsable \
        --job-name="junc-infer-sam" \
        --export=ALL,STAGE=sam \
        ${PREV_JOB:+--dependency=afterok:${PREV_JOB}} \
        "$JOB_SCRIPT" "$@")
    echo "Stage 2 (sam inference):    job ${PREV_JOB}"
else
    echo "Stage 2 (sam inference):    skipped (SAM_MODELS_RUNS is empty)"
fi

# --- Stage 3: metrics computation (waits for inference) ---
METRICS_JOB=$(sbatch --parsable \
    --job-name="junc-metrics" \
    --gres="" --gpus=0 \
    --constraint="" \
    --export=ALL,STAGE=metrics \
    --dependency=afterok:"${PREV_JOB}" \
    "$JOB_SCRIPT" "$@")
echo "Stage 3 (metrics):          job ${METRICS_JOB} (waits for job ${PREV_JOB})"

echo ""
echo "All jobs submitted. Monitor with:  squeue -u \$USER"
