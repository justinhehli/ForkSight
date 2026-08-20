#!/bin/bash
# Submit nnU-Net junction-landmark (heatmap regression) training for all folds with SLURM dependencies.
# Fold 0 runs first (plans & preprocesses if needed); folds 1-4 start in parallel once fold 0
# completes successfully.
#
# Usage (from repo root):
#   bash JunctionDetection/nnUNetLandmark/nnunet_landmark_submit_folds.sh [TRAINER]
#
# TRAINER options:
#   nnUNetTrainerHeatmapMSE (default) - MSE loss (Nonlin_MSE_loss)
#   nnUNetTrainerHeatmapAdaptiveWing  - Adaptive Wing loss (see nnUNet/nnunetv2/training/loss/adaptive_wing.py)
#
# To override the number of folds (default 5):
#   NUM_FOLDS=3 bash JunctionDetection/nnUNetLandmark/nnunet_landmark_submit_folds.sh [TRAINER]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
JOB_SCRIPT="${SCRIPT_DIR}/nnunet_landmark_fold_job.sh"

NUM_FOLDS="${NUM_FOLDS:-5}"
TRAINER="${1:-nnUNetTrainerHeatmapMSE}"

if [[ "$TRAINER" != "nnUNetTrainerHeatmapMSE" && "$TRAINER" != "nnUNetTrainerHeatmapAdaptiveWing" ]]; then
    echo "Error: unknown trainer '${TRAINER}'"
    echo "Valid options: nnUNetTrainerHeatmapMSE, nnUNetTrainerHeatmapAdaptiveWing"
    exit 1
fi

echo "Submitting nnU-Net junction-landmark training — ${NUM_FOLDS} folds, trainer: ${TRAINER}"
echo "Job script: ${JOB_SCRIPT}"
echo ""

# --- Fold 0 (must finish first) ---
FOLD0_JOB=$(sbatch --parsable \
    --job-name="nnunet-landmark-fold0" \
    --export=FOLD=0,TRAINER="${TRAINER}" \
    "$JOB_SCRIPT")

echo "Fold 0: job ${FOLD0_JOB} (runs first)"

# --- Folds 1..N-1 (parallel, depend on fold 0 succeeding) ---
for fold in $(seq 1 $(( NUM_FOLDS - 1 ))); do
    JOB_ID=$(sbatch --parsable \
        --job-name="nnunet-landmark-fold${fold}" \
        --export=FOLD="${fold}",TRAINER="${TRAINER}" \
        --dependency=afterok:"${FOLD0_JOB}" \
        "$JOB_SCRIPT")

    echo "Fold ${fold}: job ${JOB_ID} (waits for fold 0 — job ${FOLD0_JOB})"
done

echo ""
echo "All folds submitted. Monitor with:  squeue -u \$USER"
