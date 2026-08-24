#!/bin/bash
# Submit nnU-Net junction-landmark (heatmap regression) training for all folds with SLURM dependencies.
# Fold 0 runs first (plans & preprocesses if needed); folds 1-4 start in parallel once fold 0
# completes successfully.
#
# Usage (from repo root):
#   bash JunctionDetection/nnUNetLandmark/nnunet_landmark_submit_folds.sh [TRAINER]
#
# TRAINER options:
#   nnUNetTrainerHeatmapMSE (default)                    - MSE loss (Nonlin_MSE_loss)
#   nnUNetTrainerHeatmapAdaptiveWing                     - Adaptive Wing loss
#   nnUNetTrainerHeatmapAdaptiveWingFocal                - AWL + focal hard-positive upweighting
#   nnUNetTrainerHeatmapAdaptiveWingSoftSampling          - AWL + snapshot-based background down-weighting
#   nnUNetTrainerHeatmapAdaptiveWingFocalSoftSampling     - AWL + both of the above
#   (see nnUNet/nnunetv2/training/loss/adaptive_wing.py and
#    nnUNet/nnunetv2/training/nnUNetTrainer/variants/heatmap/nnUNetTrainerHeatmapAdaptiveWing*.py)
#
# To override the number of folds (default 5):
#   NUM_FOLDS=3 bash JunctionDetection/nnUNetLandmark/nnunet_landmark_submit_folds.sh [TRAINER]
#
# To override the heatmap gaussian sigma (in pixels) without touching Environment/.env:
#   NNUNET_HEATMAP_SIGMA=4.0 bash JunctionDetection/nnUNetLandmark/nnunet_landmark_submit_folds.sh [TRAINER]
# This takes precedence over any NNUNET_HEATMAP_SIGMA set in Environment/.env (see fold job script).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
JOB_SCRIPT="${SCRIPT_DIR}/nnunet_landmark_fold_job.sh"

NUM_FOLDS="${NUM_FOLDS:-5}"
TRAINER="${1:-nnUNetTrainerHeatmapMSE}"

VALID_TRAINERS=(
    nnUNetTrainerHeatmapMSE
    nnUNetTrainerHeatmapAdaptiveWing
    nnUNetTrainerHeatmapAdaptiveWingFocal
    nnUNetTrainerHeatmapAdaptiveWingSoftSampling
    nnUNetTrainerHeatmapAdaptiveWingFocalSoftSampling
)
if [[ ! " ${VALID_TRAINERS[*]} " =~ " ${TRAINER} " ]]; then
    echo "Error: unknown trainer '${TRAINER}'"
    echo "Valid options: ${VALID_TRAINERS[*]}"
    exit 1
fi

# Only forwarded to the job (and from there into NNUNET_HEATMAP_SIGMA) if explicitly set here -
# otherwise the job script falls back to whatever Environment/.env provides.
EXPORT_VARS="FOLD,TRAINER"
if [[ -n "${NNUNET_HEATMAP_SIGMA:-}" ]]; then
    EXPORT_VARS="${EXPORT_VARS},NNUNET_HEATMAP_SIGMA"
fi

echo "Submitting nnU-Net junction-landmark training — ${NUM_FOLDS} folds, trainer: ${TRAINER}"
echo "Job script: ${JOB_SCRIPT}"
if [[ -n "${NNUNET_HEATMAP_SIGMA:-}" ]]; then
    echo "NNUNET_HEATMAP_SIGMA override: ${NNUNET_HEATMAP_SIGMA}"
fi
echo ""

# --- Fold 0 (must finish first) ---
FOLD0_JOB=$(FOLD=0 TRAINER="${TRAINER}" sbatch --parsable \
    --job-name="nnunet-landmark-fold0" \
    --export="${EXPORT_VARS}" \
    "$JOB_SCRIPT")

echo "Fold 0: job ${FOLD0_JOB} (runs first)"

# --- Folds 1..N-1 (parallel, depend on fold 0 succeeding) ---
for fold in $(seq 1 $(( NUM_FOLDS - 1 ))); do
    JOB_ID=$(FOLD="${fold}" TRAINER="${TRAINER}" sbatch --parsable \
        --job-name="nnunet-landmark-fold${fold}" \
        --export="${EXPORT_VARS}" \
        --dependency=afterok:"${FOLD0_JOB}" \
        "$JOB_SCRIPT")

    echo "Fold ${fold}: job ${JOB_ID} (waits for fold 0 — job ${FOLD0_JOB})"
done

echo ""
echo "All folds submitted. Monitor with:  squeue -u \$USER"
