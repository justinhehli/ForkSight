#!/bin/bash
# Submit nnUNet training for all folds with SLURM dependencies.
# Fold 0 runs first (plans & preprocesses if needed); folds 1-4
# start in parallel once fold 0 completes successfully.
#
# Usage (from repo root):
#   bash Segmentation/nnUNet/nnunet_submit_folds.sh [TRAINER]
#
# TRAINER options:
#   nnUNetTrainerWandb      (default)
#   nnUNetTrainerClDiceLoss
#
# To override the number of folds (default 5):
#   NUM_FOLDS=3 bash Segmentation/nnUNet/nnunet_submit_folds.sh [TRAINER]
#
# To resume all folds from their latest checkpoints (e.g. after hitting the
# time limit; folds without a checkpoint start from scratch):
#   CONTINUE=1 bash Segmentation/nnUNet/nnunet_submit_folds.sh [TRAINER]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
JOB_SCRIPT="${SCRIPT_DIR}/nnunet_fold_job.sh"

NUM_FOLDS="${NUM_FOLDS:-5}"
TRAINER="${1:-nnUNetTrainerWandb}"
CONTINUE="${CONTINUE:-0}"

if [[ "$TRAINER" != "nnUNetTrainerWandb" && "$TRAINER" != "nnUNetTrainerClDiceLoss" ]]; then
    echo "Error: unknown trainer '${TRAINER}'"
    echo "Valid options: nnUNetTrainerWandb, nnUNetTrainerClDiceLoss"
    exit 1
fi

echo "Submitting nnUNet training — ${NUM_FOLDS} folds, trainer: ${TRAINER}, continue: ${CONTINUE}"
echo "Job script: ${JOB_SCRIPT}"
echo ""

# --- Fold 0 (must finish first) ---
FOLD0_JOB=$(sbatch --parsable \
    --job-name="nnunet-fold0" \
    --export=FOLD=0,TRAINER="${TRAINER}",CONTINUE="${CONTINUE}" \
    "$JOB_SCRIPT")

echo "Fold 0: job ${FOLD0_JOB} (runs first)"

# --- Folds 1..N-1 (parallel, depend on fold 0 succeeding) ---
for fold in $(seq 1 $(( NUM_FOLDS - 1 ))); do
    JOB_ID=$(sbatch --parsable \
        --job-name="nnunet-fold${fold}" \
        --export=FOLD="${fold}",TRAINER="${TRAINER}",CONTINUE="${CONTINUE}" \
        --dependency=afterok:"${FOLD0_JOB}" \
        "$JOB_SCRIPT")

    echo "Fold ${fold}: job ${JOB_ID} (waits for fold 0 — job ${FOLD0_JOB})"
done

echo ""
echo "All folds submitted. Monitor with:  squeue -u \$USER"
