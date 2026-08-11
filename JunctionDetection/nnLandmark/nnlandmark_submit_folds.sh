#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
JOB_SCRIPT="${SCRIPT_DIR}/nnlandmark_fold_job.sh"

NUM_FOLDS="${NUM_FOLDS:-5}"

echo "Submitting nnLandmark training — ${NUM_FOLDS} folds"
echo "Job script: ${JOB_SCRIPT}"
echo ""

# --- Fold 0 (must finish first) ---
FOLD0_JOB=$(sbatch --parsable \
    --job-name="nnlandmark-fold0" \
    --export=FOLD=0 \
    "$JOB_SCRIPT")

echo "Fold 0: job ${FOLD0_JOB} (runs first)"

# --- Folds 1..N-1 (parallel, depend on fold 0 succeeding) ---
for fold in $(seq 1 $(( NUM_FOLDS - 1 ))); do
    JOB_ID=$(sbatch --parsable \
        --job-name="nnlandmark-fold${fold}" \
        --export=FOLD="${fold}" \
        --dependency=afterok:"${FOLD0_JOB}" \
        "$JOB_SCRIPT")

    echo "Fold ${fold}: job ${JOB_ID} (waits for fold 0 — job ${FOLD0_JOB})"
done

echo ""
echo "All folds submitted. Monitor with:  squeue -u \$USER"
