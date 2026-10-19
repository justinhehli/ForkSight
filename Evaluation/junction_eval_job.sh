#!/bin/bash -l
#SBATCH --partition=standard
#SBATCH --gres=gpu:1
#SBATCH --gpus=1 --constraint="GPUMEM80GB|GPUMEM96GB|GPUMEM140GB"
#SBATCH --cpus-per-task=16
#SBATCH --mem=40G
#SBATCH --time=24:00:00
#SBATCH --output=/scratch/jhehli/logs/%x-%j.out
#SBATCH --error=/scratch/jhehli/logs/%x-%j.err

# Junction detection evaluation job.
# Expected env vars (passed via --export):
#   STAGE   — one of: nnunet | sam | metrics
# Any script arguments (e.g. --dataset-dir DIR [--pred-dir DIR]) are forwarded
# to the stage's python script.

set -euo pipefail

: "${STAGE:?STAGE env var is required (nnunet | sam | metrics)}"

REPO_ROOT="/home/jhehli/data/ForkSight"
cd "$REPO_ROOT"

echo "CWD: $(pwd)"
echo "STAGE: ${STAGE}"
echo "Job ${SLURM_JOB_ID} on $(hostname)"

# load shared environment variables
set -a
source "${REPO_ROOT}/Environment/.env"
set +a

mkdir -p "/scratch/jhehli/logs"

case "$STAGE" in
    nnunet)
        source ~/.nnUNet_env/bin/activate
        srun python -u -m Evaluation.infer_patches_junction_nnunet "$@"
        ;;
    sam)
        source "$SAM_LORA_VENV/bin/activate"
        srun python -u -m Evaluation.infer_patches_junction_sam --batch-size 8 "$@"
        ;;
    metrics)
        source "$SAM_LORA_VENV/bin/activate"
        srun python -u -m Evaluation.compute_metrics_junction_detection --plot --plot-skeleton --force-recompute "$@"
        ;;
    *)
        echo "Error: unknown STAGE '${STAGE}'. Valid options: nnunet | sam | metrics"
        exit 1
        ;;
esac