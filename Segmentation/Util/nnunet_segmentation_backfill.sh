#!/bin/bash -l
#SBATCH --partition=standard
#SBATCH --gres=gpu:1
#SBATCH --gpus=1 --constraint="GPUMEM80GB|GPUMEM96GB|GPUMEM140GB"
#SBATCH --cpus-per-task=16
#SBATCH --mem=40G
#SBATCH --time=10:00:00
#SBATCH --output=/scratch/jhehli/logs/%x-%j.out
#SBATCH --error=/scratch/jhehli/logs/%x-%j.err

set -euo pipefail

REPO_ROOT="/home/jhehli/data/ForkSight"
cd "$REPO_ROOT"

echo "CWD: $(pwd)"
echo "Job ${SLURM_JOB_ID} on $(hostname)"

# load shared environment variables
set -a
source "${REPO_ROOT}/Environment/.env"
set +a

mkdir -p "/scratch/jhehli/logs"

source ~/.nnUNet_env/bin/activate
srun python -u -m Evaluation.infer_patches_junction_nnunet --save-probs