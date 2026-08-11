#!/bin/bash
#SBATCH --partition=standard
#SBATCH --gres=gpu:1
#SBATCH --gpus=1 --constraint="GPUMEM80GB|GPUMEM96GB|GPUMEM140GB"
#SBATCH --cpus-per-task=16
#SBATCH --mem=40G
#SBATCH --time=10:00:00
#SBATCH --output=/scratch/jhehli/logs/%x-%j.out
#SBATCH --error=/scratch/jhehli/logs/%x-%j.err

# Single nnLandmark fold training job.
# Expected env vars (passed via --export):
#   FOLD     — the fold number (0-4)

set -euo pipefail

# export nnUNet dataset paths (nnUNet expects these env vars to be set)
# .bashrc sets these but SLURM jobs don't load .bashrc, so we set them here explicitly
export nnLM_raw="/home/jhehli/data/datasets/nnLandmark/nnLM_raw/"
export nnLM_preprocessed="/home/jhehli/data/datasets/nnLandmark/nnLM_preprocessed/"
export nnLM_results="/home/jhehli/data/datasets/nnLandmark/nnLM_results/"

: "${FOLD:?FOLD env var is required (0-4)}"

echo "CWD: $(pwd)"
echo "FOLD: ${FOLD}"
echo "Job ${SLURM_JOB_ID} on $(hostname)"

# activate nnUNet virtual environment
source ~/.nnLandmark_env/bin/activate

mkdir -p "/scratch/jhehli/logs"

nnLM_train 001 2d "$FOLD" --npz
