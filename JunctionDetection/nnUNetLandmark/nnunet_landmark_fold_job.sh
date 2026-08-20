#!/bin/bash
#SBATCH --partition=standard
#SBATCH --gres=gpu:1
#SBATCH --gpus=1 --constraint="GPUMEM80GB|GPUMEM96GB|GPUMEM140GB"
#SBATCH --cpus-per-task=16
#SBATCH --mem=40G
#SBATCH --time=10:00:00
#SBATCH --output=/scratch/jhehli/logs/%x-%j.out
#SBATCH --error=/scratch/jhehli/logs/%x-%j.err

# Single nnU-Net junction-landmark (heatmap regression) fold training job.
# Expected env vars (passed via --export):
#   FOLD     — the fold number (0-4)
#   TRAINER  — nnUNet trainer class (default: nnUNetTrainerHeatmapMSE)

set -euo pipefail

# export nnUNet dataset paths (nnUNet expects these env vars to be set)
export nnUNet_raw="/home/jhehli/data/datasets/nnUNet/nnUNet_raw"
export nnUNet_preprocessed="/home/jhehli/data/datasets/nnUNet/nnUNet_preprocessed"
export nnUNet_results="/home/jhehli/data/datasets/nnUNet/nnUNet_results"

# Hardcoded, not read from Environment/.env - must match NNUNET_LANDMARK_DATASET_ID in
# JunctionDetection/PreProcessing/create_nnunet_heatmap_dataset.py and NNUNET_LANDMARK_MODEL_DIR in
# Evaluation/compute_metrics_junction_detection_nnunet_landmark.py
NNUNET_LANDMARK_DATASET_ID="011"

: "${FOLD:?FOLD env var is required (0-4)}"
TRAINER="${TRAINER:-nnUNetTrainerHeatmapMSE}"

REPO_ROOT="/home/jhehli/data/ForkSight"
cd "$REPO_ROOT"

echo "CWD: $(pwd)"
echo "FOLD: ${FOLD}"
echo "TRAINER: ${TRAINER}"
echo "DATASET_ID: ${NNUNET_LANDMARK_DATASET_ID}"
echo "Job ${SLURM_JOB_ID} on $(hostname)"

# load environment variables (e.g. NNUNET_HEATMAP_SIGMA/THRESHOLD/MIN_DISTANCE, paths, etc.)
set -a
source "${REPO_ROOT}/Environment/.env"
set +a

# activate nnUNet virtual environment
source ~/.nnUNet_env/bin/activate

mkdir -p "/scratch/jhehli/logs"

nnUNetv2_train "$NNUNET_LANDMARK_DATASET_ID" 2d "$FOLD" -tr "$TRAINER" --npz
