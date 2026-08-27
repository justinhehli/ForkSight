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
# If TRAINER is nnUNetTrainerHeatmapAdaptiveWingFocalSoftSamplingSingleLabel (the single-combined-label
# trainer - see its docstring), this also runs patch_single_label_plans.py (idempotent) before
# nnUNetv2_train, which that trainer requires.
#
# Expected env vars (passed via --export):
#   FOLD                   — the fold number (0-4)
#   TRAINER                — nnUNet trainer class (default: nnUNetTrainerHeatmapMSE)
#   DATASET                — nnU-Net dataset ID (default: 11)
#   NNUNET_HEATMAP_SIGMA   — optional, heatmap gaussian sigma in pixels; overrides whatever
#                            Environment/.env sets, see below

set -euo pipefail

# export nnUNet dataset paths (nnUNet expects these env vars to be set)
export nnUNet_raw="/home/jhehli/data/datasets/nnUNet/nnUNet_raw"
export nnUNet_preprocessed="/home/jhehli/data/datasets/nnUNet/nnUNet_preprocessed"
export nnUNet_results="/home/jhehli/data/datasets/nnUNet/nnUNet_results"

export PYTHONUNBUFFERED=1

: "${FOLD:?FOLD env var is required (0-4)}"
TRAINER="${TRAINER:-nnUNetTrainerHeatmapMSE}"
NNUNET_LANDMARK_DATASET_ID=$(printf "%03d" "$((10#${DATASET:-11}))")

REPO_ROOT="/home/jhehli/data/ForkSight"
cd "$REPO_ROOT"

echo "CWD: $(pwd)"
echo "FOLD: ${FOLD}"
echo "TRAINER: ${TRAINER}"
echo "DATASET_ID: ${NNUNET_LANDMARK_DATASET_ID}"
echo "Job ${SLURM_JOB_ID} on $(hostname)"

# Preserve a NNUNET_HEATMAP_SIGMA passed in via sbatch --export (from the submit script's
# NNUNET_HEATMAP_SIGMA=... argument) across sourcing .env below - Environment/.env may itself
# unconditionally set NNUNET_HEATMAP_SIGMA, which would otherwise silently clobber this override.
SIGMA_OVERRIDE="${NNUNET_HEATMAP_SIGMA:-}"

# load environment variables (e.g. NNUNET_HEATMAP_SIGMA/THRESHOLD/MIN_DISTANCE, paths, etc.)
set -a
source "${REPO_ROOT}/Environment/.env"
set +a

if [[ -n "${SIGMA_OVERRIDE}" ]]; then
    export NNUNET_HEATMAP_SIGMA="${SIGMA_OVERRIDE}"
fi
echo "NNUNET_HEATMAP_SIGMA: ${NNUNET_HEATMAP_SIGMA:-<unset, using trainer default>}"

# activate nnUNet virtual environment
source ~/.nnUNet_env/bin/activate

mkdir -p "/scratch/jhehli/logs"

if [[ "$TRAINER" == "nnUNetTrainerHeatmapAdaptiveWingFocalSoftSamplingSingleLabel" ]]; then
    # register SingleHeadLabelManager on this dataset's plans, so the
    # network is sized with 1 output channel instead of the default 2 - 
    # see JunctionDetection/PreProcessing/patch_single_label_plans.py and
    # nnUNet/nnunetv2/utilities/label_handling/single_label_manager.py
    python -m JunctionDetection.PreProcessing.patch_single_label_plans --dataset-id "$NNUNET_LANDMARK_DATASET_ID"
fi

nnUNetv2_train "$NNUNET_LANDMARK_DATASET_ID" 2d "$FOLD" -tr "$TRAINER" --npz
