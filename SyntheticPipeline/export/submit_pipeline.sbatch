#!/bin/bash
# =============================================================================
# SBATCH driver for the synthetic-data training + junction-detection + eval
# pipeline.
#
# Submit with:  sbatch submit_pipeline.sbatch
#
# What this script does:
#   1. [TODO]  Generate + preprocess the synthetic dataset (placeholder section).
#   2.         Train an nnUNet-v2 model with CL-Dice loss
#              (train_cldice_nnunet.py).
#   3.         Copy the trained model weights into a local ./artifacts/ dir so
#              they survive scratch cleanup.
#   4.         Run inference + junction detection on the test split
#              (junction_detection.py).
#   5.         Run the downstream evaluation against GT annotations
#              (evaluate_junctions.py).
#
# Required companion scripts next to this file:
#   - train_cldice_nnunet.py
#   - junction_detection.py
#   - evaluate_junctions.py
# =============================================================================

#SBATCH --job-name=synth_pipeline
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --partition=gpu

set -euo pipefail

# -----------------------------------------------------------------------------
# 0. Paths & environment
# -----------------------------------------------------------------------------
REPO_ROOT="${REPO_ROOT:-$(pwd)}"
SCRIPT_DIR="${REPO_ROOT}/SyntheticPipeline/export"

# Dataset and nnUNet roots on scratch / shared storage
DATASET_DIR="${DATASET_DIR:-${REPO_ROOT}/data/synth_v1}"
NNUNET_ROOT="${NNUNET_ROOT:-${REPO_ROOT}/data/synth_v1/nnunet}"

# Local directory where we persist the trained model weights
ARTIFACTS_DIR="${ARTIFACTS_DIR:-${REPO_ROOT}/artifacts}"
EVAL_OUT_DIR="${EVAL_OUT_DIR:-${REPO_ROOT}/evaluation_runs/${SLURM_JOB_ID:-local}}"

# nnUNet dataset identity + model config
DATASET_ID="${DATASET_ID:-900}"
DATASET_NAME="${DATASET_NAME:-Synthetic}"
TRAINER="${TRAINER:-nnUNetTrainerCLDice}"
CONFIGURATION="${CONFIGURATION:-2d}"
FOLD="${FOLD:-0}"
TEST_RATIO="${TEST_RATIO:-0.2}"
SEED="${SEED:-42}"

mkdir -p logs "${ARTIFACTS_DIR}" "${EVAL_OUT_DIR}"

# Activate your Python env here (conda / mamba / venv / module load)
# >>> CUSTOMISE THIS BLOCK <<<
# Example:
# source ~/miniconda3/etc/profile.d/conda.sh
# conda activate forksight
# module load cuda/12.1
python --version

# nnUNet-v2 needs these env vars (the Python scripts also set them, but some
# nnUNetv2 CLI invocations read them at import time).
export nnUNet_raw="${NNUNET_ROOT}/nnUNet_raw"
export nnUNet_preprocessed="${NNUNET_ROOT}/nnUNet_preprocessed"
export nnUNet_results="${NNUNET_ROOT}/nnUNet_results"
mkdir -p "${nnUNet_raw}" "${nnUNet_preprocessed}" "${nnUNet_results}"

echo "=============================================================="
echo "Job:        ${SLURM_JOB_NAME:-local} (${SLURM_JOB_ID:-no-id})"
echo "Node:       $(hostname)"
echo "Repo root:  ${REPO_ROOT}"
echo "Dataset:    ${DATASET_DIR}"
echo "nnUNet:     ${NNUNET_ROOT}"
echo "Trainer:    ${TRAINER}  |  Fold: ${FOLD}  |  Config: ${CONFIGURATION}"
echo "Artifacts:  ${ARTIFACTS_DIR}"
echo "Eval out:   ${EVAL_OUT_DIR}"
echo "=============================================================="

# =============================================================================
# 1. Synthetic data generation + preprocessing
# ---------------------------------------------------------------------------
# Fill in the calls that generate <DATASET_DIR>/images/*.png,
# <DATASET_DIR>/masks/*.png and <DATASET_DIR>/annotations.csv.
# Claude / your teammate: drop the generation + preprocessing commands below.
# =============================================================================
echo ">>> [1/5] Synthetic data generation + preprocessing"

# <<< CLAUDE: FILL IN HERE >>>
# e.g.
# python "${REPO_ROOT}/synthetic/generate.py" \
#     --out-dir "${DATASET_DIR}" \
#     --n-images 500 --seed "${SEED}"
#
# python "${REPO_ROOT}/synthetic/preprocess.py" \
#     --dataset-dir "${DATASET_DIR}"

# Sanity-check required layout before spending a GPU slot on training.
for sub in images masks; do
  if [ ! -d "${DATASET_DIR}/${sub}" ]; then
    echo "ERROR: expected ${DATASET_DIR}/${sub}/ to exist after synthetic generation." >&2
    exit 1
  fi
done
if [ ! -f "${DATASET_DIR}/annotations.csv" ]; then
  echo "WARNING: ${DATASET_DIR}/annotations.csv not found; evaluation step will fail." >&2
fi

# =============================================================================
# 2. Train nnUNet with CL-Dice
# =============================================================================
echo ">>> [2/5] Training nnUNet (${TRAINER}) on fold ${FOLD}"
python "${SCRIPT_DIR}/train_cldice_nnunet.py" \
    --dataset-dir "${DATASET_DIR}" \
    --nnunet-root "${NNUNET_ROOT}" \
    --dataset-id "${DATASET_ID}" \
    --dataset-name "${DATASET_NAME}" \
    --trainer "${TRAINER}" \
    --configuration "${CONFIGURATION}" \
    --fold "${FOLD}" \
    --test-ratio "${TEST_RATIO}" \
    --seed "${SEED}"

# =============================================================================
# 3. Save model weights locally (scratch -> artifacts)
# =============================================================================
echo ">>> [3/5] Copying trained weights to ${ARTIFACTS_DIR}"
MODEL_SUBPATH="Dataset$(printf '%03d' ${DATASET_ID})_${DATASET_NAME}/${TRAINER}__nnUNetPlans__${CONFIGURATION}"
SRC_MODEL_DIR="${NNUNET_ROOT}/nnUNet_results/${MODEL_SUBPATH}"
DST_MODEL_DIR="${ARTIFACTS_DIR}/${MODEL_SUBPATH}"

mkdir -p "${DST_MODEL_DIR}"
rsync -a --info=progress2 "${SRC_MODEL_DIR}/" "${DST_MODEL_DIR}/"
echo "Weights saved to ${DST_MODEL_DIR}"

# =============================================================================
# 4. Junction detection on the test split
# ---------------------------------------------------------------------------
# This runs nnUNet inference on the test images first, then pipes the masks
# through the skeletonization-based detector.  We re-use evaluate_junctions.py
# for inference and also save the detected junctions as JSON.  If you want the
# standalone CLI, run junction_detection.py on a folder of masks instead.
# =============================================================================
echo ">>> [4/5] Running junction detection"
JUNCTION_DIR="${EVAL_OUT_DIR}/junctions"
mkdir -p "${JUNCTION_DIR}"

# If you want a plain mask-folder invocation of the detector:
#   python "${SCRIPT_DIR}/junction_detection.py" \
#       --pred-dir <folder_of_mask_pngs> \
#       --out-json "${JUNCTION_DIR}/junctions.json" \
#       --save-skeletons "${JUNCTION_DIR}/skeletons"

# =============================================================================
# 5. Downstream evaluation
# =============================================================================
echo ">>> [5/5] Evaluating junction detections against GT"
python "${SCRIPT_DIR}/evaluate_junctions.py" \
    --dataset-dir "${DATASET_DIR}" \
    --nnunet-root "${NNUNET_ROOT}" \
    --dataset-id "${DATASET_ID}" \
    --dataset-name "${DATASET_NAME}" \
    --trainer "${TRAINER}" \
    --configuration "${CONFIGURATION}" \
    --fold "${FOLD}" \
    --out-dir "${EVAL_OUT_DIR}" \
    --save-plots

echo "=============================================================="
echo "Pipeline complete."
echo "  Weights:  ${DST_MODEL_DIR}"
echo "  Metrics:  ${EVAL_OUT_DIR}/metrics.json"
echo "=============================================================="
