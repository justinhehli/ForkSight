#!/bin/bash -l
#SBATCH --job-name=nnunet-landmark-eval
#SBATCH --partition=standard
#SBATCH --gres=gpu:1
#SBATCH --gpus=1 --constraint="GPUMEM80GB|GPUMEM96GB|GPUMEM140GB"
#SBATCH --cpus-per-task=16
#SBATCH --mem=40G
#SBATCH --time=06:00:00
#SBATCH --output=/scratch/jhehli/logs/%x-%j.out
#SBATCH --error=/scratch/jhehli/logs/%x-%j.err

# nnU-Net junction-landmark (heatmap regression) evaluation job: preprocesses (optionally) the
# segmentation predictions + raw images, runs heatmap-regression inference, and matches the
# resulting point predictions against GT to compute junction detection metrics
# (Evaluation/compute_metrics_junction_detection_nnunet_landmark.py).
#
# Usage (from repo root):
#   sbatch JunctionDetection/nnUNetLandmark/nnunet_landmark_eval_job.sh --seg-model <seg_model_name> \
#       [--trainer <nnUNetTrainerHeatmapMSE|nnUNetTrainerHeatmapAdaptiveWing>] [--preprocess] [--test-run]
#
# Arguments (forwarded to compute_metrics_junction_detection_nnunet_landmark.py):
#   --seg-model  <name>  — name of the segmentation model with existing predictions under
#                          JUNCTION_PRED_DIR (required)
#   --trainer    <name>  — nnU-Net trainer to evaluate (default: nnUNetTrainerHeatmapMSE)
#   --preprocess         — (re)build the model input tifs before inference
#   --test-run           — only preprocess/predict/evaluate a single sample

set -euo pipefail

SEG_MODEL=""
TRAINER="nnUNetTrainerHeatmapMSE"
PREPROCESS=0
TEST_RUN=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --seg-model)
            SEG_MODEL="$2"
            shift 2
            ;;
        --trainer)
            TRAINER="$2"
            shift 2
            ;;
        --preprocess)
            PREPROCESS=1
            shift
            ;;
        --test-run)
            TEST_RUN=1
            shift
            ;;
        *)
            echo "Error: unknown argument '$1'"
            exit 1
            ;;
    esac
done

if [[ -z "$SEG_MODEL" ]]; then
    echo "Error: --seg-model is required"
    exit 1
fi

# export nnUNet dataset paths (nnUNet expects these env vars to be set)
export nnUNet_raw="/home/jhehli/data/datasets/nnUNet/nnUNet_raw"
export nnUNet_preprocessed="/home/jhehli/data/datasets/nnUNet/nnUNet_preprocessed"
export nnUNet_results="/home/jhehli/data/datasets/nnUNet/nnUNet_results"

export PYTHONUNBUFFERED=1

REPO_ROOT="/home/jhehli/data/ForkSight"
cd "$REPO_ROOT"

echo "CWD: $(pwd)"
echo "SEG_MODEL: ${SEG_MODEL}"
echo "TRAINER: ${TRAINER}"
echo "PREPROCESS: ${PREPROCESS}"
echo "TEST_RUN: ${TEST_RUN}"
echo "Job ${SLURM_JOB_ID} on $(hostname)"

# load shared environment variables (EVALUATION_OUTPUT_DIR, JUNCTION_DETECTION_DATASET_DIR,
# JUNCTION_PRED_DIR, JUNCTION_MATCHING_THRESHOLD, etc.)
set -a
source "${REPO_ROOT}/Environment/.env"
set +a

# activate nnUNet virtual environment
source ~/.nnUNet_env/bin/activate

mkdir -p "/scratch/jhehli/logs"

PY_ARGS=(--seg-model "$SEG_MODEL" --nnunet-trainer "$TRAINER")
[[ "$PREPROCESS" == "1" ]] && PY_ARGS+=(--preprocess)
[[ "$TEST_RUN" == "1" ]] && PY_ARGS+=(--test-run)

srun python -u -m Evaluation.compute_metrics_junction_detection_nnunet_landmark "${PY_ARGS[@]}"
