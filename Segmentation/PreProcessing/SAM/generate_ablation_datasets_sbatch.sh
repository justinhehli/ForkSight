#!/bin/bash -l
#SBATCH --job-name=generate_ablation_datasets
#SBATCH --partition=standard
#SBATCH --cpus-per-task=16
#SBATCH --mem=20G
#SBATCH --time=05:00:00
#SBATCH --output=/home/jhehli/scratch/logs/%x-%j.out
#SBATCH --error=/home/jhehli/scratch/logs/%x-%j.err

set -euo pipefail

# navigate to the repository root
cd /home/jhehli/data/ForkSight
echo "Current working directory: $(pwd)"

# load and export environment variables
set -a
source ./Environment/.env
set +a

# activate virtual environment
source "$SAM_LORA_VENV/bin/activate"

python -u -m Segmentation.PreProcessing.SAM.generate_ablation_datasets