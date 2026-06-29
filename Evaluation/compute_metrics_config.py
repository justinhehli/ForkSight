"""Central registry of models/runs to include in evaluation scripts.

Both compute_segmentation_metrics.py and compute_junction_detection_metrics.py
import from here so that adding or removing a model only requires editing one
place.
"""

# WandB SAM run names and suffix for artifact to evaluate for segmentation and junction detection
SAM_MODELS_RUNS: list[str] = [
    "sweep-skeleton-recall-lora-BEST",
    "sweep-skeleton-recall-n-blocks-BEST",
    "sweep-clDice-n-blocks-BEST",
    "sweep-cldice-lora-BEST",
    "sweep-skeleton-recall-BEST",
    "sweep-cldice-BEST",

    "SAM_LoRA_HuTopoLoss",
    "SAM_LoRA_BCE_Dice_Only",
    "SAM_LoRA_Ablation_100_100",
    "SAM_LoRA_Ablation_100_0",
    "SAM_LoRA_Ablation_100_200",
    "SAM_LoRA_Ablation_100_400",
    "SAM_LoRA_Ablation_25_75",
    "SAM_LoRA_Ablation_50_50",
    "SAM_LoRA_Ablation_75_25"
]
SAM_PARAMS_ARTIFACT_SUFFIX = "_params_minloss:v0"

# nnUNet evaluations: list of (dataset_name, trainer_class) tuples.
# pre-computed predictions in NNUNET_RESULTS_DIR/<dataset_name>/<trainer_class>__nnUNetPlans__2d/NNUNET_PRED_DIR are evaluated
NNUNET_EVALUATIONS: list[tuple[str, str]] = [
    ("Dataset001_Segmentation_v1", "nnUNetTrainerWandb"),
    ("Dataset001_Segmentation_v1", "nnUNetTrainerClDiceLoss"),
]
