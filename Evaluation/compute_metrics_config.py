"""Central registry of models/runs to include in evaluation scripts.

Both compute_segmentation_metrics.py and compute_junction_detection_metrics.py
import from here so that adding or removing a model only requires editing one
place.
"""

# WandB SAM run names and suffix for artifact to evaluate for segmentation and junction detection
SAM_PARAMS_ARTIFACT_SUFFIX = "_params_minloss:v0"
SAM_MODELS_RUNS: list[str] = []

# nnUNet evaluations: list of (dataset_name, trainer_class) tuples.
# pre-computed predictions in NNUNET_RESULTS_DIR/<dataset_name>/<trainer_class>__nnUNetPlans__2d/NNUNET_PRED_DIR are evaluated
NNUNET_EVALUATIONS: list[tuple[str, str]] = [
    #("Dataset001_Segmentation_v1", "nnUNetTrainerWandb"),
    #("Dataset001_Segmentation_v1", "nnUNetTrainerClDiceLoss"),
    ("Dataset021_SegmentationSynthetic", "nnUNetTrainerClDiceLoss"),
]
