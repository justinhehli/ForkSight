from __future__ import annotations
from pathlib import Path
from typing import Sequence, TYPE_CHECKING
import numpy as np
import torch
from PIL import Image

# assumes nnlandmark is installed
from nnlandmark.inference.nnLandmark.predict_from_raw_data import nnUNetPredictor


NNLM_DEFAULT_FOLDS = (0, 1, 2, 3, 4)
NNLM_DEFAULT_CHECKPOINT = "checkpoint_best.pth"

import torch
from pathlib import Path


def initialize_nnlandmark_predictor(
    model_dir: Path,
    device: torch.device,
    folds: Sequence[int] = NNLM_DEFAULT_FOLDS,
    checkpoint: str = NNLM_DEFAULT_CHECKPOINT,
) -> "nnUNetPredictor":
    predictor = nnUNetPredictor(
        tile_step_size=0.5,
        use_gaussian=True,
        use_mirroring=True,
        perform_everything_on_device=True,
        device=device,
        verbose=False,
        verbose_preprocessing=False,
        allow_tqdm=False,
    )

    predictor.initialize_from_trained_model_folder(
        str(model_dir),
        use_folds=tuple(folds),
        checkpoint_name=checkpoint,
    )

    return predictor


def nnlandmark_predict_from_files(
    predictor,
    input_dir: Path,
    output_dir: Path,
    save_probabilities: bool = False,
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    predictor.predict_from_files(
        str(input_dir),
        str(output_dir),
        save_probabilities=save_probabilities,
        overwrite=True,
        num_processes_preprocessing=3,
        num_processes_segmentation_export=3,
    )


def get_point_predictions_from_heatmap(reversed_crossing_combinded: bool) -> dict[str, np.ndarray]:
    """
    Compute point predictions (coordinates) from an nnLandmark heatmap regresssion prediction
    reversed_crossing_combinded: whether reversed forks and crossings were combined into one label during training
    Returns a dictionary like {label_A: [[x_A_1, y_A_1], ... ], label_B: [...]}
    """
    pass
