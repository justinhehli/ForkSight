"""
Inference utilities for the nnU-Net junction-landmark heatmap-regression model
(nnUNetTrainerHeatmapMSE)

reads back the per-case point predictions written by
nnunetv2.inference.heatmap_export.export_heatmap_prediction_from_logits, 
which finds any number (zero, one, or many) of local heatmap maxima per label.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Sequence

import numpy as np
import torch

from nnunetv2.inference.data_iterators import preprocessing_iterator_fromfiles
from nnunetv2.inference.heatmap_export import export_heatmap_prediction_from_logits
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

from Segmentation.Util.nnunet_util import initialize_nnunet_predictor

NNUNET_LANDMARK_DEFAULT_FOLDS = (0, 1, 2, 3, 4)
NNUNET_LANDMARK_DEFAULT_CHECKPOINT = "checkpoint_final.pth"
NNUNET_LANDMARK_DEFAULT_THRESHOLD = 0.5
NNUNET_LANDMARK_DEFAULT_MIN_DISTANCE = 3

RAW_CHANNEL_SUFFIX = "_0000.tif"


def initialize_nnunet_landmark_predictor(
    model_dir: Path,
    device: torch.device,
    folds: Sequence[int] = NNUNET_LANDMARK_DEFAULT_FOLDS,
    checkpoint: str = NNUNET_LANDMARK_DEFAULT_CHECKPOINT,
) -> nnUNetPredictor:
    predictor, _ = initialize_nnunet_predictor(
        model_dir, device=device, folds=folds, checkpoint=checkpoint, ensure_custom_trainer_file=True
    )
    return predictor


def nnunet_landmark_predict_from_files(
    predictor: nnUNetPredictor,
    input_dir: Path,
    output_dir: Path,
    save_probabilities: bool = False,
    num_processes_preprocessing: int = 2,
    threshold: float = NNUNET_LANDMARK_DEFAULT_THRESHOLD,
    min_distance: int = NNUNET_LANDMARK_DEFAULT_MIN_DISTANCE,
    verbose: bool = False
) -> None:
    """
    Runs heatmap-regression inference on every case in input_dir (named `<case>_0000.tif`,
    `<case>_0001.tif`, ... matching dataset.json's channel_names, as written by
    JunctionDetection/PreProcessing/create_nnunet_heatmap_dataset.py), and writes, per case:
      - `<case>.json`: detected point coordinates (in the model's input pixel space) with
        confidence scores, per label - see nnunetv2.inference.heatmap_export.export_heatmap_prediction_from_logits
      - `<case>.npz` (if save_probabilities): the full reverted heatmap probability map

    This replaces nnU-Net's default predict_from_files, which is segmentation-shaped (argmax) and
    not applicable to heatmap regression - it reuses the same preprocessing iterator internally, but
    exports through export_heatmap_prediction_from_logits instead.
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    case_ids = sorted({
        p.name[: -len(RAW_CHANNEL_SUFFIX)]
        for p in input_dir.glob(f"*{RAW_CHANNEL_SUFFIX}")
    })
    if not case_ids:
        raise FileNotFoundError(
            f"No *{RAW_CHANNEL_SUFFIX} files found in {input_dir}")

    list_of_lists = [
        sorted(str(p) for p in input_dir.glob(f"{case_id}_*.tif"))
        for case_id in case_ids
    ]
    output_filenames_truncated = [
        str(output_dir / case_id) for case_id in case_ids]

    iterator = preprocessing_iterator_fromfiles(
        list_of_lists, None, output_filenames_truncated,
        predictor.plans_manager, predictor.dataset_json, predictor.configuration_manager,
        num_processes_preprocessing, predictor.device.type == "cuda", predictor.verbose_preprocessing,
    )

    for idx, preprocessed in enumerate(iterator, start=1):
        data = preprocessed["data"]
        if isinstance(data, str):
            delfile = data
            data = torch.from_numpy(np.load(data))
            os.remove(delfile)

        properties = preprocessed["data_properties"]
        ofile = preprocessed["ofile"]

        with torch.no_grad():
            prediction = predictor.predict_logits_from_preprocessed_data(
                data).cpu().numpy()

        export_heatmap_prediction_from_logits(
            prediction, properties, predictor.configuration_manager, predictor.plans_manager,
            predictor.dataset_json, ofile, save_probabilities=save_probabilities,
            threshold=threshold, min_distance=min_distance,
        )

        if verbose:
            print(f"predicted {idx} / {len(case_ids)} samples")


def get_rescaled_point_predictions_from_model_output(
    predictions_dir: Path,
    original_shapes: dict[str, tuple[int, int]],
    image_resize: int,
) -> dict[str, dict[str, list[list[float]]]]:
    """
    Reads back the per-case point-prediction JSONs written by nnunet_landmark_predict_from_files
    (via export_heatmap_prediction_from_logits) and rescales coordinates from the
    (image_resize x image_resize) model input back to each image's original pixel space.

    original_shapes: {case_id: (orig_height, orig_width)} - e.g. as returned by
        create_nnunet_heatmap_dataset.resize_copy_input, or computed directly from the source TIFs.

    Returns {case_id: {label_name: [[x, y], ...], ...}, ...}. A label with no detections in an
    image is simply absent/empty for that image
    """
    predictions_dir = Path(predictions_dir)
    result: dict[str, dict[str, list[list[float]]]] = {}

    for json_path in sorted(predictions_dir.glob("*.json")):
        case_id = json_path.stem
        if case_id not in original_shapes:
            continue
        orig_h, orig_w = original_shapes[case_id]
        scale_y = orig_h / image_resize
        scale_x = orig_w / image_resize

        points_by_label = json.loads(json_path.read_text())
        result[case_id] = {
            label_name: [[point["x"] * scale_x, point["y"] * scale_y]
                         for point in points]
            for label_name, points in points_by_label.items()
        }

    return result
