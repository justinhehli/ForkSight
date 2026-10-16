"""
Label-agnostic (localization-only) counterpart to compute_metrics_junction_detection_nnunet_landmark.py,
for models trained with nnUNetTrainerHeatmapAdaptiveWingFocalSoftSamplingSingleLabel on one of the
single-label dataset copies (JunctionDetection/PreProcessing/create_nnunet_dataset_variants.py) -
"combined-label" or "segprob-only-combined-label" - where every landmark type is merged into one
foreground label. There's no per-class distinction left to evaluate, so this script only computes
spatial matching (precision/recall/F1 of predicted vs. ground-truth junction locations, regardless of
type) - no type accuracy, no per-class breakdown, no confusion matrix.
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import torch

import Environment.env_utils as env_utils
from Evaluation.compute_metrics_junction_detection import (
    _load_gt_annotations,
    _match_predictions_to_gt,
)
from Evaluation.compute_metrics_junction_detection_nnunet_landmark import (
    NNUNET_SEG_STITCHED_SIZE,
    NNUNET_LANDMARK_INPUT_SIZE,
    _preprocess_input,
    _sanitize_case_id,
    _check_init_paths
)
from JunctionDetection.nnUNetLandmark.nnunet_landmark_inference import (
    initialize_nnunet_landmark_predictor,
    nnunet_landmark_predict_from_files,
    get_rescaled_point_predictions_from_model_output,
)

# only trainer currently supported by this dataset/pipeline
_NNUNET_TRAINER = "nnUNetTrainerHeatmapAdaptiveWingFocalSoftSamplingSingleLabel"


def _compute_localization_metrics(
    all_pred_rows: list[dict],
    all_fn_annotations: list[dict],
) -> dict:
    """Label-agnostic localization metrics only - a prediction is a TP if it falls within the
    matching distance threshold of any (unused) GT junction, regardless of landmark type."""
    tp_loc = sum(1 for r in all_pred_rows if r["is_tp"])
    fp_loc = sum(1 for r in all_pred_rows if r["is_fp"])
    fn_loc = len(all_fn_annotations)

    precision_loc = tp_loc / \
        (tp_loc + fp_loc) if (tp_loc + fp_loc) > 0 else 0.0
    recall_loc = tp_loc / (tp_loc + fn_loc) if (tp_loc + fn_loc) > 0 else 0.0
    f1_loc = (2 * precision_loc * recall_loc / (precision_loc + recall_loc)
              if (precision_loc + recall_loc) > 0 else 0.0)

    return {
        "tp_loc": tp_loc, "fp_loc": fp_loc, "fn_loc": fn_loc,
        "precision_loc": precision_loc, "recall_loc": recall_loc, "f1_loc": f1_loc,
    }


def _evaluate_predictions(
    test_tifs_paths: list[Path],
    gt_by_image: dict[str, list[dict]],
    nnunet_landmark_out_dir: Path,
    matching_threshold: float,
    eval_out_dir: Path,
    verbose: bool = True
) -> dict:
    """Match nnU-Net landmark point predictions (single combined label) against GT and compute
    label-agnostic localization metrics."""
    case_id_to_stem = {_sanitize_case_id(
        p.stem): p.stem for p in test_tifs_paths}
    original_shapes = {case_id: (NNUNET_SEG_STITCHED_SIZE, NNUNET_SEG_STITCHED_SIZE)
                       for case_id in case_id_to_stem}

    points_by_case = get_rescaled_point_predictions_from_model_output(
        nnunet_landmark_out_dir, original_shapes, image_resize=NNUNET_LANDMARK_INPUT_SIZE)

    all_pred_rows: list[dict] = []
    all_fn_annotations: list[dict] = []
    pred_csv_rows: list[dict] = []

    for idx, (case_id, stem) in enumerate(case_id_to_stem.items(), start=1):
        gt_annotations = gt_by_image.get(stem)
        if gt_annotations is None:
            raise ValueError(
                f"No GT annotations found for image stem '{stem}' in CSV.")

        # single-label predictions JSON has exactly one key (the combined landmark label) -
        # grab its point list regardless of the label's name
        points_by_label = points_by_case.get(case_id, {})
        pred_coords_list = next(iter(points_by_label.values()), [])
        pred_coords = np.array(
            pred_coords_list) if pred_coords_list else np.empty((0, 2))
        # matching is spatial-only (type-agnostic); this placeholder type is never inspected
        pred_types = ["Landmark"] * len(pred_coords_list)

        pred_rows, fn_annotations = _match_predictions_to_gt(
            pred_coords, pred_types, gt_annotations, matching_threshold)

        all_pred_rows.extend(pred_rows)
        all_fn_annotations.extend(fn_annotations)
        for r in pred_rows:
            pred_csv_rows.append({"image": stem, **r})

        if verbose:
            print(f"evaluated {idx} / {len(case_id_to_stem)} samples")

    pred_df = pd.DataFrame(pred_csv_rows)
    pred_path = eval_out_dir / "predictions_nnunet_landmark_single_label.csv"
    pred_df.to_csv(pred_path, index=False)
    print(f"\nSaved predictions as {pred_path}")

    metrics = _compute_localization_metrics(all_pred_rows, all_fn_annotations)
    metrics_df = pd.DataFrame([metrics])
    metrics_path = eval_out_dir / "metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)
    print(f"Saved metrics as {metrics_path}")

    print(f"\nloc P={metrics['precision_loc']:.3f} "
          f"R={metrics['recall_loc']:.3f} F1={metrics['f1_loc']:.3f} "
          f"(TP={metrics['tp_loc']}, FP={metrics['fp_loc']}, FN={metrics['fn_loc']})")

    return metrics


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,)
    parser.add_argument("--seg-model", type=str, required=True,
                        help="name of the segmentation model used for the evaluation,"
                        "where segmentation predictions with this model were already made")
    parser.add_argument("--dataset", type=str, required=True,
                        help="nnunet dataset ID for the model")
    parser.add_argument("--preprocess", action="store_true",
                        help="enable input image preprocessing (if disabled, we assume these exist already)")
    parser.add_argument("--test-run", action="store_true",
                        help="whether this is just a test run that only preprocesses (and thus predicts) one sample")
    args = parser.parse_args()

    env_utils.load_forksight_env()
    JUNCTION_MATCHING_THRESHOLD = env_utils.load_as(
        "JUNCTION_MATCHING_THRESHOLD", float, 75.0)

    test_tifs_paths, test_labels_csv, seg_pred_dir, eval_out_dir,  \
        nnunet_landmark_model_dir, nnunet_landmark_in_dir, nnunet_landmark_out_dir, num_input_channels = \
        _check_init_paths(args.seg_model, _NNUNET_TRAINER, args.dataset)

    assert torch.cuda.is_available(), "torch CUDA is not available"

    # stitch, resize and copy segmentation probability maps,
    # resize and copy raw TIF images for model input (unless the model was trained
    # with the segmentation probability map as its sole input channel)
    if args.preprocess:
        _preprocess_input(test_tifs_paths, seg_pred_dir,
                          nnunet_landmark_in_dir, args.test_run,
                          single_channel=(num_input_channels == 1))

    # compute heatmap regression predictions with the trained nnUNetTrainerHeatmapAdaptiveWingFocalSoftSamplingSingleLabel model
    nnunet_landmark_predictor = initialize_nnunet_landmark_predictor(
        nnunet_landmark_model_dir, device=torch.device("cuda"))
    nnunet_landmark_predict_from_files(
        nnunet_landmark_predictor, input_dir=nnunet_landmark_in_dir, output_dir=nnunet_landmark_out_dir,
        save_probabilities=True, single_label_channel=True, verbose=True)

    gt_by_image = _load_gt_annotations(test_labels_csv)
    _evaluate_predictions(
        test_tifs_paths, gt_by_image, nnunet_landmark_out_dir,
        JUNCTION_MATCHING_THRESHOLD, eval_out_dir,
    )


if __name__ == "__main__":
    main()
