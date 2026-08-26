from datetime import datetime
import json
import os
import argparse
from pathlib import Path
from skimage.transform import resize
import numpy as np
import pandas as pd
import tifffile
import torch

import Environment.env_utils as env_utils
from Evaluation.compute_metrics_junction_detection import (
    _load_gt_annotations,
    _match_predictions_to_gt,
    _compute_metrics,
)
from JunctionDetection.nnUNetLandmark.nnunet_landmark_inference import (
    initialize_nnunet_landmark_predictor,
    nnunet_landmark_predict_from_files,
    get_rescaled_point_predictions_from_model_output,
)
from JunctionDetection.PreProcessing.create_nnunet_heatmap_dataset import (
    NORMAL_FORK_LABEL,
    REVERSED_FORK_CROSSING_COMBINED_LABEL,
)
from Segmentation.PostProcessing.segmentation_postprocessing import stitch_mask_tiles
from Segmentation.Util.patch_grid_util import GRID_SIZE, PATCH_SIZE, load_probability_pred_patches


_JUNCTION_TYPE_3_WAY = "3-way"
_JUNCTION_TYPE_4_WAY = "4-way"

NNUNET_SEG_STITCHED_SIZE = 4096
NNUNET_LANDMARK_INPUT_SIZE = 1024

RAW_CHANNEL_ID = "0000"
SEGPROB_CHANNEL_ID = "0001"
TIF_FILE_ENDING = ".tif"

# hardcoded dataset ID must match NNUNET_LANDMARK_DATASET_ID in
# JunctionDetection/PreProcessing/create_nnunet_heatmap_dataset.py and
# JunctionDetection/nnUNetLandmark/nnunet_landmark_fold_job.sh
NNUNET_LANDMARK_MODEL_BASE_DIR = "/home/jhehli/data/datasets/nnUNet/nnUNet_results"
NNUNET_LANDMARK_MODEL_INPUT_DIR = "/home/jhehli/data/nnUNet_landmark_eval/model_input"
NNUNET_LANDMARK_MODEL_OUTPUT_DIR = "/home/jhehli/data/nnUNet_landmark_eval/model_output/<DATASET><TRAINER>"


def _check_init_paths(seg_model: str, nnunet_trainer: str, dataset_id: str):
    EVALUATION_OUTPUT_DIR = os.getenv("EVALUATION_OUTPUT_DIR")
    JUNCTION_DETECTION_DATASET_DIR = os.getenv(
        "JUNCTION_DETECTION_DATASET_DIR")
    JUNCTION_PRED_DIR = os.getenv("JUNCTION_PRED_DIR")

    if EVALUATION_OUTPUT_DIR is None:
        raise ValueError(
            "EVALUATION_OUTPUT_DIR environment variable must be set.")
    if JUNCTION_DETECTION_DATASET_DIR is None:
        raise ValueError(
            "JUNCTION_DETECTION_DATASET_DIR environment variable must be set.")
    if JUNCTION_PRED_DIR is None:
        raise ValueError("JUNCTION_PRED_DIR environment variable must be set.")

    test_dir = Path(JUNCTION_DETECTION_DATASET_DIR)
    test_tifs_dir = test_dir / "images_tif"
    test_labels_csv = test_dir / "relabeling_data.csv"
    if not test_tifs_dir.is_dir():
        raise FileNotFoundError(
            f"Images directory not found: {test_tifs_dir}")
    if not test_labels_csv.is_file():
        raise FileNotFoundError(f"Annotation CSV not found: {test_labels_csv}")

    test_tifs_paths = sorted(p for p in test_tifs_dir.glob("*.tif"))
    if not test_tifs_paths:
        raise FileNotFoundError(f"No image files found in {test_tifs_dir}")

    seg_pred_dir: Path = Path(JUNCTION_PRED_DIR) / seg_model
    if not seg_pred_dir.is_dir() or not (seg_pred_dir / "metadata.json").exists():
        raise FileNotFoundError(
            f"Segmentation prediction directory (with metadata.json) does not exist: "
            f"{seg_pred_dir}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dataset_prefix = f"Dataset{dataset_id.zfill(3)}_"

    eval_out_dir = Path(EVALUATION_OUTPUT_DIR) / \
        "junction_detection_nnUNetLandmark" / \
        f"{dataset_prefix}{nnunet_trainer}" / timestamp
    eval_out_dir.mkdir(parents=True)

    nnunet_landmark_results_dir = Path(NNUNET_LANDMARK_MODEL_BASE_DIR)
    assert nnunet_landmark_results_dir.is_dir(
    ), f"nnU-Net results directory does not exist ({nnunet_landmark_results_dir})"

    dataset_dir = next((p for p in nnunet_landmark_results_dir.iterdir(
    ) if p.is_dir() and p.name.startswith(dataset_prefix)), None)
    assert dataset_dir is not None and dataset_dir.is_dir(
    ), f"nnU-Net dataset directory for dataset ID {dataset_id} does not exist in ({nnunet_landmark_results_dir})"

    nnunet_landmark_model_dir = dataset_dir / \
        f"{nnunet_trainer}__nnUNetPlans__2d"
    nnunet_landmark_in_dir = Path(NNUNET_LANDMARK_MODEL_INPUT_DIR)
    nnunet_landmark_out_dir = Path(NNUNET_LANDMARK_MODEL_OUTPUT_DIR.replace(
        "<DATASET>", dataset_prefix).replace("<TRAINER>", nnunet_trainer)) / timestamp

    assert nnunet_landmark_model_dir.is_dir() and nnunet_landmark_in_dir.is_dir()
    nnunet_landmark_out_dir.mkdir(parents=True)
    print(f"Model input dir: {nnunet_landmark_in_dir}")
    print(f"Model output dir: {nnunet_landmark_out_dir}")

    return test_tifs_paths, test_labels_csv, seg_pred_dir, eval_out_dir, \
        nnunet_landmark_model_dir, nnunet_landmark_in_dir, nnunet_landmark_out_dir


def _resize_array(arr: np.ndarray, target_size: tuple[int, int], out_dtype: np.dtype) -> np.ndarray:
    return resize(
        arr,
        target_size,
        # bilinear; avoids blocky (order=0) or overshoot (order=3) artifacts
        order=1,
        mode='reflect',
        anti_aliasing=True,
        preserve_range=True,
    ).astype(out_dtype)


def _sanitize_case_id(img_name: str) -> str:
    return img_name.replace("_", "-").replace(" ", "-")


def _stitch_resize_copy_segmentation_prob_map(seg_pred_dir: Path, img_stem: str, nnunet_landmark_input_dir: Path):
    # load segmentation probability map patches and stitch them
    seg_prob_map_patches, _ = load_probability_pred_patches(
        seg_pred_dir, img_stem)
    stitched_seg_prob_map = stitch_mask_tiles(
        seg_prob_map_patches, grid_size=GRID_SIZE,
        original_input_patch_img_size=PATCH_SIZE,
        as_uint=False,
    )

    seg_prob_map_np = stitched_seg_prob_map.squeeze(0).detach().cpu().numpy()
    assert seg_prob_map_np.shape == (
        NNUNET_SEG_STITCHED_SIZE, NNUNET_SEG_STITCHED_SIZE), f"sample {img_stem}: expected squeezed probability map shape ({NNUNET_SEG_STITCHED_SIZE}, {NNUNET_SEG_STITCHED_SIZE}), got {seg_prob_map_np.shape}"

    # resize to the heatmap-regression model's input size (1024x1024)
    # use same resizing as in create_nnunet_heatmap_dataset.py
    seg_prob_map_np = _resize_array(seg_prob_map_np, target_size=(
        NNUNET_LANDMARK_INPUT_SIZE, NNUNET_LANDMARK_INPUT_SIZE), out_dtype=np.float32)

    # save as TIF with sanitized name (no underscores, since "_" separates the nnU-Net channel suffix)
    img_stem_sanitized = _sanitize_case_id(img_stem)
    tifffile.imwrite(
        nnunet_landmark_input_dir / f"{img_stem_sanitized}_{SEGPROB_CHANNEL_ID}{TIF_FILE_ENDING}", seg_prob_map_np)


def _resize_copy_raw_tif(tif_path: Path, nnunet_landmark_input_dir: Path):
    tif_img_np = tifffile.imread(tif_path)

    assert tif_img_np.shape == (
        NNUNET_SEG_STITCHED_SIZE, NNUNET_SEG_STITCHED_SIZE), f"sample {tif_path.stem}: expected raw TIF shape ({NNUNET_SEG_STITCHED_SIZE}, {NNUNET_SEG_STITCHED_SIZE}), got {tif_img_np.shape}"

    # resize to the heatmap-regression model's input size (1024x1024)
    # use same resizing as in create_nnunet_heatmap_dataset.py
    tif_img_np = _resize_array(tif_img_np, target_size=(
        NNUNET_LANDMARK_INPUT_SIZE, NNUNET_LANDMARK_INPUT_SIZE), out_dtype=tif_img_np.dtype)

    # save TIF with sanitized name (no underscores, since "_" separates the nnU-Net channel suffix)
    img_stem_sanitized = _sanitize_case_id(tif_path.stem)
    tifffile.imwrite(
        nnunet_landmark_input_dir / f"{img_stem_sanitized}_{RAW_CHANNEL_ID}{TIF_FILE_ENDING}", tif_img_np)


def _preprocess_input(test_tifs_paths: list[Path], seg_pred_dir: Path, nnunet_landmark_input_dir: Path, is_test_run: bool):
    num_samples = len(test_tifs_paths) if not is_test_run else 1
    for idx, tif_path in enumerate(test_tifs_paths, start=1):
        # load segmentation prediction PROBABILITY MAP patches, stitch them and
        # resize to heatmap regression model input size (1024x1024)
        _stitch_resize_copy_segmentation_prob_map(
            seg_pred_dir, tif_path.stem, nnunet_landmark_input_dir)

        # resize and copy raw TIF images
        _resize_copy_raw_tif(tif_path, nnunet_landmark_input_dir)

        print(f"preprocessed {idx} / {num_samples} samples")
        if is_test_run:
            break


def _evaluate_predictions(
    test_tifs_paths: list[Path],
    gt_by_image: dict[str, list[dict]],
    nnunet_landmark_out_dir: Path,
    matching_threshold: float,
    eval_out_dir: Path,
    verbose: bool = True
) -> dict:
    """Match nnU-Net landmark point predictions against GT and compute aggregate metrics.

    Models are trained with the Reversed Fork / Crossing labels combined (see
    create_nnunet_heatmap_dataset.py), so "Normal Fork" predictions are matched against
    3-way GT junctions, and the combined "Reversed Fork / Crossing" predictions are
    matched against 4-way GT junctions
    """
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

        points_by_label = points_by_case.get(case_id, {})
        points_3way = points_by_label.get(NORMAL_FORK_LABEL, [])
        points_4way = points_by_label.get(
            REVERSED_FORK_CROSSING_COMBINED_LABEL, [])

        pred_coords = (np.array(points_3way + points_4way)
                       if points_3way or points_4way else np.empty((0, 2)))
        pred_types = ([_JUNCTION_TYPE_3_WAY] * len(points_3way)
                      + [_JUNCTION_TYPE_4_WAY] * len(points_4way))

        pred_rows, fn_annotations = _match_predictions_to_gt(
            pred_coords, pred_types, gt_annotations, matching_threshold)

        all_pred_rows.extend(pred_rows)
        all_fn_annotations.extend(fn_annotations)
        for r in pred_rows:
            pred_csv_rows.append({"image": stem, **r})

        if verbose:
            print(f"evaluated {idx} / {len(case_id_to_stem)} samples")

    pred_df = pd.DataFrame(pred_csv_rows)
    pred_path = eval_out_dir / "predictions_nnunet_landmark.csv"
    pred_df.to_csv(pred_path, index=False)
    print(f"\nSaved predictions as {pred_path}")

    metrics = _compute_metrics(all_pred_rows, all_fn_annotations)
    metrics_df = pd.DataFrame([metrics])
    metrics_path = eval_out_dir / "metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)
    print(f"Saved metrics as {metrics_path}")

    print(f"\nloc P={metrics['precision_loc']:.3f} "
          f"R={metrics['recall_loc']:.3f} F1={metrics['f1_loc']:.3f} "
          f"| type acc={metrics['type_accuracy']:.3f}")
    print(f"3-way P={metrics['class_precision_3way']:.3f} "
          f"R={metrics['class_recall_3way']:.3f} "
          f"F1={metrics['class_f1_3way']:.3f} "
          f"(TP={metrics['class_tp_3way']}, "
          f"FP={metrics['class_fp_3way']}, "
          f"FN={metrics['class_fn_3way']})")
    print(f"4-way P={metrics['class_precision_4way']:.3f} "
          f"R={metrics['class_recall_4way']:.3f} "
          f"F1={metrics['class_f1_4way']:.3f} "
          f"(TP={metrics['class_tp_4way']}, "
          f"FP={metrics['class_fp_4way']}, "
          f"FN={metrics['class_fn_4way']})")

    return metrics


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,)
    parser.add_argument("--seg-model", type=str, required=True,
                        help="name of the segmentation model used for the evaluation,"
                        "where segmentation predictions with this model were already made")
    parser.add_argument("--nnunet-trainer", type=str, required=True,
                        help="name of the nU-Net trainer to evaluate")
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

    # nnunet_trainers = ["nnUNetTrainerHeatmapMSE",
    #                   "nnUNetTrainerHeatmapAdaptiveWing",
    #                   "nnUNetTrainerHeatmapAdaptiveWingFocal",
    #                   "nnUNetTrainerHeatmapAdaptiveWingSoftSampling",
    #                   "nnUNetTrainerHeatmapAdaptiveWingFocalSoftSampling"]
    # assert args.nnunet_trainer in nnunet_trainers, f"nnU-Net trainer must be in {nnunet_trainers}"

    test_tifs_paths, test_labels_csv, seg_pred_dir, eval_out_dir,  \
        nnunet_landmark_model_dir, nnunet_landmark_in_dir, nnunet_landmark_out_dir = \
        _check_init_paths(args.seg_model, args.nnunet_trainer, args.dataset)

    assert torch.cuda.is_available(), "torch CUDA is not available"

    # stitch, resize and copy segmentation probability maps,
    # resize and copy raw TIF images for model input
    if args.preprocess:
        _preprocess_input(test_tifs_paths, seg_pred_dir,
                          nnunet_landmark_in_dir, args.test_run)

    # compute heatmap regression predictions with the trained nnUNetTrainerHeatmapMSE model
    nnunet_landmark_predictor = initialize_nnunet_landmark_predictor(
        nnunet_landmark_model_dir, device=torch.device("cuda"))
    nnunet_landmark_predict_from_files(
        nnunet_landmark_predictor, input_dir=nnunet_landmark_in_dir, output_dir=nnunet_landmark_out_dir,
        save_probabilities=True, verbose=True)

    # evaluate predictions against ground truth
    gt_by_image = _load_gt_annotations(test_labels_csv)
    _evaluate_predictions(
        test_tifs_paths, gt_by_image, nnunet_landmark_out_dir,
        JUNCTION_MATCHING_THRESHOLD, eval_out_dir,
    )


if __name__ == "__main__":
    main()
