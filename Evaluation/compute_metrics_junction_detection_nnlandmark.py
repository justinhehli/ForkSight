from datetime import datetime
import json
import os
import argparse
from pathlib import Path
from skimage.transform import resize
import numpy as np
import tifffile
import torch

import Environment.env_utils as env_utils
from Evaluation.compute_metrics_junction_detection import _load_gt_annotations
from JunctionDetection.nnLandmark.nnlandmark_inference import initialize_nnlandmark_predictor, nnlandmark_predict_from_files
from Segmentation.PostProcessing.segmentation_postprocessing import postprocess_segmentation_masks, stitch_mask_tiles
from Segmentation.Util.patch_grid_util import GRID_SIZE, PATCH_SIZE, load_probability_pred_patches


_JUNCTION_TYPE_3_WAY = "3-way"
_JUNCTION_TYPE_4_WAY = "4-way"

NNUNET_SEG_STITCHED_SIZE = 4096
NNLM_INPUT_SIZE = 1024

RAW_CHANNEL_ID = "0000"
SEGPROB_CHANNEL_ID = "0001"
TIF_FILE_ENDING = ".tif"

NNLM_MODEL_DIR = "/home/jhehli/data/datasets/nnLandmark/nnLM_results/Dataset001_JunctionDetection_v1_nnLandmark/nnLandmark__nnUNetPlans__2d"
NNLM_MODEL_INPUT_DIR = "/home/jhehli/data/nnLM_eval/model_input"
NNLM_MODEL_OUTPUT_DIR = "/home/jhehli/data/nnLM_eval/model_output"


def _check_init_paths(seg_model: str, do_plot: bool):
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
    eval_out_dir = Path(EVALUATION_OUTPUT_DIR) / \
        "junction_detection_nnLM" / timestamp
    eval_out_dir.mkdir(parents=True)
    eval_out_plt_dir = eval_out_dir / "plots" if do_plot else None
    eval_out_plt_dir.mkdir()

    return test_tifs_paths, test_labels_csv, seg_pred_dir, eval_out_dir


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


def _stitch_resize_copy_segmentation_prob_map(seg_pred_dir: Path, img_stem: str, nnLM_input_dir: Path):
    # load segmentation probability map patches and stitch them
    seg_prob_map_patches, _ = load_probability_pred_patches(
        seg_pred_dir, img_stem)
    stitched_seg_prob_map, _ = stitch_mask_tiles(
        seg_prob_map_patches, grid_size=GRID_SIZE,
        original_input_patch_img_size=PATCH_SIZE,
        as_uint=False,
    )

    seg_prob_map_np = stitched_seg_prob_map.squeeze(0).detach().cpu().numpy()
    assert seg_prob_map_np.shape == (
        NNUNET_SEG_STITCHED_SIZE, NNUNET_SEG_STITCHED_SIZE), f"sample {img_stem}: expected squeezed probability map shape ({NNUNET_SEG_STITCHED_SIZE}, {NNUNET_SEG_STITCHED_SIZE}), got {seg_prob_map_np.shape}"

    # resize to nnLandmark input size (1024x1024)
    # use same resizing as in create_nnlandmark_heatmap_dataset.py
    seg_prob_map_np = _resize_array(seg_prob_map_np, target_size=(
        NNLM_INPUT_SIZE, NNLM_INPUT_SIZE), out_dtype=np.float32)

    # save as TIF with sanitized name (no underscores because of nnLandmark channel suffix)
    img_stem_sanitized = _sanitize_case_id(img_stem)
    tifffile.imwrite(
        nnLM_input_dir / f"{img_stem_sanitized}_{SEGPROB_CHANNEL_ID}{TIF_FILE_ENDING}", seg_prob_map_np)


def _resize_copy_raw_tif(tif_path: Path, nnLM_input_dir: Path):
    tif_img_np = tifffile.imread(tif_path)

    assert tif_img_np.shape == (
        NNUNET_SEG_STITCHED_SIZE, NNUNET_SEG_STITCHED_SIZE), f"sample {tif_path.stem}: expected raw TIF shape ({NNUNET_SEG_STITCHED_SIZE}, {NNUNET_SEG_STITCHED_SIZE}), got {tif_img_np.shape}"

    # resize to nnLandmark input size (1024x1024)
    # use same resizing as in create_nnlandmark_heatmap_dataset.py
    tif_img_np = _resize_array(tif_img_np, target_size=(
        NNLM_INPUT_SIZE, NNLM_INPUT_SIZE), out_dtype=tif_img_np.dtype)

    # save TIF with sanitized name (no underscores because of nnLandmark channel suffix)
    img_stem_sanitized = _sanitize_case_id(tif_path.stem)
    tifffile.imwrite(
        nnLM_input_dir / f"{img_stem_sanitized}_{RAW_CHANNEL_ID}{TIF_FILE_ENDING}", tif_img_np)


def _preprocess_input(test_tifs_paths: list[Path], seg_pred_dir: Path, nnLM_input_dir: Path, is_test_run: bool):
    num_samples = len(test_tifs_paths) if not is_test_run else 1
    for idx, tif_path in enumerate(test_tifs_paths, start=1):
        # load segmentation prediction PROBABILITY MAP patches, stitch them and
        # resize to heatmap regression model input size (1024x1024)
        _stitch_resize_copy_segmentation_prob_map(
            seg_pred_dir, tif_path.stem, nnLM_input_dir)

        # resize and copy raw TIF images
        _resize_copy_raw_tif(tif_path, nnLM_input_dir)

        print(f"preprocessed {idx} / {num_samples} samples")
        if is_test_run:
            break


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,)
    parser.add_argument("--seg-model", type=str, required=True,
                        help="name of the segmentation model used for the evaluation,"
                        "where segmentation predictions with this model were already made")
    parser.add_argument("--preprocess", action="store_true",
                        help="enable input image preprocessing (if disabled, we assume these exist already)")
    parser.add_argument("--test-run", action="store_true",
                        help="whether this is just a test run that only preprocesses (and thus predicts) one sample")
    args = parser.parse_args()

    env_utils.load_forksight_env()
    JUNCTION_MATCHING_THRESHOLD = env_utils.load_as(
        "JUNCTION_MATCHING_THRESHOLD", float, 75.0)

    nnLM_model_dir = Path(NNLM_MODEL_DIR)
    nnLM_in_dir = Path(NNLM_MODEL_INPUT_DIR)
    nnLM_out_dir = Path(NNLM_MODEL_INPUT_DIR)
    assert nnLM_model_dir.is_dir() and nnLM_in_dir.is_dir() and nnLM_out_dir.is_dir()

    assert torch.cuda.is_available(), "torch CUDA is not available"

    test_tifs_paths, test_labels_csv, seg_pred_dir = _check_init_paths(
        args.seg_model, args.plot)
    gt_by_image = _load_gt_annotations(test_labels_csv)

    # stitch, resize and copy segmentation probability maps,
    # resize and copy raw TIF images for model input
    if args.preprocess:
        _preprocess_input(test_tifs_paths, seg_pred_dir,
                          nnLM_in_dir, args.test_run)

    # compute heatmap regression predictions with trained nnLandmark model
    nnLM_predictor = initialize_nnlandmark_predictor(
        nnLM_model_dir, device=torch.device("cuda"))
    nnlandmark_predict_from_files(
        nnLM_predictor, input_dir=nnLM_in_dir, output_dir=nnLM_out_dir, save_probabilities=True)


if __name__ == "__main__":
    main()
