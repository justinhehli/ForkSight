"""
Build an nnU-Net raw dataset for junction (landmark) heatmap-regression detection, to be trained
with the nnUNetTrainerHeatmapMSE trainer (nnUNet/nnunetv2/training/nnUNetTrainer/variants/heatmap/),
which supports zero, one, or many landmarks per label per image - unlike nnLandmark
(https://github.com/MIC-DKFZ/nnLandmark), which hard-codes exactly one per label.

IMPORTANT: the labels this script writes to labelsTr/ are NOT the multi-channel Gaussian heatmaps.
They are single-channel discrete label maps - a small square blob, painted with its class id, at
each annotated landmark's coordinate (the exact same on-disk format nnLandmark itself used). This
is intentional, not a stand-in that still needs finishing:
  - Storing discrete blobs (instead of precomputed float heatmaps) lets this dataset flow through
    nnU-Net's normal cropping/resampling/foreground-oversampling pipeline unmodified - all of that
    assumes discrete integer labels.
  - The actual multi-channel Gaussian heatmap targets are built ON THE FLY, per training batch,
    by ConvertSegToMultiChannelHeatmap
    (nnUNet/nnunetv2/training/data_augmentation/custom_transforms/heatmap_regression.py), which is
    appended as the last step of nnUNetTrainerHeatmapMSE.get_training_transforms /
    get_validation_transforms. That transform is what turns these blobs into
    (num_classes + 1, H, W) float heatmaps (channel 0 = combined, channels 1..N = per class) -
    nothing in this file or in labelsTr/ ever looks like a heatmap.
  - Multiple same-class instances are already representable in this on-disk format (one blob per
    annotations.csv row); the "one landmark per label" limitation nnLandmark had lived entirely in
    its trainer-side target construction, not in this dataset format.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile
from skimage.transform import resize

ANNOTATIONS_FILE_NAME = "annotations.csv"
IMAGES_SUBDIR_NAME = "images"
SEGMENTATION_PROBABILITIES_SUBDIR = "segmentation_probabilities"

NNUNET_DATASET_PREFIX = "DatasetXXX_"

# Hardcoded nnU-Net dataset ID for the junction-landmark heatmap-regression task (kept fixed, rather
# than configurable, so this script, JunctionDetection/nnUNetLandmark/nnunet_landmark_fold_job.sh,
# and Evaluation/compute_metrics_junction_detection_nnlandmark.py always agree on the same dataset).
NNUNET_LANDMARK_DATASET_ID = 11

NORMAL_FORK_LABEL = "Normal Fork"
REVERSED_FORK_LABEL = "Reversed Fork"
CROSSING_LABEL = "Crossing"
NEGATIVE_LABEL = "Negative"
REVERSED_FORK_CROSSING_COMBINED_LABEL = "Reversed Fork / Crossing"

DEFAULT_SEED = 42

NNUNET_IMAGES_DIR = "imagesTr"
NNUNET_LABELS_DIR = "labelsTr"

RAW_CHANNEL_ID = "0000"
SEGPROB_CHANNEL_ID = "0001"
FILE_ENDING = ".tif"

COORD_X_COL = "x"
COORD_Y_COL = "y"
IMAGE_COL = "image"
LABEL_COL = "label"

# Half-size of the small discrete blob drawn around each landmark in the label map (half=1 -> 3x3),
# mirroring nnLandmark's "3x3x3 cube" convention for 3D. This is NOT the Gaussian heatmap's sigma -
# it's a safety margin against the label being dropped by spatial augmentation (rotation/scaling/
# low-res simulation) before ConvertSegToMultiChannelHeatmap ever sees it. Measured empirically by
# running single labeled pixels vs. 3x3 blobs through nnUNetTrainer's actual training transform
# pipeline (300 trials each, full rotation range): a lone pixel survived only ~95% of the time
# (interpolation/resampling artifacts silently erase it the rest of the time), a 3x3 blob survived
# 100%. Also large enough that connected-component labeling (run later, on the fly, by
# ConvertSegToMultiChannelHeatmap) reliably finds one component per landmark instance.
LANDMARK_HALF_SIZE = 1


def validate_input_initialize_output(input_dir: Path, dataset_id: int) -> Path:
    if not input_dir.is_dir():
        raise ValueError("input directory does not exist")
    if not (input_dir / ANNOTATIONS_FILE_NAME).is_file():
        raise ValueError("input directory doesn't contain annotations.csv")
    if not (input_dir / IMAGES_SUBDIR_NAME).is_dir() or not (input_dir / SEGMENTATION_PROBABILITIES_SUBDIR).is_dir():
        raise ValueError(
            f"input directory doesn't contain the {IMAGES_SUBDIR_NAME} and {SEGMENTATION_PROBABILITIES_SUBDIR} sub-directories")

    output_dir = input_dir.parent / (NNUNET_DATASET_PREFIX.replace(
        "XXX", f"{dataset_id:03d}") + input_dir.name)

    output_dir.mkdir()
    (output_dir / NNUNET_IMAGES_DIR).mkdir()
    (output_dir / NNUNET_LABELS_DIR).mkdir()

    return output_dir


def combine_reversed_and_crossing(df: pd.DataFrame, seed: int) -> pd.DataFrame:
    reversed_crossing_mask = df[LABEL_COL].isin(
        [REVERSED_FORK_LABEL, CROSSING_LABEL])
    reversed_crossing_rows = df[reversed_crossing_mask]
    other_rows = df[~reversed_crossing_mask]

    num_replication = (df[LABEL_COL] == NORMAL_FORK_LABEL).sum()
    reversed_crossing_rows_undersampled = reversed_crossing_rows.sample(
        n=num_replication, random_state=seed)

    reversed_crossing_rows_undersampled[LABEL_COL] = REVERSED_FORK_CROSSING_COMBINED_LABEL

    return pd.concat([reversed_crossing_rows_undersampled, other_rows], ignore_index=True)


def get_label_mapping(combine_reversed_crossing: bool) -> dict[str, int]:
    if combine_reversed_crossing:
        return {
            "background": 0,
            NORMAL_FORK_LABEL: 1,
            REVERSED_FORK_CROSSING_COMBINED_LABEL: 2,
        }
    return {
        "background": 0,
        NORMAL_FORK_LABEL: 1,
        REVERSED_FORK_LABEL: 2,
        CROSSING_LABEL: 3,
    }


def _sanitize_case_id(img_name: str) -> str:
    return img_name.replace("_", "-").replace(" ", "-")


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


def resize_copy_input(input_dir: Path, output_dir: Path, df: pd.DataFrame,
                      image_resize: int) -> dict[str, tuple[int, int]]:
    target_size = (image_resize, image_resize)
    images_out_dir = output_dir / NNUNET_IMAGES_DIR

    original_shapes: dict[str, tuple[int, int]] = {}

    for idx, img_name in enumerate(df[IMAGE_COL].unique(), start=1):
        img_path = input_dir / IMAGES_SUBDIR_NAME / f"{img_name}.tif"
        npy_path = input_dir / \
            SEGMENTATION_PROBABILITIES_SUBDIR / f"{img_name}.npy"

        if not img_path.exists():
            print(f"Missing raw image: {img_path}")
            continue
        if not npy_path.exists():
            print(f"Missing probability map: {npy_path}")
            continue

        raw = tifffile.imread(img_path)
        prob = np.load(npy_path)

        if raw.ndim != 2:
            raise ValueError(
                f"Expected a single-channel 2D raw image for {img_name}, got shape {raw.shape}")
        if prob.ndim != 2:
            raise ValueError(
                f"Expected a single-channel 2D probability map for {img_name}, got shape {prob.shape}")
        if raw.shape != prob.shape:
            raise ValueError(f"Raw image and probability map shapes differ for {img_name}: "
                             f"{raw.shape} vs {prob.shape}")

        original_shapes[img_name] = raw.shape

        raw_resized = _resize_array(raw, target_size, raw.dtype)
        prob_resized = _resize_array(prob, target_size, np.float32)

        case_id = _sanitize_case_id(img_name)
        tifffile.imwrite(
            images_out_dir / f"{case_id}_{RAW_CHANNEL_ID}{FILE_ENDING}", raw_resized)
        tifffile.imwrite(
            images_out_dir / f"{case_id}_{SEGPROB_CHANNEL_ID}{FILE_ENDING}", prob_resized)

        print(f"Copied {idx}/{df[IMAGE_COL].nunique()} images ({img_name})")

    return original_shapes


def create_labels(df: pd.DataFrame, output_dir: Path, original_shapes: dict[str, tuple[int, int]],
                  image_resize: int, combine_reversed_crossing: bool):
    """
    Writes labelsTr/<case>.tif as a single-channel, uint8, discrete class-id label map - one small
    blob per landmark, NOT a heatmap. See the module docstring: the multi-channel Gaussian heatmap
    used as the actual training target is built later, on the fly, by ConvertSegToMultiChannelHeatmap
    in the nnUNet submodule, from exactly this kind of blob map.
    """
    label_mapping = get_label_mapping(combine_reversed_crossing)
    labels_out_dir = output_dir / NNUNET_LABELS_DIR

    idx = 1
    num_images = df[IMAGE_COL].nunique()
    for img_name, group in df.groupby(IMAGE_COL):
        orig_h, orig_w = original_shapes[img_name]
        scale_y = image_resize / orig_h
        scale_x = image_resize / orig_w

        # discrete class-id map (0 = background, 1..N = landmark classes) - not a heatmap
        seg = np.zeros((image_resize, image_resize), dtype=np.uint8)

        for _, row in group.iterrows():
            label_name = row[LABEL_COL]
            if label_name == NEGATIVE_LABEL:
                continue
            if label_name not in label_mapping:
                raise ValueError(f"Unknown label '{label_name}' for image '{img_name}'; "
                                 f"expected one of {list(label_mapping.keys())}")
            if pd.isna(row[COORD_X_COL]) or pd.isna(row[COORD_Y_COL]):
                raise ValueError(f"Row with label '{label_name}' for image '{img_name}' is missing "
                                 f"x/y coordinates (only '{NEGATIVE_LABEL}' rows may omit them)")

            y = int(round(row[COORD_Y_COL] * scale_y))
            x = int(round(row[COORD_X_COL] * scale_x))
            half = LANDMARK_HALF_SIZE
            seg[max(y - half, 0):y + half + 1, max(x - half, 0)
                    :x + half + 1] = label_mapping[label_name]

        tifffile.imwrite(
            labels_out_dir / f"{_sanitize_case_id(img_name)}{FILE_ENDING}", seg)

        print(f"Created label for {idx}/{num_images} images")
        idx += 1


def create_dataset_json(dir: Path, combine_reversed_crossing: bool, num_training: int):
    label_mapping = get_label_mapping(combine_reversed_crossing)

    dataset_json = {
        "channel_names": {
            "0": "rawImage",
            "1": "segmentationProbability",
        },
        "labels": label_mapping,
        "numTraining": num_training,
        "file_ending": FILE_ENDING,
        "overwrite_image_reader_writer": "NaturalImage2DIO",
    }

    with open(dir / "dataset.json", "w") as f:
        json.dump(dataset_json, f, indent=2)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", required=True, type=Path,
                        help="Path to a (balanced) dataset directory with annotations.csv, images/, segmentation/, segmentation_probabilities/")
    parser.add_argument("--combine-reversed-crossing", action=argparse.BooleanOptionalAction, default=True,
                        help="Whether to combine the crossing and reversed fork labels into one, and undersampling them (randomly) to match the replication fork count. "
                             "Use --no-combine-reversed-crossing to disable.")
    parser.add_argument("--image-resize", type=int, default=1024,
                        help="resize dimension (side length) for images")
    parser.add_argument("--seed", type=int,
                        default=DEFAULT_SEED, help="Random seed")
    args = parser.parse_args()

    output_dir = validate_input_initialize_output(args.input, NNUNET_LANDMARK_DATASET_ID)
    df = pd.read_csv(args.input / ANNOTATIONS_FILE_NAME)

    print(
        f"Found {df.shape[0]} total annotations in {df[IMAGE_COL].nunique()} images")

    if args.combine_reversed_crossing:
        df = combine_reversed_and_crossing(df, args.seed)
        print(
            f"Combined reversed and crossing labels, resulting in {df.shape[0]} total annotations in {df[IMAGE_COL].nunique()} images")

    original_shapes = resize_copy_input(
        args.input, output_dir, df, args.image_resize)
    create_labels(df, output_dir, original_shapes,
                  args.image_resize, args.combine_reversed_crossing)
    create_dataset_json(output_dir, args.combine_reversed_crossing,
                        num_training=df[IMAGE_COL].nunique())

    print(f"Wrote nnU-Net dataset to {output_dir}")


if __name__ == "__main__":
    main()
