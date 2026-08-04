import argparse
import shutil
from pathlib import Path
import pandas as pd
import numpy as np
from PIL import Image

ANNOTATIONS_FILE_NAME = "annotations.csv"
IMAGES_SUBDIR_NAME = "images"
SEGMENTATION_SUBDIR_NAME = "segmentation"

NNUNET_DATASET_PREFIX = "DatasetXXX_"
NNUNET_DATASET_SUFFIX = "_nnUNet"

NORMAL_FORK_LABEL = "Normal Fork"
REVERSED_FORK_LABEL = "Reversed Fork"
CROSSING_LABEL = "Crossing"
NEGATIVE_LABEL = "Negative"
REVERSED_FORK_CROSSING_COMBINED_LABEL = "Reversed Fork / Crossing"

DEFAULT_SEED = 42

NNUNET_IMAGES_DIR = "imagesTr"
NNUNET_LABELS_DIR = "labelsTr"


def validate_input_initialize_output(input_dir: Path, dataset_id: int) -> Path:
    if not input_dir.is_dir():
        raise ValueError("input directory does not exist")
    if not (input_dir / ANNOTATIONS_FILE_NAME).is_file():
        raise ValueError("input directory doesn't contain annotations.csv")
    if not (input_dir / IMAGES_SUBDIR_NAME).is_dir() or not (input_dir / SEGMENTATION_SUBDIR_NAME).is_dir():
        raise ValueError(
            "input directory doesn't contain the images and segmentation sub-directories")

    output_dir = input_dir.parent / (NNUNET_DATASET_PREFIX.replace(
        "XXX", f"{dataset_id:03d}") + input_dir.name + NNUNET_DATASET_SUFFIX)

    output_dir.mkdir()
    (output_dir / NNUNET_IMAGES_DIR).mkdir()
    (output_dir / NNUNET_LABELS_DIR).mkdir()

    return output_dir


def combine_reversed_and_crossing(df: pd.DataFrame, seed: int) -> pd.DataFrame:
    reversed_crossing_mask = df['label'].isin(
        [REVERSED_FORK_LABEL, CROSSING_LABEL])
    reversed_crossing_rows = df[reversed_crossing_mask]
    other_rows = df[~reversed_crossing_mask]

    num_replication = (df['label'] == NORMAL_FORK_LABEL).sum()
    reversed_crossing_rows_undersampled = reversed_crossing_rows.sample(
        n=num_replication, random_state=seed)

    reversed_crossing_rows_undersampled['label'] = REVERSED_FORK_CROSSING_COMBINED_LABEL

    return pd.concat([reversed_crossing_rows_undersampled, other_rows], ignore_index=True)


def combine_image_channels_resize_copy(input_dir: Path, output_dir: Path, df: pd.DataFrame):
    in_images_dir = input_dir / IMAGES_SUBDIR_NAME
    in_segmentation_dir = input_dir / SEGMENTATION_SUBDIR_NAME

    out_images_dir = output_dir / NNUNET_IMAGES_DIR

    for img_name in df['image'].unique():
        img_filename = f"{img_name}.png"
        raw_img = np.array(Image.open(
            in_images_dir / img_filename) .convert("L"), dtype=np.float32)
        segmentation_img = np.array(Image.open(
            in_segmentation_dir / img_filename).convert("L"), dtype=np.float32)

        # normalize raw image and segmentation mask to comparable scales (binarize segmentation mask)
        raw_img = raw_img / 255.0
        segmentation_img = (segmentation_img > 0).astype(np.float32)

        # channel 0 = mask, channel 1 = raw
        combined = np.stack([raw_img, segmentation_img], axis=0)


def create_dataset_json(dir: Path, combine_reversed_crossing: bool):
    pass


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", required=True, type=Path,
                        help="Path to a (balanced) dataset directory with annotations.csv, images/ and segmentation/")
    parser.add_argument("--combine-reversed-crossing", type=bool, default=True,
                        help="Whether to combine the crossing and reversed fork labels into one, and undersampling them (randomly) to match the replication fork count")
    parser.add_argument("--image-resize", type=int, default=1024,
                        help="resize dimension (side length) for images")
    parser.add_argument("--dataset-id", type=int, default=1,
                        help="ID for nnU-Net dataset identification")
    parser.add_argument("--seed", type=int,
                        default=DEFAULT_SEED, help="Random seed")
    args = parser.parse_args()

    output_dir = validate_input_initialize_output(args.input, args.dataset_id)
    df = pd.read_csv(args.input / ANNOTATIONS_FILE_NAME)

    if args.combine_reversed_crossing:
        df = combine_reversed_and_crossing(df, args.seed)


if __name__ == "__main__":
    main()
