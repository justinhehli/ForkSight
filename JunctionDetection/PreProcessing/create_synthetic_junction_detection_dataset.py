"""Build a junction detection test dataset from the synthetic DNA images.

Downloads all full 4096x4096 images of the test split of the HuggingFace
dataset repo used for the synthetic nnUNet dataset (see
Segmentation/PreProcessing/nnUNet/generate_nnUNet_synthetic_dataset.py) and
writes them in the same layout as the real junction detection test set
(RawData/junction_detection), so the evaluation scripts can consume it via
--dataset-dir:

    <output-dir>/
        images/<name>.png         8-bit grayscale, 4096x4096
        relabeling_data.csv       image, label, x, y, source
        images_tif/<name>.tif     (--tif) uint16 copies, as in the real dataset
        junction_plots/<name>.png (--plot-junctions)

Labels: the manifest stores one junction (junction_x, junction_y) per positive
image, which is mapped to the real dataset's labels
(replication_fork -> "Normal Fork" (3-way), reversed_fork -> "Reversed Fork"
(4-way)); negative images get a single "Negative" row without coordinates.
Note that only this one generated junction is annotated per image; incidental
fiber crossings in the synthetic images are not.

Run from the repo root:
    python -m JunctionDetection.PreProcessing.create_synthetic_junction_detection_dataset \\
        --output-dir DIR [--tif] [--plot-junctions]
"""
import argparse
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile
from PIL import Image

from Segmentation.PreProcessing.General.preprocessing_util import init_dir
from Segmentation.PreProcessing.nnUNet.generate_nnUNet_synthetic_dataset import (
    fetch,
    load_manifest,
)

SPLIT = "test"
IMAGE_SIZE = 4096
LABELS_FILE_NAME = "relabeling_data.csv"
SOURCE = "synthetic"

CLASS_TO_LABEL = {
    "replication_fork": "Normal Fork",
    "reversed_fork": "Reversed Fork",
    "negative": "Negative",
}


def image_stem(row: dict) -> str:
    return Path(row["tem"]).stem


def label_rows(row: dict) -> list[dict]:
    """relabeling_data.csv rows (same columns as the real dataset) for one image."""
    label = CLASS_TO_LABEL[row["class"]]
    if label == "Negative":
        x, y = np.nan, np.nan
    else:
        x, y = float(row["junction_x"]), float(row["junction_y"])
        if not (0 <= x < IMAGE_SIZE and 0 <= y < IMAGE_SIZE):
            raise ValueError(
                f"Junction ({x}, {y}) of {row['tem']} lies outside the image")
    return [{"image": image_stem(row), "label": label, "x": x, "y": y,
             "source": SOURCE}]


def save_images(rows: list[dict], dataset_dir: Path, save_tif: bool):
    images_dir = dataset_dir / "images"
    images_dir.mkdir()
    tifs_dir = dataset_dir / "images_tif"
    if save_tif:
        tifs_dir.mkdir()

    for idx, row in enumerate(rows, start=1):
        img = np.array(Image.open(fetch(f"{SPLIT}/{row['tem']}")).convert("L"))
        if img.shape != (IMAGE_SIZE, IMAGE_SIZE):
            raise ValueError(
                f"Expected a {IMAGE_SIZE}x{IMAGE_SIZE} image, got {img.shape} "
                f"for {row['tem']}")

        Image.fromarray(img).save(images_dir / f"{image_stem(row)}.png")
        if save_tif:
            # real dataset TIFs are uint16; scale 8-bit values to the full range
            tifffile.imwrite(tifs_dir / f"{image_stem(row)}.tif",
                             img.astype(np.uint16) * 257)
        print(f"Saved {idx}/{len(rows)}: {image_stem(row)}")


def create_dataset(dataset_dir: Path, save_tif: bool, plot: bool):
    rows = sorted(load_manifest(SPLIT), key=lambda row: row["tem"])
    print(f"{len(rows)} {SPLIT} images: "
          f"{dict(Counter(row['class'] for row in rows))}")

    init_dir(dataset_dir)
    save_images(rows, dataset_dir, save_tif)

    labels_df = pd.DataFrame([r for row in rows for r in label_rows(row)])
    labels_path = dataset_dir / LABELS_FILE_NAME
    labels_df.to_csv(labels_path, index=False)
    print(f"\nWrote {len(labels_df)} label rows to {labels_path}")

    if plot:
        # imported lazily: the module pulls in the ForkSight environment utils
        from JunctionDetection.PreProcessing.merge_final_relabeling_annotations import (
            plot_junctions,
        )
        plot_junctions(labels_df, dataset_dir)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="dataset directory to create")
    parser.add_argument("--tif", action="store_true",
                        help="also write uint16 TIFs to images_tif/ (only needed "
                        "for the nnUNet landmark evaluation)")
    parser.add_argument("--plot-junctions", action="store_true",
                        help="save per-image junction plots to junction_plots/")
    parser.add_argument("--overwrite", action="store_true",
                        help="replace the output directory if it already exists")
    args = parser.parse_args()

    dataset_dir = args.output_dir
    if dataset_dir.exists() and any(dataset_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(
            f"{dataset_dir} already exists and is not empty; pass --overwrite to replace it")

    create_dataset(dataset_dir, save_tif=args.tif, plot=args.plot_junctions)
    print(f"\nSynthetic junction detection dataset written to {dataset_dir}")


if __name__ == "__main__":
    main()
