"""
Balance a junction-detection dataset.

Balancing works on *rows* of annotations.csv (one row per point annotation),
not on image files:
  1. Oversample whichever of "Normal Fork" / "Reversed Fork" has fewer rows
     so both roughly match the larger of the two.
  2. Undersample "Crossing" rows down to roughly that same balanced count.
  3. Keep only "Negative" rows whose image is listed in --negatives (one
     image name per line); drop the rest.

Does not modify the input dataset - writes a full copy (balanced
annotations.csv plus the images/ and segmentation/ files still referenced
afterwards) to the given --output directory.

Usage:
    python create_balanced_dataset.py --input <dataset_dir> --negatives <negatives_file> --output <output_dir>
"""

import argparse
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

ANNOTATIONS_CSV_NAME = "annotations.csv"
FORK_LABELS = ("Normal Fork", "Reversed Fork")
CROSSING_LABEL = "Crossing"
NEGATIVE_LABEL = "Negative"


def load_and_clean(csv_path: Path) -> pd.DataFrame:
    """Load annotations.csv and drop exact duplicate rows."""
    df = pd.read_csv(csv_path)
    before = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    dropped = before - len(df)
    if dropped:
        print(f"Dropped {dropped} duplicate rows")
    return df


def copy_referenced_files(image_names, dataset_dir: Path, dest_dir: Path, subfolder: str) -> tuple[list[str], list[str]]:
    """Copy each image's file from dataset_dir/subfolder into dest_dir, matching by stem (any extension).

    Returns (copied_names, missing_names).
    """
    src_dir = dataset_dir / subfolder
    dest_dir.mkdir(parents=True, exist_ok=True)

    copied = []
    missing = []
    for name in image_names:
        matches = list(src_dir.glob(f"{name}.*")) if src_dir.is_dir() else []
        if not matches:
            missing.append(name)
            continue
        for match in matches:
            shutil.copy2(match, dest_dir / match.name)
        copied.append(name)

    return copied, missing


def oversample_to(df: pd.DataFrame, label: str, target: int, rng: np.random.RandomState) -> pd.DataFrame:
    rows = df[df['label'] == label]
    deficit = target - len(rows)
    if deficit <= 0:
        return df
    extra = rows.sample(n=deficit, replace=True, random_state=rng)
    return pd.concat([df, extra], ignore_index=True)


def undersample_to(df: pd.DataFrame, label: str, target: int, rng: np.random.RandomState) -> pd.DataFrame:
    rows = df[df['label'] == label]
    if len(rows) <= target:
        return df
    keep = rows.sample(n=target, replace=False, random_state=rng)
    return pd.concat([df[df['label'] != label], keep], ignore_index=True)


def balance_forks_and_crossing(df: pd.DataFrame, rng: np.random.RandomState) -> pd.DataFrame:
    fork_counts = {label: (df['label'] == label).sum()
                   for label in FORK_LABELS}
    target = max(fork_counts.values())

    for label in FORK_LABELS:
        df = oversample_to(df, label, target, rng)

    return undersample_to(df, CROSSING_LABEL, target, rng)


def filter_negatives(df: pd.DataFrame, negatives_file: Path) -> pd.DataFrame:
    keep_images = {line.strip()
                   for line in negatives_file.read_text().splitlines() if line.strip()}
    drop_mask = (df['label'] == NEGATIVE_LABEL) & ~df['image'].isin(
        keep_images)
    return df[~drop_mask]


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", required=True, type=Path,
                        help="Path to a dataset directory with annotations.csv, images/ and segmentation/")
    parser.add_argument("--negatives", required=True, type=Path,
                        help="Path to a text file listing (one per line) the image names of Negative samples to keep")
    parser.add_argument("--output", required=True, type=Path,
                        help="Path to the output directory where the balanced dataset is created")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for oversampling/undersampling")
    args = parser.parse_args()

    rng = np.random.RandomState(args.seed)

    df = load_and_clean(args.input / ANNOTATIONS_CSV_NAME)
    print("Input label counts:")
    print(df['label'].value_counts().to_string())

    df = filter_negatives(df, args.negatives)
    df = balance_forks_and_crossing(df, rng)
    df = df.reset_index(drop=True)

    print("\nBalanced label counts:")
    print(df['label'].value_counts().to_string())

    args.output.mkdir(parents=True, exist_ok=True)

    output_csv = args.output / ANNOTATIONS_CSV_NAME
    df.to_csv(output_csv, index=False)
    print(f"\nWrote {len(df)} rows to {output_csv}")

    image_names = df['image'].dropna().unique()
    for subfolder in ("images", "segmentation"):
        copied, missing = copy_referenced_files(
            image_names, args.input, args.output / subfolder, subfolder)
        print(f"Copied {len(copied)} files to {args.output / subfolder}")
        if missing:
            print(
                f"Warning: {len(missing)} images have no {subfolder} file: {missing}")


if __name__ == "__main__":
    main()
