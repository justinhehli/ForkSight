"""
Merge point annotations from final re-labeling into one CSV.

Output is written to $JUNCTION_DETECTION_DIR/$JUNCTION_DETECTION_RELABELING_DATA/merged_annotations.csv

Usage:
    python merge_final_relabeling_annotations.py --json Final.json --csv agreed.csv
"""

import argparse
import json
import os
import re
import shutil
from pathlib import Path
from urllib.parse import unquote

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

import Environment.env_utils as env_utils

# Directories to collect source PNG images from
SOURCE_IMAGE_DIRS: list[str] = [
    "Z:\\imcrdata\\2025_EM_Deep_Learning_Project\\FinalRelabelling\\Rescoring_Images_Agreed",
    "Z:\\imcrdata\\2025_EM_Deep_Learning_Project\\FinalRelabelling\\Rescoring_Images_Disagreed"
]


def strip_float_suffix(label: str) -> str:
    """Remove trailing float suffix from a label, e.g. 'Normal Fork 0.67' -> 'Normal Fork'."""
    return re.sub(r"\s+\d+(\.\d+)?\s*$", "", label).strip()


def load_final_json(path: str) -> pd.DataFrame:
    """Load Label Studio JSON annotations and convert x/y from percentages to pixel coords."""
    with open(path, "r") as f:
        data = json.load(f)

    rows = []
    for entry in data:
        image_path = entry.get("data", {}).get("image", "")
        image_name = Path(unquote(image_path)).stem.replace(
            "-000000_0-000", "")

        for annotation in entry.get("annotations", []):
            for result in annotation.get("result", []):
                orig_w = result.get("original_width")
                orig_h = result.get("original_height")
                value = result.get("value", {})
                x_pct = value.get("x")
                y_pct = value.get("y")
                keypointlabels = value.get("keypointlabels", [])
                if not keypointlabels:
                    continue
                label = strip_float_suffix(keypointlabels[0])
                if label == "Unsure":
                    label = "Negative"
                    x = y = None
                else:
                    x = x_pct / 100 * orig_w
                    y = y_pct / 100 * orig_h
                rows.append({
                    "image": image_name,
                    "label": label,
                    "x": x,
                    "y": y,
                    "source": "json",
                })

    return pd.DataFrame(rows, columns=["image", "label", "x", "y", "source"])


def load_agreed_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["image"] = df["image"].apply(lambda i: i.replace(
        "-000000_0-000", "").replace(".png", ""))
    df["label"] = df["label"].apply(strip_float_suffix)
    result = df[["image", "label", "mean_x_px", "mean_y_px"]].copy()
    result.columns = ["image", "label", "x", "y"]
    result["source"] = "csv"
    return result


def setup_images_dir(junction_detection_dir: Path):
    """Delete and recreate images/, then populate it with PNGs from SOURCE_IMAGE_DIRS."""
    image_dir = junction_detection_dir / "images"
    if image_dir.exists():
        shutil.rmtree(image_dir)
    image_dir.mkdir(parents=True)

    copied = 0
    for src in SOURCE_IMAGE_DIRS:
        src_path = Path(src)
        for png in src_path.glob("*.png"):
            shutil.copy2(png, image_dir / png.name)
            copied += 1
            print(f"Copied {png.name} to {image_dir}")
    print(
        f"Populated {image_dir} with {copied} PNG files from {len(SOURCE_IMAGE_DIRS)} source directories\n")


def delete_excluded_images(junction_detection_dir: Path):
    """Delete images whose stems appear in excluded_images.txt."""
    excluded_txt = Path(__file__).parent / "excluded_images.txt"
    if not excluded_txt.exists():
        return
    excluded_stems = {
        line.strip() for line in excluded_txt.read_text().splitlines() if line.strip()}
    image_dir = junction_detection_dir / "images"
    deleted = 0
    for img_path in list(image_dir.iterdir()):
        if img_path.stem in excluded_stems:
            img_path.unlink()
            deleted += 1
            print(f"Deleted excluded image {img_path.name} from {image_dir}")
    if deleted:
        print(f"Deleted {deleted} excluded images from {image_dir}\n")


def normalize_image_filenames(junction_detection_dir: Path):
    """Remove '-000000_0-000' suffix from image filenames in images/ in-place."""
    image_dir = junction_detection_dir / "images"
    suffix = "-000000_0-000"
    renamed = 0
    for img_path in image_dir.iterdir():
        if suffix in img_path.stem:
            new_name = img_path.stem.replace(suffix, "") + img_path.suffix
            img_path.rename(img_path.parent / new_name)
            renamed += 1
    if renamed:
        print(f"Renamed {renamed} image files (removed '{suffix}')\n")


def _get_image_points(df: pd.DataFrame, image_name: str):
    """Return (has_any_annotation, point_rows) for an image name (stem, suffix stripped)."""
    rows = df[df["image"] == image_name]
    points = rows[rows["label"] != "Negative"].dropna(subset=["x", "y"])
    return rows, points


def move_excluded(df: pd.DataFrame, junction_detection_dir: Path):
    """Move images with no annotations at all to images_excluded/. Negative-labelled images stay."""
    image_dir = junction_detection_dir / "images"
    excluded_dir = junction_detection_dir / "images_excluded"
    excluded_dir.mkdir(exist_ok=True)

    unannotated = []
    for img_path in sorted(image_dir.iterdir()):
        rows, points = _get_image_points(df, img_path.stem)
        if rows.empty or points.empty and not (rows["label"] == "Negative").any():
            unannotated.append(img_path.stem)
            img_path.rename(excluded_dir / img_path.name)

    if unannotated:
        print(
            f"\nMoved {len(unannotated)} unannotated images to {excluded_dir}:")
        for name in unannotated:
            print(f"  {name}")
    else:
        print("No unannotated images found.")


def resize_images_to_target(
    df: pd.DataFrame, junction_detection_dir: Path, target_size: int = 4096
) -> tuple[pd.DataFrame, list[dict]]:
    """Resize images that are not target_size×target_size in-place; scale coordinates in df."""
    image_dir = junction_detection_dir / "images"
    did_resize = False

    for img_path in sorted(image_dir.iterdir()):
        pil_img = Image.open(img_path)
        w, h = pil_img.size
        if w == target_size and h == target_size:
            continue

        pil_img_resized = pil_img.resize(
            (target_size, target_size), Image.BICUBIC)
        pil_img_resized.save(img_path)

        scale_x = target_size / w
        scale_y = target_size / h
        mask = df["image"] == img_path.stem
        df.loc[mask, "x"] = df.loc[mask, "x"] * scale_x
        df.loc[mask, "y"] = df.loc[mask, "y"] * scale_y

        did_resize = True
        print(
            f"Resized {img_path.stem}: {w}x{h} to {target_size}x{target_size}")

    if not did_resize:
        print("All images already at target size; no resizing needed.")

    return df, did_resize


def plot_junctions(df: pd.DataFrame, junction_detection_dir: Path):
    """For each image in images/, save a plot with annotated junction points."""
    image_dir = junction_detection_dir / "images"
    plots_dir = junction_detection_dir / "junction_plots"
    plots_dir.mkdir(exist_ok=True)

    image_files = sorted(image_dir.iterdir())
    no_points_images = []

    for img_path in image_files:
        rows, points = _get_image_points(df, img_path.stem)

        if points.empty:
            no_points_images.append(img_path.stem)

        pil_img = Image.open(img_path)
        orig_w, orig_h = pil_img.size
        pil_img = pil_img.resize((1024, 1024), Image.LANCZOS)
        img = np.array(pil_img)
        scale_x, scale_y = 1024 / orig_w, 1024 / orig_h

        fig, ax = plt.subplots(figsize=(10, 10))
        ax.imshow(img, cmap="gray" if img.ndim == 2 else None)

        for _, row in points.iterrows():
            px, py = row["x"] * scale_x, row["y"] * scale_y
            ax.plot(px, py, "o", color="none", markersize=40, markeredgewidth=1,
                    markeredgecolor="red")
            ax.plot(px, py, "o", color="red", markersize=2)

        ax.set_title(img_path.stem, fontsize=8)
        ax.axis("off")
        fig.tight_layout()
        fig.savefig(plots_dir / (img_path.stem + ".png"), dpi=150)
        plt.close(fig)

    if no_points_images:
        print(f"\nImages with no plotted points ({len(no_points_images)}):")
        for name in no_points_images:
            print(f"  {name}")

    n_with_points = len(image_files) - len(no_points_images)
    print(
        f"\nSaved {len(image_files)} plots to {plots_dir} ({n_with_points} with points)")


LABEL_ORDER = ["Normal Fork", "Crossing", "Reversed Fork", "Negative"]


def plot_label_stats(df: pd.DataFrame, junction_detection_dir: Path):
    """Two bar plots: (1) images per label, (2) total annotations per label."""
    plots_dir = junction_detection_dir / "plots"
    plots_dir.mkdir(exist_ok=True)

    # Count distinct images per label (fixed order)
    per_label_images = (
        df.groupby("label")["image"]
        .nunique()
        .reindex(LABEL_ORDER)
        .dropna()
        .astype(int)
    )

    # Total annotations per label (fixed order)
    per_label_total = (
        df.groupby("label")
        .size()
        .reindex(LABEL_ORDER)
        .dropna()
        .astype(int)
    )

    # Images with more than one annotation row
    annotation_counts = df.groupby("image").size()
    n_multi_annot = (annotation_counts > 1).sum()

    # Images with more than one distinct label type
    label_counts = df.groupby("image")["label"].nunique()
    n_mixed_labels = (label_counts > 1).sum()

    # --- Plot 1: Images per label ---
    labels1 = list(per_label_images.index) + \
        ["Multiple annotations", "Mixed label types"]
    counts1 = list(per_label_images.values) + [n_multi_annot, n_mixed_labels]
    colors1 = ["steelblue"] * len(per_label_images) + ["orange", "tomato"]

    fig1, ax1 = plt.subplots(figsize=(4, 5))
    bars1 = ax1.bar(labels1, counts1, color=colors1)
    ax1.bar_label(bars1, padding=3)
    ax1.set_ylabel("Number of Images")
    ax1.set_title("Images per Annotation Label")
    ax1.set_ylim(top=ax1.get_ylim()[1] * 1.15)
    plt.xticks(rotation=30, ha="right")
    fig1.tight_layout()

    # --- Plot 2: Total annotations per label ---
    fig2, ax2 = plt.subplots(figsize=(4, 5))
    bars2 = ax2.bar(per_label_total.index,
                    per_label_total.values, color="steelblue")
    ax2.bar_label(bars2, padding=3)
    ax2.set_ylabel("Number of Annotations")
    ax2.set_title("Total Annotations per Label")
    ax2.set_ylim(top=ax2.get_ylim()[1] * 1.15)
    plt.xticks(rotation=30, ha="right")
    fig2.tight_layout()

    # --- Equalize bottom margins across both figures ---
    bottom = max(fig1.subplotpars.bottom, fig2.subplotpars.bottom)
    fig1.subplots_adjust(bottom=bottom)
    fig2.subplots_adjust(bottom=bottom)

    out1 = plots_dir / "junction_detection_dataset_annotations_per_image.png"
    fig1.savefig(out1, dpi=150)
    plt.close(fig1)
    print(f"Saved label stats (images) plot to {out1}")

    out2 = plots_dir / "junction_detection_dataset_annotations_total.png"
    fig2.savefig(out2, dpi=150)
    plt.close(fig2)
    print(f"Saved label stats (annotations) plot to {out2}")


def main():
    env_utils.load_forksight_env()

    RAW_DATA_DIR = os.getenv("RAW_DATA_DIR", None)
    JUNCTION_DETECTION_DIR_NAME = os.getenv(
        "JUNCTION_DETECTION_DIR_NAME", None)
    JUNCTION_DETECTION_RELABELING_FILE_NAME = os.getenv(
        "JUNCTION_DETECTION_RELABELING_FILE_NAME", None)

    if not all([RAW_DATA_DIR, JUNCTION_DETECTION_DIR_NAME, JUNCTION_DETECTION_RELABELING_FILE_NAME]):
        raise ValueError(
            "One or more required environment variables are not set: RAW_DATA_DIR, JUNCTION_DETECTION_DIR, JUNCTION_DETECTION_RELABELING_DATA")

    parser = argparse.ArgumentParser(
        description="Merge Final.json and agreed.csv annotations.")
    parser.add_argument("--json", help="Path to Final.json")
    parser.add_argument("--csv", help="Path to agreed.csv")
    parser.add_argument("--move-excluded", action="store_true", default=False,
                        help="Detect, list, and move images with no annotations to images_excluded/")
    parser.add_argument("--plot-junctions", action="store_true", default=False,
                        help="Save per-image junction plots to junction_plots/ (images must be in images/)")
    parser.add_argument("--plot-stats", action="store_true", default=False,
                        help="Save label distribution bar plot to plots/")
    parser.add_argument("--plot-only", action="store_true", default=False,
                        help="Recompute and overwrite the output CSV even if it already exists")
    args = parser.parse_args()

    junction_detection_dir = Path(RAW_DATA_DIR) / JUNCTION_DETECTION_DIR_NAME
    output_path = junction_detection_dir / JUNCTION_DETECTION_RELABELING_FILE_NAME

    if not args.plot_only:
        if not args.json or not args.csv:
            raise ValueError("--json and --csv arguments must not be empty")

        setup_images_dir(junction_detection_dir)
        normalize_image_filenames(junction_detection_dir)
        delete_excluded_images(junction_detection_dir)

        if output_path.exists():
            output_path.unlink()
        df_json = load_final_json(args.json)
        df_csv = load_agreed_csv(args.csv)
        print(f"JSON points: {len(df_json)}, CSV points: {len(df_csv)}")
        df_merged = pd.concat([df_json, df_csv], ignore_index=True)

        # remove annotations for images that were manually excluded
        excluded_txt = Path(__file__).parent / "excluded_images.txt"
        if excluded_txt.exists():
            images_manually_excluded = {
                line.strip() for line in excluded_txt.read_text().splitlines() if line.strip()}
            df_merged = df_merged[~df_merged["image"].isin(
                images_manually_excluded)]

        df_merged.to_csv(output_path, index=False)
        print(f"Wrote {len(df_merged)} rows to {output_path}")

        # Resize non-4096 images and update coordinates before any plotting or exclusion
        df_merged, did_resize = resize_images_to_target(
            df_merged, junction_detection_dir)
        if did_resize:
            df_merged.to_csv(output_path, index=False)
            print(f"Updated {output_path} with rescaled coordinates")

        if args.move_excluded:
            move_excluded(df_merged, junction_detection_dir)
    else:
        df_merged = pd.read_csv(output_path)

    if args.plot_junctions:
        plot_junctions(df_merged, junction_detection_dir)

    if args.plot_stats:
        plot_label_stats(df_merged, junction_detection_dir)


if __name__ == "__main__":
    main()
