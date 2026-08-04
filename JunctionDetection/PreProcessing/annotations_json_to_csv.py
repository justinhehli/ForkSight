"""
Convert an annotations.json produced by the AnnotationTool pipeline into a
CSV with columns: image, label, x, y.

Only images with processed = True are included; unprocessed and archived
images are skipped.

Also renames segmentation mask files in the "segmentation" folder next to
the input JSON from their raw UUID names (the annotations.json key) to
"<image>.png", using each image's "source_tif" field - so later steps can
find a mask by image name like any other per-image file.

Usage:
    python annotations_json_to_csv.py --input annotations.json --output relabeling_data.csv
"""

import argparse
import json
from pathlib import Path

import pandas as pd


def convert_annotations_to_rows(data: dict) -> list[dict]:
    label_mappings = {
        "Replication Fork 50%": "Normal Fork",
        "Replication Fork 100%": "Normal Fork",
        "Reversed Fork 50%": "Reversed Fork",
        "Reversed Fork 100%": "Reversed Fork",
        "Crossing": "Crossing"
    }

    rows = []
    for img in data["images"].values():
        if img.get("archived", False) or not img.get("processed", False):
            continue

        image_name = img["display_name"].replace(".tif", "")
        points = img.get("points", [])

        if not points:
            rows.append(
                {"image": image_name, "label": "Negative", "x": "", "y": ""})
            continue

        for point in points:
            labels = point.get("labels", [])
            if len(labels) != 1 or labels[0] not in label_mappings:
                # training data can only have replication fork, reversed fork or crossing labels
                raise ValueError(
                    f"point {point["id"]} doesn't have single label or unsuitable label: {point["labels"]}")
            label = label_mappings[labels[0]]

            rows.append({"image": image_name, "label": label,
                        "x": point["x"], "y": point["y"]})

    return rows


def rename_segmentation_masks(data: dict, segmentation_dir: Path) -> tuple[int, int, list[str]]:
    """Rename segmentation masks in segmentation_dir from "<uuid>.<ext>" to "<image>.<ext>".

    Returns (renamed_count, already_renamed_count, missing_image_names).
    """
    renamed = 0
    already_renamed = 0
    missing = []
    for uuid, img in data["images"].items():
        source_tif = img.get("source_tif")
        if not source_tif:
            continue
        image_name = Path(source_tif).stem

        if any(segmentation_dir.glob(f"{image_name}.*")):
            already_renamed += 1
            continue

        matches = list(segmentation_dir.glob(f"{uuid}.*"))
        if not matches:
            missing.append(image_name)
            continue
        for match in matches:
            match.rename(match.with_name(f"{image_name}{match.suffix}"))
            renamed += 1

    return renamed, already_renamed, missing


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", required=True, type=Path,
                        help="Path to annotations.json")
    parser.add_argument("--output", required=True, type=Path,
                        help="Path to write the output CSV")
    args = parser.parse_args()

    data = json.loads(args.input.read_text(encoding="utf-8"))
    rows = convert_annotations_to_rows(data)

    df = pd.DataFrame(rows, columns=["image", "label", "x", "y"])
    df.to_csv(args.output, index=False)
    print(
        f"Wrote {len(df)} rows ({df['image'].nunique()} images) to {args.output}")

    segmentation_dir = args.input.parent / "segmentation"
    if segmentation_dir.is_dir():
        renamed, already_renamed, missing = rename_segmentation_masks(
            data, segmentation_dir)
        suffix = f" ({already_renamed} already renamed)" if already_renamed else ""
        print(
            f"Renamed {renamed} segmentation masks in {segmentation_dir}{suffix}")
        if missing:
            print(
                f"Warning: {len(missing)} images have no segmentation mask: {missing}")


if __name__ == "__main__":
    main()
