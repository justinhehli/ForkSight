"""
Given an nnU-Net dataset directory already built by create_nnunet_heatmap_dataset.py (2 input
channels: raw image `_0000`, segmentation probability map `_0001`; multi-class landmark labels),
produces two derived "copies" of it as sibling directories:

  segprob-only    - same (multi-class) labelsTr, but imagesTr keeps only the segmentation
                    probability map, renamed to the sole channel `_0000`
  combined-label  - same (both-channel) imagesTr, but labelsTr is binarized: every landmark type is
                    collapsed into one combined foreground label.
                    Meant to be trained with nnUNetTrainerHeatmapAdaptiveWingFocalSoftSamplingSingleLabel (see
                    nnUNet/nnunetv2/training/nnUNetTrainer/variants/heatmap/) - see that trainer's
                    docstring and JunctionDetection/PreProcessing/patch_single_label_plans.py for the
                    one-time setup step required before training on the combined-label copy.
"""

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import tifffile

from JunctionDetection.PreProcessing.create_nnunet_heatmap_dataset import (
    FILE_ENDING,
    NNUNET_DATASET_PREFIX,
    NNUNET_IMAGES_DIR,
    NNUNET_LABELS_DIR,
    RAW_CHANNEL_ID,
    SEGPROB_CHANNEL_ID,
)

# Hardcoded, like NNUNET_LANDMARK_DATASET_ID in create_nnunet_heatmap_dataset.py - kept fixed (not
# configurable) so this script and the corresponding job scripts always agree on the same IDs.
NNUNET_LANDMARK_SEGPROB_ONLY_DATASET_ID = 12
NNUNET_LANDMARK_COMBINED_LABEL_DATASET_ID = 13

SEGPROB_ONLY_SUFFIX = "_SegProbOnly"
COMBINED_LABEL_SUFFIX = "_CombinedLabel"

COMBINED_LABEL_NAME = "Junction"

VARIANT_SEGPROB_ONLY = "segprob-only"
VARIANT_COMBINED_LABEL = "combined-label"
VARIANT_BOTH = "both"


def _strip_dataset_prefix(name: str) -> str:
    prefix_head = NNUNET_DATASET_PREFIX.split("XXX_")[0]  # "Dataset"
    if name.startswith(prefix_head) and "_" in name:
        return name.split("_", 1)[1]
    return name


def _make_output_dir(input_dir: Path, dataset_id: int, suffix: str) -> Path:
    base_name = _strip_dataset_prefix(input_dir.name)
    dir_name = NNUNET_DATASET_PREFIX.replace(
        "XXX", f"{dataset_id:03d}") + base_name + suffix
    output_dir = input_dir.parent / dir_name

    if output_dir.exists():
        raise ValueError(f"output directory already exists: {output_dir}")

    output_dir.mkdir()
    (output_dir / NNUNET_IMAGES_DIR).mkdir()
    (output_dir / NNUNET_LABELS_DIR).mkdir()
    return output_dir


def _load_dataset_json(input_dir: Path) -> dict:
    dataset_json_path = input_dir / "dataset.json"
    if not dataset_json_path.is_file():
        raise ValueError(f"input directory doesn't contain dataset.json: {input_dir}")
    with open(dataset_json_path) as f:
        return json.load(f)


def _case_ids_from_labels(input_dir: Path) -> list[str]:
    return sorted(p.stem for p in (input_dir / NNUNET_LABELS_DIR).glob(f"*{FILE_ENDING}"))


def create_segprob_only_variant(input_dir: Path, dataset_json: dict) -> Path:
    output_dir = _make_output_dir(
        input_dir, NNUNET_LANDMARK_SEGPROB_ONLY_DATASET_ID, SEGPROB_ONLY_SUFFIX)
    case_ids = _case_ids_from_labels(input_dir)

    for idx, case_id in enumerate(case_ids, start=1):
        src = input_dir / NNUNET_IMAGES_DIR / \
            f"{case_id}_{SEGPROB_CHANNEL_ID}{FILE_ENDING}"
        if not src.is_file():
            raise FileNotFoundError(f"Missing segmentation-probability image: {src}")
        shutil.copy2(
            src, output_dir / NNUNET_IMAGES_DIR / f"{case_id}_{RAW_CHANNEL_ID}{FILE_ENDING}")
        shutil.copy2(input_dir / NNUNET_LABELS_DIR / f"{case_id}{FILE_ENDING}",
                     output_dir / NNUNET_LABELS_DIR / f"{case_id}{FILE_ENDING}")
        print(f"[segprob-only] copied {idx}/{len(case_ids)} cases ({case_id})")

    new_dataset_json = dict(dataset_json)
    new_dataset_json["channel_names"] = {"0": "segmentationProbability"}
    new_dataset_json["numTraining"] = len(case_ids)
    with open(output_dir / "dataset.json", "w") as f:
        json.dump(new_dataset_json, f, indent=2)

    print(f"Wrote segprob-only variant to {output_dir}")
    return output_dir


def create_combined_label_variant(input_dir: Path, dataset_json: dict) -> Path:
    output_dir = _make_output_dir(
        input_dir, NNUNET_LANDMARK_COMBINED_LABEL_DATASET_ID, COMBINED_LABEL_SUFFIX)
    case_ids = _case_ids_from_labels(input_dir)

    for idx, case_id in enumerate(case_ids, start=1):
        for channel_id in (RAW_CHANNEL_ID, SEGPROB_CHANNEL_ID):
            src = input_dir / NNUNET_IMAGES_DIR / f"{case_id}_{channel_id}{FILE_ENDING}"
            if not src.is_file():
                raise FileNotFoundError(f"Missing input image: {src}")
            shutil.copy2(
                src, output_dir / NNUNET_IMAGES_DIR / f"{case_id}_{channel_id}{FILE_ENDING}")

        label = tifffile.imread(input_dir / NNUNET_LABELS_DIR / f"{case_id}{FILE_ENDING}")
        combined_label = (label > 0).astype(np.uint8)
        tifffile.imwrite(
            output_dir / NNUNET_LABELS_DIR / f"{case_id}{FILE_ENDING}", combined_label)

        print(f"[combined-label] copied {idx}/{len(case_ids)} cases ({case_id})")

    new_dataset_json = dict(dataset_json)
    new_dataset_json["labels"] = {"background": 0, COMBINED_LABEL_NAME: 1}
    new_dataset_json["numTraining"] = len(case_ids)
    with open(output_dir / "dataset.json", "w") as f:
        json.dump(new_dataset_json, f, indent=2)

    print(f"Wrote combined-label variant to {output_dir}")
    return output_dir


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", required=True, type=Path,
                        help="Path to an nnU-Net dataset directory already built by "
                             "create_nnunet_heatmap_dataset.py")
    parser.add_argument("--variant", choices=[VARIANT_SEGPROB_ONLY, VARIANT_COMBINED_LABEL, VARIANT_BOTH],
                        default=VARIANT_BOTH, help="Which variant(s) to create (default: both)")
    args = parser.parse_args()

    if not args.input.is_dir():
        raise ValueError("input directory does not exist")

    dataset_json = _load_dataset_json(args.input)

    if args.variant in (VARIANT_SEGPROB_ONLY, VARIANT_BOTH):
        create_segprob_only_variant(args.input, dataset_json)
    if args.variant in (VARIANT_COMBINED_LABEL, VARIANT_BOTH):
        create_combined_label_variant(args.input, dataset_json)


if __name__ == "__main__":
    main()
