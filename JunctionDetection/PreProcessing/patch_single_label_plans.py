"""
One-time setup step for the "combined-label" nnU-Net dataset (see create_nnunet_dataset_variants.py)
before training with nnUNetTrainerHeatmapAdaptiveWingFocalSoftSamplingSingleLabel.

Run once, after `nnUNetv2_plan_and_preprocess` and before `nnUNetv2_train`:

    python -m JunctionDetection.PreProcessing.patch_single_label_plans --dataset-id 13

Sets the "label_manager" key in the preprocessed dataset's plans file
(nnUNet_preprocessed/<dataset>/nnUNetPlans.json) to "SingleHeadLabelManager" (see
nnUNet/nnunetv2/utilities/label_handling/single_label_manager.py), so the network is built with a
single (non-redundant) output channel both at training time and whenever the trained model is later
reloaded for inference. Refuses to patch a dataset whose dataset.json doesn't have exactly one
foreground label, since applying this to a multi-label dataset would silently corrupt its outputs.

Must be re-run if the dataset is ever re-preprocessed (nnUNetv2_plan_and_preprocess regenerates
nnUNetPlans.json from scratch, dropping this key).
"""

import argparse
import json
from pathlib import Path

from nnunetv2.paths import nnUNet_preprocessed
from nnunetv2.utilities.dataset_name_id_conversion import maybe_convert_to_dataset_name

from Segmentation.Util.nnunet_util import NNUNET_PLANS

SINGLE_HEAD_LABEL_MANAGER_NAME = "SingleHeadLabelManager"
DEFAULT_LABEL_MANAGER_NAME = "LabelManager"


def patch_single_label_plans(dataset_name_or_id: str) -> None:
    if nnUNet_preprocessed is None:
        raise RuntimeError("nnUNet_preprocessed env var is not set")

    dataset_name = maybe_convert_to_dataset_name(dataset_name_or_id)
    preprocessed_dir = Path(nnUNet_preprocessed) / dataset_name

    dataset_json_path = preprocessed_dir / "dataset.json"
    if not dataset_json_path.is_file():
        raise FileNotFoundError(f"dataset.json not found: {dataset_json_path}")
    with open(dataset_json_path) as f:
        dataset_json = json.load(f)

    foreground_labels = [name for name, label_id in dataset_json["labels"].items()
                         if name != "background" and name != "ignore"]
    if len(foreground_labels) != 1:
        raise ValueError(
            f"{dataset_name}'s dataset.json has {len(foreground_labels)} foreground label(s) "
            f"({foreground_labels}), expected exactly 1. SingleHeadLabelManager only makes sense for "
            f"a single-foreground-label (combined) dataset - use nnUNetTrainerHeatmapMSE (or its "
            f"AdaptiveWing variants) with the default LabelManager for multi-label datasets instead.")

    plans_path = preprocessed_dir / f"{NNUNET_PLANS}.json"
    if not plans_path.is_file():
        raise FileNotFoundError(
            f"{plans_path} not found - run nnUNetv2_plan_and_preprocess first")
    with open(plans_path) as f:
        plans = json.load(f)

    current = plans.get("label_manager")
    if current == SINGLE_HEAD_LABEL_MANAGER_NAME:
        print(
            f"{plans_path} already has label_manager={SINGLE_HEAD_LABEL_MANAGER_NAME}, nothing to do")
        return
    if current is not None and current != DEFAULT_LABEL_MANAGER_NAME:
        raise ValueError(
            f"{plans_path} already has label_manager={current!r}, refusing to overwrite it with "
            f"{SINGLE_HEAD_LABEL_MANAGER_NAME!r}")

    plans["label_manager"] = SINGLE_HEAD_LABEL_MANAGER_NAME
    with open(plans_path, "w") as f:
        json.dump(plans, f, indent=2, sort_keys=False)

    print(f"Set label_manager={SINGLE_HEAD_LABEL_MANAGER_NAME} in {plans_path} "
          f"(foreground label: {foreground_labels[0]!r})")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset-id", required=True,
                        help="nnU-Net dataset ID (e.g. 13) or full dataset name (e.g. Dataset013_...)")
    args = parser.parse_args()

    patch_single_label_plans(args.dataset_id)


if __name__ == "__main__":
    main()
