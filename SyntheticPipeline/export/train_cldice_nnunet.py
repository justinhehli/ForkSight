"""Train an nnUNet-v2 segmentation model on a synthetic dataset with CL-Dice loss.

This script is self-contained: it has no dependencies on other files in this
repo.  It expects a dataset laid out as::

    <dataset_dir>/
        images/<stem>.png          (input images, 8-bit grayscale)
        masks/<stem>.png           (binary GT masks, same filenames)
        annotations.csv            (optional, consumed by evaluation)

Steps:
  1. Build a train/test split (random, reproducible via --seed, or loaded from
     --split-json).  The split is written back to <dataset_dir>/split.json so
     the evaluation script can pick it up.
  2. Convert images+masks into nnUNet's expected layout under
     <nnunet_root>/nnUNet_raw/Dataset<ID>_<NAME>/.
  3. Run nnUNetv2_plan_and_preprocess and nnUNetv2_train with a CL-Dice trainer.

For CL-Dice you need an nnUNet trainer class that implements the CL-Dice loss.
Pass its class name via --trainer (default nnUNetTrainerCLDice).  Two common
installable options:
  - https://github.com/jocpae/clDice  (reference implementation)
  - any local trainer placed under nnunetv2.training.nnUNetTrainer.variants

Python requirements:
    pip install nnunetv2 numpy pillow
External CLI requirements:
    nnUNetv2_plan_and_preprocess
    nnUNetv2_train
"""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset-dir", required=True, type=Path,
                   help="Root of the synthetic dataset (contains images/ and masks/).")
    p.add_argument("--nnunet-root", required=True, type=Path,
                   help="Root dir under which nnUNet_raw/preprocessed/results live.")
    p.add_argument("--dataset-id", type=int, default=900,
                   help="nnUNet dataset ID (default 900).")
    p.add_argument("--dataset-name", default="Synthetic",
                   help="nnUNet dataset name suffix (default 'Synthetic').")
    p.add_argument("--split-json", type=Path, default=None,
                   help="Optional JSON file with {'train': [...], 'test': [...]} "
                        "image filenames.  If omitted, a random split is drawn.")
    p.add_argument("--test-ratio", type=float, default=0.2,
                   help="Test fraction when --split-json is not given (default 0.2).")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--fold", type=int, default=0,
                   help="nnUNet fold to train (default 0).")
    p.add_argument("--trainer", default="nnUNetTrainerCLDice",
                   help="nnUNet trainer class implementing CL-Dice loss.")
    p.add_argument("--configuration", default="2d")
    p.add_argument("--skip-preprocess", action="store_true")
    p.add_argument("--skip-train", action="store_true")
    return p.parse_args()


def setup_nnunet_env(nnunet_root: Path) -> tuple[Path, Path, Path]:
    raw = nnunet_root / "nnUNet_raw"
    pre = nnunet_root / "nnUNet_preprocessed"
    res = nnunet_root / "nnUNet_results"
    for d in (raw, pre, res):
        d.mkdir(parents=True, exist_ok=True)
    os.environ["nnUNet_raw"] = str(raw)
    os.environ["nnUNet_preprocessed"] = str(pre)
    os.environ["nnUNet_results"] = str(res)
    return raw, pre, res


def derive_split(
    images_dir: Path,
    split_json: Path | None,
    test_ratio: float,
    seed: int,
) -> tuple[list[Path], list[Path]]:
    image_files = sorted(images_dir.glob("*.png"))
    if not image_files:
        raise SystemExit(f"No PNG images found in {images_dir}")

    if split_json is not None:
        with open(split_json) as f:
            split = json.load(f)
        by_name = {p.name: p for p in image_files}
        missing_tr = [n for n in split["train"] if n not in by_name]
        missing_te = [n for n in split["test"] if n not in by_name]
        if missing_tr or missing_te:
            raise SystemExit(
                f"Split file references missing images: "
                f"train={missing_tr[:5]}... test={missing_te[:5]}...")
        return ([by_name[n] for n in split["train"]],
                [by_name[n] for n in split["test"]])

    rng = random.Random(seed)
    shuffled = list(image_files)
    rng.shuffle(shuffled)
    n_test = max(1, int(round(len(shuffled) * test_ratio)))
    return shuffled[n_test:], shuffled[:n_test]


def write_case(
    img_path: Path,
    mask_path: Path,
    images_out: Path,
    labels_out: Path,
    case_name: str,
) -> None:
    shutil.copy(img_path, images_out / f"{case_name}_0000.png")
    mask_arr = np.array(Image.open(mask_path))
    if mask_arr.ndim == 3:
        mask_arr = mask_arr[..., 0]
    mask_bin = (mask_arr > 0).astype(np.uint8)
    Image.fromarray(mask_bin).save(labels_out / f"{case_name}.png")


def build_nnunet_dataset(
    train_paths: list[Path],
    test_paths: list[Path],
    masks_dir: Path,
    dataset_out: Path,
    case_prefix: str = "synth",
) -> tuple[int, int]:
    if dataset_out.exists():
        print(f"Removing existing dataset dir: {dataset_out}")
        shutil.rmtree(dataset_out)

    images_tr = dataset_out / "imagesTr"
    labels_tr = dataset_out / "labelsTr"
    images_ts = dataset_out / "imagesTs"
    labels_ts = dataset_out / "labelsTs"
    for d in (images_tr, labels_tr, images_ts, labels_ts):
        d.mkdir(parents=True)

    idx = 0
    for p in train_paths:
        mask_path = masks_dir / p.name
        if not mask_path.exists():
            raise SystemExit(f"Missing mask for training image: {mask_path}")
        write_case(p, mask_path, images_tr, labels_tr, f"{case_prefix}_{idx:04d}")
        idx += 1
    n_train = idx

    for p in test_paths:
        mask_path = masks_dir / p.name
        if not mask_path.exists():
            raise SystemExit(f"Missing mask for test image: {mask_path}")
        write_case(p, mask_path, images_ts, labels_ts, f"{case_prefix}_{idx:04d}")
        idx += 1
    n_test = idx - n_train

    dataset_json = {
        "channel_names": {"0": "grayscale"},
        "labels": {"background": 0, "foreground": 1},
        "numTraining": n_train,
        "file_ending": ".png",
    }
    with open(dataset_out / "dataset.json", "w") as f:
        json.dump(dataset_json, f, indent=2)
    return n_train, n_test


def run_cmd(cmd: list[str]) -> None:
    print(f"\n$ {' '.join(cmd)}")
    subprocess.run(cmd, check=True, env=os.environ)


def main() -> None:
    args = parse_args()

    images_dir = args.dataset_dir / "images"
    masks_dir = args.dataset_dir / "masks"
    if not images_dir.is_dir():
        raise SystemExit(f"Missing images directory: {images_dir}")
    if not masks_dir.is_dir():
        raise SystemExit(f"Missing masks directory: {masks_dir}")

    raw, pre, res = setup_nnunet_env(args.nnunet_root)
    print(f"nnUNet_raw:          {raw}")
    print(f"nnUNet_preprocessed: {pre}")
    print(f"nnUNet_results:      {res}")

    train, test = derive_split(images_dir, args.split_json, args.test_ratio, args.seed)
    print(f"\nTrain: {len(train)}  |  Test: {len(test)}")

    split_out = args.dataset_dir / "split.json"
    with open(split_out, "w") as f:
        json.dump({"train": [p.name for p in train],
                   "test": [p.name for p in test]}, f, indent=2)
    print(f"Wrote split to {split_out}")

    dataset_dir_name = f"Dataset{args.dataset_id:03d}_{args.dataset_name}"
    dataset_out = raw / dataset_dir_name
    n_tr, n_te = build_nnunet_dataset(train, test, masks_dir, dataset_out)
    print(f"\nBuilt nnUNet dataset at {dataset_out}: {n_tr} train / {n_te} test.")

    if args.skip_train:
        print("--skip-train set; stopping after dataset build.")
        return

    if not args.skip_preprocess:
        run_cmd(["nnUNetv2_plan_and_preprocess",
                 "-d", str(args.dataset_id),
                 "--verify_dataset_integrity"])

    run_cmd(["nnUNetv2_train",
             str(args.dataset_id),
             args.configuration,
             str(args.fold),
             "-tr", args.trainer])

    model_dir = (res / dataset_dir_name
                 / f"{args.trainer}__nnUNetPlans__{args.configuration}")
    print(f"\nTraining complete. Model dir: {model_dir}")


if __name__ == "__main__":
    main()
