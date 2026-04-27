"""Train an nnUNet segmentation model on a synthetic dataset.

Expected synthetic dataset layout::

    <dataset_dir>/
        images/<stem>.png          (input images, 8-bit grayscale)
        masks/<stem>.png           (binary GT masks, same filenames)
        annotations.csv            (optional, needed only for evaluation;
                                    columns: image, x, y, label)

The script:
  1. Selects a train / test split (from --split-json, or random with
     --test-ratio).  The split is saved back to ``<dataset_dir>/split.json``
     so the evaluation script can pick it up.
  2. Converts the train+test images into nnUNet's expected layout
     (imagesTr, labelsTr, imagesTs, labelsTs, dataset.json) under
     ``<nnunet-root>/nnUNet_raw/Dataset<ID>_<NAME>``.
  3. Runs ``nnUNetv2_plan_and_preprocess`` and ``nnUNetv2_train``.

After training, run ``evaluate_synthetic.py`` with the same arguments
(``--dataset-dir``, ``--nnunet-root``, ``--dataset-id``, ``--trainer``,
``--fold``) to run the junction-detection pipeline on the test split.

Example
-------
    python SyntheticPipeline/train_nnunet.py \\
        --dataset-dir /scratch/me/synth_v1 \\
        --nnunet-root /scratch/me/synth_v1/nnunet \\
        --dataset-id 900 --dataset-name SynthV1 \\
        --test-ratio 0.2 --fold 0 --trainer nnUNetTrainer
"""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--dataset-dir", required=True, type=Path,
        help="Root of the synthetic dataset (with images/ and masks/).")
    parser.add_argument(
        "--nnunet-root", required=True, type=Path,
        help="Root dir under which nnUNet_raw/preprocessed/results will live.")
    parser.add_argument(
        "--dataset-id", type=int, default=900,
        help="nnUNet dataset ID (default 900).")
    parser.add_argument(
        "--dataset-name", default="Synthetic",
        help="nnUNet dataset name suffix (default 'Synthetic').")
    parser.add_argument(
        "--split-json", type=Path, default=None,
        help="Optional: JSON file with {'train': [...], 'test': [...]} lists "
             "of image filenames. If not given, a random split is drawn.")
    parser.add_argument(
        "--test-ratio", type=float, default=0.2,
        help="Fraction of images in the test split when --split-json is "
             "not given (default 0.2).")
    parser.add_argument(
        "--seed", type=int, default=42,
        help="RNG seed for the random split (default 42).")
    parser.add_argument(
        "--fold", type=int, default=0,
        help="nnUNet fold to train (default 0).")
    parser.add_argument(
        "--trainer", default="nnUNetTrainer",
        help="nnUNet trainer class (default nnUNetTrainer).")
    parser.add_argument(
        "--configuration", default="2d",
        help="nnUNet configuration (default 2d).")
    parser.add_argument(
        "--skip-preprocess", action="store_true",
        help="Skip nnUNetv2_plan_and_preprocess (e.g. if rerunning).")
    parser.add_argument(
        "--skip-train", action="store_true",
        help="Skip training (only build the nnUNet dataset).")
    return parser.parse_args()


def _setup_nnunet_env(nnunet_root: Path) -> tuple[Path, Path, Path]:
    """Create nnUNet directory layout and export env vars."""
    raw = nnunet_root / "nnUNet_raw"
    pre = nnunet_root / "nnUNet_preprocessed"
    res = nnunet_root / "nnUNet_results"
    for d in [raw, pre, res]:
        d.mkdir(parents=True, exist_ok=True)
    os.environ["nnUNet_raw"] = str(raw)
    os.environ["nnUNet_preprocessed"] = str(pre)
    os.environ["nnUNet_results"] = str(res)
    return raw, pre, res


def _derive_split(
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
        return (
            [by_name[n] for n in split["train"]],
            [by_name[n] for n in split["test"]],
        )

    rng = random.Random(seed)
    shuffled = list(image_files)
    rng.shuffle(shuffled)
    n_test = max(1, int(round(len(shuffled) * test_ratio)))
    test_files = shuffled[:n_test]
    train_files = shuffled[n_test:]
    return train_files, test_files


def _write_case(
    img_path: Path,
    mask_path: Path,
    images_out: Path,
    labels_out: Path,
    case_name: str,
) -> None:
    """Copy image (as _0000.png) and save binary mask under case_name.png."""
    # Keep images as the original PNG (nnUNet accepts 8-bit grayscale)
    shutil.copy(img_path, images_out / f"{case_name}_0000.png")

    mask_arr = np.array(Image.open(mask_path))
    if mask_arr.ndim == 3:
        mask_arr = mask_arr[..., 0]
    mask_bin = (mask_arr > 0).astype(np.uint8)  # nnUNet expects 0/1 labels
    Image.fromarray(mask_bin).save(labels_out / f"{case_name}.png")


def _build_nnunet_dataset(
    train_paths: list[Path],
    test_paths: list[Path],
    masks_dir: Path,
    dataset_out: Path,
    case_prefix: str,
) -> tuple[int, int]:
    if dataset_out.exists():
        print(f"Removing existing dataset dir: {dataset_out}")
        shutil.rmtree(dataset_out)

    images_tr = dataset_out / "imagesTr"
    labels_tr = dataset_out / "labelsTr"
    images_ts = dataset_out / "imagesTs"
    labels_ts = dataset_out / "labelsTs"
    for d in [images_tr, labels_tr, images_ts, labels_ts]:
        d.mkdir(parents=True)

    idx = 0
    for p in train_paths:
        mask_path = masks_dir / p.name
        if not mask_path.exists():
            raise SystemExit(f"Missing mask for training image: {mask_path}")
        _write_case(p, mask_path, images_tr, labels_tr, f"{case_prefix}_{idx:04d}")
        idx += 1
    n_train = idx

    for p in test_paths:
        mask_path = masks_dir / p.name
        if not mask_path.exists():
            raise SystemExit(f"Missing mask for test image: {mask_path}")
        _write_case(p, mask_path, images_ts, labels_ts, f"{case_prefix}_{idx:04d}")
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


def _run_cmd(cmd: list[str]) -> None:
    print(f"\n$ {' '.join(cmd)}")
    subprocess.run(cmd, check=True, env=os.environ)


def main() -> None:
    args = _parse_args()

    images_dir = args.dataset_dir / "images"
    masks_dir = args.dataset_dir / "masks"
    if not images_dir.is_dir():
        raise SystemExit(f"Missing images directory: {images_dir}")
    if not masks_dir.is_dir():
        raise SystemExit(f"Missing masks directory: {masks_dir}")

    raw, pre, res = _setup_nnunet_env(args.nnunet_root)
    print(f"nnUNet_raw:          {raw}")
    print(f"nnUNet_preprocessed: {pre}")
    print(f"nnUNet_results:      {res}")

    train, test = _derive_split(
        images_dir, args.split_json, args.test_ratio, args.seed)
    print(f"\nTrain: {len(train)}  |  Test: {len(test)}")

    split_out = args.dataset_dir / "split.json"
    with open(split_out, "w") as f:
        json.dump({
            "train": [p.name for p in train],
            "test": [p.name for p in test],
        }, f, indent=2)
    print(f"Wrote split to {split_out}")

    dataset_dir_name = f"Dataset{args.dataset_id:03d}_{args.dataset_name}"
    dataset_out = raw / dataset_dir_name
    n_train, n_test = _build_nnunet_dataset(
        train, test, masks_dir, dataset_out, case_prefix="synth")
    print(f"\nBuilt nnUNet dataset at {dataset_out}")
    print(f"  {n_train} training cases, {n_test} test cases")

    if args.skip_train:
        print("\n--skip-train set; stopping after dataset build.")
        return

    if not args.skip_preprocess:
        _run_cmd([
            "nnUNetv2_plan_and_preprocess",
            "-d", str(args.dataset_id),
            "--verify_dataset_integrity",
        ])

    _run_cmd([
        "nnUNetv2_train",
        str(args.dataset_id),
        args.configuration,
        str(args.fold),
        "-tr", args.trainer,
    ])

    model_dir = (res / dataset_dir_name
                 / f"{args.trainer}__nnUNetPlans__{args.configuration}")
    print(f"\nTraining complete.")
    print(f"  Model directory: {model_dir}")
    print(f"\nNext step: run evaluate_synthetic.py with the same arguments.")
    print(
        f"  python SyntheticPipeline/evaluate_synthetic.py \\\n"
        f"      --dataset-dir {args.dataset_dir} \\\n"
        f"      --nnunet-root {args.nnunet_root} \\\n"
        f"      --dataset-id {args.dataset_id} \\\n"
        f"      --dataset-name {args.dataset_name} \\\n"
        f"      --trainer {args.trainer} \\\n"
        f"      --fold {args.fold}"
    )


if __name__ == "__main__":
    main()
