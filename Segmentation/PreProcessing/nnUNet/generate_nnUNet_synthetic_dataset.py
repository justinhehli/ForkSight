"""Build an nnUNet raw dataset from the synthetic DNA image / mask dataset.

Raw dataset layout (HuggingFace dataset repo jhehli/synthetic-tem-images)::

    train/
        manifest.csv     (split, class, fiber_min, fiber_max, seed, mask, tem)
        tem/{negative,replication_fork,reversed_fork}/<name>.png    (4096x4096)
        masks/{negative,replication_fork,reversed_fork}/<name>.png  (black / red)
    test/
        (same as train)

Only the manifests are read up front. Full images are then randomly sampled per
split, stratified across the fork classes and biased toward higher fiber count
ranges (more realistic), and only the sampled image/mask pairs are downloaded
(to the HuggingFace cache).
Each image is cut into 4x4 patches of 1024x1024. The number of sampled images
is derived from the target patch counts: 350 train patches -> 22 images
(352 patches), 65 test patches -> 4 images (64 patches).

Run from the repo root:
    python -m Segmentation.PreProcessing.nnUNet.generate_nnUNet_synthetic_dataset \\
        --nnunet-raw-dir /path/to/nnUNet_raw --dataset-id 10 \\
        --dataset-name Synthetic_v1 [--verify]
"""
import argparse
import csv
import json
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from huggingface_hub import hf_hub_download
from PIL import Image

from Segmentation.PreProcessing.General.preprocessing_util import init_dir

NNUNET_CASE_PREFIX = "synthetic"

HF_REPO_ID = "jhehli/synthetic-tem-images"
FORK_TYPES = ["negative", "replication_fork", "reversed_fork"]
PATCH_SIZE = 1024
PATCHES_PER_IMAGE = (4096 // PATCH_SIZE) ** 2

N_TRAIN_PATCHES = 350
N_TEST_PATCHES = 65
# sampling weight increases linearly from 1 for the lowest fiber range to
# 1 + FIBER_BIAS for the highest one (0 = uniform)
FIBER_BIAS = 1.0
SEED = 42


def fetch(rel_path: str) -> Path:
    """Download a single file of the raw dataset (path relative to the repo root)
    from HuggingFace, or return it from the cache."""
    return Path(hf_hub_download(
        repo_id=HF_REPO_ID, filename=rel_path, repo_type="dataset"))


def load_manifest(split: str) -> list[dict]:
    with open(fetch(f"{split}/manifest.csv"), newline="") as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        if row["class"] not in FORK_TYPES:
            raise ValueError(f"Unknown class '{row['class']}' in {split} manifest")
        row["fiber_min"] = int(row["fiber_min"])
        row["fiber_max"] = int(row["fiber_max"])
    return rows


def fiber_weights(rows: list[dict]) -> dict[int, float]:
    """Sampling weight per fiber range (keyed by fiber_min), increasing linearly
    from 1 for the lowest range to 1 + FIBER_BIAS for the highest."""
    bins = sorted({row["fiber_min"] for row in rows})
    if len(bins) == 1:
        return {bins[0]: 1.0}
    return {b: 1.0 + FIBER_BIAS * i / (len(bins) - 1) for i, b in enumerate(bins)}


def weighted_sample(rows: list[dict], k: int, weights: dict[int, float],
                    rng: random.Random) -> list[dict]:
    """Weighted sampling without replacement (Efraimidis-Spirakis)."""
    keyed = [(rng.random() ** (1.0 / weights[row["fiber_min"]]), row)
             for row in rows]
    keyed.sort(key=lambda x: x[0], reverse=True)
    return [row for _, row in keyed[:k]]


def sample_images(
    rows: list[dict],
    n_images: int,
    rng: random.Random,
) -> list[dict]:
    """Randomly sample n_images manifest rows, spread as evenly as possible across
    fork classes and biased toward higher fiber ranges. If a class runs out of
    images, the rest is filled from the others."""
    if n_images > len(rows):
        raise ValueError(
            f"Requested {n_images} images but only {len(rows)} are available")

    by_class = {t: [row for row in rows if row["class"] == t] for t in FORK_TYPES}
    counts = {t: 0 for t in FORK_TYPES}
    remaining = n_images
    while remaining > 0:
        available = [t for t in FORK_TYPES if counts[t] < len(by_class[t])]
        # hand out one image per available class per round, in random order
        rng.shuffle(available)
        for t in available[:remaining]:
            counts[t] += 1
            remaining -= 1

    weights = fiber_weights(rows)
    sampled = []
    for t in FORK_TYPES:
        sampled += sorted(weighted_sample(by_class[t], counts[t], weights, rng),
                          key=lambda row: row["tem"])
    return sampled


def mask_to_label(mask_path: Path) -> np.ndarray:
    """Convert a black / red mask to a 0/1 label map (red = DNA)."""
    mask = np.array(Image.open(mask_path).convert("RGB"))
    return (mask[..., 0] > 127).astype(np.uint8)


def save_cases(
    samples: list[dict],
    images_out_dir: Path,
    labels_out_dir: Path,
    start_idx: int,
    split: str,
    mapping: list[dict],
) -> int:
    """Fetch each sampled image/mask pair, cut it into 1024x1024 patches and save
    them as nnUNet cases; returns the next available case index. Appends one row
    per saved case to mapping."""
    case_idx = start_idx
    for row in samples:
        img_path = fetch(f"{split}/{row['tem']}")
        mask_path = fetch(f"{split}/{row['mask']}")

        img = np.array(Image.open(img_path).convert("L"))
        label = mask_to_label(mask_path)
        if img.shape != label.shape:
            raise ValueError(
                f"Image / mask shape mismatch for {row['tem']}: {img.shape} vs {label.shape}")
        if img.shape[0] % PATCH_SIZE or img.shape[1] % PATCH_SIZE:
            raise ValueError(
                f"Image {row['tem']} of shape {img.shape} is not divisible into {PATCH_SIZE}px patches")

        # row-major patch order, same as create_patches_from_img
        n_rows, n_cols = img.shape[0] // PATCH_SIZE, img.shape[1] // PATCH_SIZE
        for r in range(n_rows):
            for c in range(n_cols):
                patch_idx = r * n_cols + c
                ys = slice(r * PATCH_SIZE, (r + 1) * PATCH_SIZE)
                xs = slice(c * PATCH_SIZE, (c + 1) * PATCH_SIZE)

                case_name = f"{NNUNET_CASE_PREFIX}_{case_idx:04d}"
                Image.fromarray(img[ys, xs]).save(
                    images_out_dir / f"{case_name}_0000.png")
                Image.fromarray(label[ys, xs]).save(
                    labels_out_dir / f"{case_name}.png")

                mapping.append({
                    "original_filename": Path(row["tem"]).name,
                    "fork_type": row["class"],
                    "fiber_min": row["fiber_min"],
                    "fiber_max": row["fiber_max"],
                    "patch": patch_idx,
                    "nnunet_case": case_name,
                    "split": split,
                })
                case_idx += 1

        print(f"Saved {n_rows * n_cols} patches of {row['tem']} "
              f"({split}), up to case {case_idx - 1:04d}")

    return case_idx


def inspect_masks(dataset_dir: Path):
    """Plot 9 random image/mask pairs from the generated dataset to verify mask
    quality; saved as verify_samples.png in the dataset dir."""
    images_tr_dir = dataset_dir / "imagesTr"
    labels_tr_dir = dataset_dir / "labelsTr"

    all_cases = sorted(labels_tr_dir.glob("*.png"))

    sample = random.sample(all_cases, min(9, len(all_cases)))
    cols = 3
    rows = (len(sample) + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols * 2, figsize=(cols * 2 * 4, rows * 4))
    axes = axes.flatten()

    for i, mask_path in enumerate(sample):
        case_name = mask_path.stem
        img_path = images_tr_dir / f"{case_name}_0000.png"

        img = np.array(Image.open(img_path))
        mask = np.array(Image.open(mask_path))  # values 0 or 1

        ax_img = axes[i * 2]
        ax_mask = axes[i * 2 + 1]

        ax_img.imshow(img, cmap="gray")
        ax_img.set_title(case_name, fontsize=7)
        ax_img.axis("off")

        ax_mask.imshow(mask, cmap="hot", vmin=0, vmax=1)
        ax_mask.set_title(f"{case_name} (mask)", fontsize=7)
        ax_mask.axis("off")

    for ax in axes[len(sample) * 2:]:
        ax.axis("off")

    plt.tight_layout()
    out_path = dataset_dir / "verify_samples.png"
    fig.savefig(out_path, dpi=100)
    plt.close(fig)
    print(f"Verification plot written to {out_path}")


def generate_dataset(dataset_dir: Path):
    rng = random.Random(SEED)

    n_images = {
        "train": round(N_TRAIN_PATCHES / PATCHES_PER_IMAGE),
        "test": round(N_TEST_PATCHES / PATCHES_PER_IMAGE),
    }
    samples = {}
    for split in ["train", "test"]:
        rows = load_manifest(split)
        samples[split] = sample_images(rows, n_images[split], rng)

        per_class = {t: sum(1 for row in samples[split] if row["class"] == t)
                     for t in FORK_TYPES}
        per_fiber = {}
        for row in sorted(samples[split], key=lambda row: row["fiber_min"]):
            key = f"{row['fiber_min']}-{row['fiber_max']}"
            per_fiber[key] = per_fiber.get(key, 0) + 1
        print(f"Sampled {len(samples[split])} {split} images "
              f"(of {len(rows)}): {per_class}, fibers: {per_fiber}")

    images_tr_dir = dataset_dir / "imagesTr"
    labels_tr_dir = dataset_dir / "labelsTr"
    images_ts_dir = dataset_dir / "imagesTs"
    labels_ts_dir = dataset_dir / "labelsTs"

    init_dir(dataset_dir)
    for d in [images_tr_dir, labels_tr_dir, images_ts_dir, labels_ts_dir]:
        d.mkdir(parents=True)

    mapping: list[dict] = []

    print(f"\nProcessing {len(samples['train'])} training images")
    case_idx = save_cases(
        samples["train"], images_tr_dir, labels_tr_dir,
        start_idx=0, split="train", mapping=mapping,
    )
    num_training = case_idx

    print(f"\nProcessing {len(samples['test'])} test images")
    case_idx = save_cases(
        samples["test"], images_ts_dir, labels_ts_dir,
        start_idx=case_idx, split="test", mapping=mapping,
    )
    num_test = case_idx - num_training

    dataset_json = {
        "channel_names": {"0": "grayscale"},
        "labels": {"background": 0, "DNA": 1},
        "numTraining": num_training,
        "file_ending": ".png",
    }
    with open(dataset_dir / "dataset.json", "w") as f:
        json.dump(dataset_json, f, indent=4)

    mapping_path = dataset_dir / "case_mapping.json"
    with open(mapping_path, "w") as f:
        json.dump(mapping, f, indent=4)

    print(
        f"\nDataset '{dataset_dir.name}' created with {num_training} training cases "
        f"and {num_test} test cases."
    )
    print(f"dataset.json written to {dataset_dir / 'dataset.json'}")
    print(f"case_mapping.json written to {mapping_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--nnunet-raw-dir", type=Path, required=True,
                        help="nnUNet_raw directory to create the dataset in")
    parser.add_argument("--dataset-id", type=int, required=True,
                        help="nnUNet dataset ID")
    parser.add_argument("--dataset-name", default="Synthetic",
                        help="nnUNet dataset name suffix (default 'Synthetic')")
    parser.add_argument("--verify", action="store_true",
                        help="Save a plot of random image/mask pairs of the generated dataset")
    args = parser.parse_args()

    if not args.nnunet_raw_dir.is_dir():
        raise ValueError(
            f"--nnunet-raw-dir ({args.nnunet_raw_dir}) must be a valid directory")

    dataset_dir = args.nnunet_raw_dir / \
        f"Dataset{args.dataset_id:03d}_{args.dataset_name}"

    generate_dataset(dataset_dir)
    if args.verify:
        inspect_masks(dataset_dir)
