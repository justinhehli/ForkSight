"""Save SAM segmentation patch predictions for junction detection evaluation.

Run this script in the SAM conda environment.  For each configured SAM model
and each test image, it runs patch-level inference and saves the 16 predicted
binary mask patches as PNGs alongside a metadata.json file.  The outputs can
then be consumed by compute_metrics_junction_detection.py in any environment.

Output layout under JUNCTION_PRED_DIR:
  <JUNCTION_PRED_DIR>/
    <safe_model_key>/
      metadata.json          {"model_key": str, "dataset": str}
      <image_stem>_patch_00.png
      ...
      <image_stem>_patch_15.png

Required environment variables (loaded via load_forksight_env):
  JUNCTION_DETECTION_DATASET_DIR   root of the junction detection dataset
  JUNCTION_PRED_DIR                output directory for predictions
  WANDB_ENTITY                     WandB entity
  WANDB_SAM_PROJECT                WandB SAM project name

Usage:
  python infer_patches_junction_sam.py [--device N] [--batch-size N]
                                       [--is-test] [--force-rerun]
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
import wandb
from PIL import Image

import Environment.env_utils as env_utils
from Segmentation.SAM.sam_lora_util import (
    get_params_from_artifact,
    initialize_sam_lora_with_params,
)
from Evaluation.pipeline_evaluation_shared import (
    load_full_image_as_patches,
    predict_patches_batched,
)
from Segmentation.Util.patch_grid_util import pred_patch_filename
from Evaluation.compute_metrics_config import (
    SAM_MODELS_RUNS,
    SAM_PARAMS_ARTIFACT_SUFFIX,
)


def _save_pred_patches(
    patches: torch.Tensor,
    model_pred_dir: Path,
    image_stem: str,
) -> None:
    """Save (N, 1, H, W) binary float32 tensor as per-patch PNGs (0/255)."""
    for idx in range(patches.shape[0]):
        arr = (patches[idx, 0].cpu().numpy() * 255).astype(np.uint8)
        Image.fromarray(arr).save(
            model_pred_dir / pred_patch_filename(image_stem, idx))


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--device", type=int, default=0,
                        help="CUDA device index (default: 0)")
    parser.add_argument("--batch-size", type=int, default=4,
                        help="Patches per SAM forward pass (default: 4)")
    parser.add_argument("--is-test", action="store_true",
                        help="Only infer the first test image per model")
    parser.add_argument("--force-rerun", action="store_true",
                        help="Re-run inference even if output already exists")
    args = parser.parse_args()

    env_utils.load_forksight_env()

    JUNCTION_DETECTION_DATASET_DIR = os.getenv(
        "JUNCTION_DETECTION_DATASET_DIR")
    JUNCTION_PRED_DIR = os.getenv("JUNCTION_PRED_DIR")
    WANDB_ENTITY = os.getenv("WANDB_ENTITY", "EM_IMCR_BIOVSION")
    WANDB_SAM_PROJECT = os.getenv("WANDB_SAM_PROJECT", "ForkSight-SAM")

    if JUNCTION_DETECTION_DATASET_DIR is None:
        raise ValueError(
            "JUNCTION_DETECTION_DATASET_DIR environment variable must be set.")
    if JUNCTION_PRED_DIR is None:
        raise ValueError("JUNCTION_PRED_DIR environment variable must be set.")

    test_images_dir = (Path(JUNCTION_DETECTION_DATASET_DIR) / "images")
    if not test_images_dir.is_dir():
        raise FileNotFoundError(
            f"Test images directory not found: {test_images_dir}")

    test_image_paths = sorted(p for p in test_images_dir.glob("*.png"))
    if not test_image_paths:
        raise FileNotFoundError(
            f"No PNG files found in {test_images_dir}")

    device = torch.device(
        f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Found {len(test_image_paths)} test image(s).")

    pred_base = Path(JUNCTION_PRED_DIR)
    pred_base.mkdir(parents=True, exist_ok=True)

    api = wandb.Api()
    all_runs = list(api.runs(f"{WANDB_ENTITY}/{WANDB_SAM_PROJECT}"))
    runs_to_run = [
        r for r in all_runs
        if r.state == "finished" and r.name in SAM_MODELS_RUNS
    ]

    if not runs_to_run:
        print("No matching SAM runs found.")
        return

    print(f"Will process {len(runs_to_run)} SAM model(s): "
          + ", ".join(r.name for r in runs_to_run))

    for run in runs_to_run:
        safe_name = run.name.replace("/", "_")
        model_pred_dir = pred_base / safe_name
        metadata_path = model_pred_dir / "metadata.json"

        if metadata_path.is_file() and not args.force_rerun:
            print(f"\n  Already inferred: {run.name} — skipping. "
                  f"(use --force-rerun to override)")
            continue

        print(f"\n{'='*60}")
        print(f"Inferring SAM: {run.name}")
        print(f"{'='*60}")

        run_artifacts = [a for a in run.logged_artifacts()
                         if a.type == "model"]
        artifact = next(
            (a for a in run_artifacts
             if a.name.endswith(SAM_PARAMS_ARTIFACT_SUFFIX)),
            None,
        )
        if artifact is None:
            print(f"  No artifact ending with '{SAM_PARAMS_ARTIFACT_SUFFIX}', "
                  f"skipping.")
            continue
        print(f"  Artifact: {artifact.name}")

        params, _ = get_params_from_artifact(artifact, device)
        model = initialize_sam_lora_with_params(run.config, params, device)
        model.eval()

        model_pred_dir.mkdir(parents=True, exist_ok=True)
        metadata_path.write_text(json.dumps({
            "model_key": run.name,
            "dataset": run.config.get("dataset", ""),
        }))

        for img_path in test_image_paths:
            print(f"  {img_path.name}")
            patches, _ = load_full_image_as_patches(img_path)
            mask_patches, _ = predict_patches_batched(
                model, patches, device, args.batch_size)
            _save_pred_patches(mask_patches, model_pred_dir, img_path.stem)
            if args.is_test:
                break

        del model, params
        torch.cuda.empty_cache()

    print("\nDone.")


if __name__ == "__main__":
    main()
