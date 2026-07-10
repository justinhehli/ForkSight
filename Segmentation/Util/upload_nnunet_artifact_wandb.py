"""Upload an nnUNet trained-model folder to WandB as a reusable artifact.
Run this once per model on the machine where training happened.

Required arguments:
    --dataset
    --trainer

Example usage:
python upload_nnunet_to_wandb.py \
    --dataset Dataset001_Segmentation_v1 \
    --trainer nnUNetTrainerClDiceLoss \

The resulting artifact contains:
    plans.json
    dataset.json
    dataset_fingerprint.json
    fold_0/checkpoint_final.pth
    fold_1/checkpoint_final.pth
    ...
"""

import argparse
import os
from pathlib import Path
import wandb

from Environment.env_utils import load_forksight_env
from Segmentation.Util.nnunet_util import nnunet_folder_name, nnunet_artifact_name, NNUNET_DEFAULT_FOLDS, NNUNET_DEFAULT_CHECKPOINT


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--dataset", type=str, required=True,
        help="nnUNet dataset name, e.g. Dataset001_Segmentation_v1",
    )
    parser.add_argument(
        "--trainer", type=str, required=True,
        help="nnUNet trainer class name, e.g. nnUNetTrainerClDiceLoss",
    )
    args = parser.parse_args()

    load_forksight_env()
    NNUNET_RESULTS_DIR = os.getenv("NNUNET_RESULTS_DIR")
    WANDB_ENTITY = os.getenv("WANDB_ENTITY")
    WANDB_NNUNET_PROJECT = os.getenv("WANDB_NNUNET_PROJECT")

    folds = NNUNET_DEFAULT_FOLDS
    checkpoint = NNUNET_DEFAULT_CHECKPOINT
    artifact_name = nnunet_artifact_name(args.dataset, args.trainer)
    model_dir = Path(NNUNET_RESULTS_DIR) / args.dataset / \
        nnunet_folder_name(args.trainer)

    if not model_dir.is_dir():
        raise FileNotFoundError(f"Model directory not found: {model_dir}")

    # check if required files exist
    plans_path = model_dir / "plans.json"
    dataset_path = model_dir / "dataset.json"
    dataset_fingerprint_path = model_dir / "dataset_fingerprint.json"  # optional
    if not plans_path.is_file() or not dataset_path.is_file():
        raise FileNotFoundError(f"Required files not found in {model_dir}")

    fold_dirs: list[Path] = []
    for f in folds:
        fold_name = f"fold_{f}"
        fold_path = model_dir / fold_name
        ckpt_path = fold_path / checkpoint
        if not ckpt_path.is_file():
            raise FileNotFoundError(
                f"Checkpoint not found: {ckpt_path}. "
                f"Make sure fold '{f}' was trained and the checkpoint exists."
            )
        fold_dirs.append(fold_path)

    run = wandb.init(
        entity=WANDB_ENTITY,
        project=WANDB_NNUNET_PROJECT,
        job_type="upload-nnunet-model",
        name=f"artifact_upload_{artifact_name}",
    )

    artifact = wandb.Artifact(
        name=artifact_name,
        type="model",
        description=f"nnUNet model: {args.dataset} / {args.trainer}",
        metadata={
            "dataset": args.dataset,
            "trainer": args.trainer,
            "source_dir": str(model_dir),
            "folds": folds,
            "checkpoint": checkpoint,
            "framework": "nnUNet",
        },
    )

    artifact.add_file(str(plans_path), name="plans.json")
    artifact.add_file(str(dataset_path), name="dataset.json")
    if dataset_fingerprint_path.is_file():
        artifact.add_file(str(dataset_fingerprint_path),
                          name="dataset_fingerprint.json")

    for fold_path in fold_dirs:
        ckpt_path = fold_path / checkpoint
        artifact_name_in_dir = f"{fold_path.name}/{checkpoint}"
        artifact.add_file(str(ckpt_path), name=artifact_name_in_dir)

    run.log_artifact(artifact)
    print(
        f"\nUploaded artifact '{artifact_name}' with folds {folds} to {WANDB_NNUNET_PROJECT}.")
    run.finish()


if __name__ == "__main__":
    main()
