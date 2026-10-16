"""Utilities for loading nnUNet models and running patch-level inference on 2D images.

nnUNetPredictor.initialize_from_trained_model_folder needs a directory with:
    plans.json
    dataset.json
    fold_*/checkpoint_*.pth
these files are all included in the artifact uploaded by upload_nnunet_artifact_wandb.py

Naming conventions:
    nnUNet folder:   {trainer}__nnUNetPlans__2d
    WandB artifact:  nnunet-{dataset}-{trainer}
    metrics CSV key: nnunet/{dataset}/{trainer}
"""

from __future__ import annotations
from pathlib import Path
from typing import Sequence, TYPE_CHECKING
import numpy as np
import torch
from PIL import Image

from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor


NNUNET_CONFIGURATION = "2d"
NNUNET_PLANS = "nnUNetPlans"
NNUNET_DEFAULT_FOLDS = (0, 1, 2, 3, 4)
NNUNET_DEFAULT_CHECKPOINT = "checkpoint_final.pth"


def nnunet_folder_name(trainer: str) -> str:
    return f"{trainer}__{NNUNET_PLANS}__{NNUNET_CONFIGURATION}"


def nnunet_artifact_name(dataset: str, trainer: str) -> str:
    return f"nnunet-{dataset}-{trainer}"


def nnunet_model_key(dataset: str, trainer: str) -> str:
    return f"nnunet/{dataset}/{trainer}"


import torch
import nnunetv2
from pathlib import Path


def get_trainer_name_from_checkpoint(checkpoint_path) -> str:
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    return ckpt["trainer_name"]


def ensure_trainer_class_exists(trainer_name: str):
    trainer_dir = Path(nnunetv2.__path__[0]) / "training" / "nnUNetTrainer"
    target_file = trainer_dir / f"{trainer_name}.py"

    if target_file.exists() or trainer_name == "nnUNetTrainer":
        return None

    source_file = trainer_dir / "nnUNetTrainer.py"
    source_code = source_file.read_text()

    new_code = source_code.replace(
        "class nnUNetTrainer(", f"class {trainer_name}(", 1)

    target_file.write_text(new_code)
    return target_file


def ensure_trainer_for_model(model_dir, check_fold=0):
    """
    Ensures that a trainer file/class matching the one that nnU-Net trained with exists.If 
    it doesn't (which is the case if nnunetv2 was installed as a normal package and the
    model was trained with custom trainer class), we simply copy the default nnUNetTrainer
    class file under the custom trainer class name (in our case, this works since our custom
    trainer classes don't change any network architecture, only wandb logging and loss functions).
    """
    model_dir = Path(model_dir)
    ckpt_path = next(
        (model_dir / f"fold_{check_fold}").glob("checkpoint_*.pth"))
    trainer_name = get_trainer_name_from_checkpoint(ckpt_path)
    trainer_file_path = ensure_trainer_class_exists(trainer_name)
    return trainer_name, trainer_file_path


def initialize_nnunet_predictor(
    model_dir: Path,
    device: torch.device,
    folds: Sequence[int] = NNUNET_DEFAULT_FOLDS,
    checkpoint: str = NNUNET_DEFAULT_CHECKPOINT,
    ensure_custom_trainer_file=False
) -> tuple["nnUNetPredictor", Path | None]:
    """Initialize an nnUNetPredictor from a local model directory.

    model_dir must contain plans.json, dataset.json, and fold_*/checkpoint_*.pth,
    as produced by download_nnunet_artifact().
    """
    predictor = nnUNetPredictor(
        tile_step_size=0.5,
        use_gaussian=True,
        use_mirroring=True,
        perform_everything_on_device=True,
        device=device,
        verbose=False,
        verbose_preprocessing=False,
        allow_tqdm=False,
    )

    custom_trainer_file_path = None
    if ensure_custom_trainer_file:
        _, custom_trainer_file_path = ensure_trainer_for_model(model_dir)

    predictor.initialize_from_trained_model_folder(
        str(model_dir),
        use_folds=tuple(folds),
        checkpoint_name=checkpoint,
    )

    return predictor, custom_trainer_file_path


def convert_labelmap_pngs_to_binary(dir_path: Path) -> None:
    """Convert nnUNet's class-label PNGs (0/1) in-place to 0/255 PNGs"""
    for png in Path(dir_path).glob("*.png"):
        arr = np.array(Image.open(png))
        Image.fromarray((arr * 255).astype(np.uint8)).save(png)


def run_nnunet_predict_from_patches(
    predictor,
    patch_file_lists: list[list[str]],
    output_dir: Path,
    save_probabilities: bool = False,
) -> None:
    """Run nnUNet inference on pre-saved patch PNGs and write (0/255)
    prediction PNGs to output_dir, named after the input patch files.

    patch_file_lists : one-item-per-channel file lists, as required by
                        nnUNetPredictor.predict_from_files (one entry per patch).
    save_probabilities : if True, additionally write a per-patch .npz (softmax
                        probabilities, under the "probabilities" key, shape
                        (num_classes, H, W)) and .pkl (properties) file
                        alongside each PNG.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    predictor.predict_from_files(
        patch_file_lists,
        str(output_dir),
        save_probabilities=save_probabilities,
        overwrite=True,
        num_processes_preprocessing=2,
        num_processes_segmentation_export=2,
    )
    convert_labelmap_pngs_to_binary(output_dir)
