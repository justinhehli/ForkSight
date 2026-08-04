"""Segmentation-stage worker for the annotation tool pipeline

Runs as its ow subprocess, separate from both the FastAPI backend and the
detection-stage worker. The trained model is assumed to be present on disk
at NNUNET_MODEL_DIR (AnnotationTool/.pipeline_env), containing plans.json,
dataset.json, and fold_*/checkpoint_*.pth.

Usage (must be invoked with the pipeline venv's python, from the repo root):
    python -m AnnotationTool.backend.pipeline.segmentation_worker \\
        --project-dir <base folder> \\
        --manifest <manifest.json listing tiles to process> \\
        --patch-output-dir <dir to write binary 0/255 nnU-Net prediction patches, \\
                            plus per-patch probability .npz/.pkl files> \\
        --model-dir <local directory with the pretrained nnU-Net model> \\
        [--device 0]

manifest.json format:
    {"tiles": [{"id": "<uuid>", "source_tif": "<relative path>", "display_name": "..."}]}
"""

import argparse
import json
import tempfile
import threading
from pathlib import Path

import torch
import torchvision.transforms.functional as TF

from AnnotationTool.backend.pipeline.discovery import SEGMENTATION_TMP_DIR_PREFIX
from AnnotationTool.backend.pipeline.progress_util import write_progress
from Segmentation.PreProcessing.General.preprocessing_util import create_patches_from_img
from Segmentation.PreProcessing.General.tif_to_png import convert_tif_to_png
from Segmentation.Util.patch_grid_util import (
    N_PATCHES,
    PATCH_SIZE,
    nnunet_input_patch_filename,
    pred_patch_filename,
)

# set nnuNet directory env vars to prevent warning logs
import os
_tmp_dir = Path(tempfile.gettempdir())
os.environ.setdefault("nnUNet_raw", str(_tmp_dir / "nnunet_raw"))
os.environ.setdefault("nnUNet_preprocessed", str(
    _tmp_dir / "nnunet_preprocessed"))
os.environ.setdefault("nnUNet_results", str(_tmp_dir / "nnunet_results"))

from Segmentation.Util.nnunet_util import (
    initialize_nnunet_predictor,
    run_nnunet_predict_from_patches,
    NNUNET_DEFAULT_FOLDS,
    NNUNET_DEFAULT_CHECKPOINT,
)


def _save_tif_as_png(tif_path: Path, out_path: Path) -> None:
    img = convert_tif_to_png(tif_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path, format="PNG")


def _count_completed_tiles(patch_output_dir: Path, tile_ids: list[str]) -> int:
    """A tile is done once all its predicted patches have been written out."""
    return sum(
        1 for tile_id in tile_ids
        if all((patch_output_dir / pred_patch_filename(tile_id, idx)).is_file()
               for idx in range(N_PATCHES))
    )


def _poll_segmentation_progress(
    project_dir: Path, patch_output_dir: Path, tile_ids: list[str], stop_event: threading.Event,
    interval: float = 2.0,
) -> None:
    """predict_from_files runs as one long blocking call with no per-tile hook,
    so progress is inferred from how many tiles have all their output patches
    on disk yet - runs in a background thread alongside the predict call."""
    while not stop_event.is_set():
        completed = _count_completed_tiles(patch_output_dir, tile_ids)
        write_progress(project_dir, "segmentation", completed, len(tile_ids))
        stop_event.wait(interval)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--project-dir", required=True)
    parser.add_argument("--manifest", required=True,
                        help="manifest.json listing tiles to process")
    parser.add_argument("--patch-output-dir", required=True,
                        help="Directory to write binary (0/255) nnU-Net prediction patches")
    parser.add_argument("--model-dir", required=True,
                        help="Local directory with the pretrained nnU-Net model "
                             "(plans.json, dataset.json, fold_*/checkpoint_*.pth)")
    parser.add_argument("--device", type=int, default=0,
                        help="CUDA device index")
    args = parser.parse_args()

    project_dir = Path(args.project_dir)
    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    tiles = manifest["tiles"]
    if not tiles:
        print("No tiles to process.")
        return

    patch_output_dir = Path(args.patch_output_dir)

    device = torch.device(
        f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model_dir = Path(args.model_dir)
    if not model_dir.is_dir():
        raise FileNotFoundError(
            f"NNUNET_MODEL_DIR does not exist: {model_dir}. The annotation "
            "tool pipeline expects a pretrained model stored in this location")

    predictor, custom_trainer_file_path = initialize_nnunet_predictor(
        model_dir, device,
        folds=NNUNET_DEFAULT_FOLDS,
        checkpoint=NNUNET_DEFAULT_CHECKPOINT,
        ensure_custom_trainer_file=True
    )

    with tempfile.TemporaryDirectory(prefix=SEGMENTATION_TMP_DIR_PREFIX) as tmp:
        patch_input_dir = Path(tmp)
        input_file_lists = []

        for i, tile in enumerate(tiles, start=1):
            print(
                f"Preparing patches for tile {i}/{len(tiles)}: {tile['display_name']}")
            tif_path = project_dir / tile["source_tif"]
            png_path = patch_input_dir / f"{tile['id']}.png"
            _save_tif_as_png(tif_path, png_path)

            patches = create_patches_from_img(
                png_path, patch_size=PATCH_SIZE[0])
            for idx in range(patches.shape[0]):
                fname = nnunet_input_patch_filename(tile['id'], idx)
                TF.to_pil_image(patches[idx]).save(patch_input_dir / fname)
                input_file_lists.append([str(patch_input_dir / fname)])
            write_progress(project_dir, "preprocessing", i, len(tiles))

        print(f"Running predict_from_files on {len(input_file_lists)} patches")

        patch_output_dir.mkdir(parents=True, exist_ok=True)
        tile_ids = [tile["id"] for tile in tiles]
        stop_event = threading.Event()
        progress_thread = threading.Thread(
            target=_poll_segmentation_progress,
            args=(project_dir, patch_output_dir, tile_ids, stop_event),
            daemon=True,
        )
        progress_thread.start()
        try:
            run_nnunet_predict_from_patches(
                predictor, input_file_lists, patch_output_dir, save_probabilities=True)
        finally:
            stop_event.set()
            progress_thread.join()
        write_progress(project_dir, "segmentation", len(tile_ids), len(tile_ids))

    if custom_trainer_file_path is not None:
        custom_trainer_file_path.unlink()

    print("Done.")


if __name__ == "__main__":
    main()
