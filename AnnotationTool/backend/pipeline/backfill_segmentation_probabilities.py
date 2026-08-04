"""One-off script: re-run nnU-Net segmentation for every already-PROCESSED
image in a project's annotations.json, purely to backfill the stitched
foreground probability map that the regular pipeline workers now save
alongside the binary mask.

Does NOT touch annotations.json (points, labels, processed flags are all left
alone) and does not touch the existing Segmentation/ binary masks - it only
(re)writes AutomaticForkDetection/SegmentationProbabilities/<image_id>.npy
(float32, foreground probability, same H×W as the source tile).

Usage (must be invoked with the pipeline venv's python, from the repo root):
    python -m AnnotationTool.backend.pipeline.backfill_segmentation_probabilities \\
        --project-dir <base folder> \\
        --model-dir <local directory with the pretrained nnU-Net model> \\
        [--device 0] [--force]
"""

import argparse
import tempfile
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms.functional as TF

from AnnotationTool.backend.pipeline.annotations_store import load_annotations
from AnnotationTool.backend.pipeline.discovery import (
    SEGMENTATION_PROBABILITIES_DIR_NAME,
    SEGMENTATION_TMP_DIR_PREFIX,
    fork_detection_dir,
)
from Segmentation.PostProcessing.segmentation_postprocessing import stitch_mask_tiles
from Segmentation.PreProcessing.General.preprocessing_util import create_patches_from_img
from Segmentation.PreProcessing.General.tif_to_png import convert_tif_to_png
from Segmentation.Util.patch_grid_util import (
    GRID_SIZE,
    PATCH_SIZE,
    load_probability_pred_patches,
    nnunet_input_patch_filename,
)

# set nnUNet directory env vars to prevent warning logs
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


def _segment_tile_probabilities(tile_path: Path, image_id: str, predictor) -> torch.Tensor:
    with tempfile.TemporaryDirectory(prefix=SEGMENTATION_TMP_DIR_PREFIX) as tmp:
        tmp_dir = Path(tmp)
        png_path = tmp_dir / f"{image_id}.png"
        convert_tif_to_png(tile_path).save(png_path, format="PNG")

        patches = create_patches_from_img(png_path, patch_size=PATCH_SIZE[0])
        input_file_lists = []
        for idx in range(patches.shape[0]):
            fname = nnunet_input_patch_filename(image_id, idx)
            TF.to_pil_image(patches[idx]).save(tmp_dir / fname)
            input_file_lists.append([str(tmp_dir / fname)])

        patch_output_dir = tmp_dir / "pred"
        run_nnunet_predict_from_patches(
            predictor, input_file_lists, patch_output_dir, save_probabilities=True)

        prob_patches, _ = load_probability_pred_patches(
            patch_output_dir, image_id)
        stitched_prob = stitch_mask_tiles(
            prob_patches, grid_size=GRID_SIZE,
            original_input_patch_img_size=PATCH_SIZE, as_uint=False)
        return stitched_prob.detach().cpu()


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--project-dir", required=True)
    parser.add_argument("--model-dir", required=True,
                        help="Local directory with the pretrained nnU-Net model "
                             "(plans.json, dataset.json, fold_*/checkpoint_*.pth)")
    parser.add_argument("--device", type=int, default=0,
                        help="CUDA device index")
    parser.add_argument("--force", action="store_true",
                        help="Recompute even for images that already have a probability map on disk")
    args = parser.parse_args()

    project_dir = Path(args.project_dir)
    annotations = load_annotations(project_dir)
    processed_images = {
        image_id: img for image_id, img in annotations["images"].items()
        if img.get("processed")
    }
    print(f"{len(processed_images)} processed image(s) found in annotations.json")
    if not processed_images:
        return

    prob_out_dir = fork_detection_dir(project_dir) / SEGMENTATION_PROBABILITIES_DIR_NAME
    prob_out_dir.mkdir(parents=True, exist_ok=True)

    if not args.force:
        before = len(processed_images)
        processed_images = {
            image_id: img for image_id, img in processed_images.items()
            if not (prob_out_dir / f"{image_id}.npy").is_file()
        }
        skipped = before - len(processed_images)
        if skipped:
            print(f"Skipping {skipped} image(s) that already have a probability map "
                  f"(use --force to redo them too)")

    if not processed_images:
        print("Nothing to do.")
        return

    device = torch.device(
        f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model_dir = Path(args.model_dir)
    if not model_dir.is_dir():
        raise FileNotFoundError(f"Model dir does not exist: {model_dir}")

    predictor, custom_trainer_file_path = initialize_nnunet_predictor(
        model_dir, device,
        folds=NNUNET_DEFAULT_FOLDS,
        checkpoint=NNUNET_DEFAULT_CHECKPOINT,
        ensure_custom_trainer_file=True,
    )

    try:
        for i, (image_id, img) in enumerate(processed_images.items(), start=1):
            tile_path = project_dir / img["source_tif"]
            print(f"[{i}/{len(processed_images)}] {img.get('display_name', image_id)}")
            stitched_prob = _segment_tile_probabilities(
                tile_path, image_id, predictor)
            np.save(prob_out_dir / f"{image_id}.npy",
                    stitched_prob.squeeze(0).numpy().astype(np.float32))
    finally:
        if custom_trainer_file_path is not None:
            custom_trainer_file_path.unlink()

    print("Done.")


if __name__ == "__main__":
    main()
