"""Sequential-mode worker for the annotation tool pipeline.

Alternates segmentation and junction detection for randomly-ordered,
not-yet-annotated tiles of a project, one tile at a time, until a target
total junction count is reached (or tiles run out).

Unlike the staged pipeline (segmentation_worker.py + detection_worker.py),
which processes a whole batch of tiles before writing anything to
annotations.json, this worker persists the stitched segmentation PNG,
probability map and annotations.json after every single tile - so progress up to the point of a
crash or manual stop is never lost, and the loop can stop as soon as the
target is hit without wasting time on tiles that turn out not to be needed.

Usage (must be invoked with the pipeline venv's python, from the repo root):
    python -m AnnotationTool.backend.pipeline.sequential_worker \\
        --project-dir <base folder> \\
        --model-dir <local directory with the pretrained nnU-Net model> \\
        --target-junction-count 150 \\
        [--device 0]
"""

import argparse
import random
import tempfile
import uuid
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image

import Environment.env_utils as env_utils
from AnnotationTool.backend.pipeline.annotations_store import (
    load_annotations,
    save_annotations,
)
from AnnotationTool.backend.pipeline.detection_worker import (
    JUNCTION_LABEL_3WAY,
    JUNCTION_LABEL_3WAY_50,
    JUNCTION_LABEL_4WAY,
    JUNCTION_LABEL_4WAY_50,
    get_junction_points,
)
from AnnotationTool.backend.pipeline.discovery import (
    SEGMENTATION_DIR_NAME,
    SEGMENTATION_PROBABILITIES_DIR_NAME,
    SEGMENTATION_TMP_DIR_PREFIX,
    find_project_tiles,
    fork_detection_dir,
    get_tile_display_name,
)
from AnnotationTool.backend.pipeline.progress_util import write_progress
from Segmentation.PostProcessing.segmentation_postprocessing import (
    postprocess_segmentation_masks,
    stitch_mask_tiles,
)
from Segmentation.PreProcessing.General.preprocessing_util import create_patches_from_img
from Segmentation.PreProcessing.General.tif_to_png import convert_tif_to_png
from Segmentation.Util.patch_grid_util import (
    GRID_SIZE,
    PATCH_SIZE,
    load_binary_mask_pred_patches,
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

_JUNCTION_WEIGHTS = {
    JUNCTION_LABEL_3WAY: 1.0,
    JUNCTION_LABEL_4WAY: 1.0,
    JUNCTION_LABEL_3WAY_50: 0.5,
    JUNCTION_LABEL_4WAY_50: 0.5,
}


def _count_total_junctions(images: dict) -> float:
    return sum(
        _JUNCTION_WEIGHTS.get(l, 0.0)
        for img in images.values()
        if not img.get("archived", False)
        for p in img.get("points", [])
        for l in p.get("labels", [])
    )


def _segment_tile(tile_path: Path, image_id: str, predictor) -> tuple[torch.Tensor, torch.Tensor]:
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

        pred_patches, _ = load_binary_mask_pred_patches(
            patch_output_dir, image_id)
        stitched, _ = postprocess_segmentation_masks(
            pred_patches, grid_size=GRID_SIZE,
            original_input_patch_img_size=PATCH_SIZE,
            remove_small_objects=True,
        )

        prob_patches, _ = load_probability_pred_patches(
            patch_output_dir, image_id)
        stitched_prob = stitch_mask_tiles(
            prob_patches, grid_size=GRID_SIZE,
            original_input_patch_img_size=PATCH_SIZE, as_uint=False)

        return stitched.detach().cpu(), stitched_prob.detach().cpu()


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--project-dir", required=True)
    parser.add_argument("--model-dir", required=True,
                        help="Local directory with the pretrained nnU-Net model "
                             "(plans.json, dataset.json, fold_*/checkpoint_*.pth)")
    parser.add_argument("--target-junction-count", type=int, required=True,
                        help="Stop once this many total junctions (both types "
                             "combined) have been found across the project")
    parser.add_argument("--device", type=int, default=0,
                        help="CUDA device index")
    args = parser.parse_args()

    env_utils.load_shared_env()

    project_dir = Path(args.project_dir)
    annotations = load_annotations(project_dir)
    total_junctions = _count_total_junctions(annotations["images"])
    print(f"{total_junctions}/{args.target_junction_count} junctions found so far")
    if total_junctions >= args.target_junction_count:
        print("Target already reached; nothing to do.")
        return

    known_source_tifs = {img["source_tif"]
                         for img in annotations["images"].values()}
    remaining_tiles = [
        t for t in find_project_tiles(project_dir)
        if t.relative_to(project_dir).as_posix() not in known_source_tifs
    ]
    random.shuffle(remaining_tiles)
    print(f"{len(remaining_tiles)} unprocessed tile(s) available")
    if not remaining_tiles:
        print("No tiles left to sample from.")
        return

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
        ensure_custom_trainer_file=True,
    )

    seg_out_dir = fork_detection_dir(project_dir) / SEGMENTATION_DIR_NAME
    seg_out_dir.mkdir(parents=True, exist_ok=True)
    prob_out_dir = fork_detection_dir(
        project_dir) / SEGMENTATION_PROBABILITIES_DIR_NAME
    prob_out_dir.mkdir(parents=True, exist_ok=True)

    try:
        for i, tile_path in enumerate(remaining_tiles, start=1):
            if total_junctions >= args.target_junction_count:
                break

            image_id = str(uuid.uuid4())
            source_tif = tile_path.relative_to(project_dir).as_posix()
            display_name = get_tile_display_name(tile_path)

            stitched, stitched_prob = _segment_tile(
                tile_path, image_id, predictor)

            mask_arr = (stitched.squeeze(0).numpy() * 255).astype(np.uint8)
            Image.fromarray(mask_arr).save(seg_out_dir / f"{image_id}.png")
            np.save(prob_out_dir / f"{image_id}.npy",
                    stitched_prob.squeeze(0).numpy().astype(np.float32))

            points = get_junction_points(stitched, display_name)

            annotations["images"][image_id] = {
                "source_tif": source_tif,
                "display_name": display_name,
                "processed": False,
                "points": points,
            }
            save_annotations(project_dir, annotations)

            total_junctions = _count_total_junctions(annotations["images"])
            write_progress(project_dir, "sequential",
                           total_junctions, args.target_junction_count)
            print(f"[{i}/{len(remaining_tiles)}] {display_name}: {len(points)} junction(s) "
                  f"detected (total {total_junctions}/{args.target_junction_count})")
    finally:
        if custom_trainer_file_path is not None:
            custom_trainer_file_path.unlink()

    print("Done.")


if __name__ == "__main__":
    main()
