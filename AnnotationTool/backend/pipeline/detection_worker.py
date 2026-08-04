"""Detection-stage worker for the annotation tool pipeline
stitches and postprocesses segmentation prediction patches,
runs skeleton-based junction detection.

Usage:
    python -m AnnotationTool.backend.pipeline.detection_worker \\
        --project-dir <base folder> \\
        --manifest <manifest.json listing tiles to process> \\
        --patch-dir <dir with nnU-Net prediction patches from segmentation_worker> \\
        --results-out <path to write the resulting images JSON>
"""

import argparse
import json
import traceback
import uuid
from pathlib import Path

import numpy as np
from PIL import Image

import Environment.env_utils as env_utils
from Segmentation.PostProcessing.segmentation_postprocessing import (
    postprocess_segmentation_masks,
    stitch_mask_tiles,
)
from JunctionDetection.SkeletonizeDetect.segmentation_junction_detection import (
    detect_junctions_in_segmentation_mask,
)
from Segmentation.Util.patch_grid_util import (
    PATCH_SIZE, GRID_SIZE, load_binary_mask_pred_patches,
    load_probability_pred_patches,
)
from AnnotationTool.backend.pipeline.discovery import (
    AUTOMATIC_FORK_DETECTION_DIR_NAME,
    SEGMENTATION_DIR_NAME,
    SEGMENTATION_PROBABILITIES_DIR_NAME,
)
from AnnotationTool.backend.pipeline.progress_util import write_progress

JUNCTION_LABEL_3WAY = "Replication Fork 100%"
JUNCTION_LABEL_4WAY = "Reversed Fork 100%"

# pre-defined labels that can be set by human annotator in the UI
# and count 0.5 to the sum of detected forks/junctions
JUNCTION_LABEL_3WAY_50 = "Replication Fork 50%"
JUNCTION_LABEL_4WAY_50 = "Reversed Fork 50%"


def get_junction_points(stitched, display_name: str) -> list[dict]:
    try:
        coords_3way, coords_4way, _ = detect_junctions_in_segmentation_mask(
            stitched)
    except Exception:
        print(f"Junction detection failed for {display_name}; "
              f"assuming no junctions were found:\n{traceback.format_exc()}")
        coords_3way, coords_4way = np.empty((0, 2)), np.empty((0, 2))

    points = []
    for x, y in coords_3way:
        points.append({"id": str(uuid.uuid4()), "x": round(float(x)),
                       "y": round(float(y)), "labels": [JUNCTION_LABEL_3WAY]})
    for x, y in coords_4way:
        points.append({"id": str(uuid.uuid4()), "x": round(float(x)),
                       "y": round(float(y)), "labels": [JUNCTION_LABEL_4WAY]})
    return points


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--project-dir", required=True)
    parser.add_argument("--manifest", required=True,
                        help="manifest.json listing tiles to process")
    parser.add_argument("--patch-dir", required=True,
                        help="Directory with nnU-Net prediction patches from segmentation_worker")
    parser.add_argument("--results-out", required=True,
                        help="Path to write the resulting images JSON")
    args = parser.parse_args()

    env_utils.load_shared_env()

    project_dir = Path(args.project_dir)

    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    tiles = manifest["tiles"]

    patch_dir = Path(args.patch_dir)

    seg_out_dir = project_dir / AUTOMATIC_FORK_DETECTION_DIR_NAME / SEGMENTATION_DIR_NAME
    seg_out_dir.mkdir(parents=True, exist_ok=True)
    prob_out_dir = project_dir / AUTOMATIC_FORK_DETECTION_DIR_NAME / \
        SEGMENTATION_PROBABILITIES_DIR_NAME
    prob_out_dir.mkdir(parents=True, exist_ok=True)

    images = {}
    for i, tile in enumerate(tiles, start=1):
        image_id = tile["id"]
        pred_patches, pred_patch_paths = load_binary_mask_pred_patches(
            patch_dir, image_id)

        stitched, _ = postprocess_segmentation_masks(
            pred_patches, grid_size=GRID_SIZE,
            original_input_patch_img_size=PATCH_SIZE,
            remove_small_objects=True,
        )
        stitched = stitched.detach().cpu()

        # convert stitched full-size segmentation to black/white mask and save as PNG
        mask_arr = (stitched.squeeze(0).numpy() * 255).astype(np.uint8)
        Image.fromarray(mask_arr).save(seg_out_dir / f"{image_id}.png")

        # delete segmentation mask patches
        for p in pred_patch_paths:
            p.unlink(missing_ok=True)

        prob_patches, prob_patch_paths = load_probability_pred_patches(
            patch_dir, image_id)
        stitched_prob = stitch_mask_tiles(
            prob_patches, grid_size=GRID_SIZE,
            original_input_patch_img_size=PATCH_SIZE, as_uint=False)
        np.save(prob_out_dir / f"{image_id}.npy",
                stitched_prob.squeeze(0).detach().cpu().numpy().astype(np.float32))

        # delete probability patches (.npz) and their accompanying properties (.pkl)
        for p in prob_patch_paths:
            p.unlink(missing_ok=True)
            p.with_suffix(".pkl").unlink(missing_ok=True)

        points = get_junction_points(stitched, tile["display_name"])

        images[image_id] = {
            "source_tif": tile["source_tif"],
            "display_name": tile["display_name"],
            "processed": False,
            "points": points,
        }
        print(
            f"Processed {tile['display_name']}: {len(points)} junction(s) detected")
        write_progress(project_dir, "detection", i, len(tiles))

    Path(args.results_out).write_text(
        json.dumps({"images": images}, indent=2), encoding="utf-8")
    print("Done.")


if __name__ == "__main__":
    main()
