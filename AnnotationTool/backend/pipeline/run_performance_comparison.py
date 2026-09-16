"""Reruns automatic junction detection on already-reviewed tiles, to check
what a project's reversed-fork ratio would look like if reviewers only ever
opened images where the automatic pipeline actually predicted something.

For each given project, every image currently marked "processed": true in
AutomaticForkDetection/annotations.json is treated as reviewed ground truth
(this is the same flag the rest of the app uses to mean "a human finished
reviewing this image" - see main.py's export/summary logic). Its tile is
re-segmented and re-detected from scratch (fresh model run, new image ids),
purely to find out, for each tile, whether the pipeline would predict
anything on it today. The project's existing AutomaticForkDetection folder
(annotations.json, Segmentation/, SegmentationProbabilities/) is never
touched - all new output goes to a sibling folder,
AutomaticForkDetection_PerformanceComparison/.

The reversed-fork ratio is then computed the same way as the annotation
tool's own summary (main.py's _compute_summary: weighted reversed-fork count
/ (weighted reversed-fork count + weighted replication-fork count), 100%
labels weighing 1.0 and 50% labels weighing 0.5), but computed twice per
project so the two can be compared directly:
  - "full_reviewed": using every reviewed tile's (human-corrected) points
  - "streamlined_reviewed_predictions_only": using only the reviewed points
    of tiles where the new pipeline run found >=1 point - i.e. exactly the
    set of tiles a reviewer would still open under a "only review images
    with predictions" workflow.

Usage (must be invoked with the backend's own interpreter, not the pipeline
venv - it only orchestrates; the actual segmentation_worker/detection_worker
subprocesses it spawns run with the pipeline venv, same as run_pipeline.py):
    python -m AnnotationTool.backend.pipeline.run_performance_comparison \\
        --project-dir <path> [--project-dir <path> ...]
"""

import argparse
import json
import logging
import os
import sys
import tempfile
import uuid
from pathlib import Path

from AnnotationTool.backend.pipeline.annotations_store import load_annotations
from AnnotationTool.backend.pipeline.discovery import PIPELINE_TMP_DIR_PREFIX
from AnnotationTool.backend.pipeline.progress_util import clear_progress
from AnnotationTool.backend.pipeline.run_pipeline import PipelineConfig, _run_worker

PERFORMANCE_COMPARISON_DIR_NAME = "AutomaticForkDetection_PerformanceComparison"

# Mirrors main.py's JunctionType/FORK_WEIGHTS, avoids importing them from
# main.py, since main.py builds the whole FastAPI app on import (same reason
# pipeline_runner.py re-declares its own _MODE_SEQUENTIAL/_MODE_STAGED).
_LABEL_REPLICATION_100 = "Replication Fork 100%"
_LABEL_REPLICATION_50 = "Replication Fork 50%"
_LABEL_REVERSED_100 = "Reversed Fork 100%"
_LABEL_REVERSED_50 = "Reversed Fork 50%"

FORK_WEIGHTS = {
    _LABEL_REPLICATION_50: 0.5,
    _LABEL_REPLICATION_100: 1.0,
    _LABEL_REVERSED_50: 0.5,
    _LABEL_REVERSED_100: 1.0,
}
REPLICATION_FORK_LABELS = {_LABEL_REPLICATION_50, _LABEL_REPLICATION_100}
REVERSED_FORK_LABELS = {_LABEL_REVERSED_50, _LABEL_REVERSED_100}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)


def _fork_ratio_stats(points: list[dict]) -> dict:
    replication = sum(
        FORK_WEIGHTS.get(l, 0.0)
        for p in points for l in p.get("labels", [])
        if l in REPLICATION_FORK_LABELS)
    reversed_ = sum(
        FORK_WEIGHTS.get(l, 0.0)
        for p in points for l in p.get("labels", [])
        if l in REVERSED_FORK_LABELS)
    total = sum(1 for p in points for l in p.get("labels", [])
                if l in REPLICATION_FORK_LABELS or l in REVERSED_FORK_LABELS)
    return {
        "total_forks_count": total,
        "replication_fork_weighted_count": replication,
        "reversed_fork_weighted_count": reversed_,
        "reversed_fork_ratio": round(reversed_ / (replication + reversed_), 3)
        if (replication + reversed_) > 0 else None,
    }


def _write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def run_performance_comparison(project_dir: Path) -> None:
    project_dir = Path(project_dir)
    annotations = load_annotations(project_dir, must_exist=True)

    reviewed_images = {
        image_id: img for image_id, img in annotations["images"].items()
        if img.get("processed", False) and not img.get("archived", False)
    }
    if not reviewed_images:
        logger.info(
            "No reviewed (processed=true) images found in %s; nothing to do.", project_dir)
        return

    output_dir = project_dir / PERFORMANCE_COMPARISON_DIR_NAME
    config = PipelineConfig()

    tiles_manifest = [
        {
            "id": str(uuid.uuid4()),
            "source_tif": img["source_tif"],
            "display_name": img.get("display_name", img["source_tif"]),
        }
        for img in reviewed_images.values()
    ]
    reviewed_by_source_tif = {
        img["source_tif"]: img for img in reviewed_images.values()}

    logger.info("Rerunning detection on %d reviewed tile(s) in %s",
                len(tiles_manifest), project_dir)

    try:
        with tempfile.TemporaryDirectory(prefix=PIPELINE_TMP_DIR_PREFIX) as tmp:
            tmp_root = Path(tmp)
            manifest_path = tmp_root / "manifest.json"
            results_path = tmp_root / "results.json"
            patch_dir = tmp_root / "SegmentationPatches"
            manifest_path.write_text(json.dumps(
                {"tiles": tiles_manifest}), encoding="utf-8")

            _run_worker("segmentation_worker", [
                "--project-dir", str(project_dir),
                "--manifest", str(manifest_path),
                "--patch-output-dir", str(patch_dir),
                "--model-dir", str(config.nnunet_model_dir),
                "--device", str(config.nnunet_device),
            ], config)

            _run_worker("detection_worker", [
                "--project-dir", str(project_dir),
                "--manifest", str(manifest_path),
                "--patch-dir", str(patch_dir),
                "--results-out", str(results_path),
                "--output-dir", str(output_dir),
            ], config)

            results = json.loads(results_path.read_text(encoding="utf-8"))
    finally:
        # segmentation_worker/detection_worker report progress into the
        # project's *existing* AutomaticForkDetection folder; clear it so this
        # analysis run doesn't leave a stale progress file behind there.
        clear_progress(project_dir)

    per_tile = {}
    all_reviewed_points = []
    streamlined_reviewed_points = []
    tiles_with_predictions = 0
    tiles_without_predictions = 0

    for new_id, new_img in results["images"].items():
        source_tif = new_img["source_tif"]
        reviewed_img = reviewed_by_source_tif[source_tif]
        reviewed_points = reviewed_img.get("points", [])
        new_points = new_img.get("points", [])
        has_prediction = len(new_points) > 0

        all_reviewed_points.extend(reviewed_points)
        if has_prediction:
            tiles_with_predictions += 1
            streamlined_reviewed_points.extend(reviewed_points)
        else:
            tiles_without_predictions += 1

        per_tile[new_id] = {
            "source_tif": source_tif,
            "display_name": new_img.get("display_name", source_tif),
            "processed": False,
            "points": new_points,
            "reviewed_points": reviewed_points,
            "included_in_streamlined_ratio": has_prediction,
        }

    summary = {
        "tiles_considered": len(reviewed_images),
        "tiles_with_new_predictions": tiles_with_predictions,
        "tiles_without_new_predictions": tiles_without_predictions,
        "full_reviewed": _fork_ratio_stats(all_reviewed_points),
        "streamlined_reviewed_predictions_only": _fork_ratio_stats(streamlined_reviewed_points),
    }

    output = {
        "source_project_dir": str(project_dir),
        "summary": summary,
        "images": per_tile,
    }
    _write_json(output_dir / "annotations.json", output)

    logger.info(
        "%s: %d/%d reviewed tile(s) got a new prediction | "
        "full reviewed ratio=%s | streamlined (predictions-only) ratio=%s",
        project_dir, tiles_with_predictions, summary["tiles_considered"],
        summary["full_reviewed"]["reversed_fork_ratio"],
        summary["streamlined_reviewed_predictions_only"]["reversed_fork_ratio"],
    )
    logger.info("Wrote comparison output to %s",
                output_dir / "annotations.json")


def main():
    # parser = argparse.ArgumentParser(
    #    description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    # parser.add_argument("--project-dir", required=True, action="append",
    #                    help="Project directory to run the comparison for; repeat for multiple projects")
    # args = parser.parse_args()

    # hardcode project dir paths (as they are on the ScienceCloud VM) instead of passing them as arguments
    # since this is run only once anyways
    project_dirs = [
        "/mnt/lopesgroup/2026_Cyril_CellLines_Ola/260826_NB4+Ola_R1_Cyril",
        "/mnt/lopesgroup/2026_Cyril_CellLines_Ola/260828_THP1_Ola_R1_Cyril_20260901_0700",
        "/mnt/lopesgroup/2026_Cyril_CellLines_Ola/talos_transfer_20260819_1200/20260818_Cyril_NB4",
        "/mnt/lopesgroup/2026_Cyril_CellLines_Ola/talos_transfer_20260828_1000/260827_THP1_R1_Cyril"
    ]

    for i, project_dir in enumerate(project_dirs, start=1):
        print(
            f"{i}/{len(project_dirs)} running performance comparison for project dir {project_dir}")
        run_performance_comparison(Path(project_dir))


if __name__ == "__main__":
    main()
