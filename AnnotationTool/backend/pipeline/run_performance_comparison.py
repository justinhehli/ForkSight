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

Console/log output: the console (and each project's
performance_comparison.log, see below) only ever gets high-level
"foreground" lines - one per tile per stage (preprocessing/segmentation/
detection), plus the per-project start/end summary. The segmentation_worker
/detection_worker subprocesses' own (much chattier) stdout/stderr is
redirected straight to per-project log files instead
(AutomaticForkDetection_PerformanceComparison/logs/*.log), same layout as
performance_comparison.log itself.

Usage (must be invoked with the backend's own interpreter, not the pipeline
venv - it only orchestrates; the actual segmentation_worker/detection_worker
subprocesses it spawns run with the pipeline venv, same as run_pipeline.py):
    python -m AnnotationTool.backend.pipeline.run_performance_comparison \\
        --project-dir <path> [--project-dir <path> ...]
"""

import argparse
import contextlib
import json
import logging
import os
import subprocess
import sys
import tempfile
import threading
import uuid
from pathlib import Path

# Re-segmenting/re-detecting every reviewed tile writes a lot of per-patch
# scratch data; point tempfile (and, via TMPDIR, the segmentation_worker /
# detection_worker subprocesses spawned below) at a dedicated scratch dir
# instead of the regular, often small/tmpfs-backed /tmp - same as
# backfill_segmentation_probabilities.py.
SCRATCH_TMP_DIR = Path("/mnt/scratch")
os.environ["TMPDIR"] = str(SCRATCH_TMP_DIR)
tempfile.tempdir = str(SCRATCH_TMP_DIR)

from AnnotationTool.backend.util import get_repo_root, venv_python_executable
from AnnotationTool.backend.pipeline.annotations_store import load_annotations
from AnnotationTool.backend.pipeline.discovery import PIPELINE_TMP_DIR_PREFIX
from AnnotationTool.backend.pipeline.progress_util import clear_progress, read_progress
from AnnotationTool.backend.pipeline.run_pipeline import PipelineConfig

PERFORMANCE_COMPARISON_DIR_NAME = "AutomaticForkDetection_PerformanceComparison"
LOGS_DIR_NAME = "logs"
FOREGROUND_LOG_FILENAME = "performance_comparison.log"

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


@contextlib.contextmanager
def _log_to_file_too(log_path: Path):
    """Additionally send every foreground log line (this module's own
    logger.info calls - never the worker subprocesses' stdout, see
    _run_worker_to_log) to `log_path`, on top of the console."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(log_path, encoding="utf-8")
    handler.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
    root_logger = logging.getLogger()
    root_logger.addHandler(handler)
    try:
        yield
    finally:
        root_logger.removeHandler(handler)
        handler.close()


def _run_worker_to_log(module: str, worker_args: list[str], config: PipelineConfig,
                       log_path: Path) -> None:
    """Same subprocess invocation as run_pipeline._run_worker, but the worker's
    stdout/stderr is written straight to `log_path` instead of being piped
    through this process's own logger - keeps the console down to this
    script's own foreground per-tile/per-stage lines (see _watch_progress)."""
    repo_root = get_repo_root()
    python_exe = venv_python_executable(config.pipeline_venv)
    if not python_exe.is_file():
        raise FileNotFoundError(
            f"Pipeline venv python executable not found: {python_exe}. "
        )
    cmd = [str(python_exe), "-u", "-m",
           f"AnnotationTool.backend.pipeline.{module}", *worker_args]

    env = os.environ.copy()
    env.pop("VIRTUAL_ENV", None)
    env.pop("PYTHONHOME", None)
    env["VIRTUAL_ENV"] = str(config.pipeline_venv)
    env["PYTHONUNBUFFERED"] = "1"

    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Starting %s (output redirected to %s)", module, log_path)
    with log_path.open("w", encoding="utf-8") as log_fh:
        returncode = subprocess.run(
            cmd, cwd=str(repo_root), env=env,
            stdout=log_fh, stderr=subprocess.STDOUT, text=True,
        ).returncode
    if returncode != 0:
        raise RuntimeError(
            f"{module} subprocess failed (exit {returncode}); see {log_path} for details")
    logger.info("%s finished successfully", module)


def _watch_progress(project_dir: Path, stop_event: threading.Event,
                    interval: float = 0.5) -> None:
    """Polls the same pipeline_progress.json that segmentation_worker/
    detection_worker already write per tile (see progress_util.py) and emits
    one foreground log line per tile per stage (preprocessing/segmentation/
    detection) - runs in a background thread alongside the worker
    subprocesses, which write their own (redirected, not logged here) output
    separately."""
    last = (None, None)
    while True:
        progress = read_progress(project_dir)
        stage, completed, total = (
            progress.get("stage"), progress.get("completed"), progress.get("total"))
        key = (stage, completed)
        if stage is not None and key != last:
            logger.info("%s: %d/%d tile(s)", stage, completed, total)
            last = key
        if stop_event.wait(interval):
            return


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
    output_dir = project_dir / PERFORMANCE_COMPARISON_DIR_NAME
    logs_dir = output_dir / LOGS_DIR_NAME

    with _log_to_file_too(output_dir / FOREGROUND_LOG_FILENAME):
        annotations = load_annotations(project_dir, must_exist=True)

        reviewed_images = {
            image_id: img for image_id, img in annotations["images"].items()
            if img.get("processed", False) and not img.get("archived", False)
        }
        if not reviewed_images:
            logger.info(
                "No reviewed (processed=true) images found in %s; nothing to do.", project_dir)
            return

        config = PipelineConfig()

        # annotations.json can contain more than one reviewed image entry for
        # the same physical tile (e.g. from a rerun whose new-tile dedup
        # missed it due to a differently-represented project_dir path) -
        # collapse to one entry per source_tif before building the manifest,
        # so each unique tile is only re-segmented/re-detected once and its
        # reviewed points aren't double-counted in the ratios below.
        reviewed_by_source_tif = {
            img["source_tif"]: img for img in reviewed_images.values()}
        duplicate_count = len(reviewed_images) - len(reviewed_by_source_tif)
        if duplicate_count:
            logger.warning(
                "%d reviewed image(s) in %s share a source_tif with another "
                "reviewed image; only re-segmenting/re-detecting each of the "
                "%d unique tile(s) once", duplicate_count, project_dir,
                len(reviewed_by_source_tif))

        tiles_manifest = [
            {
                "id": str(uuid.uuid4()),
                "source_tif": source_tif,
                "display_name": img.get("display_name", source_tif),
            }
            for source_tif, img in reviewed_by_source_tif.items()
        ]

        logger.info("Rerunning detection on %d reviewed tile(s) in %s",
                    len(tiles_manifest), project_dir)

        stop_event = threading.Event()
        watcher = threading.Thread(
            target=_watch_progress, args=(project_dir, stop_event), daemon=True)
        try:
            watcher.start()
            with tempfile.TemporaryDirectory(prefix=PIPELINE_TMP_DIR_PREFIX) as tmp:
                tmp_root = Path(tmp)
                manifest_path = tmp_root / "manifest.json"
                results_path = tmp_root / "results.json"
                patch_dir = tmp_root / "SegmentationPatches"
                manifest_path.write_text(json.dumps(
                    {"tiles": tiles_manifest}), encoding="utf-8")

                _run_worker_to_log("segmentation_worker", [
                    "--project-dir", str(project_dir),
                    "--manifest", str(manifest_path),
                    "--patch-output-dir", str(patch_dir),
                    "--model-dir", str(config.nnunet_model_dir),
                    "--device", str(config.nnunet_device),
                ], config, logs_dir / "segmentation_worker.log")

                _run_worker_to_log("detection_worker", [
                    "--project-dir", str(project_dir),
                    "--manifest", str(manifest_path),
                    "--patch-dir", str(patch_dir),
                    "--results-out", str(results_path),
                    "--output-dir", str(output_dir),
                ], config, logs_dir / "detection_worker.log")

                results = json.loads(results_path.read_text(encoding="utf-8"))
        finally:
            # segmentation_worker/detection_worker report progress into the
            # project's *existing* AutomaticForkDetection folder; clear it so this
            # analysis run doesn't leave a stale progress file behind there.
            stop_event.set()
            watcher.join()
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
            "tiles_considered": len(reviewed_by_source_tif),
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
        #"/mnt/lopesgroup/2026_Cyril_CellLines_Ola/260826_NB4+Ola_R1_Cyril",
        "/mnt/lopesgroup/2026_Cyril_CellLines_Ola/260828_THP1_Ola_R1_Cyril_20260901_0700",
        "/mnt/lopesgroup/2026_Cyril_CellLines_Ola/talos_transfer_20260819_1200/20260818_Cyril_NB4",
        "/mnt/lopesgroup/2026_Cyril_CellLines_Ola/talos_transfer_20260828_1000/260827_THP1_R1_Cyril"
    ]

    for i, project_dir in enumerate(project_dirs, start=1):
        logger.info(
            "%d/%d running performance comparison for project dir %s", i, len(project_dirs), project_dir)
        run_performance_comparison(Path(project_dir))


if __name__ == "__main__":
    main()
