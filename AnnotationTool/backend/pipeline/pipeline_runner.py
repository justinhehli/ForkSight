"""Standalone runner for one project's automatic junction-detection pipeline,
spawned as a fully independent OS process by the API backend

Runs with the backend's own (lightweight) interpreter, not the pipeline venv;
this script only orchestrates the detection stages (segmentation_worker.py, detection_worker.py,
or sequential_worker.py, depending on --mode)

Usage:
    python -m AnnotationTool.backend.pipeline.pipeline_runner --project-dir <path> \\
        [--mode {sequential,staged}] [--target-junction-count N] [--sample-count N]
"""

import argparse
import logging
import sys
from pathlib import Path

from AnnotationTool.backend.pipeline.annotations_store import (
    PipelineStatus,
    load_annotations,
    save_annotations,
)
from AnnotationTool.backend.pipeline.discovery import (
    DEFAULT_STAGED_SAMPLE_COUNT,
    DEFAULT_TARGET_JUNCTION_COUNT,
)

from AnnotationTool.backend.pipeline.progress_util import clear_progress
from AnnotationTool.backend.pipeline.run_pipeline import (
    run_staged_junction_detection_pipeline,
    run_sequential_junction_detection_pipeline,
)

# Mirrors main.py's PipelineMode, avoids importing enums from main.py,
# since main.py it pulls unwanted dependencies
_MODE_SEQUENTIAL = "sequential"
_MODE_STAGED = "staged"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-dir", required=True)
    parser.add_argument("--mode", choices=[_MODE_SEQUENTIAL, _MODE_STAGED],
                        default=_MODE_SEQUENTIAL)
    parser.add_argument("--target-junction-count", type=int,
                        default=DEFAULT_TARGET_JUNCTION_COUNT)
    parser.add_argument("--sample-count", type=int,
                        default=DEFAULT_STAGED_SAMPLE_COUNT)
    args = parser.parse_args()
    project_dir = Path(args.project_dir)

    logger.info("Pipeline runner started for project: %s (mode=%s)",
                project_dir, args.mode)
    try:
        if args.mode == _MODE_SEQUENTIAL:
            run_sequential_junction_detection_pipeline(
                project_dir, args.target_junction_count)
        else:
            run_staged_junction_detection_pipeline(
                project_dir, args.sample_count)

        # reload annotations to make sure we have the current version here
        annotations = load_annotations(project_dir)
        annotations["junction_detection_pipeline_status"] = PipelineStatus.Done
        annotations["pipeline_error"] = None
        annotations["pipeline_pid"] = None
        save_annotations(project_dir, annotations)
        logger.info(
            "Pipeline finished successfully for project: %s", project_dir)
    except Exception as e:
        logger.exception("Pipeline failed for project: %s", project_dir)
        annotations = load_annotations(project_dir)
        annotations["junction_detection_pipeline_status"] = PipelineStatus.Failed
        annotations["pipeline_error"] = str(e)
        annotations["pipeline_pid"] = None
        save_annotations(project_dir, annotations)
        sys.exit(1)
    finally:
        clear_progress(project_dir)


if __name__ == "__main__":
    main()
