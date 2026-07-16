"""Standalone runner for one project's automatic junction-detection pipeline,
spawned as a fully independent OS process by the API backend

Runs with the backend's own (lightweight) interpreter, not the pipeline venv;
this script only orchestrates the detection stages (segmentation_worker.py, detection_worker.py)

Usage:
    python -m AnnotationTool.backend.pipeline.pipeline_runner --project-dir <path>
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
from AnnotationTool.backend.pipeline.progress_util import clear_progress
from AnnotationTool.backend.pipeline.run_pipeline import run_junction_detection_pipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-dir", required=True)
    args = parser.parse_args()
    project_dir = Path(args.project_dir)

    logger.info("Pipeline runner started for project: %s", project_dir)
    try:
        annotations = load_annotations(project_dir)
        updated = run_junction_detection_pipeline(project_dir, annotations)
        updated["junction_detection_pipeline_status"] = PipelineStatus.Done
        updated["pipeline_error"] = None
        updated["pipeline_pid"] = None
        save_annotations(project_dir, updated)
        logger.info("Pipeline finished successfully for project: %s", project_dir)
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
