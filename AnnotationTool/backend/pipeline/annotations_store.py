import json
import os
from enum import Enum
from pathlib import Path

from AnnotationTool.backend.pipeline.discovery import AUTOMATIC_FORK_DETECTION_DIR_NAME


class PipelineStatus(str, Enum):
    Idle = "Idle"
    Running = "Running"
    Done = "Done"
    Failed = "Failed"


def annotations_file_path(project_dir: Path) -> Path:
    return Path(project_dir) / AUTOMATIC_FORK_DETECTION_DIR_NAME / "annotations.json"


def pipeline_log_path(project_dir: Path) -> Path:
    return Path(project_dir) / AUTOMATIC_FORK_DETECTION_DIR_NAME / "pipeline.log"


def load_annotations(project_dir: Path) -> dict:
    p = annotations_file_path(project_dir)
    if not p.exists():
        return {
            "junction_detection_pipeline_status": PipelineStatus.Idle,
            "pipeline_error": None,
            "pipeline_pid": None,
            "images": {},
        }
    return json.loads(p.read_text(encoding="utf-8"))


def save_annotations(project_dir: Path, data: dict) -> None:
    p = annotations_file_path(project_dir)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + ".tmp")
    tmp.write_text(json.dumps(
        data, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, p)
