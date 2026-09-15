import json
import os
from enum import Enum
from pathlib import Path

from AnnotationTool.backend.pipeline.discovery import fork_detection_dir


class PipelineStatus(str, Enum):
    Idle = "Idle"
    Running = "Running"
    Done = "Done"
    Failed = "Failed"


def annotations_file_path(project_dir: Path) -> Path:
    return fork_detection_dir(project_dir) / "annotations.json"


def pipeline_log_path(project_dir: Path) -> Path:
    return fork_detection_dir(project_dir) / "pipeline.log"


def load_annotations(project_dir: Path, must_exist: bool = False) -> dict:
    p = annotations_file_path(project_dir)
    if not p.exists():
        if must_exist:
            raise ValueError(f"annoations file {p} does not exist!")
        return {
            "junction_detection_pipeline_status": PipelineStatus.Idle,
            "pipeline_error": None,
            "pipeline_pid": None,
            "pipeline_mode": None,
            "additional_labels": [],
            "images": {},
        }
    
    data = json.loads(p.read_text(encoding="utf-8"))
    data.setdefault("pipeline_mode", None)
    data.setdefault("additional_labels", [])
    return data


def save_annotations(project_dir: Path, data: dict) -> None:
    p = annotations_file_path(project_dir)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + ".tmp")
    tmp.write_text(json.dumps(
        data, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, p)
