import json
import os
from pathlib import Path

from AnnotationTool.backend.pipeline.discovery import fork_detection_dir

PROGRESS_FILENAME = "pipeline_progress.json"


def progress_path(project_dir: Path) -> Path:
    return fork_detection_dir(project_dir) / PROGRESS_FILENAME


def write_progress(project_dir: Path, stage: str, completed: int, total: int) -> None:
    p = progress_path(project_dir)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + ".tmp")
    tmp.write_text(json.dumps(
        {"stage": stage, "completed": completed, "total": total}), encoding="utf-8")
    os.replace(tmp, p)


def read_progress(project_dir: Path) -> dict:
    p = progress_path(project_dir)
    if not p.is_file():
        return {"stage": None, "completed": 0, "total": 0}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {"stage": None, "completed": 0, "total": 0}


def clear_progress(project_dir: Path) -> None:
    progress_path(project_dir).unlink(missing_ok=True)
