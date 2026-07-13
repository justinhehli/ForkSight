import io
import json
import os
import subprocess
import sys
import threading
from enum import Enum
from pathlib import Path


def _find_repo_root() -> Path:
    start = Path(__file__).resolve()
    for parent in [start] + list(start.parents):
        if (parent / ".git").exists():
            return parent
    raise RuntimeError("No repo root ('.git' folder) found")


REPO_ROOT = _find_repo_root()
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from pydantic import BaseModel

from AnnotationTool.backend.pipeline.discovery import (
    list_candidate_dirs,
    load_registered_projects,
    save_registered_projects,
)
from AnnotationTool.backend.pipeline.annotations_store import (
    PipelineStatus,
    load_annotations,
    pipeline_log_path,
    save_annotations,
)
from AnnotationTool.backend.pipeline.process_util import is_pid_running, terminate_process_tree
from AnnotationTool.backend.pipeline.run_pipeline import cleanup_stale_temp_dirs
from AnnotationTool.backend.util import get_repo_root
from Segmentation.PreProcessing.General.tif_to_png import convert_tif_to_png

load_dotenv(get_repo_root() / "AnnotationTool" / ".annotation_tool_env")
PROJECTS_PARENT_DIR = Path(os.environ["PROJECTS_PARENT_DIR"])

app = FastAPI(title="ForkSight Annotator API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# models
# ====================


class JunctionType(str, Enum):
    ReplicationFork = "Replication Fork"
    ReversedFork = "Reversed Fork"


class Point(BaseModel):
    id: str
    x: int
    y: int
    label: JunctionType


class ImageAnnotations(BaseModel):
    processed: bool
    points: list[Point]


class ProjectSelection(BaseModel):
    names: list[str]


# concurrency
# ====================
# One pipeline may run at a time globally, as determined from annotations.json
# This lock only makes the check-then-set sequence atomic within this backend process
_pipeline_start_lock = threading.Lock()

# Per-project locks guard read-modify-write of annotations.json.
_project_locks: dict[str, threading.Lock] = {}
_project_locks_guard = threading.Lock()


def _get_project_lock(project: str) -> threading.Lock:
    with _project_locks_guard:
        if project not in _project_locks:
            _project_locks[project] = threading.Lock()
        return _project_locks[project]


# helpers
# ====================

def project_dir(project: str) -> Path:
    p = PROJECTS_PARENT_DIR / project
    registered = load_registered_projects(PROJECTS_PARENT_DIR)
    if project not in registered or not p.is_dir():
        raise HTTPException(404, f"Project '{project}' not found")
    return p


# routes
# ====================


@app.get("/projects")
def list_projects() -> list[str]:
    if not PROJECTS_PARENT_DIR.exists():
        raise HTTPException(
            500, f"PROJECTS_PARENT_DIR does not exist: {PROJECTS_PARENT_DIR}")
    registered = load_registered_projects(PROJECTS_PARENT_DIR)
    return sorted(name for name in registered if (PROJECTS_PARENT_DIR / name).is_dir())


@app.get("/project-candidates")
def get_project_candidates() -> list[dict]:
    if not PROJECTS_PARENT_DIR.exists():
        raise HTTPException(
            500, f"PROJECTS_PARENT_DIR does not exist: {PROJECTS_PARENT_DIR}")
    return list_candidate_dirs(PROJECTS_PARENT_DIR)


@app.post("/project-candidates")
def set_project_candidates(selection: ProjectSelection) -> list[dict]:
    save_registered_projects(PROJECTS_PARENT_DIR, selection.names)
    return list_candidate_dirs(PROJECTS_PARENT_DIR)


@app.get("/projects/{project:path}/images")
def list_images(project: str):
    pd = project_dir(project)
    ann = load_annotations(pd)
    images = [
        {"id": image_id, "name": img["display_name"],
            "processed": img.get("processed", False)}
        for image_id, img in ann["images"].items()
    ]
    images.sort(key=lambda i: i["name"])
    return images


@app.get("/projects/{project:path}/images/{image_id}")
def serve_image(project: str, image_id: str):
    pd = project_dir(project)
    ann = load_annotations(pd)
    img_ann = ann["images"].get(image_id)
    if img_ann is None:
        raise HTTPException(404, "Image not found")

    tif_path = pd / img_ann["source_tif"]
    if not tif_path.exists():
        raise HTTPException(404, "Source TIF not found")

    png = convert_tif_to_png(tif_path)
    buf = io.BytesIO()
    png.save(buf, format="PNG")
    return Response(content=buf.getvalue(), media_type="image/png")


@app.get("/projects/{project:path}/annotations")
def get_annotations(project: str) -> dict:
    return load_annotations(project_dir(project))


@app.put("/projects/{project:path}/annotations/{image_id}", status_code=204)
def save_image_annotations(project: str, image_id: str, data: ImageAnnotations):
    pd = project_dir(project)
    with _get_project_lock(project):
        ann = load_annotations(pd)
        if image_id not in ann["images"]:
            raise HTTPException(404, f"Image '{image_id}' not found")
        ann["images"][image_id].update(data.model_dump())
        save_annotations(pd, ann)


@app.post("/projects/{project:path}/export")
def export_project(project: str):
    ann = load_annotations(project_dir(project))
    images = ann["images"]

    all_points = [p for img_ann in images.values()
                  for p in img_ann.get("points", [])]
    replication_forks = sum(
        1 for p in all_points if p["label"] == JunctionType.ReplicationFork)
    reversed_forks = sum(
        1 for p in all_points if p["label"] == JunctionType.ReversedFork)
    processed_count = sum(1 for img_ann in images.values()
                          if img_ann.get("processed", False))

    export_data = {
        "summary": {
            "total_images": len(images),
            "processed_images": processed_count,
            "replication_fork_count": replication_forks,
            "reversed_fork_count": reversed_forks,
            "replication_reversed_ratio": round(replication_forks / reversed_forks, 3) if reversed_forks > 0 else None,
        },
        "images": images,
    }

    content = json.dumps(export_data, indent=2, ensure_ascii=False)
    safe_name = project.replace("/", "_")
    return Response(
        content=content,
        media_type="application/json",
        headers={
            "Content-Disposition": f'attachment; filename="{safe_name}_annotations.json"'},
    )


@app.get("/projects/{project:path}/pipeline-log")
def get_pipeline_log(project: str):
    pd = project_dir(project)
    log_path = pipeline_log_path(pd)
    if not log_path.is_file():
        return Response(content="", media_type="text/plain")
    return Response(content=log_path.read_text(encoding="utf-8", errors="replace"),
                    media_type="text/plain")


def _find_running_pipeline() -> str | None:
    """Return the name of a project with an active pipeline, if any.
    Projects stuck at "Running" whose recorded pipeline_runner.py process is 
    no longer alive is set to "Failed" here so it doesn't block new runs.
    """
    for name in sorted(load_registered_projects(PROJECTS_PARENT_DIR)):
        pd = PROJECTS_PARENT_DIR / name
        if not pd.is_dir():
            continue
        ann = load_annotations(pd)
        if ann.get("junction_detection_pipeline_status") != PipelineStatus.Running:
            continue
        if is_pid_running(ann.get("pipeline_pid")):
            return name
        with _get_project_lock(name):
            ann = load_annotations(pd)
            if ann.get("junction_detection_pipeline_status") == PipelineStatus.Running:
                ann["junction_detection_pipeline_status"] = PipelineStatus.Failed
                ann["pipeline_error"] = (
                    "Pipeline process is no longer running "
                    "(backend restarted or the process crashed)."
                )
                ann["pipeline_pid"] = None
                save_annotations(pd, ann)
    return None


@app.get("/pipeline-status")
def get_pipeline_status():
    return {"running_project": _find_running_pipeline()}


@app.post("/projects/{project:path}/stop-junction-detection")
def stop_junction_detection(project: str):
    pd = project_dir(project)

    with _get_project_lock(project):
        ann = load_annotations(pd)
        if ann.get("junction_detection_pipeline_status") != PipelineStatus.Running:
            raise HTTPException(
                409, f"No junction detection pipeline is running for project '{project}'")
        pid = ann.get("pipeline_pid")

    # Terminating the process tree can take a few seconds (graceful attempt
    # before escalating to a forceful kill) - do this outside the lock.
    terminate_process_tree(pid)
    cleanup_stale_temp_dirs()

    with _get_project_lock(project):
        ann = load_annotations(pd)
        ann["junction_detection_pipeline_status"] = PipelineStatus.Failed
        ann["pipeline_error"] = "Pipeline stopped by user."
        ann["pipeline_pid"] = None
        save_annotations(pd, ann)

    return {"status": PipelineStatus.Failed, "project": project}


@app.post("/projects/{project:path}/run-junction-detection")
def run_junction_detection(project: str):
    pd = project_dir(project)

    with _pipeline_start_lock:
        running_project = _find_running_pipeline()
        if running_project is not None:
            raise HTTPException(
                409, f"A junction detection pipeline is already running for project '{running_project}'")

        # Spawn the pipeline as its own OS process, this request returns immediately
        # and pipeline_runner.py itself updates annotations.json when it's done
        log_path = pipeline_log_path(pd)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_file = open(log_path, "w", encoding="utf-8")
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        try:
            proc = subprocess.Popen(
                [sys.executable, "-u", "-m", "AnnotationTool.backend.pipeline.pipeline_runner",
                 "--project-dir", str(pd)],
                cwd=str(get_repo_root()),
                env=env,
                stdout=log_file,
                stderr=subprocess.STDOUT,
            )
        finally:
            # Popen duplicates the handle for the child; safe to close our copy.
            log_file.close()

        with _get_project_lock(project):
            ann = load_annotations(pd)
            ann["junction_detection_pipeline_status"] = PipelineStatus.Running
            ann["pipeline_error"] = None
            ann["pipeline_pid"] = proc.pid
            save_annotations(pd, ann)

    return {"status": PipelineStatus.Running, "project": project}
