import os
from pathlib import Path
import json

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel

load_dotenv(Path(__file__).parent.parent / ".env")
DATA_DIR = Path(os.environ["DATA_DIR"])

IMAGE_EXTS = {".jpg", ".jpeg", ".png"}

app = FastAPI(title="ForkSight Annotator API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# helpers
# ====================

def project_dir(project: str) -> Path:
    p = DATA_DIR / project
    if not p.is_dir():
        raise HTTPException(404, f"Project '{project}' not found")
    return p


def annotations_file_path(project_dir: Path) -> Path:
    return project_dir / "annotations.json"


def load_annotations(project_dir: Path) -> dict:
    p = annotations_file_path(project_dir)
    if not p.exists():
        return {"junction_detection_done": False, "images": {}}
    return json.loads(p.read_text(encoding="utf-8"))


def save_annotations(pd: Path, data: dict) -> None:
    annotations_file_path(pd).write_text(json.dumps(
        data, indent=2, ensure_ascii=False), encoding="utf-8")


# models
# ====================

class Point(BaseModel):
    id: str
    x: int
    y: int
    label: str


class ImageAnnotations(BaseModel):
    processed: bool
    points: list[Point]


# routes
# ====================

@app.get("/projects")
def list_projects() -> list[str]:
    if not DATA_DIR.exists():
        raise HTTPException(500, f"DATA_DIR does not exist: {DATA_DIR}")
    return sorted(p.name for p in DATA_DIR.iterdir() if p.is_dir())


@app.get("/projects/{project}/images")
def list_images(project: str):
    pd = project_dir(project)
    annotations = load_annotations(pd)
    image_names = sorted(f.name for f in pd.iterdir()
                         if f.suffix.lower() in IMAGE_EXTS)
    return [
        {"name": n, "processed": annotations["images"].get(
            n, {}).get("processed", False)}
        for n in image_names
    ]


@app.get("/projects/{project}/images/{image_name}")
def serve_image(project: str, image_name: str):
    pd = project_dir(project)
    p = pd / image_name
    if not p.exists() or p.suffix.lower() not in IMAGE_EXTS:
        raise HTTPException(404, "Image not found")
    return FileResponse(str(p))


@app.get("/projects/{project}/annotations")
def get_annotations(project: str) -> dict:
    return load_annotations(project_dir(project))


@app.put("/projects/{project}/annotations/{image_name}")
def save_image_annotations(project: str, image_name: str, data: ImageAnnotations):
    pd = project_dir(project)
    ann = load_annotations(pd)
    ann["images"][image_name] = data.model_dump()
    save_annotations(pd, ann)
    return {"ok": True}


@app.post("/projects/{project}/export")
def export_project(project: str):
    ann = load_annotations(project_dir(project))
    content = json.dumps(ann["images"], indent=2, ensure_ascii=False)
    return Response(
        content=content,
        media_type="application/json",
        headers={
            "Content-Disposition": f'attachment; filename="{project}_annotations.json"'},
    )


@app.post("/projects/{project}/run-junction-detection")
def run_junction_detection(project: str):
    project_dir(project)  # validates project exists
    # TODO: start ML pipeline (e.g. submit SLURM job, spawn subprocess, etc.)
    return {"status": "started", "project": project}
