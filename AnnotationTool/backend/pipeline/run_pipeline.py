"""Orchestrates the automatic junction-detection pipeline for a single project

Actual image processing (segmentation, postprocessing, junction detection) 
happens in separate subprocesses (segmentation_worker.py, detection_worker.py), 
run inside a dedicated pipeline venv. manifest.json/results.json handoff
on disk is the only thing exchanged between this orchestrator and the two subprocesses.
"""

import json
import logging
import os
import shutil
import subprocess
import tempfile
import uuid
from pathlib import Path

from dotenv import dotenv_values

from AnnotationTool.backend.util import get_repo_root, venv_python_executable
from Segmentation.PreProcessing.General.tile_naming_util import get_display_name

from AnnotationTool.backend.pipeline.discovery import (
    PIPELINE_TMP_DIR_PREFIX,
    SEGMENTATION_PATCHES_DIR_NAME,
    SEGMENTATION_TMP_DIR_PREFIX,
    find_project_tiles,
)
from AnnotationTool.backend.pipeline.progress_util import write_progress

logger = logging.getLogger(__name__)

_PIPELINE_ENV_PATH = Path(__file__).resolve().parents[2] / ".pipeline_env"


class PipelineConfig:
    def __init__(self, env_path: Path = _PIPELINE_ENV_PATH):
        values = dotenv_values(env_path)

        def _get(key: str, default: str | None = None, required: bool = False) -> str:
            val = values.get(key) or os.getenv(key) or default
            if required and not val:
                raise ValueError(f"{key} must be set in {env_path}")
            return val

        self.pipeline_venv = Path(_get("PIPELINE_VENV", required=True))
        self.nnunet_model_dir = Path(_get("NNUNET_MODEL_DIR", required=True))
        self.nnunet_device = int(_get("NNUNET_DEVICE", "0"))


def _run_worker(module: str, worker_args: list[str], config: PipelineConfig) -> None:
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

    logger.info("Starting %s: %s", module, " ".join(cmd))
    proc = subprocess.Popen(
        cmd, cwd=str(repo_root), env=env,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1,
    )
    for line in proc.stdout:
        logger.info("[%s] %s", module, line.rstrip())
    returncode = proc.wait()
    if returncode != 0:
        raise RuntimeError(
            f"{module} subprocess failed (exit {returncode}); see the log above for details")
    logger.info("%s finished successfully", module)


def cleanup_stale_temp_dirs() -> None:
    tmp_root = Path(tempfile.gettempdir())
    for prefix in (PIPELINE_TMP_DIR_PREFIX, SEGMENTATION_TMP_DIR_PREFIX):
        for d in tmp_root.glob(f"{prefix}*"):
            shutil.rmtree(d, ignore_errors=True)


def run_junction_detection_pipeline(project_dir: Path, annotations: dict) -> dict:
    """Run the pipeline for all project tiles not yet present in annotations.

    Mutates and returns `annotations["images"]` in place; does not persist it
    to disk itself - the caller owns the read-modify-write + status handling
    around the annotations.json file.
    """
    project_dir = Path(project_dir)
    config = PipelineConfig()

    known_source_tifs = {img["source_tif"]
                         for img in annotations["images"].values()}
    new_tiles = [
        t for t in find_project_tiles(project_dir)
        if t.relative_to(project_dir).as_posix() not in known_source_tifs
    ]
    logger.info("Found %d new tile(s) to process in %s",
                len(new_tiles), project_dir)
    if not new_tiles:
        logger.info("Nothing to do.")
        return annotations

    tiles_manifest = [
        {
            "id": str(uuid.uuid4()),
            "source_tif": t.relative_to(project_dir).as_posix(),
            "display_name": get_display_name(t),
        }
        for t in new_tiles
    ]

    with tempfile.TemporaryDirectory(prefix=PIPELINE_TMP_DIR_PREFIX) as tmp:
        tmp_root = Path(tmp)
        manifest_path = tmp_root / "manifest.json"
        results_path = tmp_root / "results.json"
        patch_dir = tmp_root / SEGMENTATION_PATCHES_DIR_NAME
        manifest_path.write_text(json.dumps(
            {"tiles": tiles_manifest}), encoding="utf-8")

        write_progress(project_dir, "segmentation", 0, len(new_tiles))
        _run_worker("segmentation_worker", [
            "--project-dir", str(project_dir),
            "--manifest", str(manifest_path),
            "--patch-output-dir", str(patch_dir),
            "--model-dir", str(config.nnunet_model_dir),
            "--device", str(config.nnunet_device),
        ], config)

        write_progress(project_dir, "detection", 0, len(new_tiles))
        _run_worker("detection_worker", [
            "--project-dir", str(project_dir),
            "--manifest", str(manifest_path),
            "--patch-dir", str(patch_dir),
            "--results-out", str(results_path),
        ], config)

        results = json.loads(results_path.read_text(encoding="utf-8"))
        annotations["images"].update(results["images"])

    logger.info("Pipeline stages complete; %d image(s) updated",
                len(results["images"]))
    return annotations
