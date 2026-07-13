"""Project discovery for the annotation tool.

A "project" is a base folder containing raw microscope tiles like
``<base_folder>/LayersData/highmag/Tile Set (N)/Tile_X-Y-000000_0-000.tif``.
A base folder can be nested arbitrarily deep anywhere under
PROJECTS_PARENT_DIR, the user registers individual ones in a small registry 
file  stored in this parent directory. Each "registered" project is 
identified by its path relative to PROJECTS_PARENT_DIR
"""

import json
import os
from pathlib import Path

from dotenv import dotenv_values

_ANNOTATION_TOOL_ENV_PATH = Path(
    __file__).resolve().parents[2] / ".annotation_tool_env"
_env_values = dotenv_values(_ANNOTATION_TOOL_ENV_PATH)


def _get_env(key: str, default: str) -> str:
    return _env_values.get(key) or os.getenv(key) or default


AUTOMATIC_FORK_DETECTION_DIR_NAME = _get_env(
    "AUTOMATIC_FORK_DETECTION_DIR_NAME", "AutomaticForkDetection")
REGISTRY_FILENAME = _get_env(
    "REGISTRY_FILENAME", ".forksight-annotator-projects.json")

TILE_GLOB_PATTERN = "LayersData/highmag/Tile Set (*)/*.tif"

SEGMENTATION_DIR_NAME = "Segmentation"
SEGMENTATION_PATCHES_DIR_NAME = "SegmentationPatches"

PIPELINE_TMP_DIR_PREFIX = "forksight_pipeline_"
SEGMENTATION_TMP_DIR_PREFIX = "forksight_seg_"


def is_valid_project_dir(path: Path) -> bool:
    return (path / "LayersData" / "highmag").is_dir()


def _registry_path(parent_dir: Path) -> Path:
    return Path(parent_dir) / AUTOMATIC_FORK_DETECTION_DIR_NAME / REGISTRY_FILENAME


def load_registered_projects(parent_dir: Path) -> set[str]:
    p = _registry_path(parent_dir)
    if not p.is_file():
        return set()
    data = json.loads(p.read_text(encoding="utf-8"))
    return set(data.get("projects", []))


def save_registered_projects(parent_dir: Path, names: list[str]) -> None:
    _registry_path(parent_dir).write_text(
        json.dumps({"projects": sorted(set(names))}, indent=2),
        encoding="utf-8",
    )


def find_candidate_project_dirs(parent_dir: Path) -> list[Path]:
    parent_dir = Path(parent_dir)
    found = []
    for dirpath, dirnames, _ in os.walk(parent_dir):
        current = Path(dirpath)
        if is_valid_project_dir(current):
            found.append(current)
            dirnames[:] = []
    return sorted(found)


def list_candidate_dirs(parent_dir: Path) -> list[dict]:
    parent_dir = Path(parent_dir)
    registered = load_registered_projects(parent_dir)
    candidates = []
    for path in find_candidate_project_dirs(parent_dir):
        rel = path.relative_to(parent_dir).as_posix()
        candidates.append({
            "name": rel,
            "valid": True,
            "registered": rel in registered,
        })
    return candidates


def find_project_tiles(base_folder: Path) -> list[Path]:
    return sorted(Path(base_folder).glob(TILE_GLOB_PATTERN))
