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
import re
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


DEFAULT_TARGET_JUNCTION_COUNT = int(
    _get_env("DEFAULT_TARGET_JUNCTION_COUNT", "100"))
DEFAULT_STAGED_SAMPLE_PERCENTAGE = float(
    _get_env("DEFAULT_STAGED_SAMPLE_PERCENTAGE", "10"))


def is_valid_project_dir(path: Path) -> bool:
    return any(p.is_file() for p in path.glob("*.mapsxml")) and (path / "LayersData" / "highmag").is_dir()


def fork_detection_dir(base_dir: Path) -> Path:
    return Path(base_dir) / AUTOMATIC_FORK_DETECTION_DIR_NAME


def _registry_path(parent_dir: Path) -> Path:
    return fork_detection_dir(parent_dir) / REGISTRY_FILENAME


def _load_registry(parent_dir: Path) -> dict:
    p = _registry_path(parent_dir)
    if not p.is_file():
        return {}
    return json.loads(p.read_text(encoding="utf-8"))


def _save_registry(parent_dir: Path, data: dict) -> None:
    p = _registry_path(parent_dir)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, indent=2), encoding="utf-8")


def load_registered_projects(parent_dir: Path) -> set[str]:
    return set(_load_registry(parent_dir).get("projects", []))


def save_registered_projects(parent_dir: Path, names: list[str]) -> None:
    data = _load_registry(parent_dir)
    data["projects"] = sorted(set(names))
    _save_registry(parent_dir, data)


def load_pipeline_settings(parent_dir: Path) -> dict:
    """Keys match the registry JSON's own field names, with defaults filled
    in for any that are missing."""
    data = _load_registry(parent_dir)
    return {
        "pipeline_mode": data.get("pipeline_mode", "sequential"),
        "sequential_target_junction_count": data.get(
            "sequential_target_junction_count", DEFAULT_TARGET_JUNCTION_COUNT),
        "staged_sample_percentage": data.get(
            "staged_sample_percentage", DEFAULT_STAGED_SAMPLE_PERCENTAGE),
    }


def save_pipeline_settings(
    parent_dir: Path, pipeline_mode: str, sequential_target_junction_count: int,
    staged_sample_percentage: float,
) -> None:
    data = _load_registry(parent_dir)
    data["pipeline_mode"] = pipeline_mode
    data["sequential_target_junction_count"] = sequential_target_junction_count
    data["staged_sample_percentage"] = staged_sample_percentage
    _save_registry(parent_dir, data)


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


_NETWORK_FS_TYPES = {
    "cifs", "smb3", "smbfs", "nfs", "nfs2", "nfs3", "nfs4", "fuse.sshfs",
}


def _unescape_mtab_field(field: str) -> str:
    # /proc/mounts octal-escapes spaces, tabs, newlines and backslashes (fstab convention).
    return re.sub(r"\\([0-7]{3})", lambda m: chr(int(m.group(1), 8)), field)


def resolve_unc_path(path: Path) -> Path:
    """If `path` lives on a mapped network mount, return the equivalent mount source"""
    try:
        lines = Path("/proc/mounts").read_text(encoding="utf-8").splitlines()
    except OSError:
        return path

    resolved = path.resolve()
    best_mount_point: Path | None = None
    best_source = ""
    best_fs_type = ""
    for line in lines:
        fields = line.split()
        if len(fields) < 3:
            continue
        source, mount_point, fs_type = (
            _unescape_mtab_field(f) for f in fields[:3])
        mount_point_path = Path(mount_point)
        try:
            resolved.relative_to(mount_point_path)
        except ValueError:
            continue
        if best_mount_point is None or len(mount_point_path.parts) > len(best_mount_point.parts):
            best_mount_point, best_source, best_fs_type = mount_point_path, source, fs_type

    if best_mount_point is None or best_fs_type not in _NETWORK_FS_TYPES:
        return path

    rel = resolved.relative_to(best_mount_point)
    return Path(best_source, *rel.parts)
