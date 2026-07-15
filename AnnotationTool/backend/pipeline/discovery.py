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


def is_valid_project_dir(path: Path) -> bool:
    return any(p.is_file() for p in path.glob("*.mapsxml")) and (path / "LayersData" / "highmag").is_dir()


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
