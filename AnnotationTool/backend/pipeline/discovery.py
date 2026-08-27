"""Project discovery for the annotation tool.

The tool runs in one of three environments (ANNOTATION_TOOL_ENV): PROD, DEV
or TRAIN - see ToolEnvironment below. Each has its own parent dir
(PROJECTS_PARENT_DIR_<ENV> in .annotation_tool_env).

In PROD/DEV, a "project" is a base folder containing raw microscope tiles,
by default at ``<base_folder>/LayersData/highmag/Tile Set (N)/Tile_X-Y-000000_0-000.tif``
(see DEFAULT_TILE_GLOB_PATTERNS - configurable, and overridable per project).
A base folder can be nested arbitrarily deep anywhere under
PROJECTS_PARENT_DIR; which directories are even offered as project
candidates is determined by DEFAULT_PROJECT_DISCOVERY_RULES (also
configurable, globally only). The user registers individual candidates in a
small registry file stored in this parent directory. Each "registered"
project is identified by its path relative to PROJECTS_PARENT_DIR.

In TRAIN, PROJECTS_PARENT_DIR directly contains the TIF images to process
(no LayersData/highmag nesting) and is itself the only "project" - there is
no discovery/registration step, see TRAIN_PROJECT_NAME.
"""

import copy
import json
import os
import re
from enum import Enum
from pathlib import Path

from dotenv import dotenv_values

_ANNOTATION_TOOL_ENV_PATH = Path(
    __file__).resolve().parents[2] / ".annotation_tool_env"
_env_values = dotenv_values(_ANNOTATION_TOOL_ENV_PATH)


def _get_env(key: str, default: str) -> str:
    return _env_values.get(key) or os.getenv(key) or default


class ToolEnvironment(str, Enum):
    PROD = "PROD"
    DEV = "DEV"
    TRAIN = "TRAIN"


def _get_tool_environment() -> ToolEnvironment:
    raw = _get_env("ANNOTATION_TOOL_ENV", "DEV").strip().upper()
    try:
        return ToolEnvironment(raw)
    except ValueError:
        valid = ", ".join(e.value for e in ToolEnvironment)
        raise ValueError(
            f"ANNOTATION_TOOL_ENV must be one of [{valid}], got '{raw}'")


def _get_projects_parent_dir(environment: ToolEnvironment) -> Path:
    key = f"PROJECTS_PARENT_DIR_{environment.value}"
    val = _get_env(key, "")
    if not val:
        raise ValueError(
            f"{key} must be set in {_ANNOTATION_TOOL_ENV_PATH}")
    return Path(val)


TOOL_ENVIRONMENT = _get_tool_environment()
IS_TRAIN_ENV = TOOL_ENVIRONMENT == ToolEnvironment.TRAIN

PROJECTS_PARENT_DIR = _get_projects_parent_dir(TOOL_ENVIRONMENT)

# In TRAIN mode, PROJECTS_PARENT_DIR is itself the only project - it's
# exposed to the frontend under this fixed name instead of going through
# discovery/registration.
TRAIN_PROJECT_NAME = "training-data"

AUTOMATIC_FORK_DETECTION_DIR_NAME = _get_env(
    "AUTOMATIC_FORK_DETECTION_DIR_NAME", "AutomaticForkDetection")
REGISTRY_FILENAME = _get_env(
    "REGISTRY_FILENAME", ".forksight-annotator-projects.json")

# PROD/DEV enforce the LayersData/highmag/Tile Set (N)/ nesting by default;
# TRAIN just discovers all TIFs directly under the project dir. Configurable
# globally (see load_pipeline_settings) and overridable per project (see
# load_project_tile_settings).
DEFAULT_TILE_GLOB_PATTERNS = ["*.tif"] if IS_TRAIN_ENV else [
    "LayersData/highmag/Tile Set (*)/*.tif"]

# A project discovery rule (nested list) is a list of conditions that must ALL be met
# (a file or a possibly-nested subfolder existing) for a directory to qualify as
# a project; a directory qualifies if it satisfies ANY configured rule (in the list of rules = outer list)
# Configurable globally only, see load_pipeline_settings.
DEFAULT_PROJECT_DISCOVERY_RULES = [
    [
        {"type": "file", "pattern": "*.mapsxml"},
        {"type": "dir", "pattern": "LayersData/highmag"},
    ],
]

SEGMENTATION_DIR_NAME = "Segmentation"
SEGMENTATION_PROBABILITIES_DIR_NAME = "SegmentationProbabilities"
SEGMENTATION_PATCHES_DIR_NAME = "SegmentationPatches"

PIPELINE_TMP_DIR_PREFIX = "forksight_pipeline_"
SEGMENTATION_TMP_DIR_PREFIX = "forksight_seg_"


DEFAULT_TARGET_JUNCTION_COUNT = int(
    _get_env("DEFAULT_TARGET_JUNCTION_COUNT", "100"))
DEFAULT_STAGED_SAMPLE_COUNT = int(
    _get_env("DEFAULT_STAGED_SAMPLE_COUNT", "300"))


def _discovery_condition_matches(path: Path, condition: dict) -> bool:
    pattern = (condition.get("pattern") or "").strip()
    if not pattern:
        return False
    kind = condition.get("type")
    if kind == "file":
        return any(p.is_file() for p in path.glob(pattern))
    if kind == "dir":
        return any(p.is_dir() for p in path.glob(pattern))
    return False


def is_valid_project_dir(path: Path, rules: list[list[dict]]) -> bool:
    return any(
        rule and all(_discovery_condition_matches(path, condition)
                     for condition in rule)
        for rule in rules
    )


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
    """Global settings, apply to every project under `parent_dir` unless a
    project overrides them (currently only tile_glob_patterns can be
    overridden, see load_project_tile_settings). Keys match the registry
    JSON's own field names, with defaults filled in for any that are
    missing."""
    data = _load_registry(parent_dir)
    return {
        "pipeline_mode": data.get("pipeline_mode", "sequential"),
        "sequential_target_junction_count": data.get(
            "sequential_target_junction_count", DEFAULT_TARGET_JUNCTION_COUNT),
        "staged_sample_count": data.get(
            "staged_sample_count", DEFAULT_STAGED_SAMPLE_COUNT),
        "tile_glob_patterns": data.get(
            "tile_glob_patterns") or list(DEFAULT_TILE_GLOB_PATTERNS),
        "project_discovery_rules": data.get(
            "project_discovery_rules") or copy.deepcopy(DEFAULT_PROJECT_DISCOVERY_RULES),
    }


def save_pipeline_settings(
    parent_dir: Path, pipeline_mode: str, sequential_target_junction_count: int,
    staged_sample_count: int, tile_glob_patterns: list[str],
    project_discovery_rules: list[list[dict]],
) -> None:
    data = _load_registry(parent_dir)
    data["pipeline_mode"] = pipeline_mode
    data["sequential_target_junction_count"] = sequential_target_junction_count
    data["staged_sample_count"] = staged_sample_count
    data["tile_glob_patterns"] = tile_glob_patterns
    data["project_discovery_rules"] = project_discovery_rules
    _save_registry(parent_dir, data)


def load_project_tile_settings(project_dir: Path) -> dict:
    """Per-project override of the global tile_glob_patterns, stored in the
    project's own AutomaticForkDetection dir rather than the global registry.
    `tile_glob_patterns_override` is None when the project just uses the
    global default patterns."""
    project_dir = Path(project_dir)
    override = _load_registry(project_dir).get("tile_glob_patterns") or None
    return {
        "tile_glob_patterns_override": override,
        "effective_tile_glob_patterns": override or load_pipeline_settings(
            PROJECTS_PARENT_DIR)["tile_glob_patterns"],
    }


def save_project_tile_glob_patterns_override(project_dir: Path, patterns: list[str] | None) -> None:
    project_dir = Path(project_dir)
    data = _load_registry(project_dir)
    if patterns:
        data["tile_glob_patterns"] = patterns
    else:
        data.pop("tile_glob_patterns", None)
    _save_registry(project_dir, data)


def find_candidate_project_dirs(parent_dir: Path) -> list[Path]:
    parent_dir = Path(parent_dir)
    rules = load_pipeline_settings(parent_dir)["project_discovery_rules"]
    found = []
    for dirpath, dirnames, _ in os.walk(parent_dir):
        current = Path(dirpath)
        if is_valid_project_dir(current, rules):
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
    base_folder = Path(base_folder)
    patterns = load_project_tile_settings(
        base_folder)["effective_tile_glob_patterns"]
    found: set[Path] = set()
    for pattern in patterns:
        found.update(base_folder.glob(pattern))
    return sorted(found)


def get_tile_display_name(tile_path: Path) -> str:
    """Name shown for a tile in the UI/exports"""
    tile_path = Path(tile_path)
    if IS_TRAIN_ENV:
        return tile_path.name

    tile_name = tile_path.stem.replace("-000000_0-000", "")
    try:
        tile_number = tile_name.split("_", 1)[-1].replace("-", " ")
        return f"{tile_path.parent.name} - Tile {tile_number}"
    except:
        return f"{tile_path.parent.name} - {tile_name}"


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
