"""Bubblewrap (bwrap) sandboxing shared by the backend and the detection pipeline.

`sandbox_prefix` builds a bwrap command prefix that mounts the entire
filesystem read-only, then re-mounts a given allowlist of directories
read-write at their own paths. Prepend it to a command's argv to run that
command unable to delete or modify anything outside the allowlist, while
still able to read everything (source TIFs, the pipeline venv, the nnU-Net
model directory, etc.) exactly as before.

Bwrap sandboxes are nestable: a process already running inside one of these
sandboxes can apply a second, narrower one to a subprocess it spawns (see
main.py's run_junction_detection, which further restricts the per-project
pipeline run beyond what the backend itself is allowed).
"""
import os
import shutil
from pathlib import Path

# Set by run_sandboxed.py so the backend can tell if it's running sandboxed
SANDBOX_ENV_VAR = "FORKSIGHT_BWRAP_SUPERVISED"

# Exit code the backend uses to ask run_sandboxed.py's supervisor loop to
# rebuild the sandbox (picking up newly-registered projects) and relaunch it.
RESTART_EXIT_CODE = 75


def is_sandboxed() -> bool:
    return os.environ.get(SANDBOX_ENV_VAR) == "1"


def _bwrap_executable() -> str:
    bwrap = shutil.which("bwrap")
    if not bwrap:
        raise RuntimeError(
            "bwrap (bubblewrap) not found on PATH; install it (e.g. `apt install "
            "bubblewrap`) before starting this process")
    return bwrap


def sandbox_prefix(read_write_dirs: list[Path]) -> list[str]:
    """Bwrap prefix: whole filesystem read-only, `read_write_dirs` read-write.

    Each directory in `read_write_dirs` must already exist - bwrap can't bind
    a path that isn't there, so create it beforehand if needed.
    """
    args = [
        _bwrap_executable(),
        "--ro-bind", "/", "/",
        "--dev-bind", "/dev", "/dev",
        "--proc", "/proc",
    ]
    for d in read_write_dirs:
        d = Path(d)
        if not d.is_dir():
            raise FileNotFoundError(
                f"Cannot grant bwrap write access to nonexistent directory: {d}")
        args += ["--bind", str(d), str(d)]
    args.append("--")
    return args
