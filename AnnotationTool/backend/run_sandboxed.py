"""Supervisor that runs the FastAPI backend inside a bubblewrap sandbox.

Replaces `uvicorn AnnotationTool.backend.main:app` as the way to start the
backend. Each (re)launch runs as:

    bwrap --ro-bind / / ... --bind <rw dir> <rw dir> ... -- python -m uvicorn AnnotationTool.backend.main:app <args>

Read-write is granted only for:
  - PROJECTS_PARENT_DIR/AutomaticForkDetection (the project registry)
  - <project>/AutomaticForkDetection, for every currently *registered* project
  - the pipeline scratch dir (used by AnnotationTool.backend.pipeline.run_pipeline
    for scratch dirs): PIPELINE_TMP_DIR from .pipeline_env if set, else /tmp.
    TMPDIR is set to match inside the sandbox, so tempfile.gettempdir() (and
    hence run_pipeline.py's scratch dirs) resolve there instead of /tmp.
  - the pipeline venv (PIPELINE_VENV in .pipeline_env) - nnU-Net writes a
    trainer-class file into its own site-packages at run time

Everything else on disk - including the project TIFs/mapsxml themselves, and
the AutomaticForkDetection dir of a not-yet-registered candidate - stays
read-only, so the backend can read but never delete or modify pre-existing
project data, and can't create its output dir for a project before that
project is actually selected in the tool.

Since bwrap's bind mounts are fixed for the life of the sandboxed process,
registering a new project (main.py's set_project_candidates) can't just start
writing into that project's AutomaticForkDetection dir - there's no bind for
it yet. Instead main.py exits with bwrap_util.RESTART_EXIT_CODE, and this
supervisor loop rebuilds the read-write list (creating the new project's dir,
now unsandboxed) and relaunches. A plain Ctrl-C/SIGTERM stops the supervisor
for good instead of relaunching.

Don't pass `--reload`: uvicorn's reload supervisor runs the app in a separate
child worker process, so the backend exiting with RESTART_EXIT_CODE would
only kill that worker - not the process this script is actually watching -
and the restart would never be noticed.

Usage:
    python -m AnnotationTool.backend.run_sandboxed [uvicorn args...]
"""
import os
import signal
import subprocess
import sys
from pathlib import Path

from dotenv import load_dotenv

from AnnotationTool.backend.bwrap_util import (
    RESTART_EXIT_CODE,
    SANDBOX_ENV_VAR,
    sandbox_prefix,
)
from AnnotationTool.backend.pipeline.discovery import (
    fork_detection_dir,
    load_registered_projects,
)
from AnnotationTool.backend.pipeline.run_pipeline import PipelineConfig
from AnnotationTool.backend.util import get_repo_root


def _projects_parent_dir() -> Path:
    load_dotenv(get_repo_root() / "AnnotationTool" / ".annotation_tool_env")
    return Path(os.environ["PROJECTS_PARENT_DIR"])


def _read_write_dirs(projects_parent_dir: Path, pipeline_config: PipelineConfig) -> list[Path]:
    global_dir = fork_detection_dir(projects_parent_dir)
    global_dir.mkdir(parents=True, exist_ok=True)

    dirs = [global_dir]
    for name in load_registered_projects(projects_parent_dir):
        d = fork_detection_dir(projects_parent_dir / name)
        d.mkdir(parents=True, exist_ok=True)
        dirs.append(d)

    dirs.append(pipeline_config.pipeline_venv)
    if pipeline_config.pipeline_tmp_dir is not None and pipeline_config.pipeline_tmp_dir.is_dir():
        dirs.append(pipeline_config.pipeline_tmp_dir)
    else:
        dirs.append(Path("/tmp"))
    return dirs


def _build_command(extra_args: list[str]) -> list[str]:
    pipeline_config = PipelineConfig()
    setenv = (
        {"TMPDIR": str(pipeline_config.pipeline_tmp_dir)}
        if pipeline_config.pipeline_tmp_dir is not None
        else None
    )
    return sandbox_prefix(
        _read_write_dirs(_projects_parent_dir(), pipeline_config), setenv=setenv
    ) + [
        sys.executable, "-m", "uvicorn",
        "AnnotationTool.backend.main:app", *extra_args,
    ]


def main():
    extra_args = sys.argv[1:]
    env = os.environ.copy()
    env[SANDBOX_ENV_VAR] = "1"

    while True:
        proc = subprocess.Popen(_build_command(extra_args), env=env)

        def _forward(signum, _frame, proc=proc):
            proc.terminate()

        signal.signal(signal.SIGINT, _forward)
        signal.signal(signal.SIGTERM, _forward)

        returncode = proc.wait()
        print(f"[run_sandboxed] backend process exited with code {returncode}",
              file=sys.stderr, flush=True)
        if returncode != RESTART_EXIT_CODE:
            sys.exit(returncode)
        print("[run_sandboxed] A project was registered; restarting the "
              "backend to grant it write access...", file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()
