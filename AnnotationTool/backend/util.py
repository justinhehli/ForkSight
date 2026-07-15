from pathlib import Path


def get_repo_root() -> Path:
    start = Path(__file__).resolve()
    for parent in [start] + list(start.parents):
        if (parent / ".git").exists():
            return parent
    raise RuntimeError("No repo root ('.git' folder) found")


def venv_python_executable(venv_dir: Path) -> Path:
    return Path(venv_dir) / "bin" / "python"
