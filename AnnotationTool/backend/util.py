import platform
from pathlib import Path


def get_repo_root() -> Path:
    start = Path(__file__).resolve()
    for parent in [start] + list(start.parents):
        if (parent / ".git").exists():
            return parent
    raise RuntimeError("No repo root ('.git' folder) found")


def venv_python_executable(venv_dir: Path) -> Path:
    venv_dir = Path(venv_dir)
    if platform.system() == "Windows":
        return venv_dir / "Scripts" / "python.exe"
    return venv_dir / "bin" / "python"
