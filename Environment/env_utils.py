import os
from dotenv import load_dotenv
from pathlib import Path


def get_repo_root() -> Path:
    start = Path(__file__).resolve()
    for parent in [start] + list(start.parents):
        if (parent / ".git").exists():
            return parent
    raise RuntimeError("No repo root ('.git' folder) found")


def get_dev_env_file() -> Path:
    """Path of environment file used exclusively during development (model training, pipeline evaluation)"""
    repo_root = get_repo_root()
    env_path = repo_root / "Environment" / ".env"
    if not env_path.exists():
        raise FileNotFoundError(
            f"Warning: .env file not found at expected location: {env_path}")
    return env_path


def get_shared_env_file() -> Path:
    """Path of shared environment file used during development AND production"""
    return get_repo_root() / "Environment" / "shared.env"


def load_shared_env():
    """env variables shared between development (training/evaluation) and 
    the deployed annotation-tool pipeline """
    env_path = get_shared_env_file()
    if env_path.exists():
        load_dotenv(dotenv_path=env_path, override=False)
        print(f"loaded shared environment variables from: {env_path}")


def load_forksight_env():
    load_shared_env()
    env_path = get_dev_env_file()
    load_dotenv(dotenv_path=env_path, override=False)
    print(f"loaded environment variables from: {env_path}")


def load_as_tuple(var: str, default=None, dtype=int) -> tuple:
    val = os.getenv(var, default)
    if val is None or val.strip() == "":
        return None
    parts = val.strip().split(',')
    return tuple(dtype(p.strip()) for p in parts)


def load_as(var: str, dtype, default=None):
    val = os.getenv(var, default)
    if val is None:
        return None
    return dtype(val)


def load_as_bool(var: str, default=False) -> bool:
    val = os.getenv(var, str(default)).lower().strip()
    if val == 'true':
        return True
    elif val == 'false':
        return False
    else:
        raise ValueError(f"Cannot convert {val} to bool")
