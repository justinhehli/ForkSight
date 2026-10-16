"""
Audit .env variables against README documentation and codebase usage.

Outputs:
  1. Variables in .env but NOT documented in README.md
  2. Variables in README.md but NOT present in .env
  3. Variables in .env but NOT referenced in any .py or .ipynb file
  4. Variables loaded in code (os.getenv, env_utils helpers) but NOT in .env or README

Usage:
    python env_audit.py [--env .env] [--readme README.md] [--repo .]
"""

import argparse
import json
import os
import re
from pathlib import Path

from Environment.env_utils import get_dev_env_file


def parse_env_file(filepath: str) -> list[str]:
    """Extract variable names from a .env file, preserving order."""
    names = []
    with open(filepath, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            match = re.match(r"^([A-Za-z_][A-Za-z0-9_]*)\s*=", line)
            if match:
                names.append(match.group(1))
    return names


def parse_readme_table(filepath: str) -> list[str]:
    """
    Extract variable names from the first column of a Markdown pipe-table
    whose header row contains 'Parameter' and 'Description'.
    """
    names = []
    in_target_table = False

    with open(filepath, encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()

            # Detect start of the target table by its header row
            if not in_target_table:
                if re.match(
                    r"^\|.*Parameter.*\|.*Description.*\|", stripped, re.IGNORECASE
                ):
                    in_target_table = True
                continue

            # Skip the separator row  |---|---|
            if re.match(r"^\|[\s\-:]+\|[\s\-:]+\|$", stripped):
                continue

            # End of table: blank line or non-table line
            if not stripped.startswith("|"):
                break

            # Extract the first column value
            cells = [c.strip() for c in stripped.split("|")]
            # split on | gives ['', 'COL1', 'COL2', ''] for |COL1|COL2|
            first_col = cells[1] if len(cells) > 1 else ""
            # Strip optional backticks
            first_col = first_col.strip("`").strip()
            if first_col and re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", first_col):
                names.append(first_col)

    return names


def find_source_files(repo_root: str) -> list[Path]:
    """Recursively find all .py, .ipynb, .sh files, skipping hidden dirs and venvs."""
    skip_dirs = {
        ".git", ".venv", "venv", "env", "__pycache__", "node_modules", ".tox",
    }
    files = []
    for dirpath, dirnames, filenames in os.walk(repo_root):
        dirnames[:] = [
            d for d in dirnames if d not in skip_dirs and not d.startswith(".")
        ]
        for fname in filenames:
            if fname.endswith(".py") or fname.endswith(".ipynb") or fname.endswith(".sh"):
                files.append(Path(dirpath) / fname)
    return files


def extract_text_from_file(filepath: Path) -> str:
    """Return searchable text content from a .py, .ipynb, or .sh file."""
    try:
        raw = filepath.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""

    if filepath.suffix == ".ipynb":
        try:
            notebook = json.loads(raw)
            chunks = []
            for cell in notebook.get("cells", []):
                if cell.get("cell_type") in ("code", "markdown", "raw"):
                    source = cell.get("source", [])
                    if isinstance(source, list):
                        chunks.append("".join(source))
                    else:
                        chunks.append(source)
            return "\n".join(chunks)
        except (json.JSONDecodeError, KeyError):
            return raw

    return raw


def find_code_loaded_vars(repo_root: str) -> list[str]:
    """
    Return all unique env variable names loaded in source files via:
      - os.getenv("VAR")
      - os.environ["VAR"] or os.environ.get("VAR")
      - load_as_tuple("VAR"), load_as("VAR"), load_as_bool("VAR")
    """
    # Patterns: each captures the variable name as group 1
    patterns = [
        re.compile(r'os\.getenv\s*\(\s*["\']([A-Za-z_][A-Za-z0-9_]*)["\']'),
        re.compile(r'os\.environ(?:\.get)?\s*[\[(\s]\s*["\']([A-Za-z_][A-Za-z0-9_]*)["\']'),
        re.compile(r'load_as(?:_tuple|_bool)?\s*\(\s*["\']([A-Za-z_][A-Za-z0-9_]*)["\']'),
    ]

    seen: set[str] = set()
    found: list[str] = []

    for filepath in find_source_files(repo_root):
        text = extract_text_from_file(filepath)
        for pattern in patterns:
            for match in pattern.finditer(text):
                var = match.group(1)
                if var not in seen:
                    seen.add(var)
                    found.append(var)

    return sorted(found)


def find_unused_vars(env_vars: list[str], repo_root: str) -> list[str]:
    """Return env vars that are never referenced in any .py or .ipynb file."""
    source_files = find_source_files(repo_root)

    # Read all source text once
    all_text = "\n".join(extract_text_from_file(f) for f in source_files)

    unused = []
    for var in env_vars:
        # Match the variable name as a whole word (not part of a longer name)
        if not re.search(
            rf"(?<![A-Za-z0-9_]){re.escape(var)}(?![A-Za-z0-9_])", all_text
        ):
            unused.append(var)
    return unused


def print_section(title: str, items: list[str]) -> None:
    """Pretty-print a numbered audit section."""
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")
    if items:
        for item in items:
            print(f"  - {item}")
        print(f"\n  Total: {len(items)}")
    else:
        print("  (none — all good!)")


def main():
    parser = argparse.ArgumentParser(
        description="Audit .env variables against README and codebase."
    )
    parser.add_argument(
        "--readme", default=None, help="Path to README.md (default: README.md)"
    )
    parser.add_argument(
        "--repo", default=None, help="Root of the repo to scan (default: current dir)"
    )
    args = parser.parse_args()

    if args.readme is None or args.repo is None:
        raise ValueError(
            "Error: --readme, and --repo arguments are required.")

    env_file_path = get_dev_env_file()

    env_vars = parse_env_file(env_file_path)
    readme_vars = parse_readme_table(args.readme)

    env_set = set(env_vars)
    readme_set = set(readme_vars)

    print(f"Found {len(env_set)} variable(s) in {str(env_file_path)}")
    print(f"Found {len(readme_set)} variable(s) in {args.readme}")

    # 1. In .env but not in README
    in_env_not_readme = [v for v in env_vars if v not in readme_set]
    print_section("1. In .env but NOT in README", in_env_not_readme)

    # 2. In README but not in .env
    in_readme_not_env = [v for v in readme_vars if v not in env_set]
    print_section("2. In README but NOT in .env", in_readme_not_env)

    # 3. In .env but not used in any .py / .ipynb
    unused = find_unused_vars(env_vars, args.repo)
    print_section("3. In .env but NOT used in any .py/.ipynb file", unused)

    # 4. Loaded in code but not declared in .env or README
    code_vars = find_code_loaded_vars(args.repo)
    known_vars = env_set | readme_set
    undeclared = [v for v in code_vars if v not in known_vars]
    print_section(
        "4. Loaded in code (os.getenv / env_utils) but NOT in .env or README",
        undeclared,
    )

    print()


if __name__ == "__main__":
    main()
