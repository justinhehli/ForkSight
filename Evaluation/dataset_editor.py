"""CLI to edit the junction detection dataset.

The dataset consists of:
  - ``<JUNCTION_DETECTION_DATASET_DIR>/images/*.png``
  - ``<JUNCTION_DETECTION_DATASET_DIR>/relabeling_data.csv``
      columns: ``image``, ``x``, ``y``, ``label``
      valid labels: "Normal Fork", "Crossing", "Reversed Fork", "Negative"

This script supports:
  1. Removing one or more images (and their CSV annotations)
  2. Adding new point labels to a specific image

Both subcommands back up ``relabeling_data.csv`` before writing and support
``--dry-run`` to preview changes.

Examples
--------
Remove images by stem (spaces or underscores both fine):

    python Evaluation/dataset_editor.py remove \
        "Dani_Funghi tileset 18 012 010" \
        "Dani Funghi tileset 18 012 011"

Add a single label:

    python Evaluation/dataset_editor.py add \
        --image "Dani_Funghi tileset 18 012 013" \
        --x 1532 --y 2044 \
        --label "Normal Fork"

Apply a batch of edits from JSON:

    python Evaluation/dataset_editor.py batch edits.json

Where ``edits.json`` is:

    {
      "remove": ["Dani_Funghi tileset 18 012 010"],
      "add": [
        {"image": "Dani_Funghi tileset 18 012 013",
         "x": 1532, "y": 2044, "label": "Normal Fork"}
      ]
    }
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

# Allow `python Evaluation/dataset_editor.py` from repo root
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from Environment import env_utils  # noqa: E402


VALID_LABELS = {"Normal Fork", "Crossing", "Reversed Fork", "Negative"}
CSV_COLUMNS = ["image", "x", "y", "label"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _normalize_stem(name: str) -> str:
    """Canonical form used for matching (underscores → spaces, stripped).

    Stems in the CSV and on disk often mix spaces and underscores, so we
    compare on a normalized form.  Also strips a trailing ``.png`` if given.
    """
    if name.endswith(".png"):
        name = name[:-4]
    return name.replace("_", " ").strip().lower()


def _find_image_files(images_dir: Path, target_stem: str) -> list[Path]:
    """Return all image files whose stem matches ``target_stem`` (normalized)."""
    target = _normalize_stem(target_stem)
    return [
        p for p in images_dir.glob("*.png")
        if _normalize_stem(p.stem) == target
    ]


def _matching_csv_stems(df: pd.DataFrame, target_stem: str) -> set[str]:
    """Return the set of raw CSV stems whose normalized form matches."""
    target = _normalize_stem(target_stem)
    return {
        s for s in df["image"].astype(str).unique()
        if _normalize_stem(s) == target
    }


def _backup_csv(csv_path: Path) -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup = csv_path.with_suffix(f".csv.bak_{ts}")
    shutil.copy2(csv_path, backup)
    return backup


def _load_csv(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    missing = [c for c in CSV_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"CSV {csv_path} is missing required columns: {missing}")
    return df


def _write_csv(df: pd.DataFrame, csv_path: Path) -> None:
    df.to_csv(csv_path, index=False)


def _resolve_paths() -> tuple[Path, Path]:
    env_utils.load_forksight_env()
    dataset_dir = os.getenv("JUNCTION_DETECTION_DATASET_DIR")
    if not dataset_dir:
        raise SystemExit(
            "JUNCTION_DETECTION_DATASET_DIR environment variable must be set.")
    dataset_dir = Path(dataset_dir)
    images_dir = dataset_dir / "images"
    csv_path = dataset_dir / "relabeling_data.csv"
    if not images_dir.is_dir():
        raise SystemExit(f"Images directory not found: {images_dir}")
    if not csv_path.is_file():
        raise SystemExit(f"Annotation CSV not found: {csv_path}")
    return images_dir, csv_path


# ---------------------------------------------------------------------------
# Operations
# ---------------------------------------------------------------------------


def remove_images(
    image_names: list[str],
    images_dir: Path,
    csv_path: Path,
    dry_run: bool,
) -> None:
    df = _load_csv(csv_path)
    n_rows_before = len(df)

    stems_to_drop: set[str] = set()
    files_to_delete: list[Path] = []

    for name in image_names:
        csv_stems = _matching_csv_stems(df, name)
        files = _find_image_files(images_dir, name)

        if not csv_stems and not files:
            print(f"  [warn] no match for '{name}' (csv or disk)")
            continue

        stems_to_drop.update(csv_stems)
        files_to_delete.extend(files)

        print(f"  '{name}':")
        print(f"    files to delete: {[p.name for p in files]}")
        print(f"    csv stems: {sorted(csv_stems)} "
              f"({(df['image'].astype(str).isin(csv_stems)).sum()} rows)")

    if not stems_to_drop and not files_to_delete:
        print("Nothing to do.")
        return

    mask = df["image"].astype(str).isin(stems_to_drop)
    new_df = df[~mask].reset_index(drop=True)
    n_removed = int(mask.sum())

    print(f"\nSummary: {n_removed} rows removed "
          f"({n_rows_before} → {len(new_df)}), "
          f"{len(files_to_delete)} image file(s) to delete.")

    if dry_run:
        print("[dry-run] no changes written.")
        return

    backup = _backup_csv(csv_path)
    print(f"Backed up CSV to {backup}")
    _write_csv(new_df, csv_path)
    for p in files_to_delete:
        p.unlink()
        print(f"  deleted {p.name}")
    print("Done.")


def add_label(
    image_name: str,
    x: float,
    y: float,
    label: str,
    images_dir: Path,
    csv_path: Path,
    dry_run: bool,
) -> None:
    if label not in VALID_LABELS:
        raise SystemExit(
            f"Invalid label '{label}'. Must be one of {sorted(VALID_LABELS)}.")

    df = _load_csv(csv_path)

    # Resolve the canonical stem to use in the CSV: prefer an existing
    # spelling already in the CSV, then an existing file on disk, then the
    # name as provided.
    csv_stems = _matching_csv_stems(df, image_name)
    disk_files = _find_image_files(images_dir, image_name)

    if csv_stems:
        stem = sorted(csv_stems)[0]
    elif disk_files:
        stem = disk_files[0].stem
    else:
        print(f"  [warn] '{image_name}' not found in CSV or on disk; "
              f"will add row with name as given.")
        stem = image_name[:-4] if image_name.endswith(".png") else image_name

    new_row = {"image": stem, "x": float(x), "y": float(y), "label": label}
    print(f"Adding row: {new_row}")

    if dry_run:
        print("[dry-run] no changes written.")
        return

    backup = _backup_csv(csv_path)
    print(f"Backed up CSV to {backup}")
    new_df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    _write_csv(new_df, csv_path)
    print("Done.")


def apply_batch(
    batch_path: Path,
    images_dir: Path,
    csv_path: Path,
    dry_run: bool,
) -> None:
    with open(batch_path) as f:
        spec = json.load(f)

    removes = spec.get("remove", []) or []
    adds = spec.get("add", []) or []

    if not removes and not adds:
        print("Batch file has no 'remove' or 'add' entries.")
        return

    df = _load_csv(csv_path)
    n_rows_before = len(df)

    # --- removes ---------------------------------------------------------
    stems_to_drop: set[str] = set()
    files_to_delete: list[Path] = []
    for name in removes:
        stems_to_drop.update(_matching_csv_stems(df, name))
        files_to_delete.extend(_find_image_files(images_dir, name))

    if stems_to_drop or files_to_delete:
        print(f"Remove: {len(stems_to_drop)} stems, "
              f"{len(files_to_delete)} files")

    # --- adds ------------------------------------------------------------
    new_rows: list[dict] = []
    for item in adds:
        try:
            name = item["image"]
            x = float(item["x"])
            y = float(item["y"])
            lab = item["label"]
        except (KeyError, TypeError, ValueError) as e:
            raise SystemExit(f"Invalid add entry {item}: {e}")
        if lab not in VALID_LABELS:
            raise SystemExit(
                f"Invalid label '{lab}' in batch. "
                f"Must be one of {sorted(VALID_LABELS)}.")

        csv_stems = _matching_csv_stems(df, name) - stems_to_drop
        disk_files = _find_image_files(images_dir, name)
        if csv_stems:
            stem = sorted(csv_stems)[0]
        elif disk_files:
            stem = disk_files[0].stem
        else:
            print(f"  [warn] add for '{name}': not found, using as-is.")
            stem = name[:-4] if name.endswith(".png") else name
        new_rows.append({"image": stem, "x": x, "y": y, "label": lab})

    if new_rows:
        print(f"Add: {len(new_rows)} rows")
        for r in new_rows:
            print(f"  {r}")

    # --- apply -----------------------------------------------------------
    mask = df["image"].astype(str).isin(stems_to_drop)
    new_df = df[~mask].reset_index(drop=True)
    if new_rows:
        new_df = pd.concat(
            [new_df, pd.DataFrame(new_rows)], ignore_index=True)

    print(f"\nSummary: {n_rows_before} → {len(new_df)} CSV rows, "
          f"{len(files_to_delete)} files to delete.")

    if dry_run:
        print("[dry-run] no changes written.")
        return

    backup = _backup_csv(csv_path)
    print(f"Backed up CSV to {backup}")
    _write_csv(new_df, csv_path)
    for p in files_to_delete:
        p.unlink()
        print(f"  deleted {p.name}")
    print("Done.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Edit the junction detection dataset (images + CSV).")
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Show what would change without writing anything.")

    sub = parser.add_subparsers(dest="cmd", required=True)

    p_rm = sub.add_parser("remove", help="Remove images (and their rows).")
    p_rm.add_argument(
        "names", nargs="+",
        help="Image stem(s), e.g. 'Dani_Funghi tileset 18 012 010'.")

    p_add = sub.add_parser("add", help="Add a single label row.")
    p_add.add_argument("--image", required=True, help="Image stem.")
    p_add.add_argument("--x", type=float, required=True)
    p_add.add_argument("--y", type=float, required=True)
    p_add.add_argument(
        "--label", required=True, choices=sorted(VALID_LABELS))

    p_batch = sub.add_parser(
        "batch", help="Apply a batch of edits from a JSON file.")
    p_batch.add_argument("path", type=Path, help="Path to JSON edits file.")

    return parser


def main() -> None:
    args = _build_parser().parse_args()
    images_dir, csv_path = _resolve_paths()

    print(f"Dataset: {images_dir.parent}")
    print(f"CSV:     {csv_path}")
    print(f"Images:  {images_dir}")
    if args.dry_run:
        print("[dry-run] enabled\n")
    else:
        print()

    if args.cmd == "remove":
        remove_images(args.names, images_dir, csv_path, args.dry_run)
    elif args.cmd == "add":
        add_label(
            args.image, args.x, args.y, args.label,
            images_dir, csv_path, args.dry_run)
    elif args.cmd == "batch":
        apply_batch(args.path, images_dir, csv_path, args.dry_run)


if __name__ == "__main__":
    main()
