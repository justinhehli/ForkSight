"""Generate a comparison plot for two models across samples.

Reads ``per_sample_<model>.csv`` files written by
``compute_metrics_junction_detection.py`` and produces a multi-panel figure
that compares the models at both the overall level and the per-sample level.

Usage
-----
    python Evaluation/plot_sample_comparison.py \
        /home/ncanov/scratch/forksight_output/evaluation/junction_detection/20260414_141718

If no argument is given, the most recent timestamped subdirectory under
``$EVALUATION_OUTPUT_DIR/junction_detection/`` is used.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from Environment import env_utils  # noqa: E402


# Metrics shown per sample (and overall)
_METRICS = [
    ("loc F1", "f1_loc"),
    ("3-way F1", "class_f1_3way"),
    ("4-way F1", "class_f1_4way"),
    ("image F1", "image_f1"),
    ("fiber F1", "fiber_f1"),
]

# Tiny single-image samples carry no statistical weight; drop them
_MIN_IMAGES_PER_SAMPLE = 3


def _find_latest_eval_dir() -> Path:
    env_utils.load_forksight_env()
    base = os.getenv("EVALUATION_OUTPUT_DIR")
    if not base:
        raise SystemExit("EVALUATION_OUTPUT_DIR is not set.")
    root = Path(base) / "junction_detection"
    candidates = sorted(p for p in root.iterdir() if p.is_dir())
    if not candidates:
        raise SystemExit(f"No timestamped evaluation dirs under {root}")
    return candidates[-1]


def _short_model_name(path: Path) -> str:
    """Pull a compact label from per_sample_<safe_key>.csv."""
    name = path.stem.replace("per_sample_", "")
    return name.split("_")[-1]  # e.g. "ClDiceLoss" / "Wandb"


def _load_per_sample(eval_dir: Path) -> dict[str, pd.DataFrame]:
    files = sorted(eval_dir.glob("per_sample_*.csv"))
    if not files:
        raise SystemExit(f"No per_sample_*.csv files in {eval_dir}")
    return {_short_model_name(f): pd.read_csv(f) for f in files}


def _load_global(eval_dir: Path) -> pd.DataFrame:
    csv = eval_dir / "metrics.csv"
    if not csv.is_file():
        raise SystemExit(f"metrics.csv not found in {eval_dir}")
    return pd.read_csv(csv)


def _plot_grouped_bars(ax, labels, values_by_model, title, ymax=1.0):
    n_models = len(values_by_model)
    n_groups = len(labels)
    x = np.arange(n_groups)
    width = 0.8 / max(n_models, 1)
    colors = plt.cm.tab10(np.linspace(0, 0.9, n_models))

    for i, (model, vals) in enumerate(values_by_model.items()):
        offset = (i - n_models / 2 + 0.5) * width
        bars = ax.bar(x + offset, vals, width=width * 0.9,
                      label=model, color=colors[i])
        ax.bar_label(bars, fmt="%.2f", fontsize=7, padding=1)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylim(0, ymax * 1.18)
    ax.set_ylabel("Score")
    ax.set_title(title, fontsize=11)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.legend(loc="upper right", fontsize=8, framealpha=0.7)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "eval_dir", nargs="?", type=Path,
        help="Timestamped evaluation dir (default: most recent).")
    parser.add_argument(
        "--out", type=Path, default=None,
        help="Output PNG path (default: <eval_dir>/sample_comparison.png)")
    args = parser.parse_args()

    eval_dir = args.eval_dir or _find_latest_eval_dir()
    print(f"Using eval dir: {eval_dir}")

    per_sample = _load_per_sample(eval_dir)
    global_df = _load_global(eval_dir)

    # --- Overall (pp) metrics per model ---
    # Column naming in metrics.csv:
    #   junction / class / image metrics are prefixed with pp_
    #   fiber metrics are stored without a prefix
    _GLOBAL_COLS = [
        "pp_f1_loc",
        "pp_class_f1_3way",
        "pp_class_f1_4way",
        "pp_image_f1",
        "fiber_f1",
    ]
    overall_by_model: dict[str, list[float]] = {}
    for _, row in global_df.iterrows():
        key = row["model"].split("/")[-1]  # e.g. nnUNetTrainerClDiceLoss
        overall_by_model[key] = [float(row[c]) for c in _GLOBAL_COLS]

    # --- Per-sample: union of samples, ordered by N images (desc) ---
    first_df = next(iter(per_sample.values()))
    sample_sizes = first_df.set_index("sample")["n_images"].to_dict()
    samples = [
        s for s, n in sorted(sample_sizes.items(), key=lambda t: -t[1])
        if n >= _MIN_IMAGES_PER_SAMPLE
    ]
    print(f"Samples kept (≥{_MIN_IMAGES_PER_SAMPLE} images): {samples}")

    # Build figure: 1 overall panel + 1 panel per metric × sample
    n_metrics = len(_METRICS)
    fig = plt.figure(figsize=(16, 4 + 3 * n_metrics))
    gs = fig.add_gridspec(n_metrics + 1, 1, hspace=0.55)

    # -- Overall --
    ax0 = fig.add_subplot(gs[0])
    _plot_grouped_bars(
        ax0,
        labels=[name for name, _ in _METRICS],
        values_by_model=overall_by_model,
        title="Overall (all images)",
    )

    # -- Per sample, one panel per metric --
    for mi, (metric_name, metric_col) in enumerate(_METRICS, start=1):
        ax = fig.add_subplot(gs[mi])
        values_by_model: dict[str, list[float]] = {}
        for model, df in per_sample.items():
            df_indexed = df.set_index("sample")
            values_by_model[model] = [
                float(df_indexed.loc[s, metric_col]) if s in df_indexed.index else 0.0
                for s in samples
            ]
        labels = [f"{s}\n(N={sample_sizes[s]})" for s in samples]
        _plot_grouped_bars(
            ax, labels, values_by_model,
            title=f"Per-sample — {metric_name}",
        )

    fig.suptitle(
        f"Model comparison — {eval_dir.name}",
        fontsize=13, y=1.0,
    )

    out_path = args.out or (eval_dir / "sample_comparison.png")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved figure to {out_path}")


if __name__ == "__main__":
    main()
