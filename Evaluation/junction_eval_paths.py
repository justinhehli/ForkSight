"""Resolve the junction detection evaluation dataset / prediction directories.

Shared by infer_patches_junction_{nnunet,sam}.py and
compute_metrics_junction_detection.py. Kept free of heavy imports so it can be
used from every model environment.

By default the directories come from JUNCTION_DETECTION_DATASET_DIR and
JUNCTION_PRED_DIR. Passing --dataset-dir evaluates on another dataset (e.g.
the synthetic one created by
JunctionDetection/PreProcessing/create_synthetic_junction_detection_dataset.py).
Predictions and metrics are cached per model, so for such a dataset they are
kept apart from the default ones by deriving the directories from the dataset
directory name:
    predictions: <JUNCTION_PRED_DIR>/<dataset dir name>/   (unless --pred-dir)
    metrics:     <EVALUATION_OUTPUT_DIR>/junction_detection_<dataset dir name>/
                 (no extra prefix if the name already starts with junction_detection)
"""
import argparse
import os
from dataclasses import dataclass
from pathlib import Path


@dataclass
class JunctionEvalPaths:
    dataset_dir: Path
    pred_dir: Path
    # None when evaluating on the default dataset (JUNCTION_DETECTION_DATASET_DIR)
    dataset_tag: str | None

    def output_dir(self, evaluation_output_dir: str | Path) -> Path:
        name = "junction_detection"
        if self.dataset_tag is not None:
            # e.g. dataset dir "junction_detection_synthetic" -> same output dir name
            name = (self.dataset_tag if self.dataset_tag.startswith(name)
                    else f"{name}_{self.dataset_tag}")
        return Path(evaluation_output_dir) / name


def add_junction_eval_path_args(parser: argparse.ArgumentParser):
    parser.add_argument("--dataset-dir", type=Path, default=None,
                        help="junction detection dataset to evaluate on (images/ + "
                        "relabeling_data.csv); overrides JUNCTION_DETECTION_DATASET_DIR")
    parser.add_argument("--pred-dir", type=Path, default=None,
                        help="directory for per-model patch predictions; overrides "
                        "JUNCTION_PRED_DIR (default with --dataset-dir: "
                        "<JUNCTION_PRED_DIR>/<dataset dir name>)")


def resolve_junction_eval_paths(args: argparse.Namespace) -> JunctionEvalPaths:
    """Call after the environment has been loaded (load_forksight_env)."""
    env_dataset_dir = os.getenv("JUNCTION_DETECTION_DATASET_DIR")
    env_pred_dir = os.getenv("JUNCTION_PRED_DIR")

    dataset_dir = args.dataset_dir or env_dataset_dir
    if dataset_dir is None:
        raise ValueError(
            "--dataset-dir or the JUNCTION_DETECTION_DATASET_DIR environment "
            "variable must be set.")
    dataset_dir = Path(dataset_dir)

    is_default_dataset = env_dataset_dir is not None and \
        dataset_dir.resolve() == Path(env_dataset_dir).resolve()
    dataset_tag = None if is_default_dataset else dataset_dir.resolve().name

    if args.pred_dir is not None:
        pred_dir = args.pred_dir
    elif env_pred_dir is None:
        raise ValueError(
            "--pred-dir or the JUNCTION_PRED_DIR environment variable must be set.")
    elif dataset_tag is None:
        pred_dir = Path(env_pred_dir)
    else:
        pred_dir = Path(env_pred_dir) / dataset_tag

    print(f"Junction detection dataset: {dataset_dir}")
    print(f"Junction prediction dir:    {pred_dir}")
    return JunctionEvalPaths(dataset_dir, pred_dir, dataset_tag)
