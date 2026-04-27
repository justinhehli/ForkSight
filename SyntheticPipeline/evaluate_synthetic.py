"""Evaluate a trained nnUNet model on the synthetic-dataset test split.

This script is the companion to ``train_nnunet.py``.  It assumes the
dataset layout::

    <dataset_dir>/
        images/<stem>.png      (input images)
        masks/<stem>.png       (binary GT masks, only used for context)
        annotations.csv        (columns: image, x, y, label)
        split.json             (written by train_nnunet.py)

and loads the trained model from::

    <nnunet_root>/nnUNet_results/Dataset<ID>_<NAME>/<trainer>__nnUNetPlans__<cfg>/

Pipeline per test image:
  1. Run the nnUNet predictor on the full image (no patching).
  2. Convert the 0/1 class output to a (1, H, W) float32 tensor.
  3. Run ``detect_junctions_in_segmentation_mask`` to get 3-way / 4-way
     coordinates + skeleton.
  4. Greedy NN match of predictions against GT annotations.
  5. Per-fiber evaluation (connected components).

Aggregated metrics (overall + per-sample) are written to a timestamped
output directory under ``<dataset_dir>/evaluation/`` (or ``--out-dir``).

Example
-------
    python SyntheticPipeline/evaluate_synthetic.py \\
        --dataset-dir /scratch/me/synth_v1 \\
        --nnunet-root /scratch/me/synth_v1/nnunet \\
        --dataset-id 900 --dataset-name SynthV1 \\
        --trainer nnUNetTrainer --fold 0
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

# Allow running as `python SyntheticPipeline/evaluate_synthetic.py`
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from JunctionDetection.SkeletonizeDetect.segmentation_junction_detection import (
    detect_junctions_in_segmentation_mask,
)
from Evaluation.compute_metrics_junction_detection import (
    _match_predictions_to_gt,
    _compute_metrics,
    _compute_image_level_metrics,
    _aggregate_fiber_metrics,
    _image_to_sample,
)
from Evaluation.fiber_evaluation import (
    label_fibers,
    match_gt_to_fibers,
    compute_fiber_metrics,
)
from Segmentation.Util.nnunet_wandb_util import (
    initialize_nnunet_predictor,
    nnunet_folder_name,
)


_JUNCTION_TYPE_3_WAY = "3-way"
_JUNCTION_TYPE_4_WAY = "4-way"

# Recognised annotation labels. Supports both the real-dataset label names
# and simple "3-way" / "4-way" labels that synthetic generators often use.
_LABEL_TO_TYPE: dict[str, str | None] = {
    "Normal Fork": _JUNCTION_TYPE_3_WAY,
    "Crossing": _JUNCTION_TYPE_4_WAY,
    "Reversed Fork": _JUNCTION_TYPE_4_WAY,
    "Negative": None,
    "3-way": _JUNCTION_TYPE_3_WAY,
    "4-way": _JUNCTION_TYPE_4_WAY,
    "3way": _JUNCTION_TYPE_3_WAY,
    "4way": _JUNCTION_TYPE_4_WAY,
}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--dataset-dir", required=True, type=Path,
                        help="Synthetic dataset root.")
    parser.add_argument("--nnunet-root", required=True, type=Path,
                        help="nnUNet root (containing nnUNet_results/).")
    parser.add_argument("--dataset-id", type=int, default=900)
    parser.add_argument("--dataset-name", default="Synthetic")
    parser.add_argument("--trainer", default="nnUNetTrainer")
    parser.add_argument("--configuration", default="2d")
    parser.add_argument("--fold", type=int, default=0,
                        help="nnUNet fold to use (single fold).")
    parser.add_argument("--checkpoint", default="checkpoint_final.pth",
                        help="Checkpoint filename to load (default "
                             "checkpoint_final.pth; use checkpoint_best.pth "
                             "for the best validation checkpoint).")
    parser.add_argument("--matching-threshold", type=float, default=75.0,
                        help="Max pixel distance for a GT↔pred match.")
    parser.add_argument("--split-json", type=Path, default=None,
                        help="Override split file (default: "
                             "<dataset-dir>/split.json).")
    parser.add_argument("--out-dir", type=Path, default=None,
                        help="Output directory (default: "
                             "<dataset-dir>/evaluation/<timestamp>/).")
    parser.add_argument("--device", type=int, default=0,
                        help="CUDA device index (default 0).")
    parser.add_argument("--save-plots", action="store_true",
                        help="Save per-image overlay plots.")
    parser.add_argument("--is-test", action="store_true",
                        help="Only evaluate the first test image.")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def _load_gt_annotations(csv_path: Path) -> dict[str, list[dict]]:
    """Load GT annotations from annotations.csv.

    CSV must have columns: ``image, x, y, label``.  Labels are mapped via
    ``_LABEL_TO_TYPE``; unknown labels raise an error.
    """
    df = pd.read_csv(csv_path)
    for col in ("image", "x", "y", "label"):
        if col not in df.columns:
            raise SystemExit(
                f"annotations.csv missing required column '{col}'")

    gt: dict[str, list[dict]] = {}
    for _, row in df.iterrows():
        stem = str(row["image"])
        if stem.endswith(".png"):
            stem = stem[:-4]
        if stem not in gt:
            gt[stem] = []
        label = str(row["label"]).strip()
        if label not in _LABEL_TO_TYPE:
            raise SystemExit(
                f"Unknown label '{label}' in {csv_path}. "
                f"Expected one of {sorted(_LABEL_TO_TYPE)}.")
        jtype = _LABEL_TO_TYPE[label]
        if jtype is not None:
            gt[stem].append({
                "x": float(row["x"]),
                "y": float(row["y"]),
                "type": jtype,
            })
    return gt


def _load_split(split_path: Path) -> list[str]:
    if not split_path.is_file():
        raise SystemExit(
            f"Split file not found: {split_path} — did you run train_nnunet.py?")
    with open(split_path) as f:
        split = json.load(f)
    if "test" not in split:
        raise SystemExit(f"Split file {split_path} has no 'test' entry.")
    return list(split["test"])


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------


def _stage_inputs(
    test_image_paths: list[Path],
    inp_dir: Path,
) -> list[list[str]]:
    """Copy each test image into ``inp_dir`` as ``<stem>_0000.png`` and
    return the file list expected by ``predict_from_files``."""
    inp_dir.mkdir(parents=True, exist_ok=True)
    file_lists: list[list[str]] = []
    for img_path in test_image_paths:
        dst = inp_dir / f"{img_path.stem}_0000.png"
        shutil.copy(img_path, dst)
        file_lists.append([str(dst)])
    return file_lists


def _run_nnunet_inference(
    predictor,
    test_image_paths: list[Path],
    tmp_root: Path,
) -> Path:
    """Run nnUNet inference on full images and return the output dir."""
    inp_dir = tmp_root / "inputs"
    out_dir = tmp_root / "predictions"
    out_dir.mkdir(parents=True, exist_ok=True)

    file_lists = _stage_inputs(test_image_paths, inp_dir)
    print(f"  Running predict_from_files on {len(file_lists)} image(s)")
    predictor.predict_from_files(
        file_lists,
        str(out_dir),
        save_probabilities=False,
        overwrite=True,
        num_processes_preprocessing=2,
        num_processes_segmentation_export=2,
    )
    return out_dir


def _load_pred_mask(pred_dir: Path, stem: str) -> torch.Tensor:
    """Load nnUNet 0/1 class PNG as a (1, H, W) float32 tensor."""
    pred_path = pred_dir / f"{stem}.png"
    if not pred_path.is_file():
        raise SystemExit(f"Missing nnUNet prediction: {pred_path}")
    arr = np.array(Image.open(pred_path))
    if arr.ndim == 3:
        arr = arr[..., 0]
    mask = (arr > 0).astype(np.float32)
    return torch.from_numpy(mask).unsqueeze(0)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def _save_plot(
    img_path: Path,
    pred_mask: torch.Tensor,
    coords_3way: np.ndarray,
    coords_4way: np.ndarray,
    skeleton: np.ndarray | None,
    gt_annotations: list[dict],
    out_path: Path,
    title: str,
) -> None:
    import matplotlib.pyplot as plt

    img = np.array(Image.open(img_path).convert("L"))
    mask_np = pred_mask.squeeze().cpu().numpy()

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(img, cmap="gray")
    ax.imshow(np.ma.masked_where(mask_np == 0, mask_np),
              cmap="autumn", alpha=0.35)
    if skeleton is not None:
        ax.imshow(np.ma.masked_where(skeleton == 0, skeleton),
                  cmap="cool", alpha=0.9)

    if len(coords_3way) > 0:
        ax.plot(coords_3way[:, 0], coords_3way[:, 1], "o",
                markerfacecolor="none", markeredgecolor="lime",
                markersize=10, label="Pred 3-way")
    if len(coords_4way) > 0:
        ax.plot(coords_4way[:, 0], coords_4way[:, 1], "o",
                markerfacecolor="none", markeredgecolor="orange",
                markersize=10, label="Pred 4-way")

    gt_3 = np.array([[a["x"], a["y"]] for a in gt_annotations
                     if a["type"] == _JUNCTION_TYPE_3_WAY])
    gt_4 = np.array([[a["x"], a["y"]] for a in gt_annotations
                     if a["type"] == _JUNCTION_TYPE_4_WAY])
    if len(gt_3) > 0:
        ax.plot(gt_3[:, 0], gt_3[:, 1], "x", color="red",
                markersize=9, label="GT 3-way")
    if len(gt_4) > 0:
        ax.plot(gt_4[:, 0], gt_4[:, 1], "x", color="yellow",
                markersize=9, label="GT 4-way")

    ax.set_title(title, fontsize=9)
    ax.legend(loc="upper right", fontsize=8)
    ax.set_axis_off()
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------


def _process_image(
    pred_mask: torch.Tensor,
    gt_annotations: list[dict],
    matching_threshold: float,
) -> tuple[list[dict], list[dict], np.ndarray, np.ndarray, np.ndarray]:
    coords_3way, coords_4way, skeleton = detect_junctions_in_segmentation_mask(
        pred_mask)
    if len(coords_3way) > 0 or len(coords_4way) > 0:
        pred_coords = np.concatenate([coords_3way, coords_4way], axis=0)
    else:
        pred_coords = np.empty((0, 2))
    pred_types = ([_JUNCTION_TYPE_3_WAY] * len(coords_3way)
                  + [_JUNCTION_TYPE_4_WAY] * len(coords_4way))
    pred_rows, fn_annots = _match_predictions_to_gt(
        pred_coords, pred_types, gt_annotations, matching_threshold)
    for r in pred_rows:
        r["source"] = "nnunet_full"
    return pred_rows, fn_annots, coords_3way, coords_4way, skeleton


def main() -> None:
    args = _parse_args()

    images_dir = args.dataset_dir / "images"
    csv_path = args.dataset_dir / "annotations.csv"
    if not images_dir.is_dir():
        raise SystemExit(f"Images directory not found: {images_dir}")
    if not csv_path.is_file():
        raise SystemExit(
            f"annotations.csv not found at {csv_path}. This script needs GT "
            "annotations to compute metrics.")

    split_path = args.split_json or (args.dataset_dir / "split.json")
    test_names = _load_split(split_path)

    test_image_paths: list[Path] = []
    for name in test_names:
        p = images_dir / name
        if not p.is_file():
            raise SystemExit(f"Test image missing on disk: {p}")
        test_image_paths.append(p)
    if args.is_test:
        test_image_paths = test_image_paths[:1]
    print(f"Evaluating on {len(test_image_paths)} test image(s).")

    gt_by_image = _load_gt_annotations(csv_path)
    print(f"Loaded GT for {len(gt_by_image)} image(s) from {csv_path.name}")

    # --- Locate trained model ---
    dataset_dir_name = f"Dataset{args.dataset_id:03d}_{args.dataset_name}"
    model_dir = (args.nnunet_root / "nnUNet_results" / dataset_dir_name
                 / nnunet_folder_name(args.trainer))
    if args.configuration != "2d":
        model_dir = (args.nnunet_root / "nnUNet_results" / dataset_dir_name
                     / f"{args.trainer}__nnUNetPlans__{args.configuration}")
    if not model_dir.is_dir():
        raise SystemExit(f"Trained model directory not found: {model_dir}")
    print(f"Model dir: {model_dir}")

    # nnUNet needs these env vars to resolve plans/preprocessed paths for
    # its own sanity checks even though we load plans directly from model_dir.
    os.environ["nnUNet_raw"] = str(args.nnunet_root / "nnUNet_raw")
    os.environ["nnUNet_preprocessed"] = str(
        args.nnunet_root / "nnUNet_preprocessed")
    os.environ["nnUNet_results"] = str(args.nnunet_root / "nnUNet_results")

    device = torch.device(
        f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    predictor = initialize_nnunet_predictor(
        model_dir, device,
        folds=(args.fold,),
        checkpoint=args.checkpoint,
    )

    # --- Output directory ---
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = args.out_dir or (
        args.dataset_dir / "evaluation" / timestamp)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output dir: {out_dir}")
    plot_dir = out_dir / "plots" if args.save_plots else None
    if plot_dir is not None:
        plot_dir.mkdir(parents=True, exist_ok=True)

    # --- Inference ---
    with tempfile.TemporaryDirectory(prefix="synth_eval_") as _tmp:
        tmp_root = Path(_tmp)
        pred_out_dir = _run_nnunet_inference(
            predictor, test_image_paths, tmp_root)

        # --- Evaluation loop ---
        pred_rows_all: list[dict] = []
        fn_all: list[dict] = []
        pred_csv_rows: list[dict] = []
        fiber_csv_rows: list[dict] = []
        fiber_metrics_all: list[dict] = []
        image_stats: list[dict] = []

        preds_by_image: dict[str, list[dict]] = {}
        fns_by_image: dict[str, list[dict]] = {}
        fiber_metrics_by_image: dict[str, dict] = {}

        for idx, img_path in enumerate(test_image_paths):
            stem = img_path.stem
            gt_annotations = gt_by_image.get(stem, [])

            pred_mask = _load_pred_mask(pred_out_dir, stem)

            pred_rows, fn_annots, coords_3way, coords_4way, skeleton = \
                _process_image(pred_mask, gt_annotations,
                               args.matching_threshold)

            pred_rows_all.extend(pred_rows)
            fn_all.extend(fn_annots)
            preds_by_image[stem] = pred_rows
            fns_by_image[stem] = fn_annots

            for r in pred_rows:
                pred_csv_rows.append({"image": stem, **r})

            image_stats.append({
                "image": stem,
                "gt_positive": int(len(gt_annotations) > 0),
                "pred_positive": int(len(pred_rows) > 0),
            })

            labeled_mask, fibers = label_fibers(
                pred_mask, coords_3way, coords_4way)
            gt_by_fiber = match_gt_to_fibers(labeled_mask, gt_annotations)
            img_fiber_rows, img_fiber_metrics = compute_fiber_metrics(
                fibers, gt_by_fiber)
            for r in img_fiber_rows:
                fiber_csv_rows.append({"image": stem, **r})
            fiber_metrics_all.append(img_fiber_metrics)
            fiber_metrics_by_image[stem] = img_fiber_metrics

            if plot_dir is not None:
                _save_plot(
                    img_path, pred_mask, coords_3way, coords_4way,
                    skeleton, gt_annotations,
                    plot_dir / f"{stem}.png",
                    title=f"{stem}  (GT={len(gt_annotations)}, "
                          f"pred={len(pred_rows)})",
                )

            print(f"  [{idx+1}/{len(test_image_paths)}] {stem}  "
                  f"GT={len(gt_annotations)} pred={len(pred_rows)} "
                  f"FN={len(fn_annots)}")

    # --- Aggregate ---
    metrics = _compute_metrics(pred_rows_all, fn_all)
    image_metrics = _compute_image_level_metrics(image_stats)
    fiber_agg = _aggregate_fiber_metrics(fiber_metrics_all)

    # --- Per-sample (if stems have a recognisable sample prefix) ---
    sample_to_stems: dict[str, list[str]] = {}
    for s in preds_by_image:
        sample_to_stems.setdefault(_image_to_sample(s), []).append(s)

    per_sample_rows: list[dict] = []
    if len(sample_to_stems) > 1:
        for sample, stems in sorted(sample_to_stems.items()):
            s_pred_rows: list[dict] = []
            s_fn: list[dict] = []
            s_img: list[dict] = []
            s_fib: list[dict] = []
            for st in stems:
                s_pred_rows.extend(preds_by_image[st])
                s_fn.extend(fns_by_image[st])
                s_img.append(next(
                    x for x in image_stats if x["image"] == st))
                if st in fiber_metrics_by_image:
                    s_fib.append(fiber_metrics_by_image[st])
            row = {"sample": sample, "n_images": len(stems)}
            row.update(_compute_metrics(s_pred_rows, s_fn))
            row.update(_compute_image_level_metrics(s_img))
            row.update(_aggregate_fiber_metrics(s_fib))
            per_sample_rows.append(row)

    # --- Save ---
    pd.DataFrame(pred_csv_rows).to_csv(
        out_dir / "predictions.csv", index=False)
    pd.DataFrame(image_stats).to_csv(
        out_dir / "image_level.csv", index=False)
    if fiber_csv_rows:
        pd.DataFrame(fiber_csv_rows).to_csv(
            out_dir / "fibers.csv", index=False)
    if per_sample_rows:
        pd.DataFrame(per_sample_rows).to_csv(
            out_dir / "per_sample.csv", index=False)

    summary = {
        "dataset_dir": str(args.dataset_dir),
        "model_dir": str(model_dir),
        "n_test_images": len(test_image_paths),
        "matching_threshold": args.matching_threshold,
        "fold": args.fold,
        "checkpoint": args.checkpoint,
        "junction_metrics": metrics,
        "image_metrics": image_metrics,
        "fiber_metrics": fiber_agg,
    }
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(summary, f, indent=2)

    # --- Print summary ---
    print("\n" + "=" * 60)
    print("Junction localisation (threshold "
          f"{args.matching_threshold:.0f}px)")
    print("=" * 60)
    print(f"  loc  P={metrics['precision_loc']:.3f} "
          f"R={metrics['recall_loc']:.3f} F1={metrics['f1_loc']:.3f}  "
          f"(TP={metrics['tp_loc']} FP={metrics['fp_loc']} "
          f"FN={metrics['fn_loc']})")
    print(f"  3-way P={metrics['class_precision_3way']:.3f} "
          f"R={metrics['class_recall_3way']:.3f} "
          f"F1={metrics['class_f1_3way']:.3f}  "
          f"(TP={metrics['class_tp_3way']} "
          f"FP={metrics['class_fp_3way']} "
          f"FN={metrics['class_fn_3way']})")
    print(f"  4-way P={metrics['class_precision_4way']:.3f} "
          f"R={metrics['class_recall_4way']:.3f} "
          f"F1={metrics['class_f1_4way']:.3f}  "
          f"(TP={metrics['class_tp_4way']} "
          f"FP={metrics['class_fp_4way']} "
          f"FN={metrics['class_fn_4way']})")
    print(f"  type accuracy: {metrics['type_accuracy']:.3f}")

    print(f"\n  [image]  P={image_metrics['image_precision']:.3f} "
          f"R={image_metrics['image_recall']:.3f} "
          f"F1={image_metrics['image_f1']:.3f} "
          f"acc={image_metrics['image_accuracy']:.3f} "
          f"(N={image_metrics['image_n']})")

    if fiber_agg:
        print(f"  [fiber]  P={fiber_agg['fiber_precision']:.3f} "
              f"R={fiber_agg['fiber_recall']:.3f} "
              f"F1={fiber_agg['fiber_f1']:.3f} "
              f"| class acc={fiber_agg['fiber_class_accuracy']:.3f}")

    if per_sample_rows:
        print("\n  Per-sample:")
        for r in per_sample_rows:
            print(f"    [{r['sample']}] N={r['n_images']:3d}  "
                  f"loc F1={r['f1_loc']:.3f}  "
                  f"3w F1={r.get('class_f1_3way', 0.0):.3f}  "
                  f"4w F1={r.get('class_f1_4way', 0.0):.3f}  "
                  f"img F1={r['image_f1']:.3f}  "
                  f"fib F1={r.get('fiber_f1', 0.0):.3f}")

    print(f"\nSaved metrics to {out_dir}")


if __name__ == "__main__":
    main()
