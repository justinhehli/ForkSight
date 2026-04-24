"""Evaluate pre-computed segmentation patch predictions on junction detection.

Reads binary prediction patches saved by the model-specific inference scripts
(infer_patches_junction_sam.py / infer_patches_junction_nnunet.py) and runs
postprocessing + junction detection + GT matching + metric computation on them.

No model or training-framework libraries are required; this script can be run
in any environment that has the ForkSight core dependencies.

Expected layout under JUNCTION_PRED_DIR:
  <JUNCTION_PRED_DIR>/
    <safe_model_key>/
      metadata.json          {"model_key": str, "dataset": str}
      <image_stem>_patch_00.png
      ...
      <image_stem>_patch_15.png

Required environment variables (loaded via load_forksight_env):
  JUNCTION_DETECTION_DATASET_DIR   root of the junction detection dataset
  JUNCTION_PRED_DIR                directory with per-model prediction subdirs
  JUNCTION_MATCHING_THRESHOLD      max pixel distance for GT↔pred match
                                   (optional, default 75)

Outputs (written to EVALUATION_OUTPUT_DIR/junction_detection/<timestamp>/):
  predictions_<model>.csv   one row per detected junction per image
  metrics.csv               per-model aggregate metrics
"""

import argparse
import json
import os
import re
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from PIL import Image

import Environment.env_utils as env_utils
from Segmentation.PostProcessing.segmentation_postprocessing import (
    postprocess_segmentation_masks,
)
from JunctionDetection.SkeletonizeDetect.segmentation_junction_detection import (
    detect_junctions_in_segmentation_mask,
)
from Evaluation.pipeline_evaluation_shared import (
    load_full_image_as_patches,
    plot_images_masks_junctions,
    PATCH_SIZE,
    GRID_SIZE,
)
from Evaluation.fiber_evaluation import (
    label_fibers,
    match_gt_to_fibers,
    compute_fiber_metrics,
)

_JUNCTION_TYPE_3_WAY = "3-way"
_JUNCTION_TYPE_4_WAY = "4-way"
_N_PATCHES = GRID_SIZE[0] * GRID_SIZE[1]


def _label_to_junction_type(label: str) -> str | None:
    """Map CSV label to '3-way', '4-way', or None (Negative / no junction)."""
    if label in ("Crossing", "Reversed Fork"):
        return _JUNCTION_TYPE_4_WAY
    elif label == "Normal Fork":
        return _JUNCTION_TYPE_3_WAY
    elif label == "Negative":
        return None
    raise ValueError(f"Unknown label in CSV: '{label}'")


def _load_gt_annotations(csv_path: Path) -> dict[str, list[dict]]:
    """Load ground-truth annotations from relabeling_data.csv.

    Returns
    -------
    dict mapping image stem to list of {x, y, type} dicts.
    Images with only Negative labels are included as empty lists.
    """
    df = pd.read_csv(csv_path)
    gt: dict[str, list[dict]] = {}
    for _, row in df.iterrows():
        stem = str(row["image"])
        if stem not in gt:
            gt[stem] = []
        jtype = _label_to_junction_type(str(row["label"]))
        if jtype is not None:
            gt[stem].append({
                "x": float(row["x"]),
                "y": float(row["y"]),
                "type": jtype,
            })
    return gt


def _match_predictions_to_gt(
    pred_coords: np.ndarray,
    pred_types: list[str],
    gt_annotations: list[dict],
    threshold: float,
) -> tuple[list[dict], list[dict]]:
    """Greedy nearest-neighbour matching of predicted junctions to GT.

    Matching is purely spatial (type-agnostic): a prediction matches a GT
    junction if it is within the distance threshold, regardless of type.
    Each GT can be matched at most once; if multiple predictions are within
    the threshold of the same GT, the closest one wins and the rest become
    false positives.

    Parameters
    ----------
    pred_coords   : (N, 2) array of (x, y) predicted junction coordinates.
    pred_types    : list of N strings, '3-way' or '4-way'.
    gt_annotations: list of {x, y, type} dicts for the GT junctions.
    threshold     : maximum pixel distance for a valid match.

    Returns
    -------
    pred_rows      : list of dicts (one per prediction)
    fn_annotations : list of GT annotation dicts that were not matched.
    """
    n_pred = len(pred_coords)
    n_gt = len(gt_annotations)

    if n_pred == 0:
        return [], list(gt_annotations)

    if n_gt == 0:
        pred_rows = [{"x": float(x), "y": float(y), "pred_type": t,
                      "matched_gt_x": None, "matched_gt_y": None,
                      "matched_gt_type": None, "distance": None,
                      "is_tp": False, "is_fp": True}
                     for (x, y), t in zip(pred_coords, pred_types)]
        return pred_rows, []

    gt_coords = np.array([[a["x"], a["y"]] for a in gt_annotations])

    # Compute all pairwise distances between predictions and GT junctions
    diff = pred_coords[:, None, :] - gt_coords[None, :, :]  # (N_pred, N_gt, 2)
    dist_matrix = np.linalg.norm(diff, axis=2)              # (N_pred, N_gt)

    # Collect all candidate (pred, gt) pairs within the threshold,
    # sorted by distance so the greedy loop always assigns the closest first.
    # Matching is type-agnostic; type correctness is evaluated separately.
    candidate_pairs = sorted(
        (dist_matrix[pred_idx, gt_idx], pred_idx, gt_idx)
        for pred_idx in range(n_pred)
        for gt_idx in range(n_gt)
        if dist_matrix[pred_idx, gt_idx] <= threshold
    )

    # Greedy one-to-one assignment: each prediction and each GT can be matched
    # at most once; closer pairs take priority over farther ones
    pred_matched_to_gt: list[int | None] = [None] * n_pred
    gt_matched = [False] * n_gt
    for _, pred_idx, gt_idx in candidate_pairs:
        if pred_matched_to_gt[pred_idx] is None and not gt_matched[gt_idx]:
            pred_matched_to_gt[pred_idx] = gt_idx
            gt_matched[gt_idx] = True

    # Build one output row per prediction
    pred_rows: list[dict] = []
    for pred_idx, (x, y) in enumerate(pred_coords):
        matched_gt_idx = pred_matched_to_gt[pred_idx]
        if matched_gt_idx is not None:
            pred_rows.append({
                "x": float(x),
                "y": float(y),
                "pred_type": pred_types[pred_idx],
                "matched_gt_x": float(gt_coords[matched_gt_idx, 0]),
                "matched_gt_y": float(gt_coords[matched_gt_idx, 1]),
                "matched_gt_type": gt_annotations[matched_gt_idx]["type"],
                "distance": float(dist_matrix[pred_idx, matched_gt_idx]),
                "is_tp": True,
                "is_fp": False,
            })
        else:
            pred_rows.append({
                "x": float(x),
                "y": float(y),
                "pred_type": pred_types[pred_idx],
                "matched_gt_x": None,
                "matched_gt_y": None,
                "matched_gt_type": None,
                "distance": None,
                "is_tp": False,
                "is_fp": True,
            })

    fn_annotations = [a for gt_idx, a in enumerate(gt_annotations)
                      if not gt_matched[gt_idx]]
    return pred_rows, fn_annotations


def _compute_metrics(
    all_pred_rows: list[dict],
    all_fn_annotations: list[dict],
) -> dict:
    """Compute aggregate junction detection metrics across all images."""
    def _prf(tp, fp, fn):
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        return prec, rec, f1

    metrics: dict = {}

    tp_loc = sum(1 for r in all_pred_rows if r["is_tp"])
    fp_loc = sum(1 for r in all_pred_rows if r["is_fp"])
    fn_loc = len(all_fn_annotations)
    prec_loc, rec_loc, f1_loc = _prf(tp_loc, fp_loc, fn_loc)
    metrics.update({
        "tp_loc": tp_loc, "fp_loc": fp_loc, "fn_loc": fn_loc,
        "precision_loc": prec_loc, "recall_loc": rec_loc, "f1_loc": f1_loc,
    })

    matched_rows = [r for r in all_pred_rows if r["is_tp"]]
    type_correct = sum(
        1 for r in matched_rows if r["pred_type"] == r["matched_gt_type"])
    type_incorrect = tp_loc - type_correct
    type_accuracy = type_correct / tp_loc if tp_loc > 0 else 0.0

    # 2×2 type-classification CM (only matched TPs)
    cm_3way_3way = sum(1 for r in matched_rows
                       if r["matched_gt_type"] == _JUNCTION_TYPE_3_WAY
                       and r["pred_type"] == _JUNCTION_TYPE_3_WAY)
    cm_3way_4way = sum(1 for r in matched_rows
                       if r["matched_gt_type"] == _JUNCTION_TYPE_3_WAY
                       and r["pred_type"] == _JUNCTION_TYPE_4_WAY)
    cm_4way_3way = sum(1 for r in matched_rows
                       if r["matched_gt_type"] == _JUNCTION_TYPE_4_WAY
                       and r["pred_type"] == _JUNCTION_TYPE_3_WAY)
    cm_4way_4way = sum(1 for r in matched_rows
                       if r["matched_gt_type"] == _JUNCTION_TYPE_4_WAY
                       and r["pred_type"] == _JUNCTION_TYPE_4_WAY)

    # Per-type FN: GT junctions that were never localised (missed entirely)
    fn_3way = sum(1 for a in all_fn_annotations
                  if a["type"] == _JUNCTION_TYPE_3_WAY)
    fn_4way = sum(1 for a in all_fn_annotations
                  if a["type"] == _JUNCTION_TYPE_4_WAY)

    # Per-type FP: predicted junctions with no matching GT, broken down by pred type
    fp_rows = [r for r in all_pred_rows if r["is_fp"]]
    fp_3way = sum(1 for r in fp_rows if r["pred_type"] == _JUNCTION_TYPE_3_WAY)
    fp_4way = sum(1 for r in fp_rows if r["pred_type"] == _JUNCTION_TYPE_4_WAY)

    # Detection recall per GT type: fraction of GT junctions of that type that were
    # localised at all (regardless of whether the type was classified correctly)
    gt_3way_total = cm_3way_3way + cm_3way_4way + fn_3way
    gt_4way_total = cm_4way_3way + cm_4way_4way + fn_4way
    detection_recall_3way = (cm_3way_3way + cm_3way_4way) / \
        gt_3way_total if gt_3way_total > 0 else 0.0
    detection_recall_4way = (cm_4way_3way + cm_4way_4way) / \
        gt_4way_total if gt_4way_total > 0 else 0.0

    prec_3way, rec_3way, f1_3way = _prf(
        cm_3way_3way, cm_4way_3way, cm_3way_4way)
    prec_4way, rec_4way, f1_4way = _prf(
        cm_4way_4way, cm_3way_4way, cm_4way_3way)

    # Full per-class P/R/F1 at the junction level.
    # For class C (junction type):
    #   TP_C = pred C matched to GT C
    #   FP_C = pred C not matched to a GT C (either wrong type, or no GT)
    #   FN_C = GT C not matched by a pred C (either wrong type, or missed)
    # This captures both localization AND classification errors in a single
    # score, unlike type_precision/recall which only counts type confusion.
    class_tp_3way = cm_3way_3way
    class_fp_3way = cm_4way_3way + fp_3way
    class_fn_3way = cm_3way_4way + fn_3way
    class_tp_4way = cm_4way_4way
    class_fp_4way = cm_3way_4way + fp_4way
    class_fn_4way = cm_4way_3way + fn_4way
    class_prec_3way, class_rec_3way, class_f1_3way = _prf(
        class_tp_3way, class_fp_3way, class_fn_3way)
    class_prec_4way, class_rec_4way, class_f1_4way = _prf(
        class_tp_4way, class_fp_4way, class_fn_4way)

    metrics.update({
        "type_correct": type_correct,
        "type_incorrect": type_incorrect,
        "type_accuracy": type_accuracy,
        "cm_gt3_pred3": cm_3way_3way,
        "cm_gt3_pred4": cm_3way_4way,
        "cm_gt4_pred3": cm_4way_3way,
        "cm_gt4_pred4": cm_4way_4way,
        "fn_3way": fn_3way,
        "fn_4way": fn_4way,
        "fp_3way": fp_3way,
        "fp_4way": fp_4way,
        "gt_3way_total": gt_3way_total,
        "gt_4way_total": gt_4way_total,
        "detection_recall_3way": detection_recall_3way,
        "detection_recall_4way": detection_recall_4way,
        "type_precision_3way": prec_3way,
        "type_recall_3way": rec_3way,
        "type_f1_3way": f1_3way,
        "type_precision_4way": prec_4way,
        "type_recall_4way": rec_4way,
        "type_f1_4way": f1_4way,
        "class_tp_3way": class_tp_3way,
        "class_fp_3way": class_fp_3way,
        "class_fn_3way": class_fn_3way,
        "class_precision_3way": class_prec_3way,
        "class_recall_3way": class_rec_3way,
        "class_f1_3way": class_f1_3way,
        "class_tp_4way": class_tp_4way,
        "class_fp_4way": class_fp_4way,
        "class_fn_4way": class_fn_4way,
        "class_precision_4way": class_prec_4way,
        "class_recall_4way": class_rec_4way,
        "class_f1_4way": class_f1_4way,
    })

    return metrics


_SAMPLE_DATE_PREFIX_RE = re.compile(r"^\d{8}_")


def _image_to_sample(stem: str) -> str:
    """Map an image stem to its sample name.

    Strips any leading 8-digit date prefix and takes everything before
    ``_tileset_`` (case-insensitive).  Examples:
      Dani_Funghi_TileSet_11_Tile_014-015     -> dani_funghi
      20240425_andrea_lila_tileset_14_tile... -> andrea_lila
      Veronica_Sample1_TileSet_19_Tile_011-008 -> veronica_sample1
    """
    s = stem.lower()
    s = _SAMPLE_DATE_PREFIX_RE.sub("", s)
    if "_tileset_" in s:
        return s.split("_tileset_")[0]
    return s


def _aggregate_fiber_metrics(per_image: list[dict]) -> dict:
    """Sum fiber count metrics across images and recompute rates."""
    if not per_image:
        return {}
    agg: dict = {}
    sum_keys = ["fiber_tp", "fiber_fp", "fiber_fn",
                "fiber_class_correct", "fiber_class_ambiguous",
                "fiber_class_incorrect", "fiber_n_unmatched_gt"]
    for k in sum_keys:
        agg[k] = sum(m[k] for m in per_image)
    tp = agg["fiber_tp"]
    fp = agg["fiber_fp"]
    fn = agg["fiber_fn"]
    agg["fiber_precision"] = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    agg["fiber_recall"] = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    p, r = agg["fiber_precision"], agg["fiber_recall"]
    agg["fiber_f1"] = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    agg["fiber_class_accuracy"] = (
        (agg["fiber_class_correct"] + agg["fiber_class_ambiguous"]) / tp
        if tp > 0 else 0.0
    )
    return agg


def _compute_image_level_metrics(image_stats: list[dict]) -> dict:
    """Per-image binary detection stats.

    An image has ``gt_positive = 1`` iff it has ≥1 GT junction (of any type),
    and ``pred_positive = 1`` iff the model predicted ≥1 junction on it.
    Aggregate across all images as a binary classification task.

    Parameters
    ----------
    image_stats : list of ``{'gt_positive': 0/1, 'pred_positive': 0/1}`` dicts.
    """
    n = len(image_stats)
    tp = sum(1 for s in image_stats if s["gt_positive"] and s["pred_positive"])
    fp = sum(1 for s in image_stats
             if not s["gt_positive"] and s["pred_positive"])
    fn = sum(1 for s in image_stats
             if s["gt_positive"] and not s["pred_positive"])
    tn = sum(1 for s in image_stats
             if not s["gt_positive"] and not s["pred_positive"])
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    acc = (tp + tn) / n if n > 0 else 0.0
    return {
        "image_n": n,
        "image_tp": tp,
        "image_fp": fp,
        "image_fn": fn,
        "image_tn": tn,
        "image_precision": prec,
        "image_recall": rec,
        "image_f1": f1,
        "image_accuracy": acc,
    }


def _load_pred_patches(model_pred_dir: Path, image_stem: str) -> torch.Tensor:
    """Load _N_PATCHES patch PNGs for one full image as a (N, 1, H, W) tensor."""
    patches = []
    for idx in range(_N_PATCHES):
        patch_path = model_pred_dir / f"{image_stem}_patch_{idx:02d}.png"
        arr = np.array(Image.open(patch_path))
        if arr.ndim == 3:
            arr = arr[..., 0]
        mask = torch.from_numpy((arr > 0).astype(np.float32)).unsqueeze(0)
        patches.append(mask)
    return torch.stack(patches)


def _process_image(
    pred_mask_patches: torch.Tensor,
    gt_annotations: list[dict],
    matching_threshold: float,
) -> tuple:
    """Run postprocessing + junction detection + GT matching for one full image.

    Parameters
    ----------
    pred_mask_patches : (N, 1, H, W) binary float32 tensor in row-major patch order.

    Returns
    -------
    raw_pred_rows, raw_fn_annots, pp_pred_rows, pp_fn_annots,
    pp_stitched, pp_coords_3way, pp_coords_4way, pp_skeleton
    """
    raw_stitched, _ = postprocess_segmentation_masks(
        pred_mask_patches, grid_size=GRID_SIZE,
        original_input_patch_img_size=PATCH_SIZE,
        remove_small_objects=False,
    )
    raw_stitched = raw_stitched.detach().cpu()

    pp_stitched, _ = postprocess_segmentation_masks(
        pred_mask_patches, grid_size=GRID_SIZE,
        original_input_patch_img_size=PATCH_SIZE,
        remove_small_objects=True,
    )
    pp_stitched = pp_stitched.detach().cpu()

    def _detect_and_match(mask: torch.Tensor, source: str):
        coords_3way, coords_4way, skeleton = detect_junctions_in_segmentation_mask(
            mask)
        if len(coords_3way) > 0 or len(coords_4way) > 0:
            pred_coords = np.concatenate([coords_3way, coords_4way], axis=0)
        else:
            pred_coords = np.empty((0, 2))
        pred_types = ([_JUNCTION_TYPE_3_WAY] * len(coords_3way)
                      + [_JUNCTION_TYPE_4_WAY] * len(coords_4way))
        pred_rows, fn_annots = _match_predictions_to_gt(
            pred_coords, pred_types, gt_annotations, matching_threshold,
        )
        for r in pred_rows:
            r["source"] = source
        return pred_rows, fn_annots, coords_3way, coords_4way, skeleton

    raw_pred_rows, raw_fn_annots, _, _, _ = _detect_and_match(raw_stitched, "raw")
    pp_pred_rows, pp_fn_annots, pp_coords_3way, pp_coords_4way, pp_skeleton = _detect_and_match(
        pp_stitched, "pp")

    return (raw_pred_rows, raw_fn_annots, pp_pred_rows, pp_fn_annots,
            pp_stitched, pp_coords_3way, pp_coords_4way, pp_skeleton)


def _save_junction_detection_plot(
    full_img: torch.Tensor,
    pp_stitched: torch.Tensor,
    coords_3way: np.ndarray,
    coords_4way: np.ndarray,
    pp_skeleton: np.ndarray,
    gt_annotations: list[dict],
    plot_path: Path,
    title: str = "",
) -> None:
    """Save a full-image plot: stitched image + pp mask overlay + junction markers.

    Predicted 3-way: lime open circles.  Predicted 4-way: orange open circles.
    GT 3-way: lime Xs.  GT 4-way: orange Xs.
    """
    fig, ax = plt.subplots(figsize=(14, 14))
    if title:
        ax.set_title(title, fontsize=10)

    plot_images_masks_junctions(
        full_img,
        predicted_mask=pp_stitched.numpy(),
        groundtruth_mask=None,
        junction_coords_3way=coords_3way if len(coords_3way) > 0 else None,
        junction_coords_4way=coords_4way if len(coords_4way) > 0 else None,
        skeleton=pp_skeleton,
        ax=ax,
        plot_grid=False,
    )

    gt_3way = np.array([[a["x"], a["y"]] for a in gt_annotations
                        if a["type"] == _JUNCTION_TYPE_3_WAY])
    gt_4way = np.array([[a["x"], a["y"]] for a in gt_annotations
                        if a["type"] == _JUNCTION_TYPE_4_WAY])
    if len(gt_3way) > 0:
        ax.plot(gt_3way[:, 0], gt_3way[:, 1], "x",
                color="red", markersize=8, markeredgewidth=1, label="GT 3-way")
    if len(gt_4way) > 0:
        ax.plot(gt_4way[:, 0], gt_4way[:, 1], "x",
                color="orange", markersize=8, markeredgewidth=1, label="GT 4-way")

    ax.legend(loc="upper right", fontsize=8)
    plt.tight_layout()
    fig.savefig(plot_path, bbox_inches="tight", dpi=300)
    plt.close(fig)


def _evaluate_model(
    model_key: str,
    model_dataset: str,
    model_pred_dir: Path,
    test_image_paths: list[Path],
    gt_by_image: dict[str, list[dict]],
    matching_threshold: float,
    out_dir: Path,
    is_test: bool = False,
    plot_dir: Path | None = None,
    plot_skeleton: bool = False,
) -> dict:
    """Run the full evaluation loop for one model and return a metrics row dict."""
    raw_pred_rows_all: list[dict] = []
    raw_fn_all: list[dict] = []
    pp_pred_rows_all: list[dict] = []
    pp_fn_all: list[dict] = []
    pred_csv_rows: list[dict] = []
    fiber_csv_rows: list[dict] = []
    fiber_metrics_all: list[dict] = []
    image_stats_raw: list[dict] = []
    image_stats_pp: list[dict] = []

    # Per-image bookkeeping for sample-level aggregation
    raw_preds_by_image: dict[str, list[dict]] = {}
    raw_fns_by_image: dict[str, list[dict]] = {}
    pp_preds_by_image: dict[str, list[dict]] = {}
    pp_fns_by_image: dict[str, list[dict]] = {}
    fiber_metrics_by_image: dict[str, dict] = {}

    if plot_dir is not None:
        plot_dir.mkdir(parents=True, exist_ok=True)

    for idx, img_path in enumerate(test_image_paths):
        stem = img_path.stem
        gt_annotations = gt_by_image.get(stem, None)
        if gt_annotations is None:
            raise ValueError(
                f"No GT annotations found for image stem '{stem}' in CSV.")

        pred_mask_patches = _load_pred_patches(model_pred_dir, stem)

        raw_preds, raw_fns, pp_preds, pp_fns, \
            pp_stitched, pp_coords_3way, pp_coords_4way, pp_skeleton = _process_image(
                pred_mask_patches, gt_annotations, matching_threshold,
            )

        raw_pred_rows_all.extend(raw_preds)
        raw_fn_all.extend(raw_fns)
        pp_pred_rows_all.extend(pp_preds)
        pp_fn_all.extend(pp_fns)

        raw_preds_by_image[stem] = raw_preds
        raw_fns_by_image[stem] = raw_fns
        pp_preds_by_image[stem] = pp_preds
        pp_fns_by_image[stem] = pp_fns

        for r in raw_preds + pp_preds:
            pred_csv_rows.append({"image": stem, **r})

        # --- Per-image binary detection stats ---
        gt_positive = int(len(gt_annotations) > 0)
        image_stats_raw.append({
            "image": stem,
            "gt_positive": gt_positive,
            "pred_positive": int(len(raw_preds) > 0),
        })
        image_stats_pp.append({
            "image": stem,
            "gt_positive": gt_positive,
            "pred_positive": int(len(pp_preds) > 0),
        })

        # --- Per-fiber evaluation ---
        labeled_mask, fibers = label_fibers(
            pp_stitched, pp_coords_3way, pp_coords_4way)
        gt_by_fiber = match_gt_to_fibers(labeled_mask, gt_annotations)
        img_fiber_rows, img_fiber_metrics = compute_fiber_metrics(
            fibers, gt_by_fiber)
        for r in img_fiber_rows:
            fiber_csv_rows.append({"image": stem, **r})
        fiber_metrics_all.append(img_fiber_metrics)
        fiber_metrics_by_image[stem] = img_fiber_metrics

        if plot_dir is not None:
            _, full_img = load_full_image_as_patches(img_path)
            _save_junction_detection_plot(
                full_img, pp_stitched,
                pp_coords_3way, pp_coords_4way,
                pp_skeleton if plot_skeleton else None,
                gt_annotations,
                plot_dir / f"{stem}.png",
                title=f"{model_key} — {stem}",
            )

        print(
            f"  Processed {idx + 1}/{len(test_image_paths)} ({img_path.name})")

        if is_test:
            break

    safe_name = model_key.replace("/", "_")
    pred_df = pd.DataFrame(pred_csv_rows)
    pred_path = out_dir / f"predictions_{safe_name}.csv"
    pred_df.to_csv(pred_path, index=False)
    print(f"\n  Saved predictions as {pred_path}")

    # Save per-fiber CSV
    if fiber_csv_rows:
        fiber_df = pd.DataFrame(fiber_csv_rows)
        fiber_path = out_dir / f"fibers_{safe_name}.csv"
        fiber_df.to_csv(fiber_path, index=False)
        print(f"  Saved fiber predictions as {fiber_path}")

    raw_metrics = _compute_metrics(raw_pred_rows_all, raw_fn_all)
    pp_metrics = _compute_metrics(pp_pred_rows_all, pp_fn_all)

    raw_image_metrics = _compute_image_level_metrics(image_stats_raw)
    pp_image_metrics = _compute_image_level_metrics(image_stats_pp)

    # Save per-image binary stats CSV (pp version)
    image_df = pd.DataFrame(image_stats_pp)
    image_path = out_dir / f"image_level_{safe_name}.csv"
    image_df.to_csv(image_path, index=False)
    print(f"  Saved per-image stats as {image_path}")

    # Aggregate fiber metrics across images (sum counts, recompute rates)
    agg_fiber = _aggregate_fiber_metrics(fiber_metrics_all)

    # --- Per-sample stats (pp pipeline only) ---
    processed_stems = list(pp_preds_by_image.keys())
    sample_to_stems: dict[str, list[str]] = {}
    for s in processed_stems:
        sample_to_stems.setdefault(_image_to_sample(s), []).append(s)

    per_sample_rows: list[dict] = []
    for sample, stems in sorted(sample_to_stems.items()):
        s_pred_rows: list[dict] = []
        s_fn_annots: list[dict] = []
        s_img_stats: list[dict] = []
        s_fiber_metrics: list[dict] = []
        for st in stems:
            s_pred_rows.extend(pp_preds_by_image[st])
            s_fn_annots.extend(pp_fns_by_image[st])
            s_img_stats.append(next(
                x for x in image_stats_pp if x["image"] == st))
            if st in fiber_metrics_by_image:
                s_fiber_metrics.append(fiber_metrics_by_image[st])

        s_metrics = _compute_metrics(s_pred_rows, s_fn_annots)
        s_image_metrics = _compute_image_level_metrics(s_img_stats)
        s_fiber = _aggregate_fiber_metrics(s_fiber_metrics)

        s_row = {"sample": sample, "n_images": len(stems)}
        s_row.update(s_metrics)
        s_row.update(s_image_metrics)
        s_row.update(s_fiber)
        per_sample_rows.append(s_row)

    if per_sample_rows:
        per_sample_df = pd.DataFrame(per_sample_rows)
        per_sample_path = out_dir / f"per_sample_{safe_name}.csv"
        per_sample_df.to_csv(per_sample_path, index=False)
        print(f"  Saved per-sample stats as {per_sample_path}")

    print(f"\n  [raw]  loc P={raw_metrics['precision_loc']:.3f} "
          f"R={raw_metrics['recall_loc']:.3f} F1={raw_metrics['f1_loc']:.3f} "
          f"| type acc={raw_metrics['type_accuracy']:.3f}")
    print(f"  [pp]   loc P={pp_metrics['precision_loc']:.3f} "
          f"R={pp_metrics['recall_loc']:.3f} F1={pp_metrics['f1_loc']:.3f} "
          f"| type acc={pp_metrics['type_accuracy']:.3f}")
    print(f"  [pp 3-way] P={pp_metrics['class_precision_3way']:.3f} "
          f"R={pp_metrics['class_recall_3way']:.3f} "
          f"F1={pp_metrics['class_f1_3way']:.3f} "
          f"(TP={pp_metrics['class_tp_3way']}, "
          f"FP={pp_metrics['class_fp_3way']}, "
          f"FN={pp_metrics['class_fn_3way']})")
    print(f"  [pp 4-way] P={pp_metrics['class_precision_4way']:.3f} "
          f"R={pp_metrics['class_recall_4way']:.3f} "
          f"F1={pp_metrics['class_f1_4way']:.3f} "
          f"(TP={pp_metrics['class_tp_4way']}, "
          f"FP={pp_metrics['class_fp_4way']}, "
          f"FN={pp_metrics['class_fn_4way']})")
    print(f"  [image] P={pp_image_metrics['image_precision']:.3f} "
          f"R={pp_image_metrics['image_recall']:.3f} "
          f"F1={pp_image_metrics['image_f1']:.3f} "
          f"acc={pp_image_metrics['image_accuracy']:.3f} "
          f"(TP={pp_image_metrics['image_tp']}, "
          f"FP={pp_image_metrics['image_fp']}, "
          f"FN={pp_image_metrics['image_fn']}, "
          f"TN={pp_image_metrics['image_tn']}, "
          f"N={pp_image_metrics['image_n']})")
    if agg_fiber:
        print(f"  [fiber] P={agg_fiber['fiber_precision']:.3f} "
              f"R={agg_fiber['fiber_recall']:.3f} F1={agg_fiber['fiber_f1']:.3f} "
              f"| class acc={agg_fiber['fiber_class_accuracy']:.3f} "
              f"(correct={agg_fiber['fiber_class_correct']}, "
              f"ambiguous={agg_fiber['fiber_class_ambiguous']}, "
              f"incorrect={agg_fiber['fiber_class_incorrect']})")

    if per_sample_rows:
        print(f"\n  Per-sample (pp):")
        for s_row in per_sample_rows:
            f1_3 = s_row.get("class_f1_3way", 0.0)
            f1_4 = s_row.get("class_f1_4way", 0.0)
            print(f"    [{s_row['sample']}] N={s_row['n_images']:3d}  "
                  f"loc P={s_row['precision_loc']:.3f} "
                  f"R={s_row['recall_loc']:.3f} F1={s_row['f1_loc']:.3f}  "
                  f"| 3w F1={f1_3:.3f} 4w F1={f1_4:.3f}  "
                  f"| img F1={s_row['image_f1']:.3f} "
                  f"| fib F1={s_row.get('fiber_f1', 0.0):.3f}")

    row: dict = {"model": model_key, "dataset": model_dataset}
    for k, v in raw_metrics.items():
        row[f"raw_{k}"] = v
    for k, v in pp_metrics.items():
        row[f"pp_{k}"] = v
    for k, v in raw_image_metrics.items():
        row[f"raw_{k}"] = v
    for k, v in pp_image_metrics.items():
        row[f"pp_{k}"] = v
    for k, v in agg_fiber.items():
        row[k] = v
    return row


def plot_confusion_matrix_per_model(df_metrics: pd.DataFrame, out_dir: Path) -> None:
    """Save a 3×3 confusion-matrix figure for each model (pp results).

    Rows = GT outcome  : 3-way junction | 4-way junction | no GT (FP)
    Cols = pred outcome: predicted 3-way | predicted 4-way | not found (FN)

    The top-left 2×2 block covers matched TPs with type breakdown.
    The right column shows per-type FN (missed junctions).
    The bottom row shows per-type FP (spurious detections).
    The bottom-right cell is undefined and masked out.
    """
    cm_dir = out_dir / "confusion_matrices"
    cm_dir.mkdir(exist_ok=True)

    _NAN = float("nan")

    for model_key, row in df_metrics.iterrows():
        # Build 3×3 matrix; [2,2] is N/A
        cm = np.array(
            [
                [row["pp_cm_gt3_pred3"], row["pp_cm_gt3_pred4"], row["pp_fn_3way"]],
                [row["pp_cm_gt4_pred3"], row["pp_cm_gt4_pred4"], row["pp_fn_4way"]],
                [row["pp_fp_3way"], row["pp_fp_4way"], _NAN],
            ],
            dtype=float,
        )

        # For colouring use only the defined cells
        valid = cm[:2, :]  # bottom-right is N/A, exclude from colour scale
        vmax = np.nanmax(valid) if np.nanmax(valid) > 0 else 1

        fig, ax = plt.subplots(figsize=(6, 5))

        # Draw defined cells manually so we can mask [2,2]
        masked = np.ma.masked_invalid(cm)
        im = ax.imshow(masked, cmap="Blues", vmin=0, vmax=vmax, aspect="auto")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        # Shade the N/A cell in grey
        ax.add_patch(plt.Rectangle(
            (1.5, 1.5), 1, 1, color="#cccccc", zorder=2))
        ax.text(2, 2, "N/A", ha="center", va="center",
                fontsize=10, color="#666666", zorder=3)

        col_labels = ["Pred 3-way", "Pred 4-way", "Not found\n(FN)"]
        row_labels = ["GT 3-way", "GT 4-way", "No GT\n(FP)"]
        ax.set_xticks([0, 1, 2])
        ax.set_yticks([0, 1, 2])
        ax.set_xticklabels(col_labels, fontsize=9)
        ax.set_yticklabels(row_labels, fontsize=9)
        ax.set_xlabel("Prediction", fontsize=10)
        ax.set_ylabel("Ground truth", fontsize=10)

        # Annotate each defined cell with count + row-normalised %
        row_totals = [
            row["pp_gt_3way_total"],
            row["pp_gt_4way_total"],
            row["pp_fp_3way"] + row["pp_fp_4way"],  # total FP (no GT row)
        ]
        for r in range(3):
            for c in range(3):
                if r == 2 and c == 2:
                    continue
                val = cm[r, c]
                if np.isnan(val):
                    continue
                denom = row_totals[r]
                pct_str = f"\n({100 * val / denom:.0f}%)" if denom > 0 else ""
                text_color = "white" if val > vmax * 0.6 else "black"
                ax.text(c, r, f"{int(val)}{pct_str}",
                        ha="center", va="center",
                        fontsize=10, color=text_color, fontweight="bold", zorder=4)

        # Draw dividing lines to separate the FP/FN margins from the main block
        ax.axhline(1.5, color="black", linewidth=1.5, linestyle="--")
        ax.axvline(1.5, color="black", linewidth=1.5, linestyle="--")

        det_rec_3 = row["pp_detection_recall_3way"]
        det_rec_4 = row["pp_detection_recall_4way"]
        type_acc = row["pp_type_accuracy"]
        ax.set_title(
            f"{model_key}\n"
            f"Detection recall — 3-way: {det_rec_3:.2f}  4-way: {det_rec_4:.2f}"
            f"  |  type acc (TP only): {type_acc:.2f}",
            fontsize=9,
        )

        fig.tight_layout()
        safe = model_key.replace("/", "_")
        out = cm_dir / f"cm_{safe}.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)

    print(f"Saved {len(df_metrics)} confusion-matrix plot(s) to {cm_dir}")


def plot_cross_model_comparison(df_metrics: pd.DataFrame, out_dir: Path) -> None:
    """Grouped bar chart comparing key metrics across all models (pp results)."""
    metrics_to_plot = {
        "Precision\n(loc)": "pp_precision_loc",
        "Recall\n(loc)": "pp_recall_loc",
        "F1\n(loc)": "pp_f1_loc",
        "Det. recall\n3-way": "pp_detection_recall_3way",
        "Det. recall\n4-way": "pp_detection_recall_4way",
        "Type acc\n(TP only)": "pp_type_accuracy",
        "F1 3-way\n(type)": "pp_type_f1_3way",
        "F1 4-way\n(type)": "pp_type_f1_4way",
    }

    n_metrics = len(metrics_to_plot)
    n_models = len(df_metrics)
    x = np.arange(n_metrics)
    width = 0.8 / max(n_models, 1)
    colors = plt.cm.tab10(np.linspace(0, 0.9, n_models))

    fig, ax = plt.subplots(figsize=(max(10, n_metrics * 2), 5))

    for i, (model_key, row) in enumerate(df_metrics.iterrows()):
        vals = [float(row.get(col, 0.0)) for col in metrics_to_plot.values()]
        offset = (i - n_models / 2 + 0.5) * width
        bars = ax.bar(x + offset, vals, width=width * 0.9,
                      label=model_key, color=colors[i])
        ax.bar_label(bars, fmt="%.2f", fontsize=6, padding=2)

    ax.set_xticks(x)
    ax.set_xticklabels(list(metrics_to_plot.keys()))
    ax.set_ylim(0, 1.18)
    ax.set_ylabel("Score")
    ax.set_title("Junction detection metrics by model (post-processed)")
    ax.legend(loc="upper right", fontsize=8, framealpha=0.7)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    fig.tight_layout()
    out = out_dir / "model_comparison.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved model comparison plot to {out}")


def _load_previous_metrics(base_dir: Path) -> pd.DataFrame:
    """Return the most recent metrics.csv across all timestamped sub-dirs, or
    an empty DataFrame if none exists."""
    candidates = [(p.parent.name, p) for p in base_dir.glob("*/metrics.csv")]
    if not candidates:
        return pd.DataFrame()
    _, latest = max(candidates, key=lambda t: t[0])
    df = pd.read_csv(latest, index_col="model")
    print(f"Loaded previous metrics from: {latest} ({len(df)} model(s))")
    return df


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--force-recompute", action="store_true",
                        help="Re-evaluate all models, ignoring cached results")
    parser.add_argument("--is-test", action="store_true",
                        help="Break after computing metrics for the first image")
    parser.add_argument("--plot", action="store_true",
                        help="Save stitched-image plots with junction markers per model")
    parser.add_argument("--plot-skeleton", action="store_true",
                        help="Overlay skeleton on plots (only if --plot is set)")
    args = parser.parse_args()

    env_utils.load_forksight_env()

    EVALUATION_OUTPUT_DIR = os.getenv("EVALUATION_OUTPUT_DIR")
    JUNCTION_DETECTION_DATASET_DIR = os.getenv(
        "JUNCTION_DETECTION_DATASET_DIR")
    JUNCTION_PRED_DIR = os.getenv("JUNCTION_PRED_DIR")
    JUNCTION_MATCHING_THRESHOLD = env_utils.load_as(
        "JUNCTION_MATCHING_THRESHOLD", float, 75.0)

    if EVALUATION_OUTPUT_DIR is None:
        raise ValueError(
            "EVALUATION_OUTPUT_DIR environment variable must be set.")
    if JUNCTION_DETECTION_DATASET_DIR is None:
        raise ValueError(
            "JUNCTION_DETECTION_DATASET_DIR environment variable must be set.")
    if JUNCTION_PRED_DIR is None:
        raise ValueError("JUNCTION_PRED_DIR environment variable must be set.")

    test_dir = Path(JUNCTION_DETECTION_DATASET_DIR)
    test_images_dir = test_dir / "images"
    test_labels_csv = test_dir / "relabeling_data.csv"

    if not test_images_dir.is_dir():
        raise FileNotFoundError(
            f"Images directory not found: {test_images_dir}")
    if not test_labels_csv.is_file():
        raise FileNotFoundError(f"Annotation CSV not found: {test_labels_csv}")

    gt_by_image = _load_gt_annotations(test_labels_csv)
    test_image_paths = sorted(p for p in test_images_dir.glob("*.png"))
    if not test_image_paths:
        raise FileNotFoundError(f"No image files found in {test_images_dir}")

    pred_base = Path(JUNCTION_PRED_DIR)
    model_dirs = sorted(
        d for d in pred_base.iterdir()
        if d.is_dir() and (d / "metadata.json").is_file()
    )
    if not model_dirs:
        raise FileNotFoundError(
            f"No model prediction directories (with metadata.json) found in "
            f"{pred_base}")

    print(f"Found {len(test_image_paths)} test image(s), "
          f"{len(model_dirs)} model prediction dir(s).")
    print(f"Matching threshold: {JUNCTION_MATCHING_THRESHOLD} px")

    out_base = Path(EVALUATION_OUTPUT_DIR) / "junction_detection"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = out_base / timestamp
    out_dir.mkdir(parents=True, exist_ok=True)
    _PLOT_DIR = out_dir / "plots" if args.plot else None

    df_prev = (_load_previous_metrics(out_base)
               if not args.force_recompute else pd.DataFrame())
    computed_models = set(df_prev.index) if not df_prev.empty else set()

    new_metrics_rows: list[dict] = []

    for model_dir in model_dirs:
        with open(model_dir / "metadata.json") as f:
            meta = json.load(f)
        model_key = meta["model_key"]
        model_dataset = meta.get("dataset", "")

        if model_key in computed_models:
            print(f"\n  Already cached: {model_key}")
            continue

        print(f"\n{'='*60}")
        print(f"Evaluating: {model_key}")
        print(f"  Predictions: {model_dir}")
        print(f"{'='*60}")

        safe_name = model_key.replace("/", "_")
        row = _evaluate_model(
            model_key=model_key,
            model_dataset=model_dataset,
            model_pred_dir=model_dir,
            test_image_paths=test_image_paths,
            gt_by_image=gt_by_image,
            matching_threshold=JUNCTION_MATCHING_THRESHOLD,
            out_dir=out_dir,
            is_test=args.is_test,
            plot_dir=_PLOT_DIR / safe_name if _PLOT_DIR else None,
            plot_skeleton=args.plot_skeleton,
        )
        new_metrics_rows.append(row)

    if not new_metrics_rows:
        print("\nNo new models evaluated, all results already cached.")
        return

    df_new = pd.DataFrame(new_metrics_rows).set_index("model")
    if not df_prev.empty:
        df_new = pd.concat([df_prev, df_new])
    metrics_path = out_dir / "metrics.csv"
    df_new.to_csv(metrics_path)
    print(f"\nSaved metrics as {metrics_path}")

    plot_confusion_matrix_per_model(df_new, out_dir)
    plot_cross_model_comparison(df_new, out_dir)


if __name__ == "__main__":
    main()
