"""Downstream evaluation: run an nnUNet model on the test split and score
junction detections against GT annotations.

Self-contained (only depends on ``junction_detection.py`` sitting next to it).

Inputs (produced by train_cldice_nnunet.py + junction_detection.py):
    <dataset_dir>/images/<stem>.png
    <dataset_dir>/annotations.csv   (cols: image, x, y, label)
    <dataset_dir>/split.json        ({'train': [...], 'test': [...]})
    <nnunet_root>/nnUNet_results/Dataset<ID>_<NAME>/<trainer>__nnUNetPlans__<cfg>/
        plans.json, dataset.json, fold_<k>/checkpoint_*.pth

Outputs (in <out_dir>/):
    predictions.csv, image_level.csv, fibers.csv, metrics.json

Python requirements:
    pip install nnunetv2 numpy torch pandas pillow scipy scikit-image skan networkx matplotlib
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
from scipy.ndimage import distance_transform_edt, generate_binary_structure, label

# junction_detection.py must sit next to this file
sys.path.insert(0, str(Path(__file__).resolve().parent))
from junction_detection import detect_junctions_in_segmentation_mask  # noqa: E402

from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor  # noqa: E402


# ---------------------------------------------------------------------------
# Label mapping
# ---------------------------------------------------------------------------
_J3 = "3-way"
_J4 = "4-way"

LABEL_NORMAL = "Normal Fork"
LABEL_REVERSED = "Reversed Fork"
LABEL_AMBIGUOUS = "Ambiguous"
_NEAREST_COMPONENT_MAX_DISTANCE = 75.0

# Accept both real-dataset labels and simple "3-way"/"4-way" labels
_LABEL_TO_TYPE = {
    "Normal Fork": _J3,
    "Crossing": _J4,
    "Reversed Fork": _J4,
    "Negative": None,
    "3-way": _J3, "4-way": _J4, "3way": _J3, "4way": _J4,
}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset-dir", required=True, type=Path)
    p.add_argument("--nnunet-root", required=True, type=Path)
    p.add_argument("--dataset-id", type=int, default=900)
    p.add_argument("--dataset-name", default="Synthetic")
    p.add_argument("--trainer", default="nnUNetTrainerCLDice")
    p.add_argument("--configuration", default="2d")
    p.add_argument("--fold", type=int, default=0)
    p.add_argument("--checkpoint", default="checkpoint_final.pth")
    p.add_argument("--matching-threshold", type=float, default=75.0)
    p.add_argument("--split-json", type=Path, default=None)
    p.add_argument("--out-dir", type=Path, default=None)
    p.add_argument("--device", type=int, default=0)
    p.add_argument("--save-plots", action="store_true")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def load_gt_annotations(csv_path: Path) -> dict[str, list[dict]]:
    df = pd.read_csv(csv_path)
    for col in ("image", "x", "y", "label"):
        if col not in df.columns:
            raise SystemExit(f"annotations.csv missing required column '{col}'")
    gt: dict[str, list[dict]] = {}
    for _, row in df.iterrows():
        stem = str(row["image"])
        if stem.endswith(".png"):
            stem = stem[:-4]
        label_str = str(row["label"]).strip()
        if label_str not in _LABEL_TO_TYPE:
            raise SystemExit(
                f"Unknown label '{label_str}' in {csv_path}. "
                f"Expected one of {sorted(_LABEL_TO_TYPE)}.")
        jtype = _LABEL_TO_TYPE[label_str]
        gt.setdefault(stem, [])
        if jtype is not None:
            gt[stem].append({"x": float(row["x"]),
                             "y": float(row["y"]),
                             "type": jtype})
    return gt


def load_split(split_path: Path) -> list[str]:
    if not split_path.is_file():
        raise SystemExit(f"Split file not found: {split_path}")
    with open(split_path) as f:
        split = json.load(f)
    if "test" not in split:
        raise SystemExit(f"Split file {split_path} has no 'test' entry.")
    return list(split["test"])


# ---------------------------------------------------------------------------
# nnUNet predictor
# ---------------------------------------------------------------------------

def initialize_predictor(model_dir: Path, device: torch.device,
                         fold: int, checkpoint: str) -> nnUNetPredictor:
    predictor = nnUNetPredictor(
        tile_step_size=0.5,
        use_gaussian=True,
        use_mirroring=True,
        perform_everything_on_device=True,
        device=device,
        verbose=False,
        verbose_preprocessing=False,
        allow_tqdm=False,
    )
    predictor.initialize_from_trained_model_folder(
        str(model_dir), use_folds=(fold,), checkpoint_name=checkpoint,
    )
    return predictor


def run_inference(predictor: nnUNetPredictor,
                  test_image_paths: list[Path],
                  tmp_root: Path) -> Path:
    inp = tmp_root / "inputs"; inp.mkdir(parents=True, exist_ok=True)
    out = tmp_root / "predictions"; out.mkdir(parents=True, exist_ok=True)
    file_lists: list[list[str]] = []
    for p in test_image_paths:
        dst = inp / f"{p.stem}_0000.png"
        shutil.copy(p, dst)
        file_lists.append([str(dst)])
    predictor.predict_from_files(
        file_lists, str(out),
        save_probabilities=False, overwrite=True,
        num_processes_preprocessing=2, num_processes_segmentation_export=2,
    )
    return out


def load_pred_mask(pred_dir: Path, stem: str) -> torch.Tensor:
    path = pred_dir / f"{stem}.png"
    if not path.is_file():
        raise SystemExit(f"Missing nnUNet prediction: {path}")
    arr = np.array(Image.open(path))
    if arr.ndim == 3:
        arr = arr[..., 0]
    return torch.from_numpy((arr > 0).astype(np.float32)).unsqueeze(0)


# ---------------------------------------------------------------------------
# Metrics (junction + image + fiber)
# ---------------------------------------------------------------------------

def match_predictions_to_gt(pred_coords, pred_types, gt_annotations, threshold):
    n_pred, n_gt = len(pred_coords), len(gt_annotations)
    if n_pred == 0:
        return [], list(gt_annotations)
    if n_gt == 0:
        rows = [{"x": float(x), "y": float(y), "pred_type": t,
                 "matched_gt_x": None, "matched_gt_y": None,
                 "matched_gt_type": None, "distance": None,
                 "is_tp": False, "is_fp": True}
                for (x, y), t in zip(pred_coords, pred_types)]
        return rows, []

    gt_coords = np.array([[a["x"], a["y"]] for a in gt_annotations])
    diff = pred_coords[:, None, :] - gt_coords[None, :, :]
    dist = np.linalg.norm(diff, axis=2)

    candidates = sorted(
        (dist[i, j], i, j)
        for i in range(n_pred) for j in range(n_gt) if dist[i, j] <= threshold)

    matched: list[int | None] = [None] * n_pred
    gt_matched = [False] * n_gt
    for _, i, j in candidates:
        if matched[i] is None and not gt_matched[j]:
            matched[i] = j; gt_matched[j] = True

    rows: list[dict] = []
    for i, (x, y) in enumerate(pred_coords):
        j = matched[i]
        if j is not None:
            rows.append({
                "x": float(x), "y": float(y), "pred_type": pred_types[i],
                "matched_gt_x": float(gt_coords[j, 0]),
                "matched_gt_y": float(gt_coords[j, 1]),
                "matched_gt_type": gt_annotations[j]["type"],
                "distance": float(dist[i, j]),
                "is_tp": True, "is_fp": False,
            })
        else:
            rows.append({
                "x": float(x), "y": float(y), "pred_type": pred_types[i],
                "matched_gt_x": None, "matched_gt_y": None,
                "matched_gt_type": None, "distance": None,
                "is_tp": False, "is_fp": True,
            })
    fn = [a for j, a in enumerate(gt_annotations) if not gt_matched[j]]
    return rows, fn


def compute_metrics(pred_rows, fn_annots) -> dict:
    def prf(tp, fp, fn):
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        return prec, rec, f1

    m: dict = {}
    tp_loc = sum(1 for r in pred_rows if r["is_tp"])
    fp_loc = sum(1 for r in pred_rows if r["is_fp"])
    fn_loc = len(fn_annots)
    p, r, f = prf(tp_loc, fp_loc, fn_loc)
    m.update({"tp_loc": tp_loc, "fp_loc": fp_loc, "fn_loc": fn_loc,
              "precision_loc": p, "recall_loc": r, "f1_loc": f})

    matched = [r for r in pred_rows if r["is_tp"]]
    cm33 = sum(1 for r in matched if r["matched_gt_type"] == _J3 and r["pred_type"] == _J3)
    cm34 = sum(1 for r in matched if r["matched_gt_type"] == _J3 and r["pred_type"] == _J4)
    cm43 = sum(1 for r in matched if r["matched_gt_type"] == _J4 and r["pred_type"] == _J3)
    cm44 = sum(1 for r in matched if r["matched_gt_type"] == _J4 and r["pred_type"] == _J4)
    type_correct = cm33 + cm44
    m["type_accuracy"] = type_correct / tp_loc if tp_loc else 0.0

    fn_3 = sum(1 for a in fn_annots if a["type"] == _J3)
    fn_4 = sum(1 for a in fn_annots if a["type"] == _J4)
    fp_rows = [r for r in pred_rows if r["is_fp"]]
    fp_3 = sum(1 for r in fp_rows if r["pred_type"] == _J3)
    fp_4 = sum(1 for r in fp_rows if r["pred_type"] == _J4)

    class_tp_3, class_fp_3, class_fn_3 = cm33, cm43 + fp_3, cm34 + fn_3
    class_tp_4, class_fp_4, class_fn_4 = cm44, cm34 + fp_4, cm43 + fn_4
    p3, r3, f3 = prf(class_tp_3, class_fp_3, class_fn_3)
    p4, r4, f4 = prf(class_tp_4, class_fp_4, class_fn_4)
    m.update({
        "class_tp_3way": class_tp_3, "class_fp_3way": class_fp_3,
        "class_fn_3way": class_fn_3,
        "class_precision_3way": p3, "class_recall_3way": r3, "class_f1_3way": f3,
        "class_tp_4way": class_tp_4, "class_fp_4way": class_fp_4,
        "class_fn_4way": class_fn_4,
        "class_precision_4way": p4, "class_recall_4way": r4, "class_f1_4way": f4,
    })
    return m


def compute_image_level_metrics(stats: list[dict]) -> dict:
    n = len(stats)
    tp = sum(1 for s in stats if s["gt_positive"] and s["pred_positive"])
    fp = sum(1 for s in stats if not s["gt_positive"] and s["pred_positive"])
    fn = sum(1 for s in stats if s["gt_positive"] and not s["pred_positive"])
    tn = sum(1 for s in stats if not s["gt_positive"] and not s["pred_positive"])
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f = 2 * p * r / (p + r) if (p + r) else 0.0
    acc = (tp + tn) / n if n else 0.0
    return {"image_n": n, "image_tp": tp, "image_fp": fp, "image_fn": fn,
            "image_tn": tn, "image_precision": p, "image_recall": r,
            "image_f1": f, "image_accuracy": acc}


# ----- Fiber metrics -----

def label_fibers(mask: torch.Tensor, coords_3, coords_4):
    mask_np = mask.squeeze().cpu().numpy()
    structure = generate_binary_structure(2, 2)
    labeled, n = label(mask_np > 0, structure=structure)

    comp: dict[int, dict] = {}
    for cid in range(1, n + 1):
        ys, xs = np.where(labeled == cid)
        comp[cid] = {"junctions_3way": [], "junctions_4way": [],
                     "centroid": (float(xs.mean()), float(ys.mean())),
                     "area": int(len(ys))}

    def assign(x, y):
        iy, ix = int(round(y)), int(round(x))
        if 0 <= iy < labeled.shape[0] and 0 <= ix < labeled.shape[1]:
            cid = int(labeled[iy, ix])
            if cid > 0:
                return cid
        r = 80
        y0, y1 = max(0, iy - r), min(labeled.shape[0], iy + r + 1)
        x0, x1 = max(0, ix - r), min(labeled.shape[1], ix + r + 1)
        patch = labeled[y0:y1, x0:x1]
        fy, fx = np.where(patch > 0)
        if len(fy) == 0:
            return None
        d = np.sqrt((fx + x0 - x) ** 2 + (fy + y0 - y) ** 2)
        k = int(np.argmin(d))
        if d[k] <= _NEAREST_COMPONENT_MAX_DISTANCE:
            return int(patch[fy[k], fx[k]])
        return None

    for xy in coords_3:
        cid = assign(xy[0], xy[1])
        if cid and cid in comp:
            comp[cid]["junctions_3way"].append((float(xy[0]), float(xy[1])))
    for xy in coords_4:
        cid = assign(xy[0], xy[1])
        if cid and cid in comp:
            comp[cid]["junctions_4way"].append((float(xy[0]), float(xy[1])))

    fibers: dict[int, dict] = {}
    for cid, info in comp.items():
        h3 = len(info["junctions_3way"]) > 0
        h4 = len(info["junctions_4way"]) > 0
        if h3 and h4:
            info["label"] = LABEL_AMBIGUOUS
        elif h4:
            info["label"] = LABEL_REVERSED
        elif h3:
            info["label"] = LABEL_NORMAL
        else:
            continue
        fibers[cid] = info
    return labeled, fibers


def match_gt_to_fibers(labeled_mask: np.ndarray,
                       gt_annotations: list[dict]) -> dict[int, list[dict]]:
    result: dict[int, list[dict]] = {0: []}
    bg = labeled_mask == 0
    if bg.any() and (~bg).any():
        dist_map, nearest_idx = distance_transform_edt(bg, return_indices=True)
    else:
        dist_map, nearest_idx = None, None
    for gt in gt_annotations:
        iy, ix = int(round(gt["y"])), int(round(gt["x"]))
        cid = 0
        if 0 <= iy < labeled_mask.shape[0] and 0 <= ix < labeled_mask.shape[1]:
            cid = int(labeled_mask[iy, ix])
        if cid == 0 and dist_map is not None:
            if 0 <= iy < labeled_mask.shape[0] and 0 <= ix < labeled_mask.shape[1]:
                if dist_map[iy, ix] <= _NEAREST_COMPONENT_MAX_DISTANCE:
                    ny, nx_ = int(nearest_idx[0, iy, ix]), int(nearest_idx[1, iy, ix])
                    cid = int(labeled_mask[ny, nx_])
        result.setdefault(cid, []).append(gt)
    return result


def _gt_fiber_label(annots: list[dict]) -> str | None:
    types = {a["type"] for a in annots}
    has3, has4 = _J3 in types, _J4 in types
    if has3 and has4:
        return LABEL_AMBIGUOUS
    if has4:
        return LABEL_REVERSED
    if has3:
        return LABEL_NORMAL
    return None


def compute_fiber_metrics(fibers, gt_by_fiber):
    rows: list[dict] = []
    gt_labels = {cid: lbl for cid, annots in gt_by_fiber.items()
                 if cid != 0 and (lbl := _gt_fiber_label(annots))}

    pred_ids = set(fibers.keys())
    gt_ids = set(gt_labels.keys())
    tp_ids = pred_ids & gt_ids
    fp_ids = pred_ids - gt_ids
    fn_ids = gt_ids - pred_ids

    correct = ambiguous = incorrect = 0
    for cid in tp_ids:
        pred_lbl = fibers[cid]["label"]
        gt_lbl = gt_labels[cid]
        if pred_lbl == gt_lbl:
            correct += 1; cls = "correct"
        elif pred_lbl == LABEL_AMBIGUOUS and gt_lbl in (LABEL_NORMAL, LABEL_REVERSED):
            ambiguous += 1; cls = "ambiguous"
        else:
            incorrect += 1; cls = "incorrect"
        rows.append({
            "fiber_id": cid, "pred_label": pred_lbl, "gt_label": gt_lbl,
            "is_tp": True, "is_fp": False, "classification": cls,
            "n_junctions_3way": len(fibers[cid]["junctions_3way"]),
            "n_junctions_4way": len(fibers[cid]["junctions_4way"]),
            "area": fibers[cid]["area"],
        })
    for cid in fp_ids:
        rows.append({"fiber_id": cid, "pred_label": fibers[cid]["label"],
                     "gt_label": None, "is_tp": False, "is_fp": True,
                     "classification": None,
                     "n_junctions_3way": len(fibers[cid]["junctions_3way"]),
                     "n_junctions_4way": len(fibers[cid]["junctions_4way"]),
                     "area": fibers[cid]["area"]})
    for cid in fn_ids:
        rows.append({"fiber_id": cid, "pred_label": None, "gt_label": gt_labels[cid],
                     "is_tp": False, "is_fp": False, "classification": None,
                     "n_junctions_3way": 0, "n_junctions_4way": 0, "area": 0})

    unmatched = sum(1 for a in gt_by_fiber.get(0, [])
                    if a["type"] in (_J3, _J4))
    tp, fp, fn = len(tp_ids), len(fp_ids), len(fn_ids)
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    metrics = {
        "fiber_tp": tp, "fiber_fp": fp, "fiber_fn": fn,
        "fiber_precision": prec, "fiber_recall": rec, "fiber_f1": f1,
        "fiber_class_correct": correct,
        "fiber_class_ambiguous": ambiguous,
        "fiber_class_incorrect": incorrect,
        "fiber_class_accuracy": ((correct + ambiguous) / tp) if tp else 0.0,
        "fiber_n_unmatched_gt": unmatched,
    }
    return rows, metrics


def aggregate_fiber_metrics(per_image: list[dict]) -> dict:
    if not per_image:
        return {}
    agg: dict = {}
    keys = ["fiber_tp", "fiber_fp", "fiber_fn",
            "fiber_class_correct", "fiber_class_ambiguous",
            "fiber_class_incorrect", "fiber_n_unmatched_gt"]
    for k in keys:
        agg[k] = sum(m[k] for m in per_image)
    tp, fp, fn = agg["fiber_tp"], agg["fiber_fp"], agg["fiber_fn"]
    agg["fiber_precision"] = tp / (tp + fp) if (tp + fp) else 0.0
    agg["fiber_recall"] = tp / (tp + fn) if (tp + fn) else 0.0
    p, r = agg["fiber_precision"], agg["fiber_recall"]
    agg["fiber_f1"] = 2 * p * r / (p + r) if (p + r) else 0.0
    agg["fiber_class_accuracy"] = (
        (agg["fiber_class_correct"] + agg["fiber_class_ambiguous"]) / tp
        if tp else 0.0)
    return agg


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def save_plot(img_path, pred_mask, coords_3, coords_4, skeleton,
              gt_annotations, out_path, title):
    import matplotlib.pyplot as plt

    img = np.array(Image.open(img_path).convert("L"))
    mask_np = pred_mask.squeeze().cpu().numpy()

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(img, cmap="gray")
    ax.imshow(np.ma.masked_where(mask_np == 0, mask_np), cmap="autumn", alpha=0.35)
    if skeleton is not None:
        ax.imshow(np.ma.masked_where(skeleton == 0, skeleton), cmap="cool", alpha=0.9)

    if len(coords_3) > 0:
        ax.plot(coords_3[:, 0], coords_3[:, 1], "o",
                markerfacecolor="none", markeredgecolor="lime",
                markersize=10, label="Pred 3-way")
    if len(coords_4) > 0:
        ax.plot(coords_4[:, 0], coords_4[:, 1], "o",
                markerfacecolor="none", markeredgecolor="orange",
                markersize=10, label="Pred 4-way")
    gt3 = np.array([[a["x"], a["y"]] for a in gt_annotations if a["type"] == _J3])
    gt4 = np.array([[a["x"], a["y"]] for a in gt_annotations if a["type"] == _J4])
    if len(gt3):
        ax.plot(gt3[:, 0], gt3[:, 1], "x", color="red", markersize=9, label="GT 3-way")
    if len(gt4):
        ax.plot(gt4[:, 0], gt4[:, 1], "x", color="yellow", markersize=9, label="GT 4-way")
    ax.set_title(title, fontsize=9)
    ax.legend(loc="upper right", fontsize=8)
    ax.set_axis_off()
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    images_dir = args.dataset_dir / "images"
    csv_path = args.dataset_dir / "annotations.csv"
    if not images_dir.is_dir():
        raise SystemExit(f"Images directory not found: {images_dir}")
    if not csv_path.is_file():
        raise SystemExit(f"annotations.csv not found at {csv_path}")

    split_path = args.split_json or (args.dataset_dir / "split.json")
    test_names = load_split(split_path)

    test_paths = []
    for name in test_names:
        p = images_dir / name
        if not p.is_file():
            raise SystemExit(f"Test image missing on disk: {p}")
        test_paths.append(p)
    print(f"Evaluating on {len(test_paths)} test image(s).")

    gt_by_image = load_gt_annotations(csv_path)
    print(f"Loaded GT for {len(gt_by_image)} image(s) from {csv_path.name}")

    dataset_dir_name = f"Dataset{args.dataset_id:03d}_{args.dataset_name}"
    model_dir = (args.nnunet_root / "nnUNet_results" / dataset_dir_name
                 / f"{args.trainer}__nnUNetPlans__{args.configuration}")
    if not model_dir.is_dir():
        raise SystemExit(f"Trained model directory not found: {model_dir}")
    print(f"Model dir: {model_dir}")

    os.environ["nnUNet_raw"] = str(args.nnunet_root / "nnUNet_raw")
    os.environ["nnUNet_preprocessed"] = str(args.nnunet_root / "nnUNet_preprocessed")
    os.environ["nnUNet_results"] = str(args.nnunet_root / "nnUNet_results")

    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    predictor = initialize_predictor(model_dir, device, args.fold, args.checkpoint)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = args.out_dir or (args.dataset_dir / "evaluation" / timestamp)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output dir: {out_dir}")
    plot_dir = out_dir / "plots" if args.save_plots else None
    if plot_dir is not None:
        plot_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="synth_eval_") as _tmp:
        pred_out_dir = run_inference(predictor, test_paths, Path(_tmp))

        pred_rows_all, fn_all = [], []
        pred_csv, fiber_csv = [], []
        fiber_metrics_all = []
        image_stats: list[dict] = []

        for idx, img_path in enumerate(test_paths):
            stem = img_path.stem
            gt = gt_by_image.get(stem, [])
            pred_mask = load_pred_mask(pred_out_dir, stem)

            c3, c4, skel = detect_junctions_in_segmentation_mask(pred_mask)
            if len(c3) > 0 or len(c4) > 0:
                pred_coords = np.concatenate([c3, c4], axis=0)
            else:
                pred_coords = np.empty((0, 2))
            pred_types = [_J3] * len(c3) + [_J4] * len(c4)

            pred_rows, fn_annots = match_predictions_to_gt(
                pred_coords, pred_types, gt, args.matching_threshold)
            for r in pred_rows:
                r["source"] = "nnunet_full"

            pred_rows_all.extend(pred_rows); fn_all.extend(fn_annots)
            for r in pred_rows:
                pred_csv.append({"image": stem, **r})
            image_stats.append({
                "image": stem,
                "gt_positive": int(len(gt) > 0),
                "pred_positive": int(len(pred_rows) > 0),
            })

            labeled_mask, fibers = label_fibers(pred_mask, c3, c4)
            gt_by_fiber = match_gt_to_fibers(labeled_mask, gt)
            frows, fmetrics = compute_fiber_metrics(fibers, gt_by_fiber)
            for r in frows:
                fiber_csv.append({"image": stem, **r})
            fiber_metrics_all.append(fmetrics)

            if plot_dir is not None:
                save_plot(img_path, pred_mask, c3, c4, skel, gt,
                          plot_dir / f"{stem}.png",
                          title=f"{stem}  (GT={len(gt)}, pred={len(pred_rows)})")

            print(f"  [{idx+1}/{len(test_paths)}] {stem}  "
                  f"GT={len(gt)} pred={len(pred_rows)} FN={len(fn_annots)}")

    metrics = compute_metrics(pred_rows_all, fn_all)
    image_metrics = compute_image_level_metrics(image_stats)
    fiber_agg = aggregate_fiber_metrics(fiber_metrics_all)

    pd.DataFrame(pred_csv).to_csv(out_dir / "predictions.csv", index=False)
    pd.DataFrame(image_stats).to_csv(out_dir / "image_level.csv", index=False)
    if fiber_csv:
        pd.DataFrame(fiber_csv).to_csv(out_dir / "fibers.csv", index=False)

    summary = {
        "dataset_dir": str(args.dataset_dir),
        "model_dir": str(model_dir),
        "n_test_images": len(test_paths),
        "matching_threshold": args.matching_threshold,
        "fold": args.fold,
        "checkpoint": args.checkpoint,
        "junction_metrics": metrics,
        "image_metrics": image_metrics,
        "fiber_metrics": fiber_agg,
    }
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 60)
    print(f"Junction localisation (threshold {args.matching_threshold:.0f}px)")
    print("=" * 60)
    print(f"  loc  P={metrics['precision_loc']:.3f} "
          f"R={metrics['recall_loc']:.3f} F1={metrics['f1_loc']:.3f}  "
          f"(TP={metrics['tp_loc']} FP={metrics['fp_loc']} FN={metrics['fn_loc']})")
    print(f"  3-way F1={metrics['class_f1_3way']:.3f} | "
          f"4-way F1={metrics['class_f1_4way']:.3f} | "
          f"type acc={metrics['type_accuracy']:.3f}")
    print(f"  [image] P={image_metrics['image_precision']:.3f} "
          f"R={image_metrics['image_recall']:.3f} "
          f"F1={image_metrics['image_f1']:.3f}")
    if fiber_agg:
        print(f"  [fiber] P={fiber_agg['fiber_precision']:.3f} "
              f"R={fiber_agg['fiber_recall']:.3f} "
              f"F1={fiber_agg['fiber_f1']:.3f} "
              f"| class acc={fiber_agg['fiber_class_accuracy']:.3f}")
    print(f"\nSaved metrics to {out_dir}")


if __name__ == "__main__":
    main()
