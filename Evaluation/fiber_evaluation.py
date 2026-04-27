"""Per-fiber (connected component) evaluation for junction detection.

Groups detected junctions by the segmentation mask connected component they
belong to, assigns a single label per fiber, and matches against ground-truth
annotations to compute fiber-level metrics.

Fiber labels
------------
- "Normal Fork"    : fiber contains only 3-way junctions
- "Reversed Fork"  : fiber contains only 4-way junctions
- "Ambiguous"      : fiber contains both 3-way and 4-way junctions
                     (still counts as correctly identified "interesting")
- None             : fiber has no detected junctions (excluded)
"""

import numpy as np
import torch
from scipy.ndimage import label, generate_binary_structure, distance_transform_edt

LABEL_NORMAL = "Normal Fork"
LABEL_REVERSED = "Reversed Fork"
LABEL_AMBIGUOUS = "Ambiguous"

_JUNCTION_TYPE_3_WAY = "3-way"
_JUNCTION_TYPE_4_WAY = "4-way"

# maximum pixel distance when a GT point doesn't land directly on a fiber
_NEAREST_COMPONENT_MAX_DISTANCE = 75.0


def label_fibers(
    mask: torch.Tensor,
    coords_3way: np.ndarray,
    coords_4way: np.ndarray,
) -> tuple[np.ndarray, dict[int, dict]]:
    """Map junctions to connected components and assign per-fiber labels.

    Parameters
    ----------
    mask        : (1, H, W) binary segmentation mask.
    coords_3way : (N, 2) array of 3-way junction coords in (x, y).
    coords_4way : (M, 2) array of 4-way junction coords in (x, y).

    Returns
    -------
    labeled_mask : (H, W) integer array of connected component IDs.
    fibers       : dict mapping comp_id to fiber info dict.  Only fibers
                   with at least one junction are included.
    """
    mask_np = mask.squeeze().cpu().numpy()
    structure = generate_binary_structure(2, 2)  # 8-connectivity
    labeled, num_components = label(mask_np > 0, structure=structure)

    # Build per-component info
    comp_info: dict[int, dict] = {}
    for comp_id in range(1, num_components + 1):
        ys, xs = np.where(labeled == comp_id)
        comp_info[comp_id] = {
            "junctions_3way": [],
            "junctions_4way": [],
            "centroid": (float(xs.mean()), float(ys.mean())),
            "area": int(len(ys)),
        }

    def _assign(x: float, y: float) -> int | None:
        """Return component ID for (x, y), with nearest-component fallback."""
        iy, ix = int(round(y)), int(round(x))
        if 0 <= iy < labeled.shape[0] and 0 <= ix < labeled.shape[1]:
            cid = int(labeled[iy, ix])
            if cid > 0:
                return cid
        # Fallback: search a small window for nearest foreground pixel
        r = 80
        y0 = max(0, iy - r)
        y1 = min(labeled.shape[0], iy + r + 1)
        x0 = max(0, ix - r)
        x1 = min(labeled.shape[1], ix + r + 1)
        patch = labeled[y0:y1, x0:x1]
        fg_ys, fg_xs = np.where(patch > 0)
        if len(fg_ys) == 0:
            return None
        dists = np.sqrt((fg_xs + x0 - x) ** 2 + (fg_ys + y0 - y) ** 2)
        min_idx = int(np.argmin(dists))
        if dists[min_idx] <= _NEAREST_COMPONENT_MAX_DISTANCE:
            return int(patch[fg_ys[min_idx], fg_xs[min_idx]])
        return None

    for xy in coords_3way:
        cid = _assign(xy[0], xy[1])
        if cid and cid in comp_info:
            comp_info[cid]["junctions_3way"].append((float(xy[0]), float(xy[1])))

    for xy in coords_4way:
        cid = _assign(xy[0], xy[1])
        if cid and cid in comp_info:
            comp_info[cid]["junctions_4way"].append((float(xy[0]), float(xy[1])))

    # Assign labels and keep only fibers with junctions
    fibers: dict[int, dict] = {}
    for comp_id, info in comp_info.items():
        has_3 = len(info["junctions_3way"]) > 0
        has_4 = len(info["junctions_4way"]) > 0
        if has_3 and has_4:
            info["label"] = LABEL_AMBIGUOUS
        elif has_4:
            info["label"] = LABEL_REVERSED
        elif has_3:
            info["label"] = LABEL_NORMAL
        else:
            continue  # no junctions — skip
        fibers[comp_id] = info

    return labeled, fibers


def match_gt_to_fibers(
    labeled_mask: np.ndarray,
    gt_annotations: list[dict],
) -> dict[int, list[dict]]:
    """Assign GT annotations to predicted fibers via spatial containment.

    Falls back to nearest component within ``_NEAREST_COMPONENT_MAX_DISTANCE``
    if the GT point lands on background.

    Parameters
    ----------
    labeled_mask   : (H, W) integer array from ``label_fibers()``.
    gt_annotations : list of ``{x, y, type}`` dicts.

    Returns
    -------
    dict mapping comp_id → list of GT annotation dicts.
    Key ``0`` collects GT annotations that could not be matched.
    """
    gt_by_fiber: dict[int, list[dict]] = {0: []}

    # Precompute distance transform for background-pixel fallback
    bg_mask = labeled_mask == 0
    if bg_mask.any() and (~bg_mask).any():
        dist_map, nearest_idx = distance_transform_edt(bg_mask, return_indices=True)
    else:
        dist_map, nearest_idx = None, None

    for gt in gt_annotations:
        x, y = gt["x"], gt["y"]
        iy, ix = int(round(y)), int(round(x))

        comp_id = 0
        if 0 <= iy < labeled_mask.shape[0] and 0 <= ix < labeled_mask.shape[1]:
            comp_id = int(labeled_mask[iy, ix])

        if comp_id == 0 and dist_map is not None:
            if 0 <= iy < labeled_mask.shape[0] and 0 <= ix < labeled_mask.shape[1]:
                if dist_map[iy, ix] <= _NEAREST_COMPONENT_MAX_DISTANCE:
                    ny, nx_ = int(nearest_idx[0, iy, ix]), int(nearest_idx[1, iy, ix])
                    comp_id = int(labeled_mask[ny, nx_])

        gt_by_fiber.setdefault(comp_id, []).append(gt)

    return gt_by_fiber


def _gt_fiber_label(annotations: list[dict]) -> str | None:
    """Derive a single fiber label from the GT annotations on it."""
    types = set()
    for a in annotations:
        if a["type"] == _JUNCTION_TYPE_3_WAY:
            types.add("3")
        elif a["type"] == _JUNCTION_TYPE_4_WAY:
            types.add("4")
    has_3 = "3" in types
    has_4 = "4" in types
    if has_3 and has_4:
        return LABEL_AMBIGUOUS
    elif has_4:
        return LABEL_REVERSED
    elif has_3:
        return LABEL_NORMAL
    return None


def compute_fiber_metrics(
    fibers: dict[int, dict],
    gt_by_fiber: dict[int, list[dict]],
) -> tuple[list[dict], dict]:
    """Compute per-fiber detection and classification metrics.

    A predicted fiber is a **TP** if it has predicted junctions AND at least
    one GT junction maps to it.  It is **FP** if no GT junction maps to it.
    GT annotations that don't land on any predicted fiber with junctions
    are counted as unmatched (potential FN).

    Among TPs, classification is evaluated:
    - *correct*  : pred_label == gt_label
    - *ambiguous* : pred_label is Ambiguous but gt_label is Normal or Reversed
                    (still counts as "interesting" — correctly flagged for review)
    - *incorrect* : pred_label != gt_label and pred is not Ambiguous

    Returns
    -------
    fiber_rows : list of dicts (one per fiber, for CSV export).
    metrics    : aggregate metrics dict.
    """
    fiber_rows: list[dict] = []

    # Derive GT labels for fibers that have GT annotations
    gt_fiber_labels: dict[int, str] = {}
    for cid, annots in gt_by_fiber.items():
        if cid == 0:
            continue
        gt_label = _gt_fiber_label(annots)
        if gt_label is not None:
            gt_fiber_labels[cid] = gt_label

    pred_fiber_ids = set(fibers.keys())
    gt_fiber_ids = set(gt_fiber_labels.keys())

    tp_ids = pred_fiber_ids & gt_fiber_ids
    fp_ids = pred_fiber_ids - gt_fiber_ids
    # GT fibers that mapped to a component but that component has no predicted junctions
    fn_ids = gt_fiber_ids - pred_fiber_ids

    classification_correct = 0
    classification_ambiguous = 0
    classification_incorrect = 0

    for cid in tp_ids:
        info = fibers[cid]
        pred_label = info["label"]
        gt_label = gt_fiber_labels[cid]
        is_exact = pred_label == gt_label
        is_ambig_interesting = (
            pred_label == LABEL_AMBIGUOUS
            and gt_label in (LABEL_NORMAL, LABEL_REVERSED)
        )

        if is_exact:
            classification_correct += 1
        elif is_ambig_interesting:
            classification_ambiguous += 1
        else:
            classification_incorrect += 1

        fiber_rows.append({
            "fiber_id": cid,
            "pred_label": pred_label,
            "gt_label": gt_label,
            "is_tp": True,
            "is_fp": False,
            "classification": "correct" if is_exact else ("ambiguous" if is_ambig_interesting else "incorrect"),
            "n_junctions_3way": len(info["junctions_3way"]),
            "n_junctions_4way": len(info["junctions_4way"]),
            "area": info["area"],
        })

    for cid in fp_ids:
        info = fibers[cid]
        fiber_rows.append({
            "fiber_id": cid,
            "pred_label": info["label"],
            "gt_label": None,
            "is_tp": False,
            "is_fp": True,
            "classification": None,
            "n_junctions_3way": len(info["junctions_3way"]),
            "n_junctions_4way": len(info["junctions_4way"]),
            "area": info["area"],
        })

    for cid in fn_ids:
        gt_label = gt_fiber_labels[cid]
        fiber_rows.append({
            "fiber_id": cid,
            "pred_label": None,
            "gt_label": gt_label,
            "is_tp": False,
            "is_fp": False,
            "classification": None,
            "n_junctions_3way": 0,
            "n_junctions_4way": 0,
            "area": 0,
        })

    # Count unmatched GT (points that didn't land on ANY component)
    unmatched_gt = gt_by_fiber.get(0, [])
    n_unmatched_gt = sum(
        1 for a in unmatched_gt if a["type"] in (_JUNCTION_TYPE_3_WAY, _JUNCTION_TYPE_4_WAY)
    )

    tp = len(tp_ids)
    fp = len(fp_ids)
    fn = len(fn_ids)
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

    metrics = {
        "fiber_tp": tp,
        "fiber_fp": fp,
        "fiber_fn": fn,
        "fiber_precision": prec,
        "fiber_recall": rec,
        "fiber_f1": f1,
        "fiber_class_correct": classification_correct,
        "fiber_class_ambiguous": classification_ambiguous,
        "fiber_class_incorrect": classification_incorrect,
        "fiber_class_accuracy": (
            (classification_correct + classification_ambiguous) / tp if tp > 0 else 0.0
        ),
        "fiber_n_unmatched_gt": n_unmatched_gt,
    }

    return fiber_rows, metrics
