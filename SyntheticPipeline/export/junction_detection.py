"""Skeletonization-based junction detection on binary segmentation masks.

Self-contained: no imports from other files in this repo.  Exposes
``detect_junctions_in_segmentation_mask(mask)`` as the primary entry point, and
a small CLI that runs it over a folder of prediction PNGs.

The pipeline per mask:
  1. Drop tiny connected components (by bounding-box size).
  2. Skeletonize.
  3. Prune short terminal branches (spurs).
  4. Re-connect skeleton gaps whose tip directions align.
  5. Collapse tiny cycles.
  6. Classify the remaining degree>2 nodes as 3-way or 4-way junctions,
     with H-shape handling and 3-way-priority suppression of nearby 4-ways.
  7. Merge very close junctions.

Python requirements:
    pip install numpy torch scikit-image skan networkx scipy pillow
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import networkx as nx
import numpy as np
import torch
from PIL import Image
from scipy.ndimage import generate_binary_structure, label
from scipy.sparse.csgraph import connected_components
from scipy.spatial import KDTree
from skan.csr import Skeleton, skeleton_to_csgraph, summarize
from skimage.draw import line as draw_line
from skimage.morphology import skeletonize


# ---------------------------------------------------------------------------
# Tunables (kept as module-level constants so they're easy to tweak)
# ---------------------------------------------------------------------------
MIN_BRANCH_LENGTH = 100
MAX_JUNCTION_CONNECTOR_LENGTH = 40
MIN_TERMINAL_BRANCH_LENGTH = 40
MIN_CYCLE_LENGTH = 4000
JUNCTION_CYCLE_PROXIMITY_DISTANCE = 150
MAX_3WAY_PRIORITY_DISTANCE = 500
JUNCTION_MERGE_DISTANCE = 50
MAX_SKELETON_GAP_DISTANCE = 40
MAX_GAP_ANGLE_DEG = 20
GAP_DIRECTION_SAMPLE_LENGTH = 20
MIN_SMALL_CYCLE_PRUNE_LENGTH = 100

# Components whose bounding box W and H are both below this are dropped before
# skeletonization.  Set to None to disable.
SMALL_BBOX_THRESHOLD: int | None = None


# ---------------------------------------------------------------------------
# Small-object removal (was: Segmentation.PostProcessing.segmentation_postprocessing)
# ---------------------------------------------------------------------------

def _get_connected_components(mask: torch.Tensor) -> tuple[np.ndarray, int]:
    mask_np = (mask[0] > 0).cpu().numpy()
    structure = generate_binary_structure(2, 2)  # 8-connectivity
    return label(mask_np, structure=structure)


def remove_small_bbox_objects(mask: torch.Tensor,
                              threshold: int | None = SMALL_BBOX_THRESHOLD) -> torch.Tensor:
    """Zero out components whose bbox W and H are both < threshold."""
    assert mask.ndim == 3 and mask.shape[0] == 1, "Expected mask shape (1, H, W)"
    if threshold is None:
        return mask

    labeled, n = _get_connected_components(mask)
    cleaned = np.zeros_like(labeled, dtype=np.uint8)
    for cid in range(1, n + 1):
        ys, xs = np.where(labeled == cid)
        w = int(xs.max()) - int(xs.min()) + 1
        h = int(ys.max()) - int(ys.min()) + 1
        if w < threshold and h < threshold:
            continue
        cleaned[ys, xs] = 1

    out = torch.zeros_like(mask)
    out[0] = torch.from_numpy(cleaned)
    return out


# ---------------------------------------------------------------------------
# Skeleton helpers
# ---------------------------------------------------------------------------

def skeletonize_mask(segmentation_mask: torch.Tensor) -> np.ndarray:
    mask_np = segmentation_mask.detach().cpu().squeeze().numpy().astype(np.uint8)
    return skeletonize((mask_np > 0).astype(np.uint8)).astype(np.uint8)


def get_graph_coordinates_degrees(skeleton: np.ndarray):
    graph, coordinates = skeleton_to_csgraph(skeleton)
    degrees = np.diff(graph.indptr)
    coordinates = np.stack(np.asarray(coordinates), axis=1)
    return graph, coordinates, degrees


def prune_skeleton(skeleton: np.ndarray, iterations: int = 3) -> np.ndarray:
    current = skeleton.copy()
    for _ in range(iterations):
        skel_obj = Skeleton(current)
        stats = summarize(skel_obj)
        _, _, degrees = get_graph_coordinates_degrees(current)

        junction_to_spurs: dict[int, list] = {}
        for i, row in stats.iterrows():
            u, v = int(row["node-id-src"]), int(row["node-id-dst"])
            u_tip, v_tip = degrees[u] == 1, degrees[v] == 1
            u_jnc, v_jnc = degrees[u] > 2, degrees[v] > 2
            if (u_tip and v_jnc) or (v_tip and u_jnc):
                if row["branch-distance"] < MIN_TERMINAL_BRANCH_LENGTH:
                    junc = v if u_tip else u
                    tip = u if u_tip else v
                    junction_to_spurs.setdefault(junc, []).append(
                        (row["branch-distance"], i, tip))

        paths_to_delete = []
        for _, spurs in junction_to_spurs.items():
            spurs.sort(key=lambda x: x[0], reverse=True)
            for _, path_idx, _ in (spurs[1:] if len(spurs) >= 2 else spurs):
                paths_to_delete.append(path_idx)

        if not paths_to_delete:
            break

        for path_idx in paths_to_delete:
            coords = skel_obj.path_coordinates(path_idx).astype(int)
            node_src = int(stats.loc[path_idx, "node-id-src"])
            pixels = coords[:-1] if degrees[node_src] == 1 else coords[1:]
            for r, c in pixels:
                current[r, c] = 0

        current = skeletonize(current > 0).astype(np.uint8)
    return current


def _fit_tip_direction(path_coords, sample_length, from_src):
    if from_src:
        segment = path_coords[:min(sample_length, len(path_coords))]
        tip, interior = segment[0], segment[-1]
    else:
        segment = path_coords[max(0, len(path_coords) - sample_length):]
        tip, interior = segment[-1], segment[0]
    if len(segment) < 2:
        return None
    _, _, vh = np.linalg.svd(segment - segment.mean(axis=0), full_matrices=False)
    direction = vh[0]
    if np.dot(tip - interior, direction) < 0:
        direction = -direction
    return direction


def reconnect_skeleton_gaps(skeleton: np.ndarray) -> np.ndarray:
    skel_obj = Skeleton(skeleton)
    stats = summarize(skel_obj)
    stats = stats[stats["node-id-src"] != stats["node-id-dst"]]
    _, coordinates, degrees = get_graph_coordinates_degrees(skeleton)

    tip_indices = np.where(degrees == 1)[0]
    if len(tip_indices) < 2:
        return skeleton

    tip_directions: dict[int, np.ndarray] = {}
    for tip_idx in tip_indices:
        tip_branches = stats[(stats["node-id-src"] == tip_idx)
                             | (stats["node-id-dst"] == tip_idx)]
        if len(tip_branches) != 1:
            continue
        branch = tip_branches.iloc[0]
        path_coords = skel_obj.path_coordinates(branch.name).astype(float)
        from_src = int(branch["node-id-src"]) == tip_idx
        direction = _fit_tip_direction(path_coords, GAP_DIRECTION_SAMPLE_LENGTH, from_src)
        if direction is not None:
            tip_directions[tip_idx] = direction

    tip_list = list(tip_directions.keys())
    if len(tip_list) < 2:
        return skeleton

    tip_coords = np.array([coordinates[t] for t in tip_list])
    pairs = KDTree(tip_coords).query_pairs(MAX_SKELETON_GAP_DISTANCE)
    cos_threshold = np.cos(np.deg2rad(MAX_GAP_ANGLE_DEG))

    scored = []
    for i, j in pairs:
        c1, c2 = tip_coords[i], tip_coords[j]
        d1, d2 = tip_directions[tip_list[i]], tip_directions[tip_list[j]]
        gap_vec = c2 - c1
        gap_norm = np.linalg.norm(gap_vec)
        if gap_norm == 0:
            continue
        gap_unit = gap_vec / gap_norm
        if np.dot(d1, gap_unit) < cos_threshold or np.dot(d2, -gap_unit) < cos_threshold:
            continue
        avg_axis = d1 - d2
        axis_norm = np.linalg.norm(avg_axis)
        if axis_norm < 1e-8 or abs(np.dot(avg_axis / axis_norm, gap_unit)) < cos_threshold:
            continue
        scored.append((np.dot(d1, gap_unit) + np.dot(d2, -gap_unit), i, j))

    scored.sort(key=lambda x: -x[0])
    used: set[int] = set()
    result = skeleton.copy()
    for _, i, j in scored:
        if i in used or j in used:
            continue
        rr, cc = draw_line(int(tip_coords[i][0]), int(tip_coords[i][1]),
                           int(tip_coords[j][0]), int(tip_coords[j][1]))
        result[rr, cc] = 1
        used.add(i); used.add(j)

    return skeletonize(result > 0).astype(np.uint8)


def prune_small_cycles(skeleton: np.ndarray) -> np.ndarray:
    skel_obj = Skeleton(skeleton)
    stats = summarize(skel_obj)
    stats = stats[stats["node-id-src"] != stats["node-id-dst"]]
    nx_graph = nx.from_scipy_sparse_array(skel_obj.graph)

    branches_to_remove: set[tuple[int, int]] = set()
    for cycle in nx.cycle_basis(nx_graph):
        cycle_length = sum(
            nx_graph[cycle[i]][cycle[(i + 1) % len(cycle)]].get("weight", 0)
            for i in range(len(cycle))
        )
        if cycle_length >= MIN_SMALL_CYCLE_PRUNE_LENGTH:
            continue
        min_w, min_edge = float("inf"), None
        for i in range(len(cycle)):
            u, v = cycle[i], cycle[(i + 1) % len(cycle)]
            w = nx_graph[u][v].get("weight", 0)
            if w < min_w:
                min_w, min_edge = w, (min(u, v), max(u, v))
        if min_edge is not None:
            branches_to_remove.add(min_edge)

    if not branches_to_remove:
        return skeleton

    result = skeleton.copy()
    for idx, row in stats.iterrows():
        edge = (min(int(row["node-id-src"]), int(row["node-id-dst"])),
                max(int(row["node-id-src"]), int(row["node-id-dst"])))
        if edge in branches_to_remove:
            coords = skel_obj.path_coordinates(idx).astype(int)
            for r, c in coords[1:-1]:
                result[r, c] = 0
    return skeletonize(result > 0).astype(np.uint8)


# ---------------------------------------------------------------------------
# Junction classification
# ---------------------------------------------------------------------------

def filter_junctions_by_length(skeleton, junction_indices, degrees,
                               verbose=False) -> tuple[np.ndarray, np.ndarray]:
    skel_obj = Skeleton(skeleton)
    skel_stats = summarize(skel_obj)
    skel_stats = skel_stats[skel_stats["node-id-src"] != skel_stats["node-id-dst"]]

    nx_graph = nx.from_scipy_sparse_array(skel_obj.graph)
    nodes_in_cycles: set[int] = set()
    for cycle in nx.cycle_basis(nx_graph):
        cycle_length = sum(
            nx_graph[cycle[i]][cycle[(i + 1) % len(cycle)]].get("weight", 0)
            for i in range(len(cycle))
        )
        if cycle_length < MIN_CYCLE_LENGTH:
            nodes_in_cycles.update(cycle)

    cycle_coords = np.array([skel_obj.coordinates[n] for n in nodes_in_cycles]) \
        if nodes_in_cycles else None
    cycle_kdt = KDTree(cycle_coords) if cycle_coords is not None else None

    def is_close_to_cycle(coord) -> bool:
        if cycle_kdt is None:
            return False
        d, _ = cycle_kdt.query(coord)
        return d < JUNCTION_CYCLE_PROXIMITY_DISTANCE

    valid_3, valid_4 = [], []
    handled: set[int] = set()

    def check_path_significance(branch_row, source_node, cur_len, depth):
        if depth > 3:
            return False
        new_len = cur_len + branch_row["branch-distance"]
        if new_len >= MIN_BRANCH_LENGTH:
            return True
        u, v = int(branch_row["node-id-src"]), int(branch_row["node-id-dst"])
        neighbor = v if u == source_node else u
        if neighbor in nodes_in_cycles:
            return False
        if degrees[neighbor] > 2:
            branches = skel_stats[(skel_stats["node-id-src"] == neighbor)
                                  | (skel_stats["node-id-dst"] == neighbor)]
            for _, nb in branches.iterrows():
                if nb.name == branch_row.name:
                    continue
                if check_path_significance(nb, neighbor, new_len, depth + 1):
                    return True
        return False

    hshape_candidates: list[tuple[int, int, np.ndarray]] = []
    for _, branch in skel_stats.iterrows():
        n1, n2 = int(branch["node-id-src"]), int(branch["node-id-dst"])
        if degrees[n1] > 2 and degrees[n2] > 2 \
                and branch["branch-distance"] <= MAX_JUNCTION_CONNECTOR_LENGTH:
            if n1 in handled or n2 in handled:
                continue
            if n1 in nodes_in_cycles or n2 in nodes_in_cycles \
                    or is_close_to_cycle(skel_obj.coordinates[n1]) \
                    or is_close_to_cycle(skel_obj.coordinates[n2]):
                handled.update([n1, n2])
                continue
            cluster_arms = skel_stats[
                ((skel_stats["node-id-src"].isin([n1, n2]))
                 | (skel_stats["node-id-dst"].isin([n1, n2])))
                & (skel_stats.index != branch.name)]
            n_significant = sum(1 for _, a in cluster_arms.iterrows()
                                if a["branch-distance"] >= MIN_BRANCH_LENGTH)
            if n_significant == 4:
                midpoint = (skel_obj.coordinates[n1] + skel_obj.coordinates[n2]) / 2
                hshape_candidates.append((n1, n2, midpoint))
            handled.update([n1, n2])

    results = []
    for j in junction_indices:
        if j in handled or j in nodes_in_cycles or is_close_to_cycle(skel_obj.coordinates[j]):
            continue
        branches = skel_stats[(skel_stats["node-id-src"] == j)
                              | (skel_stats["node-id-dst"] == j)]
        n_sig, sig_len, sig_junc = 0, [], []
        for _, b in branches.iterrows():
            if b["branch-distance"] >= MIN_BRANCH_LENGTH:
                n_sig += 1; sig_len.append(b)
            else:
                u, v = int(b["node-id-src"]), int(b["node-id-dst"])
                other = v if u == j else u
                if degrees[other] > 2 and check_path_significance(b, j, 0, 0):
                    n_sig += 1; sig_junc.append(b)
        if n_sig in (3, 4):
            arm_sum = sum(a["branch-distance"] for a in sig_len) \
                + sum(MIN_BRANCH_LENGTH for _ in sig_junc)
            results.append((j, skel_obj.coordinates[j], n_sig, arm_sum))

    valid_3_sorted = sorted(
        [(j, al) for j, _, n, al in results if n == 3],
        key=lambda x: x[1], reverse=True)
    valid_4_set = {j for j, _, n, _ in results if n == 4}
    hshape_nodes = {n for n1, n2, _ in hshape_candidates for n in (n1, n2)}
    suppressed: set[int] = set()
    for j, _ in valid_3_sorted:
        if j in suppressed:
            continue
        reachable = nx.single_source_dijkstra_path_length(
            nx_graph, j, cutoff=MAX_3WAY_PRIORITY_DISTANCE)
        for node in reachable:
            if node != j and node not in suppressed and (
                    node in valid_4_set
                    or node in {k for k, _ in valid_3_sorted}
                    or node in hshape_nodes):
                suppressed.add(node)

    for j, coord, n_arms, _ in results:
        if j in suppressed:
            continue
        (valid_3 if n_arms == 3 else valid_4).append(coord)
    for n1, n2, midpoint in hshape_candidates:
        if n1 not in suppressed and n2 not in suppressed:
            valid_4.append(midpoint)

    coords_3 = np.array(valid_3) if valid_3 else np.empty((0, 2))
    coords_4 = np.array(valid_4) if valid_4 else np.empty((0, 2))
    return coords_3, coords_4


def merge_nearby_junctions(coords_3, coords_4):
    if len(coords_3) == 0 and len(coords_4) == 0:
        return coords_3, coords_4
    all_coords = np.concatenate([coords_3, coords_4], axis=0)
    all_types = np.array([3] * len(coords_3) + [4] * len(coords_4))
    tree = KDTree(all_coords)
    adj = tree.sparse_distance_matrix(
        tree, max_distance=JUNCTION_MERGE_DISTANCE, output_type="coo_matrix")
    n_comp, labels = connected_components(adj, directed=False)
    merged_3, merged_4 = [], []
    for lbl in range(n_comp):
        m = labels == lbl
        centroid = all_coords[m].mean(axis=0)
        if np.sum(all_types[m] == 4) > np.sum(all_types[m] == 3):
            merged_4.append(centroid)
        else:
            merged_3.append(centroid)
    return (np.array(merged_3) if merged_3 else np.empty((0, 2)),
            np.array(merged_4) if merged_4 else np.empty((0, 2)))


def detect_junctions_in_segmentation_mask(
    segmentation_mask: torch.Tensor,
    verbose: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns (coords_3way, coords_4way, skeleton) with coords in (x, y)."""
    segmentation_mask = remove_small_bbox_objects(segmentation_mask)
    skel = skeletonize_mask(segmentation_mask)
    skel = prune_skeleton(skel)
    skel = reconnect_skeleton_gaps(skel)
    skel = prune_small_cycles(skel)

    _, _, degrees = get_graph_coordinates_degrees(skel)
    junctions = np.where(degrees > 2)[0]
    coords_3, coords_4 = filter_junctions_by_length(skel, junctions, degrees, verbose)
    coords_3, coords_4 = merge_nearby_junctions(coords_3, coords_4)
    return coords_3[:, ::-1], coords_4[:, ::-1], skel


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _load_mask_png(path: Path) -> torch.Tensor:
    arr = np.array(Image.open(path))
    if arr.ndim == 3:
        arr = arr[..., 0]
    return torch.from_numpy((arr > 0).astype(np.float32)).unsqueeze(0)


def main() -> None:
    p = argparse.ArgumentParser(
        description="Run junction detection on a folder of binary mask PNGs.",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pred-dir", required=True, type=Path,
                   help="Directory with binary segmentation PNGs (one per image).")
    p.add_argument("--out-json", required=True, type=Path,
                   help="Path to write {<stem>: {'3way': [...], '4way': [...]}}.")
    p.add_argument("--save-skeletons", type=Path, default=None,
                   help="Optional folder to write skeleton PNGs to.")
    args = p.parse_args()

    pred_dir: Path = args.pred_dir
    if not pred_dir.is_dir():
        raise SystemExit(f"Not a directory: {pred_dir}")
    if args.save_skeletons is not None:
        args.save_skeletons.mkdir(parents=True, exist_ok=True)

    results: dict[str, dict] = {}
    mask_paths = sorted(pred_dir.glob("*.png"))
    for i, p in enumerate(mask_paths):
        mask = _load_mask_png(p)
        c3, c4, skel = detect_junctions_in_segmentation_mask(mask)
        results[p.stem] = {
            "3way": c3.tolist(),
            "4way": c4.tolist(),
        }
        if args.save_skeletons is not None:
            Image.fromarray((skel * 255).astype(np.uint8)).save(
                args.save_skeletons / f"{p.stem}.png")
        print(f"  [{i+1}/{len(mask_paths)}] {p.stem}: "
              f"{len(c3)} 3-way, {len(c4)} 4-way")

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved junctions to {args.out_json}")


if __name__ == "__main__":
    main()
