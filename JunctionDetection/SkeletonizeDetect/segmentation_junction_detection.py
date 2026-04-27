import torch
import numpy as np
from skimage.morphology import skeletonize
from skimage.draw import line as draw_line
from skan.csr import skeleton_to_csgraph, summarize, Skeleton
import networkx as nx
from scipy.spatial import KDTree
from scipy.sparse.csgraph import connected_components

from Segmentation.PostProcessing.segmentation_postprocessing import remove_small_bbox_objects


# minimum length of significan branch for a junction
MIN_BRANCH_LENGTH = 100
# upper length limit for a branch connecting two junctions to be considered part of a 4-way junction (H shape)
MAX_JUNCTION_CONNECTOR_LENGTH = 40
# terminal branches shorter than this are pruned (if a junction has multiple terminal branches, the longest is preserved to avoid over-pruning)
MIN_TERMINAL_BRANCH_LENGTH = 40
# junctions on cycles with total length below this are excluded, junctions on larger cycles may be valid
MIN_CYCLE_LENGTH = 4000
# junctions within this pixel distance of any node on a (small) cycle are excluded
JUNCTION_CYCLE_PROXIMITY_DISTANCE = 150
# 4-way junctions within this distance of a VALID 3-way junction (replication fork) are suppressed
MAX_3WAY_PRIORITY_DISTANCE = 500
# junctions of any type within this pixel distance of each other are merged into one
JUNCTION_MERGE_DISTANCE = 50
# skeleton tip pairs within this pixel distance whose trajectories align are reconnected (gap repair)
MAX_SKELETON_GAP_DISTANCE = 40
# maximum angle (degrees) between a branch trajectory and the gap vector for reconnection
MAX_GAP_ANGLE_DEG = 20
# number of pixels along a branch used to estimate its tip direction for gap reconnection
GAP_DIRECTION_SAMPLE_LENGTH = 20
# skeleton cycles with total perimeter below this are collapsed by removing the shortest edge
MIN_SMALL_CYCLE_PRUNE_LENGTH = 100


def skeletonize_mask(segmentation_mask: torch.Tensor) -> np.ndarray:
    '''
    Skeletonize a single segmentation mask
    segmentation_mask: torch.Tensor of shape (C, H, W)
    Returns: np.ndarray of shape (C, H, W) representing the skeletonized mask
    '''
    mask_np = segmentation_mask.detach().cpu().squeeze().numpy().astype(np.uint8)
    binary_mask = (mask_np > 0).astype(np.uint8)
    return skeletonize(binary_mask).astype(np.uint8)


def prune_skeleton(skeleton: np.ndarray, iterations: int = 3) -> np.ndarray:
    """
    Removes short terminal branches (spurs) iteratively. 
    If a junction has multiple spurs, the longest is preserved.
    """
    current_skeleton = skeleton.copy()

    for _ in range(iterations):
        skel_obj = Skeleton(current_skeleton)
        stats = summarize(skel_obj)
        _, _, degrees = get_graph_coordinates_degrees(current_skeleton)

        # dict track terminal branches attached to each junction
        # key: junction_node_id, value: list of (branch_length, path_index, tip_node_id)
        junction_to_spurs = {}

        for i, row in stats.iterrows():
            u, v = int(row['node-id-src']), int(row['node-id-dst'])

            u_is_tip, v_is_tip = degrees[u] == 1, degrees[v] == 1
            u_is_junc, v_is_junc = degrees[u] > 2, degrees[v] > 2

            if (u_is_tip and v_is_junc) or (v_is_tip and u_is_junc):
                if row['branch-distance'] < MIN_TERMINAL_BRANCH_LENGTH:
                    junc_id = v if u_is_tip else u
                    tip_id = u if u_is_tip else v

                    if junc_id not in junction_to_spurs:
                        junction_to_spurs[junc_id] = []
                    junction_to_spurs[junc_id].append(
                        (row['branch-distance'], i, tip_id))

        paths_to_delete = []
        for junc_id, spurs in junction_to_spurs.items():
            # sort spurs from junction by length and keep the longest
            spurs.sort(key=lambda x: x[0], reverse=True)
            spurs_to_remove = spurs[1:] if len(spurs) >= 2 else spurs
            for _, path_idx, _ in spurs_to_remove:
                paths_to_delete.append(path_idx)

        if not paths_to_delete:
            break

        for path_idx in paths_to_delete:
            coords = skel_obj.path_coordinates(path_idx).astype(int)
            node_src = int(stats.loc[path_idx, 'node-id-src'])

            # Remove all but the junction pixel to prevent breaks
            pixels_to_remove = coords[:-
                                      1] if degrees[node_src] == 1 else coords[1:]
            for r, c in pixels_to_remove:
                current_skeleton[r, c] = 0

        # clean up stubs and re-thin before next iteration
        current_skeleton = skeletonize(current_skeleton > 0).astype(np.uint8)

    return current_skeleton


def _fit_tip_direction(path_coords, sample_length, from_src):
    if from_src:
        segment = path_coords[:min(sample_length, len(path_coords))]
        tip, interior = segment[0], segment[-1]
    else:
        segment = path_coords[max(0, len(path_coords) - sample_length):]
        tip, interior = segment[-1], segment[0]
    if len(segment) < 2:
        return None
    _, _, vh = np.linalg.svd(
        segment - segment.mean(axis=0), full_matrices=False)
    direction = vh[0]
    if np.dot(tip - interior, direction) < 0:
        direction = -direction
    return direction


def reconnect_skeleton_gaps(skeleton: np.ndarray) -> np.ndarray:
    skel_obj = Skeleton(skeleton)
    stats = summarize(skel_obj)
    stats = stats[stats['node-id-src'] != stats['node-id-dst']]
    _, coordinates, degrees = get_graph_coordinates_degrees(skeleton)

    tip_indices = np.where(degrees == 1)[0]
    if len(tip_indices) < 2:
        return skeleton

    tip_directions = {}
    for tip_idx in tip_indices:
        tip_branches = stats[(stats['node-id-src'] == tip_idx)
                             | (stats['node-id-dst'] == tip_idx)]
        if len(tip_branches) != 1:
            continue
        branch = tip_branches.iloc[0]
        path_coords = skel_obj.path_coordinates(branch.name).astype(float)
        from_src = (int(branch['node-id-src']) == tip_idx)
        direction = _fit_tip_direction(
            path_coords, GAP_DIRECTION_SAMPLE_LENGTH, from_src)
        if direction is not None:
            tip_directions[tip_idx] = direction

    tip_list = list(tip_directions.keys())
    if len(tip_list) < 2:
        return skeleton

    tip_coords = np.array([coordinates[t] for t in tip_list])
    pairs = KDTree(tip_coords).query_pairs(MAX_SKELETON_GAP_DISTANCE)
    cos_threshold = np.cos(np.deg2rad(MAX_GAP_ANGLE_DEG))

    scored_candidates = []
    for i, j in pairs:
        c1, c2 = tip_coords[i], tip_coords[j]
        d1, d2 = tip_directions[tip_list[i]], tip_directions[tip_list[j]]

        gap_vec = c2 - c1
        gap_norm = np.linalg.norm(gap_vec)
        if gap_norm == 0:
            continue
        gap_unit = gap_vec / gap_norm

        dot1 = np.dot(d1, gap_unit)
        dot2 = np.dot(d2, -gap_unit)
        if dot1 < cos_threshold or dot2 < cos_threshold:
            continue

        avg_axis = d1 - d2
        axis_norm = np.linalg.norm(avg_axis)
        if axis_norm < 1e-8 or abs(np.dot(avg_axis / axis_norm, gap_unit)) < cos_threshold:
            continue

        scored_candidates.append((dot1 + dot2, i, j))

    scored_candidates.sort(key=lambda x: -x[0])

    used = set()
    result = skeleton.copy()
    for _, i, j in scored_candidates:
        if i in used or j in used:
            continue
        rr, cc = draw_line(int(tip_coords[i][0]), int(tip_coords[i][1]),
                           int(tip_coords[j][0]), int(tip_coords[j][1]))
        result[rr, cc] = 1
        used.add(i)
        used.add(j)

    return skeletonize(result > 0).astype(np.uint8)


def prune_small_cycles(skeleton: np.ndarray) -> np.ndarray:
    """Collapse tiny cycles (perimeter < MIN_SMALL_CYCLE_PRUNE_LENGTH) by
    removing the shortest edge in each, reducing small loops to single paths."""
    skel_obj = Skeleton(skeleton)
    stats = summarize(skel_obj)
    stats = stats[stats['node-id-src'] != stats['node-id-dst']]

    nx_graph = nx.from_scipy_sparse_array(skel_obj.graph)

    branches_to_remove = set()
    for cycle in nx.cycle_basis(nx_graph):
        cycle_length = sum(
            nx_graph[cycle[i]][cycle[(i + 1) % len(cycle)]].get('weight', 0)
            for i in range(len(cycle))
        )
        if cycle_length >= MIN_SMALL_CYCLE_PRUNE_LENGTH:
            continue

        # Break the cycle by removing its shortest edge
        min_weight = float('inf')
        min_edge = None
        for i in range(len(cycle)):
            u, v = cycle[i], cycle[(i + 1) % len(cycle)]
            w = nx_graph[u][v].get('weight', 0)
            if w < min_weight:
                min_weight = w
                min_edge = (min(u, v), max(u, v))
        if min_edge is not None:
            branches_to_remove.add(min_edge)

    if not branches_to_remove:
        return skeleton

    result = skeleton.copy()
    for idx, row in stats.iterrows():
        edge = (min(int(row['node-id-src']), int(row['node-id-dst'])),
                max(int(row['node-id-src']), int(row['node-id-dst'])))
        if edge in branches_to_remove:
            coords = skel_obj.path_coordinates(idx).astype(int)
            # Remove inner pixels, preserve junction endpoints
            for r, c in coords[1:-1]:
                result[r, c] = 0

    return skeletonize(result > 0).astype(np.uint8)


def get_graph_coordinates_degrees(skeleton: np.ndarray):
    '''
    Convert skeleton to graph and compute node degrees
    skeleton: np.ndarray of shape (H, W)
    Returns: graph in CSR format and degrees of each node
    '''
    graph, coordinates = skeleton_to_csgraph(skeleton)
    degrees = np.diff(graph.indptr)
    coordinates = np.stack(np.asarray(coordinates), axis=1)
    return graph, coordinates, degrees


def filter_junctions_by_length(skeleton: np.ndarray, junction_indices: np.ndarray, degrees: np.ndarray,
                               verbose=False) -> tuple[np.ndarray, np.ndarray]:
    skel_obj = Skeleton(skeleton)
    skel_stats = summarize(skel_obj)
    skel_stats = skel_stats[skel_stats['node-id-src']
                            != skel_stats['node-id-dst']]

    # find all nodes in small cycles (large cycles aren't excluded)
    nx_graph = nx.from_scipy_sparse_array(skel_obj.graph)
    nodes_in_cycles = set()
    for cycle in nx.cycle_basis(nx_graph):
        cycle_length = sum(
            nx_graph[cycle[i]][cycle[(i + 1) % len(cycle)]].get('weight', 0)
            for i in range(len(cycle))
        )
        if cycle_length < MIN_CYCLE_LENGTH:
            nodes_in_cycles.update(cycle)

    # KDTree over cycle-node coordinates for fast proximity queries
    _cycle_coords = np.array([skel_obj.coordinates[n]
                             for n in nodes_in_cycles]) if nodes_in_cycles else None
    _cycle_kdtree = KDTree(
        _cycle_coords) if _cycle_coords is not None else None

    def is_close_to_cycle(coord):
        if _cycle_kdtree is None:
            return False
        dist, _ = _cycle_kdtree.query(coord)
        return dist < JUNCTION_CYCLE_PROXIMITY_DISTANCE

    valid_coords_3way = []
    valid_coords_4way = []
    nodes_handled = set()

    def check_path_significance_by_length(branch_row, source_node_idx, current_length, current_depth):
        # Prevent infinite recursion
        if current_depth > 3:
            return False

        new_length = current_length + branch_row['branch-distance']
        if new_length >= MIN_BRANCH_LENGTH:
            return True

        u, v = int(branch_row['node-id-src']), int(branch_row['node-id-dst'])
        neighbor_idx = v if u == source_node_idx else u

        if neighbor_idx in nodes_in_cycles:
            return False

        # If the neighbor is a junction, recursively check it's other branches
        if degrees[neighbor_idx] > 2:
            neighbor_branches = skel_stats[(skel_stats['node-id-src'] == neighbor_idx) |
                                           (skel_stats['node-id-dst'] == neighbor_idx)]

            for _, n_branch in neighbor_branches.iterrows():
                # skip the branch we came from
                if n_branch.name == branch_row.name:
                    continue

                if check_path_significance_by_length(n_branch, neighbor_idx, new_length, current_depth + 1):
                    return True

        return False

    # handle 4-way artifacts and junctions connected by a short branch (H shape)
    hshape_candidates = []  # (n1, n2, midpoint)
    for _, branch in skel_stats.iterrows():
        n1, n2 = int(branch['node-id-src']), int(branch['node-id-dst'])

        # check if short branch connects two junctions
        if (degrees[n1] > 2 and degrees[n2] > 2 and branch['branch-distance'] <= MAX_JUNCTION_CONNECTOR_LENGTH):
            if n1 in nodes_handled or n2 in nodes_handled:
                continue

            # If the junction is part of or close to a loop, it's an artifact
            if n1 in nodes_in_cycles or n2 in nodes_in_cycles or \
                    is_close_to_cycle(skel_obj.coordinates[n1]) or is_close_to_cycle(skel_obj.coordinates[n2]):
                nodes_handled.update([n1, n2])
                continue

            # Check significant arms for both junctions (junction connector ends)
            cluster_arms = skel_stats[((skel_stats['node-id-src'].isin([n1, n2])) |
                                       (skel_stats['node-id-dst'].isin([n1, n2]))) &
                                      (skel_stats.index != branch.name)]

            nof_significant_arms = sum(1 for _, arm in cluster_arms.iterrows()
                                       if arm['branch-distance'] >= MIN_BRANCH_LENGTH)

            # if the cluster connected by the short branch has 4 significant arms, keep as candidate
            if nof_significant_arms == 4:
                midpoint = (
                    skel_obj.coordinates[n1] + skel_obj.coordinates[n2]) / 2
                hshape_candidates.append((n1, n2, midpoint))

            nodes_handled.update([n1, n2])

    # handle standard 3-way and 4-way junctions
    junction_results = []
    for j_idx in junction_indices:
        if j_idx in nodes_handled or j_idx in nodes_in_cycles or is_close_to_cycle(skel_obj.coordinates[j_idx]):
            continue

        node_branches = skel_stats[(skel_stats['node-id-src'] == j_idx) |
                                   (skel_stats['node-id-dst'] == j_idx)]

        # an arm is only significant if it is long enough or leads to a distant junction
        nof_significant_arms = 0
        significant_arms_by_length = []
        significant_arms_to_junction = []
        for _, branch in node_branches.iterrows():
            if branch['branch-distance'] >= MIN_BRANCH_LENGTH:
                nof_significant_arms += 1
                significant_arms_by_length.append(branch)
            else:
                u, v = int(branch['node-id-src']), int(branch['node-id-dst'])
                other = v if u == j_idx else u
                if degrees[other] > 2 and check_path_significance_by_length(branch, j_idx, 0, 0):
                    # and branch['branch-distance'] > MAX_JUNCTION_CONNECTOR_LENGTH and
                    nof_significant_arms += 1
                    significant_arms_to_junction.append(branch)

        if nof_significant_arms == 3 or nof_significant_arms == 4:
            arm_length_sum = (
                sum(arm['branch-distance'] for arm in significant_arms_by_length) +
                # arms to junctions are usually paths with multiple branches, so we only add the minimum significant branch length to the sum
                sum(MIN_BRANCH_LENGTH for _ in significant_arms_to_junction)
            )
            junction_results.append(
                (j_idx, skel_obj.coordinates[j_idx], nof_significant_arms, arm_length_sum))

            if verbose:
                print(
                    f"candidate junction at node {j_idx} (degree {degrees[j_idx]})")
                print("significant arms by length:",
                      len(significant_arms_by_length))
                for arm in significant_arms_by_length:
                    print(
                        f"    length: {arm['branch-distance']}, nodes: {arm['node-id-src']}->{arm['node-id-dst']}, type: {arm['branch-type']}")
                print("significant arms to junction:",
                      len(significant_arms_to_junction))
                for arm in significant_arms_to_junction:
                    print(
                        f"    length: {arm['branch-distance']}, nodes: {arm['node-id-src']}->{arm['node-id-dst']}, type: {arm['branch-type']}")

    # suppress 4-way / 3-way junctions that are close to a 3-way junction
    # for two 3-way junctions, the one with the longer sum of significant arms has priority and is kept
    # uses graph shortest-path distance, to find junctions that are close to the 3-way junction
    valid_3way_indices_by_arm_length_sum = sorted(
        [(j_idx, arm_len)
         for j_idx, _, nof_arms, arm_len in junction_results if nof_arms == 3],
        key=lambda x: x[1],
        reverse=True,
    )
    valid_4way_indices = {j_idx for j_idx, _, nof_arms,
                          _ in junction_results if nof_arms == 4}
    hshape_node_indices = {n for n1, n2,
                           _ in hshape_candidates for n in (n1, n2)}
    suppressed_by_3way = set()
    for j_idx, _ in valid_3way_indices_by_arm_length_sum:
        # skip already suppressed junctions s.t. junctions don't suppress each other
        if j_idx in suppressed_by_3way:
            continue
        reachable = nx.single_source_dijkstra_path_length(
            nx_graph, j_idx, cutoff=MAX_3WAY_PRIORITY_DISTANCE)
        for node, _ in reachable.items():
            if node != j_idx and node not in suppressed_by_3way and \
                    (node in valid_4way_indices or node in valid_3way_indices_by_arm_length_sum or node in hshape_node_indices):
                suppressed_by_3way.add(node)

    for j_idx, coord, nof_arms, _ in junction_results:
        if j_idx not in suppressed_by_3way:
            if nof_arms == 3:
                valid_coords_3way.append(coord)
            else:
                valid_coords_4way.append(coord)
            if verbose:
                print(f"valid junction at node {j_idx} ({nof_arms}-way)")
    for n1, n2, midpoint in hshape_candidates:
        if n1 not in suppressed_by_3way and n2 not in suppressed_by_3way:
            valid_coords_4way.append(midpoint)
            if verbose:
                print(f"valid H-shape junction between nodes {n1} and {n2}")

    coords_3way = np.array(
        valid_coords_3way) if valid_coords_3way else np.empty((0, 2))
    coords_4way = np.array(
        valid_coords_4way) if valid_coords_4way else np.empty((0, 2))
    return coords_3way, coords_4way


def merge_nearby_junctions(coords_3way: np.ndarray, coords_4way: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if len(coords_3way) == 0 and len(coords_4way) == 0:
        return coords_3way, coords_4way

    all_coords = np.concatenate([coords_3way, coords_4way], axis=0)
    all_types = np.array([3] * len(coords_3way) + [4] * len(coords_4way))

    # build adjacency from all pairs within merge distance
    tree = KDTree(all_coords)
    adj = tree.sparse_distance_matrix(
        tree, max_distance=JUNCTION_MERGE_DISTANCE, output_type='coo_matrix')
    n_components, labels = connected_components(adj, directed=False)

    merged_3way, merged_4way = [], []
    for label in range(n_components):
        mask = labels == label
        centroid = all_coords[mask].mean(axis=0)
        if np.sum(all_types[mask] == 4) > np.sum(all_types[mask] == 3):
            merged_4way.append(centroid)
        else:
            merged_3way.append(centroid)

    return (
        np.array(merged_3way) if merged_3way else np.empty((0, 2)),
        np.array(merged_4way) if merged_4way else np.empty((0, 2)),
    )


def detect_junctions_in_segmentation_mask(
        segmentation_mask: torch.Tensor, verbose=False
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    '''
    Returns (coords_3way, coords_4way, skeleton) where each coords array is
    (N, 2) in (x, y) image coordinates.
    '''
    segmentation_mask = remove_small_bbox_objects(segmentation_mask)
    skeleton = skeletonize_mask(segmentation_mask)
    skeleton = prune_skeleton(skeleton)
    skeleton = reconnect_skeleton_gaps(skeleton)
    skeleton = prune_small_cycles(skeleton)

    _, _, degrees = get_graph_coordinates_degrees(skeleton)

    junction_indices = np.where(degrees > 2)[0]
    coords_3way, coords_4way = filter_junctions_by_length(
        skeleton, junction_indices, degrees, verbose=verbose)

    coords_3way, coords_4way = merge_nearby_junctions(coords_3way, coords_4way)

    # Convert (y, x) to (x, y) to match image coordinate system
    return coords_3way[:, ::-1], coords_4way[:, ::-1], skeleton
