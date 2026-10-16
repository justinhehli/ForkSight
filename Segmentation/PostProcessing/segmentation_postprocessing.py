import torch
import numpy as np
import torch.nn.functional as F
from scipy.ndimage import label, generate_binary_structure

from Environment.env_utils import load_as, load_as_bool, load_shared_env

# Read env lazily inside functions rather than as module-level constants to
# avoid issues, and load shared.env here defensively to guarantee they're loaded
load_shared_env()


def _min_obj_size() -> int | None:
    return load_as("POSTPROCESSING_MIN_OBJ_SIZE", int, 100)


def _connect_diagonally() -> bool:
    return load_as_bool("POSTPROCESSING_CONNECT_DIAGONALLY", True)


def _small_bbox_threshold() -> int | None:
    return load_as("POSTPROCESSING_SMALL_BBOX_THRESHOLD", int, None)


def get_connected_components(mask: torch.Tensor) -> tuple[np.ndarray, int]:
    """
    :param mask: mask of shape (1, H, W)
    :return: tuple of np.ndarray of shape (H, W) with connected component labels and number of components
    """
    mask = (mask[0] > 0).cpu().numpy()

    binary_structure = None
    if _connect_diagonally():
        binary_structure = generate_binary_structure(2, 2)

    return label(mask, structure=binary_structure)


def remove_small_objects_from_batch(masks: torch.Tensor) -> torch.Tensor:
    """
    :param masks: Tensor of shape (B, 1, H, W)
    :return: Tensor of same shape, with small connected components removed
    """

    assert masks.ndim == 4 and masks.shape[
        1] == 1, f"Expected mask shape (B, 1, H, W), got {masks.shape}"

    B = masks.shape[0]
    output = torch.zeros_like(masks)
    min_obj_size = _min_obj_size()

    for b in range(B):
        labeled_mask, num_components = get_connected_components(masks[b])
        cleaned_mask = np.zeros_like(labeled_mask, dtype=np.uint8)

        for component_idx in range(1, num_components + 1):
            ys, xs = np.where(labeled_mask == component_idx)

            if min_obj_size is not None and ys.size < min_obj_size:
                continue

            cleaned_mask[ys, xs] = 1

        output[b, 0] = torch.from_numpy(cleaned_mask)

    return output


def stitch_mask_tiles(masks: torch.Tensor, grid_size: tuple[int, int], original_input_patch_img_size: tuple[int, int], as_uint: bool = True) -> torch.Tensor:
    """
    :param masks: Tensor of shape (B, 1, H, W)
    :param grid_size: tuple of (grid_rows, grid_cols) indicating how patches were cut from the original full image
    :param original_input_patch_img_size: tuple of (H, W) indicating the original size of the input image patches before resizing for SAM input
    :return: Tensor of shape (1, H*grid_rows, W*grid_cols)
    """

    assert masks.ndim == 4 and masks.shape[
        1] == 1, f"Expected mask shape (B, 1, H, W), got {masks.shape}"
    assert masks.shape[0] == grid_size[0] * \
        grid_size[1], f"Number of masks does not match grid size, got {masks.shape[0]}"

    masks = F.interpolate(
        masks.float(),
        size=(original_input_patch_img_size[0],
              original_input_patch_img_size[1]),
        mode="nearest"
    )

    if as_uint:
        masks = masks.to(torch.uint8)

    _, _, H, W = masks.shape

    masks = masks.view(grid_size[0], grid_size[1], H, W)
    masks = masks.permute(0, 2, 1, 3)
    return masks.reshape(1, grid_size[0] * H, grid_size[1] * W)


def extract_mask_elements_bboxes(mask: torch.Tensor) -> list[tuple[int, int, int, int]]:
    '''
    :param mask: SINGLE segmentation mask, shape (1, H, W)
    :return: list of bounding boxes (x1, y1, x2, y2)
    '''

    assert mask.ndim == 3 and mask.shape[0] == 1, "Expected mask shape [1, H, W]"

    labeled_mask, num_components = get_connected_components(mask)

    boxes = []
    for component_idx in range(1, num_components + 1):
        ys, xs = np.where(labeled_mask == component_idx)

        x1 = int(xs.min())
        y1 = int(ys.min())
        x2 = int(xs.max())
        y2 = int(ys.max())

        boxes.append((x1, y1, x2, y2))

    return boxes


def remove_small_bbox_objects(mask: torch.Tensor) -> torch.Tensor:
    '''
    Remove connected components whose bounding box width AND height are both
    strictly below POSTPROCESSING_SMALL_BBOX_THRESHOLD.

    :param mask: SINGLE segmentation mask, shape (1, H, W)
    :return: mask of same shape with small-bbox components zeroed out
    '''
    assert mask.ndim == 3 and mask.shape[0] == 1, "Expected mask shape [1, H, W]"

    small_bbox_threshold = _small_bbox_threshold()
    if small_bbox_threshold is None:
        return mask

    labeled_mask, num_components = get_connected_components(mask)
    cleaned = np.zeros_like(labeled_mask, dtype=np.uint8)

    for component_idx in range(1, num_components + 1):
        ys, xs = np.where(labeled_mask == component_idx)
        w = int(xs.max()) - int(xs.min()) + 1
        h = int(ys.max()) - int(ys.min()) + 1
        if w < small_bbox_threshold and h < small_bbox_threshold:
            continue
        cleaned[ys, xs] = 1

    result = torch.zeros_like(mask)
    result[0] = torch.from_numpy(cleaned)
    return result


def postprocess_segmentation_masks(masks: torch.Tensor, grid_size: tuple[int, int],
                                   original_input_patch_img_size: tuple[int, int],
                                   remove_small_objects: bool = True) -> tuple[torch.Tensor, list[tuple[int, int, int, int]]]:
    '''
    :param masks: batch of segmentation masks, shape (B, 1, H, W), where B = grid_rows * grid_cols (e.g. 16 for 4x4 grid) of ONE image
    :param grid_size: tuple of (grid_rows, grid_cols) indicating how patches were cut from the original full image
    :param original_input_patch_img_size: tuple of (H, W) indicating the original size of the individual input image patches before resizing for SAM input
    :return: tuple of the stitched masks (with small elements removed) tensor and list of bounding boxes for each mask element
    '''
    cleaned_masks = remove_small_objects_from_batch(
        masks) if remove_small_objects else masks
    stitched_mask = stitch_mask_tiles(
        cleaned_masks, grid_size, original_input_patch_img_size)
    boxes = extract_mask_elements_bboxes(stitched_mask)

    return stitched_mask, boxes
