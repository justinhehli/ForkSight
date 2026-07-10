from pathlib import Path

import numpy as np
import torch
from PIL import Image

PATCH_SIZE = (1024, 1024)
GRID_SIZE = (4, 4)
N_PATCHES = GRID_SIZE[0] * GRID_SIZE[1]


def patch_stem(image_stem: str, idx: int) -> str:
    return f"{image_stem}_patch_{idx:02d}"


def pred_patch_filename(image_stem: str, idx: int) -> str:
    return f"{patch_stem(image_stem, idx)}.png"


def nnunet_input_patch_filename(image_stem: str, idx: int) -> str:
    return f"{patch_stem(image_stem, idx)}_0000.png"


def load_binary_mask_pred_patches(pred_dir: Path, image_stem: str, n_patches: int = N_PATCHES) -> torch.Tensor:
    patches, patch_paths = [], []
    for idx in range(n_patches):
        patch_path = Path(pred_dir) / pred_patch_filename(image_stem, idx)
        arr = np.array(Image.open(patch_path))
        if arr.ndim == 3:
            arr = arr[..., 0]
        mask = torch.from_numpy((arr > 0).astype(np.float32)).unsqueeze(0)
        patches.append(mask)
        patch_paths.append(patch_path)
    return torch.stack(patches), patch_paths
