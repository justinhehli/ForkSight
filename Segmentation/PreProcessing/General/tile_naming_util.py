from pathlib import Path


def get_exp_dir_name(img_path: Path) -> str:
    parts = img_path.parts
    idx = parts.index("LayersData")
    return parts[idx - 1].lower()


def get_tileset_tile_name(img_path: Path) -> str:
    parent_name = img_path.parent.name.lower().replace(
        " ", "_").replace("(", "").replace(")", "").replace("-", "_")
    tile = img_path.stem.replace("-000000_0-000", "").replace("-", "_")
    return f"{parent_name}_{tile}".lower()


def get_new_name(img_path: Path, suffix: str = None) -> str:
    suffix = f'_{suffix}' if suffix else ''
    exp_dir_name = get_exp_dir_name(img_path)
    tileset_tile_name = get_tileset_tile_name(img_path)
    return f"{exp_dir_name}_{tileset_tile_name}{suffix}.png"
