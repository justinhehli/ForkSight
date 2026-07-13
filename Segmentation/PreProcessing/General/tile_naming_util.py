import re
from pathlib import Path


def get_display_name(img_path: Path) -> str:
    tileset_match = re.search(r"\d+", img_path.parent.name)
    tileset_num = tileset_match.group() if tileset_match else img_path.parent.name

    tile_stem = img_path.stem.replace("-000000_0-000", "")
    tile_part = tile_stem.split("_", 1)[-1].replace("-", " ")

    return f"Tile Set {tileset_num} - Tile {tile_part}"


def get_exp_dir_name(img_path: Path) -> str:
    parts = img_path.parts
    idx = parts.index("LayersData")
    return parts[idx - 1].lower()


def get_tileset_tile_name(img_path: Path) -> str:
    tileset = img_path.parent.name
    tileset = tileset.replace("Tile Set ", "tileset_").replace(
        " ", "_").replace("(", "").replace(")", "")
    tile = img_path.stem.replace("-000000_0-000", "").replace("-", "_")
    return f"{tileset}_{tile}".lower()


def get_new_name(img_path: Path, suffix: str = None) -> str:
    suffix = f'_{suffix}' if suffix else ''
    exp_dir_name = get_exp_dir_name(img_path)
    tileset_tile_name = get_tileset_tile_name(img_path)
    return f"{exp_dir_name}_{tileset_tile_name}{suffix}.png"
