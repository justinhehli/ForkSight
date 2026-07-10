import os
from pathlib import Path
from PIL import Image
import numpy as np
from typing import Optional, Tuple

FULL_IMAGE_SIZE = (4096, 4096)

IN_IMAGES = [
    ("Z:\\imcrdata\\2024_Andrea_ETP_R2\\20240911_Andrea_Black\\LayersData\\highmag\\Tile Set (14)\\Tile_007-004-000000_0-000.tif", None),
    ("Z:\\imcrdata\\2024_Andrea_NBS1\\2024_Andrea_NBS1_R1\\20240523_Andrea_Orange\\LayersData\\highmag\\Tile Set (17)\\Tile_016-003-000000_0-000.tif", None),
    ("Z:\\imcrdata\\2025_Dani_SMC1\\R1\\20250811_Dani_Vegetariana\\LayersData\\highmag\\Tile Set (2)\\Tile_005-009-000000_0-000.tif", None),
    ("Z:\\imcrdata\\2025_Filip_RNF20\\2025_Filip_RNF20_R2\\20250107_Filip_Zurich\\LayersData\\highmag\\Tile Set (13)\\Tile_002-006-000000_0-000.tif", None),
    ("Z:\\imcrdata\\2025_Jana_MRE11_Inh\\20250812_Jana_Ant\\LayersData\\highmag\\Tile Set (24)\\Tile_002-006-000000_0-000.tif", None),
    ("Z:\\imcrdata\\2025_Veronica_Lamin\\2024_Veronica_Lamin\\2024_Veronica_Lamin_R1\\20240529_Veronica_Sample5\\LayersData\\highmag\\Tile Set (12)\\Tile_001-003-000000_0-000.tif", None),
    ("Z:\\imcrdata\\2025_Veronica_Lamin\\2025_Veronica_G9ai\\R2\\20250708_Vero_Larch_Part2\\LayersData\\highmag\\Tile Set (8)\\Tile_009-011-000000_0-000.tif", None),

    ("Z:\\imcrdata\\2024_Andrea_ETP_R2\\20250808_Andrea_Magenta\\LayersData\\highmag\\Tile Set (6)\\Tile_005-010-000000_0-000.tif", None),
    ("Z:\\imcrdata\\2024_Andrea_NBS1\\2024_Andrea_NBS1_R1\\20240407_Andrea_Red\\LayersData\\highmag\\Tile Set (18)\\Tile_007-016-000000_0-000.tif", None),
    ("Z:\\imcrdata\\2024_Andrea_NBS1\\2024_Andrea_NBS1_R2\\20240425_Andrea_lila\\LayersData\\highmag\\Tile Set (14)\\Tile_014-015-000000_0-000.tif", None),
    ("Z:\\imcrdata\\2024_Andrea_NBS1\\2024_Andrea_NBS1_R2\\20240613_Andrea_Green\\LayersData\\highmag\\Tile Set (15)\\Tile_006-011-000000_0-000.tif", None),
    ("Z:\\imcrdata\\2025_Cyril_AraC\\20250528_Cyril_Alpaka\\LayersData\\highmag\\Tile Set (21)\\Tile_016-013-000000_0-000.tif", None),
    ("Z:\\imcrdata\\2025_Dani_SMC1\\R1\\20250305_Dani_Margherita\\LayersData\\highmag\\Tile Set (14)\\Tile_016-003-000000_0-000.tif", None),
    ("Z:\\imcrdata\\2025_Dani_SMC1\\R2\\20251023_Dani_Thor\\LayersData\\highmag\\Tile Set (13)\\Tile_014-008-000000_0-000.tif", None),
    ("Z:\\imcrdata\\2025_Filip_RNF20\\2025_Filip_RNF20_R2\\20250108_Filip_Basel\\LayersData\\highmag\\Tile Set (3)\\Tile_005-015-000000_0-000.tif", None),
    ("Z:\\imcrdata\\2025_Jana_MRE11_Inh\\20250915_Jana_Wasp\\LayersData\\highmag\\Tile Set (11)\\Tile_010-012-000000_0-000.tif", None),
    ("Z:\\imcrdata\\2025_Veronica_Lamin\\2025_Veronica_G9ai\\R1\\20250709_Veronica_Cedar\\LayersData\\highmag\\Tile Set (15)\\Tile_006-012-000000_0-000.tif", None),

    ("Z:\\imcrdata\\2024_Andrea_NBS1\\2024_Andrea_NBS1_R1\\20240912_Andrea_Yellow\\LayersData\\highmag\\Tile Set (14)\\Tile_006-004-000000_0-000.tif", None),
    ("Z:\\imcrdata\\2024_Andrea_NBS1\\2024_Andrea_NBS1_R2\\20240425_Andrea_lila\\LayersData\\highmag\\Tile Set (22)\\Tile_004-009-000000_0-000.tif", None),
    ("Z:\\imcrdata\\2025_Dani_SMC1\\R1\\20250818_Dani_Bianca\\LayersData\\highmag\\Tile Set (12)\\Tile_015-001-000000_0-000.tif", None),
    ("Z:\\imcrdata\\2025_Dani_SMC1\\R2\\20250819_Dani_Odin\\LayersData\\highmag\\Tile Set (6)\\Tile_011-016-000000_0-000.tif", None),
    ("Z:\\imcrdata\\2025_Dani_SMC1\\R2\\20250929_Dani_Tyr\\LayersData\\highmag\\Tile Set (9)\\Tile_006-002-000000_0-000.tif", None),
    ("Z:\\imcrdata\\2025_Veronica_Lamin\\2025_Veronica_G9ai\\R1\\20250409_Veronica_Apple\\LayersData\\highmag\\Tile Set (14)\\Tile_014-006-000000_0-000.tif", None),
    ("Z:\\imcrdata\\2025_Cyril_AraC\\20250602_Cyril_Lama\\LayersData\\highmag\\Tile Set (3)\\Tile_012-007-000000_0-000.tif", None),

    ("Z:\\imcrdata\\2024_Andrea_NBS1\\2024_Andrea_NBS1_R1\\20240407_Andrea_Red\\LayersData\\highmag\\Tile Set (6)\\Tile_006-012-000000_0-000.tif", (1766, 1121)),
    ("Z:\\imcrdata\\2024_Andrea_NBS1\\2024_Andrea_NBS1_R1\\20240407_Andrea_Red\\LayersData\\highmag\\Tile Set (21)\\Tile_006-016-000000_0-000.tif", (3071, 2996)),
    ("Z:\\imcrdata\\2025_Dani_SMC1\\R1\\20250311_Dani_Funghi\\LayersData\\highmag\\Tile Set (6)\\Tile_014-005-000000_0-000.tif", (1892, 749)),
    ("Z:\\imcrdata\\2025_Dani_SMC1\\R1\\20250311_Dani_Funghi\\LayersData\\highmag\\Tile Set (25)\\Tile_009-003-000000_0-000.tif", (2255, 1767)),
    ("Z:\\imcrdata\\2025_Veronica_Lamin\\2024_Veronica_Lamin\\2024_Veronica_Lamin_R2\\20240813_Veronica_Blue\\LayersData\\highmag\\Tile Set (22)\\Tile_004-015-000000_0-000.tif", (2659, 926)),
    ("Z:\\imcrdata\\2025_Veronica_Lamin\\2024_Veronica_Lamin\\2024_Veronica_Lamin_R2\\20240813_Veronica_Blue\\LayersData\\highmag\\Tile Set (29)\\Tile_014-001-000000_0-000.tif", (2736, 314)),
    ("Z:\\imcrdata\\2024_Andrea_NBS1\\2024_Andrea_NBS1_R2\\20240425_Andrea_lila\\LayersData\\highmag\\Tile Set (9)\\Tile_006-012-000000_0-000.tif", (2435, 593)),
    ("Z:\\imcrdata\\2024_Andrea_NBS1\\2024_Andrea_NBS1_R2\\20240425_Andrea_lila\\LayersData\\highmag\\Tile Set (15)\\Tile_004-009-000000_0-000.tif", (638, 750)),
    ("Z:\\imcrdata\\2025_Dani_SMC1\\R2\\20251027_Dani_Balder\\LayersData\\highmag\\Tile Set (15)\\Tile_001-011-000000_0-000.tif", (2067, 3232)),
    ("Z:\\imcrdata\\2025_Dani_SMC1\\R2\\20250819_Dani_Odin\\LayersData\\highmag\\Tile Set (23)\\Tile_005-005-000000_0-000.tif", (928, 1115)),
    ("Z:\\imcrdata\\2025_Dani_SMC1\\R2\\20250923_Dani_Skaldi\\LayersData\\highmag\\Tile Set (27)\\Tile_014-012-000000_0-000.tif", (1335, 1000)),
    ("Z:\\imcrdata\\2025_Dani_SMC1\\R2\\20250929_Dani_Tyr\\LayersData\\highmag\\Tile Set (7)\\Tile_002-008-000000_0-000.tif", (802, 3644)),
    ("Z:\\imcrdata\\2025_Dani_SMC1\\R2\\20251023_Dani_Thor\\LayersData\\highmag\\Tile Set (3)\\Tile_010-014-000000_0-000.tif", (3188, 2699)),
    ("Z:\\imcrdata\\2025_Dani_SMC1\\R2\\20251023_Dani_Thor\\LayersData\\highmag\\Tile Set (2)\\Tile_007-002-000000_0-000.tif", (3383, 3385)),
    ("Z:\\imcrdata\\2025_Veronica_Lamin\\2024_Veronica_Lamin\\2024_Veronica_Lamin_R1\\20240606_Veronica_Sample1\\LayersData\\highmag\\Tile Set (11)\\Tile_011-005-000000_0-000.tif", (752, 1508)),
    ("Z:\\imcrdata\\2025_Veronica_Lamin\\2025_Veronica_G9ai\\R1\\20250409_Veronica_Apple\\LayersData\\highmag\\Tile Set (12)\\Tile_009-012-000000_0-000.tif", (205, 1547))
]


def save_image_as_png(img: Image.Image, out_dir: Path, out_name: str, resize: tuple = None):
    if resize is not None:
        img = img.resize(resize, Image.Resampling.BILINEAR)
    out_path = out_dir / out_name
    img.save(out_path, format="PNG")


def normalize_convert_uint8(img: Image.Image, soi_coords: Optional[Tuple[int, int]] = None, patch_size: int = 1024) -> Image.Image:
    img_arr = np.array(img).astype(np.float32)

    p_low, p_high = np.percentile(img_arr, (1, 99))
    p_low, p_high = float(p_low), float(p_high)
    img_arr = (img_arr - p_low) * (255.0 / (p_high - p_low))
    img_arr = np.clip(img_arr, 0, 255).astype(np.uint8)

    out_img = Image.fromarray(img_arr)

    if soi_coords is not None:
        left = soi_coords[0] - patch_size // 2
        upper = soi_coords[1] - patch_size // 2
        left = max(0, min(left, out_img.width - patch_size))
        top = max(0, min(upper, out_img.height - patch_size))
        right = left + patch_size
        bottom = top + patch_size

        out_img = out_img.crop((left, top, right, bottom))

    return out_img


def convert_tif_to_png(tif_path: Path) -> Image.Image:
    img = Image.open(tif_path)
    img = normalize_convert_uint8(img)
    return img.resize(FULL_IMAGE_SIZE, Image.Resampling.BILINEAR)


def main():
    from Environment.env_utils import load_forksight_env
    from Segmentation.PreProcessing.General.preprocessing_util import init_dir
    from Segmentation.PreProcessing.General.tile_naming_util import get_new_name

    load_forksight_env()
    raw_data_dir = os.getenv("RAW_DATA_DIR")
    highres_img_dir_name = os.getenv("HIGHRES_IMG_DIR_NAME", "images_4096")

    if raw_data_dir is None:
        raise RuntimeError("RAW_DATA_DIR environment variable not set")

    out_dir_path = Path(raw_data_dir) / highres_img_dir_name
    init_dir(out_dir_path)

    total_images = 0

    for img_path_str, soi_coords in IN_IMAGES:
        img_path = Path(img_path_str)
        try:
            img = Image.open(img_path)
            img = normalize_convert_uint8(
                img=img, soi_coords=soi_coords, patch_size=1024)

            is_soi_img = soi_coords is not None
            new_name = get_new_name(
                img_path, "soi" if is_soi_img else None)
            # ensure all full images are same size (some are 4000x400)
            save_image_as_png(img, out_dir_path, new_name,
                              resize=FULL_IMAGE_SIZE if not is_soi_img else None)

            print(f"{new_name}")
            total_images += 1

        except Exception as e:
            print(f"Failed to convert {img_path}: {e}")

    print(f"Created {total_images} PNG images")


if __name__ == "__main__":
    main()
