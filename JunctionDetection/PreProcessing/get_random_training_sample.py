import random
import shutil
from pathlib import Path

from Segmentation.PreProcessing.General.tile_naming_util import get_new_name

source_dirs = [
    "\\\\imcr-fs1.d.uzh.ch\\Lopesgroup\\2024_Andrea_ETP_R2\\20240911_Andrea_Black\\LayersData\\highmag",
    "\\\\imcr-fs1.d.uzh.ch\\Lopesgroup\\2024_Andrea_ETP_R2\\20250808_Andrea_Magenta\\LayersData\\highmag",

    "\\\\imcr-fs1.d.uzh.ch\\Lopesgroup\\2024_Andrea_NBS1\\2024_Andrea_NBS1_R1\\20240407_Andrea_Red\\LayersData\\highmag",
    "\\\\imcr-fs1.d.uzh.ch\\Lopesgroup\\2024_Andrea_NBS1\\2024_Andrea_NBS1_R1\\20240523_Andrea_Orange\\LayersData\\highmag",
    "\\\\imcr-fs1.d.uzh.ch\\Lopesgroup\\2024_Andrea_NBS1\\2024_Andrea_NBS1_R1\\20240912_Andrea_Yellow\\LayersData\\highmag",
    "\\\\imcr-fs1.d.uzh.ch\\Lopesgroup\\2024_Andrea_NBS1\\2024_Andrea_NBS1_R2\\20240306_Andrea_Blue\\LayersData\\highmag",
    "\\\\imcr-fs1.d.uzh.ch\\Lopesgroup\\2024_Andrea_NBS1\\2024_Andrea_NBS1_R2\\20240425_Andrea_lila\\LayersData\\highmag",
    "\\\\imcr-fs1.d.uzh.ch\\Lopesgroup\\2024_Andrea_NBS1\\2024_Andrea_NBS1_R2\\20240613_Andrea_Green\\LayersData\\highmag",

    "\\\\imcr-fs1.d.uzh.ch\\Lopesgroup\\2025_Cyril_AraC\\20250528_Cyril_Alpaka\\LayersData\\highmag",
    "\\\\imcr-fs1.d.uzh.ch\\Lopesgroup\\2025_Cyril_AraC\\20250602_Cyril_Lama\\LayersData\\highmag",

    "\\\\imcr-fs1.d.uzh.ch\\Lopesgroup\\2025_Dani_SMC1\\R1\\20250818_Dani_Bianca\\LayersData\\highmag",
    "\\\\imcr-fs1.d.uzh.ch\\Lopesgroup\\2025_Dani_SMC1\\R1\\20250305_Dani_Margherita\\LayersData\\highmag",
    "\\\\imcr-fs1.d.uzh.ch\\Lopesgroup\\2025_Dani_SMC1\\R1\\20250307_Dani_Prosciutto\\LayersData\\highmag",
    "\\\\imcr-fs1.d.uzh.ch\\Lopesgroup\\2025_Dani_SMC1\\R1\\20250311_Dani_Funghi\\LayersData\\highmag",
    "\\\\imcr-fs1.d.uzh.ch\\Lopesgroup\\2025_Dani_SMC1\\R1\\20250317_Dani_Diavolo\\LayersData\\highmag",
    "\\\\imcr-fs1.d.uzh.ch\\Lopesgroup\\2025_Dani_SMC1\\R1\\20250811_Dani_Vegetariana\\LayersData\\highmag",
    "\\\\imcr-fs1.d.uzh.ch\\Lopesgroup\\2025_Dani_SMC1\\R2\\20251027_Dani_Balder\\LayersData\\highmag",
    "\\\\imcr-fs1.d.uzh.ch\\Lopesgroup\\2025_Dani_SMC1\\R2\\20250819_Dani_Odin\\LayersData\\highmag",
    "\\\\imcr-fs1.d.uzh.ch\\Lopesgroup\\2025_Dani_SMC1\\R2\\20250923_Dani_Skaldi\\LayersData\\highmag",
    "\\\\imcr-fs1.d.uzh.ch\\Lopesgroup\\2025_Dani_SMC1\\R2\\20250925_Dani_Freyja\\LayersData\\highmag",
    "\\\\imcr-fs1.d.uzh.ch\\Lopesgroup\\2025_Dani_SMC1\\R2\\20250929_Dani_Tyr\\LayersData\\highmag",
    "\\\\imcr-fs1.d.uzh.ch\\Lopesgroup\\2025_Dani_SMC1\\R2\\20251006_Dani_Loki\\LayersData\\highmag",
    "\\\\imcr-fs1.d.uzh.ch\\Lopesgroup\\2025_Dani_SMC1\\R2\\20251023_Dani_Thor\\LayersData\\highmag",

    "\\\\imcr-fs1.d.uzh.ch\\Lopesgroup\\2025_Filip_RNF20\\2025_Filip_RNF20_R1\\20250210_Filip_Appenzell\\LayersData\\highmag",
    "\\\\imcr-fs1.d.uzh.ch\\Lopesgroup\\2025_Filip_RNF20\\2025_Filip_RNF20_R1\\20250207_Filip_StGallen\\LayersData\\highmag",
    "\\\\imcr-fs1.d.uzh.ch\\Lopesgroup\\2025_Filip_RNF20\\2025_Filip_RNF20_R2\\20250108_Filip_Basel\\LayersData\\highmag",
    "\\\\imcr-fs1.d.uzh.ch\\Lopesgroup\\2025_Filip_RNF20\\2025_Filip_RNF20_R2\\20250107_Filip_Zurich\\LayersData\\highmag",
    "\\\\imcr-fs1.d.uzh.ch\\Lopesgroup\\2025_Filip_RNF20\\2025_Filip_RNF20_R3\\20250728_Filip_Bern\\LayersData\\highmag",
    "\\\\imcr-fs1.d.uzh.ch\\Lopesgroup\\2025_Filip_RNF20\\2025_Filip_RNF20_R3\\20250729_Filip_Luzern\\LayersData\\highmag",

    "\\\\imcr-fs1.d.uzh.ch\\Lopesgroup\\2025_Jana_MRE11_Inh\\20250915_Jana_Wasp\\LayersData\\highmag",
    "\\\\imcr-fs1.d.uzh.ch\\Lopesgroup\\2025_Jana_MRE11_Inh\\20250812_Jana_Ant\\LayersData\\highmag",

    "\\\\imcr-fs1.d.uzh.ch\\Lopesgroup\\2025_Patri\\20250324_Patri_Alpha\\LayersData\\highmag",
    "\\\\imcr-fs1.d.uzh.ch\\Lopesgroup\\2025_Patri\\20251204_Patri_Forkolino\\LayersData\\highmag",
    "\\\\imcr-fs1.d.uzh.ch\\Lopesgroup\\2025_Patri\\20260112_Patri_ForkaMiseria\\LayersData\\highmag",

    "\\\\imcr-fs1.d.uzh.ch\\Lopesgroup\\2025_Veronica_Lamin\\2024_Veronica_Lamin\\2024_Veronica_Lamin_R1\\20240821_Veronica_Sample5\\LayersData\\highmag",
    "\\\\imcr-fs1.d.uzh.ch\\Lopesgroup\\2025_Veronica_Lamin\\2024_Veronica_Lamin\\2024_Veronica_Lamin_R1\\20230828_Vero_sample 2\\LayersData\\highmag",
    "\\\\imcr-fs1.d.uzh.ch\\Lopesgroup\\2025_Veronica_Lamin\\2024_Veronica_Lamin\\2024_Veronica_Lamin_R1\\20230919_sample 3-\\LayersData\\highmag",
    "\\\\imcr-fs1.d.uzh.ch\\Lopesgroup\\2025_Veronica_Lamin\\2024_Veronica_Lamin\\2024_Veronica_Lamin_R1\\20230920_Vero sample 4\\LayersData\\highmag",
    "\\\\imcr-fs1.d.uzh.ch\\Lopesgroup\\2025_Veronica_Lamin\\2024_Veronica_Lamin\\2024_Veronica_Lamin_R1\\20240529_Veronica_Sample5\\LayersData\\highmag",
    "\\\\imcr-fs1.d.uzh.ch\\Lopesgroup\\2025_Veronica_Lamin\\2024_Veronica_Lamin\\2024_Veronica_Lamin_R1\\20240606_Veronica_Sample1\\LayersData\\highmag",
    "\\\\imcr-fs1.d.uzh.ch\\Lopesgroup\\2025_Veronica_Lamin\\2024_Veronica_Lamin\\2024_Veronica_Lamin_R1\\20240614_Veronica_Sample6-\\LayersData\\highmag",
    "\\\\imcr-fs1.d.uzh.ch\\Lopesgroup\\2025_Veronica_Lamin\\2024_Veronica_Lamin\\2024_Veronica_Lamin_R1\\20240704_Veronica_Sample6\\LayersData\\highmag",
    "\\\\imcr-fs1.d.uzh.ch\\Lopesgroup\\2025_Veronica_Lamin\\2024_Veronica_Lamin\\2024_Veronica_Lamin_R2\\20240807_Veronica_Red\\LayersData\\highmag",
    "\\\\imcr-fs1.d.uzh.ch\\Lopesgroup\\2025_Veronica_Lamin\\2024_Veronica_Lamin\\2024_Veronica_Lamin_R2\\20240808_Veronica_Yellow\\LayersData\\highmag",
    "\\\\imcr-fs1.d.uzh.ch\\Lopesgroup\\2025_Veronica_Lamin\\2024_Veronica_Lamin\\2024_Veronica_Lamin_R2\\20240813_Veronica_Blue\\LayersData\\highmag",
    "\\\\imcr-fs1.d.uzh.ch\\Lopesgroup\\2025_Veronica_Lamin\\2024_Veronica_Lamin\\2024_Veronica_Lamin_R2\\20240820_Veronica_Green\\LayersData\\highmag",
    "\\\\imcr-fs1.d.uzh.ch\\Lopesgroup\\2025_Veronica_Lamin\\2025_Veronica_G9ai\\R1\\20250709_Veronica_Cedar\\LayersData\\highmag",
    "\\\\imcr-fs1.d.uzh.ch\\Lopesgroup\\2025_Veronica_Lamin\\2025_Veronica_G9ai\\R1\\20250401_Veronica_Plum\\LayersData\\highmag",
    "\\\\imcr-fs1.d.uzh.ch\\Lopesgroup\\2025_Veronica_Lamin\\2025_Veronica_G9ai\\R1\\20250409_Veronica_Apple\\LayersData\\highmag",
    "\\\\imcr-fs1.d.uzh.ch\\Lopesgroup\\2025_Veronica_Lamin\\2025_Veronica_G9ai\\R1\\20250416_Veronica_Peach\\LayersData\\highmag",
    "\\\\imcr-fs1.d.uzh.ch\\Lopesgroup\\2025_Veronica_Lamin\\2025_Veronica_G9ai\\R1\\20250612_Vero_Orange\\LayersData\\highmag",
    "\\\\imcr-fs1.d.uzh.ch\\Lopesgroup\\2025_Veronica_Lamin\\2025_Veronica_G9ai\\R1\\20250619_Veronica_Cherry\\LayersData\\highmag",
    "\\\\imcr-fs1.d.uzh.ch\\Lopesgroup\\2025_Veronica_Lamin\\2025_Veronica_G9ai\\R2\\20250708_Vero_Larch_Part2\\LayersData\\highmag",
    "\\\\imcr-fs1.d.uzh.ch\\Lopesgroup\\2025_Veronica_Lamin\\2025_Veronica_G9ai\\R2\\20250604_Vero_Pine\\LayersData\\highmag",
    "\\\\imcr-fs1.d.uzh.ch\\Lopesgroup\\2025_Veronica_Lamin\\2025_Veronica_G9ai\\R2\\20250605_Vero_Larch\\LayersData\\highmag",
    "\\\\imcr-fs1.d.uzh.ch\\Lopesgroup\\2025_Veronica_Lamin\\2025_Veronica_G9ai\\R2\\20250611_Vero_Willow\\LayersData\\highmag",
    "\\\\imcr-fs1.d.uzh.ch\\Lopesgroup\\2025_Veronica_Lamin\\2025_Veronica_G9ai\\R2\\20250701_Veronica_Oak\\LayersData\\highmag",
    "\\\\imcr-fs1.d.uzh.ch\\Lopesgroup\\2025_Veronica_Lamin\\2025_Veronica_G9ai\\R2\\20250702_Veronica_Maple\\LayersData\\highmag",
]

target_dir = "C:\\Users\\juhe9\\repos\\MasterThesis\\ForkSight\\Data\\RawData\\junction_detection_training_new"

N_SAMPLES = 400
RANDOM_SEED = 42


def main():
    random.seed(RANDOM_SEED)

    tif_paths = []
    for source_dir in source_dirs:
        source_path = Path(source_dir)
        if not source_path.exists():
            print(f"Skipping missing source dir: {source_dir}")
            continue
        tif_paths.extend(source_path.rglob("*.tif"))

    print(
        f"Found {len(tif_paths)} TIF images across {len(source_dirs)} source dirs")

    if len(tif_paths) < N_SAMPLES:
        raise ValueError(
            f"Only found {len(tif_paths)} TIF images, cannot sample {N_SAMPLES}")

    target_dir_path = Path(target_dir)
    target_dir_path.mkdir(parents=True, exist_ok=True)

    def get_new_tif_name(tif_path: Path) -> str:
        # get_new_name always appends ".png" - keep the original TIF extension
        return Path(get_new_name(tif_path)).with_suffix(".tif").name

    existing_sample_names = [p.name for p in target_dir_path.rglob("*.tif")]
    tif_paths = list(filter(lambda p: get_new_tif_name(
        p) not in existing_sample_names, tif_paths))

    sampled_paths = random.sample(tif_paths, N_SAMPLES)

    for tif_path in sampled_paths:
        new_name = get_new_tif_name(tif_path)
        shutil.copy2(tif_path, target_dir_path / new_name)
        print(f"Copied {tif_path} -> {new_name}")

    print(f"Copied {len(sampled_paths)} images to {target_dir_path}")


if __name__ == "__main__":
    main()
