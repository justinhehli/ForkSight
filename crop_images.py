import sys
from PIL import Image
from pathlib import Path

# Define crop region (left, upper, right, lower) and image paths
x1, y1, x2, y2 = 3760, 3250, 4080, 3500
paths = [
    "C:\\Users\\juhe9\\repos\\MasterThesis\\ForkSight\\Data\\evaluation_output\\junction_detection\\20260428_114907\\plots\\sweep-cldice-BEST\\Dani_Funghi_TileSet_9_Tile_002-009.png",
    "C:\\Users\\juhe9\\repos\\MasterThesis\\ForkSight\\Data\\evaluation_output\\junction_detection\\20260428_114907\\plots\\nnunet_Dataset001_Segmentation_v1_nnUNetTrainerClDiceLoss\\Dani_Funghi_TileSet_9_Tile_002-009.png"
]

for idx, path in enumerate(paths):
    img = Image.open(path)
    crop = img.crop((x1, y1, x2, y2))
    path_obj = Path(path)
    crop.save(f"{path_obj.parent.name}_{path_obj.stem}_crop_{idx}{path_obj.suffix}")
    print(f"Saved {path_obj.stem}_crop_{idx}.{path_obj.suffix}")
