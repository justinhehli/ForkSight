import sys
from PIL import Image

# Define crop region (left, upper, right, lower)
x1, y1, x2, y2 = 120, 150, 400, 350

# Input images
paths = sys.argv[1:4]
if len(paths) != 3:
    print("Usage: python crop.py image1 image2 image3")
    sys.exit(1)

for path in paths:
    img = Image.open(path)
    crop = img.crop((x1, y1, x2, y2))
    out = path.rsplit(".", 1)
    crop.save(f"{out[0]}_crop.{out[1]}")
    print(f"Saved {out[0]}_crop.{out[1]}")
