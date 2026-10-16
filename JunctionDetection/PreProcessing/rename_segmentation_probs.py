import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--probs-dir", required=True, type=Path)
    parser.add_argument("--ann-pth", required=True, type=Path)
    args = parser.parse_args()

    images = (json.loads(args.ann_pth.read_text(encoding="utf-8")))['images']

    for npy_path in args.probs_dir.rglob("*.npy"):
        if npy_path.stem not in images:
            print(f"image with ID {npy_path.stem} not in annotations")
            continue

        new_filename = images[npy_path.stem]['source_tif'].replace(".tif", ".npy")
        npy_path.rename(npy_path.parent / new_filename)



if __name__ == "__main__":
    main()
