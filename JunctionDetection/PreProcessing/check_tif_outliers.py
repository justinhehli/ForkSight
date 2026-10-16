import argparse

import tifffile
import numpy as np
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="Scan a directory of TIFs and report images whose min/max are pulled far "
        "outside their 1st/99th percentile range (i.e. outlier pixels present, which "
        "would distort a plain mean/std z-score normalization).",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dir", required=True, type=Path)
    parser.add_argument("--factor", type=float, default=3.0,
                        help="Flag an image if (max - p99) or (p1 - min) exceeds factor * (p99 - p1). "
                        "Higher = stricter (fewer flags). Default: 3.0")
    args = parser.parse_args()

    n_checked = 0
    n_flagged = 0

    for p in sorted(args.dir.rglob("*.tif")):
        n_checked += 1
        img = tifffile.imread(p).astype(np.float32)
        img_min, img_max = img.min(), img.max()
        mean, std = img.mean(), img.std()
        p_low, p_high = np.percentile(img, [1, 99])
        core_range = p_high - p_low

        if core_range <= 0:
            # 1st-99th percentile range is degenerate (near-constant image) but min/max differ
            # -> whatever variation exists is entirely outlier pixels
            is_outlier = img_max > img_min
            upper_ratio = lower_ratio = float("inf") if is_outlier else 0.0
        else:
            upper_ratio = (img_max - p_high) / core_range
            lower_ratio = (p_low - img_min) / core_range
            is_outlier = upper_ratio > args.factor or lower_ratio > args.factor

        if is_outlier:
            n_flagged += 1
            print(f"{p}")
            print(
                f"  min={img_min:.1f} max={img_max:.1f} mean={mean:.1f} std={std:.1f}")
            print(
                f"  p1={p_low:.1f} p99={p_high:.1f} (core range={core_range:.1f})")
            print(f"  upper_ratio={upper_ratio:.2f} lower_ratio={lower_ratio:.2f} "
                  f"(threshold={args.factor})")

    print(f"\nChecked {n_checked} TIFs, flagged {n_flagged} with outliers "
          f"(factor={args.factor}).")


if __name__ == "__main__":
    main()
