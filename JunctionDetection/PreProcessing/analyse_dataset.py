import argparse
from pathlib import Path
import pandas as pd


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--csv", required=True, type=Path,
                        help="Path to the dataset CSV file with point annotations")
    args = parser.parse_args()

    df = pd.read_csv(args.csv)

    num_negative = (df['label'] == 'Negative').sum()
    num_replication = (df['label'] == 'Normal Fork').sum()
    num_reversed = (df['label'] == 'Reversed Fork').sum()
    num_crossing = (df['label'] == 'Crossing').sum()
    num_reversed_or_crossing = num_reversed + num_crossing

    num_img_replication = df[df['label'] == 'Normal Fork']['image'].nunique()
    num_img_reversed = df[df['label'] == 'Reversed Fork']['image'].nunique()
    num_img_crossing = df[df['label'] == 'Crossing']['image'].nunique()
    num_img_reverved_or_crossing = df[df['label'].isin(
        ['Reversed Fork', 'Crossing'])]['image'].nunique()

    num_img_total = df['image'].nunique()
    percentage_negative = num_negative / num_img_total * 100

    print(f"total number of annotations incl. Negative: {len(df)}")
    print(
        f"total number of annotations excl. Negative: {(df['label'] != 'Negative').sum()}")
    print(f"total number of images: {num_img_total}")

    print(f"\nnegative samples: {num_negative}")
    print(f"replication forks: {num_replication}")
    print(f"reversed forks: {num_reversed}")
    print(f"crossings: {num_crossing}")
    print(
        f"combined number of reversed forks and crossings: {num_reversed_or_crossing}")

    print(f"\nimages with replication forks: {num_img_replication}")
    print(f"images with reversed forks: {num_img_reversed}")
    print(f"images with crossings: {num_img_crossing}")
    print(
        f"images with reversed forks OR crossings: {num_img_reverved_or_crossing}")

    print(f"\npercentage of negative sample images: {round(percentage_negative)}%")


if __name__ == "__main__":
    main()
