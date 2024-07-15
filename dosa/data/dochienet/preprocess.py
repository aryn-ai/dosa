import json
import os
from pathlib import Path
from typing import Optional

from dosa.data.dochienet.chunk import build_chunks
from dosa.data.dochienet.merge import merge_labels
from dosa.data.dochienet.resize import resize_images
from dosa.data.dochienet.utils import get_train_test_prefix, get_zh_en_prefix


def get_prefix(in_path: Path):
    train, test = get_train_test_prefix(in_path / "train_test_split.json")
    en, zh = get_zh_en_prefix(in_path / "en_zh_split.json")
    return [p for p in train if p in en], [p for p in test if p in en]


def process(
    in_path: Path,
    out_path: Path,
    resolution: tuple[int, int] = (800, 800),
    limit: int = 256,
    window: int = 8,
    skips: Optional[list[str]] = None,
):
    train_prefix, test_prefix = get_train_test_prefix(
        in_path / "train_test_split.json"
    )
    en, zh = get_zh_en_prefix(in_path / "en_zh_split.json")

    # resize images and annotations
    resize_images(train_prefix, in_path, out_path, resolution)
    resize_images(test_prefix, in_path, out_path, resolution)

    # Group objects per page, generate DOSA format annotation
    train = merge_labels(train_prefix, in_path, resolution, en)
    test = merge_labels(test_prefix, in_path, resolution, en)

    # write to benchmark without splitting
    os.makedirs(out_path / "annotations", exist_ok=True)
    with open(f"{out_path}/annotations/benchmark.json", "w") as ofd:
        json.dump(test, ofd, indent=2)

    # Split into chunks
    build_chunks(
        train, f"{out_path}/annotations/train.json", limit, window, skips
    )
    build_chunks(test, f"{out_path}/annotations/val.json", limit, window, skips)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Preprocess DocHieNet data for DOSA."
    )
    parser.add_argument(
        "--in_path", type=str, required=True, help="Input directory path"
    )
    parser.add_argument(
        "--out_path", type=str, required=True, help="Output directory path"
    )
    parser.add_argument(
        "--resolution",
        type=int,
        nargs=2,
        default=[800, 800],
        metavar=("W", "H"),
        help="Image resolution, e.g., 800 800",
    )
    parser.add_argument(
        "--limit", type=int, default=256, help="Chunk limit (default: 256)"
    )
    parser.add_argument(
        "--window", type=int, default=8, help="Chunk window size (default: 8)"
    )
    parser.add_argument(
        "--skips",
        type=str,
        nargs="*",
        default=None,
        help="List of prefixes to skip (space separated)",
    )

    args = parser.parse_args()

    in_path = Path(args.in_path)
    out_path = Path(args.out_path)
    resolution = tuple(args.resolution)
    limit = args.limit
    window = args.window
    skips = args.skips if args.skips else None

    process(
        in_path=in_path,
        out_path=out_path,
        resolution=resolution,
        limit=limit,
        window=window,
        skips=skips,
    )
