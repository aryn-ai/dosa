import json
import os
from pathlib import Path
from tqdm import tqdm

from PIL import Image


def resize_images(
    prefixes: list[str], in_path: Path, out_path: Path, target: tuple[int, int]
):
    """
    Resize images in original DocHieNet dataset to target shape
    """
    for prefix in tqdm(prefixes, "Resizing images"):
        path = in_path / f"labels/{prefix}.json"
        with open(path) as ifd:
            pages = json.load(ifd)["pages"]

            os.makedirs(out_path / f"images/{prefix}/", exist_ok=True)
            for k in pages.keys():
                image = Image.open(in_path / f"hres_images/{prefix}/{k}.jpg")
                resized_image = image.resize(target)
                ow = pages[k]["width"]
                oh = pages[k]["height"]
                w, h = image.size
                if w != ow or h != oh:
                    # declared size in labels should be same as actual size
                    # but there are hundreds of images with size error of 1
                    print(
                        f"{prefix} and {k} not equal, declared {ow, oh} but actual {w, h}"
                    )
                resized_image.save(out_path / f"images/{prefix}/{k}.png")
