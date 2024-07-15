import json
from itertools import groupby
from pathlib import Path
from tqdm import tqdm

from dosa.data.dochienet.utils import class2number, number2class


def build_prompt(label: int, text: str):
    try:
        if label == 2:
            return "A figure"
        elif label == 15:
            return "Table of Content"
        elif text:
            return text
        else:
            return number2class[label]
    except KeyError:
        raise Exception(f"Unknown label {label}")


def merge_labels(
    prefixes: list[str],
    in_path: Path,
    resolution: tuple[int, int],
    en: list[str],
):
    # Take DocHieNet label input and generate DOSA compatible label output.
    # Each DocHieNet object includes fields: box, text, page, label, linking, id, order with ids > 0, order id >= 0,
    # and linking(parent->self) 0 or -1
    # The output would still reuse DocHieNet id, but with parent id set 0 if it's top level object, sibling id set to
    # 0 if it's first object
    result = []
    for prefix in tqdm(prefixes, "Converting DocHieNet labels to Dosa"):
        path = in_path / f"labels/{prefix}.json"
        with open(path) as ifd:
            labels = json.load(ifd)
            contents = labels["contents"]

            # attention, here order id is not continuous in original annotations,
            # sort obj by order, then find each object's predecessor based on order,
            # also, page headers and footers have order of 0, we need to handle them
            contents.sort(key=lambda obj: (obj["page"], obj["order"]))
            siblings = {}
            for i, obj in enumerate(contents):
                # set the first object's sibling to 0, set objects to 0 if order is 0
                siblings[obj["id"]] = (
                    0 if i == 0 or obj["order"] == 0 else contents[i - 1]["id"]
                )

            page_objects = []
            for obj in contents:
                bbox = obj["box"]

                # convert bbox to target resolution
                page_id = obj["page"]
                ow = labels["pages"][f"page{page_id}"]["width"]
                oh = labels["pages"][f"page{page_id}"]["height"]
                tw, th = resolution
                bbox = [
                    bbox[0] * tw / ow,
                    bbox[1] * th / oh,
                    bbox[2] * tw / ow,
                    bbox[3] * th / oh,
                ]

                # from x1, y1, x2, y2 to x1, y1, w, h
                new_bbox = [
                    bbox[0],
                    bbox[1],
                    bbox[2] - bbox[0],
                    bbox[3] - bbox[1],
                ]

                # construct new page object by converting parent id, sibling id, category text to id and
                # generating prompt
                page_object = {
                    "id": obj["id"],
                    # some object has parent -1 and some has 0, we unify them to 0
                    "parent_id": (
                        obj["linking"][0][0]
                        if obj["linking"][0][0] != -1
                        else 0
                    ),
                    "sibling_id": siblings[obj["id"]],
                    "category_id": class2number[obj["label"]],
                    "bbox": new_bbox,
                    "content": build_prompt(
                        class2number[obj["label"]], obj["text"]
                    ),
                    "page_id": obj["page"],
                    "category": obj["label"],
                    "order": obj["order"],
                }
                page_objects.append(page_object)

            # group objects into pages
            page_objects.sort(key=lambda obj: (obj["page_id"], obj["order"]))
            pages = []
            for page_id, page_anns in groupby(
                page_objects, key=lambda x: x["page_id"]
            ):
                pages.append(
                    {
                        "file_path": f"{prefix}/page{page_id}.png",
                        "objects": list(page_anns),
                    }
                )

        result.append(
            {
                "id": prefix,
                "pages": pages,
                "language": "en" if prefix in en else "zh",
                "count": len(contents),
            }
        )
    return result
