import json
from collections import defaultdict
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset

from dosa.core.sample import Samples
from dosa.models import PreProcessor


def collate_fn(batch):
    """
    Collate samples for batch training
    :param batch: a list of tuple with (input, target) for each sample
    :return: batched inputs and batched targets
    """
    batch = list(zip(*batch))

    samples = Samples.collate(batch[0])

    targets = defaultdict(list)
    for target in batch[1]:
        for key, value in target.items():
            targets[key].append(value)
    targets = {key: torch.stack(values) for key, values in targets.items()}

    return samples, targets


def generate_sample(ann: dict, preprocessor: PreProcessor, img_folder: Path):
    """
    Create a training sample based on one chunked annotation
    :param ann: a chunk of annotations
    :param preprocessor:
    :param img_folder: image folder for docs
    :return: one Samples object
    """
    pages = []
    for page in ann["pages"]:
        image = Image.open(Path(img_folder, page["file_path"])).convert("RGB")
        content = [object_ann["content"] for object_ann in page["objects"]]
        bbox = [object_ann["bbox"] for object_ann in page["objects"]]
        category = [object_ann["category_id"] for object_ann in page["objects"]]
        order = [
            object_ann["id"] / ann["count"] for object_ann in page["objects"]
        ]
        pages.append(
            {
                "image": image,
                "content": content,
                "bbox": torch.Tensor(bbox),
                "category": torch.tensor(category, dtype=torch.int8),
                "order": torch.Tensor(order),
            }
        )
    preprocessed = preprocessor.process(pages)
    return preprocessed


def generate_target(ann: dict, capacity: int):
    """
    :param ann: a chunk of annotations
    :param capacity
    :return: a list of tensor, each tensor is of shape [capacity]
    """
    parent = []
    context = []
    for page in ann["pages"]:
        parent.extend([po["parent_id"] for po in page["objects"]])
        context.extend([po["context"] for po in page["objects"]])

    target = {
        "parent": torch.tensor(parent, dtype=torch.long),
        "context": torch.tensor(context, dtype=torch.long),
    }

    # padding the target
    n_padding = capacity - len(parent)
    target = {
        key: torch.concat([target, torch.zeros(n_padding, dtype=torch.long)])
        for key, target in target.items()
    }
    return target


class DOSADataset(Dataset):

    def __init__(self, img_folder, ann_file, preprocessor, capacity=128):
        """
        Build a dataset for document structure detection training

        :param img_folder: root folder for data, it expects to contain:
            an image folder with sub-folder for each document, each sub-folder
            containing doc images
        :param ann_file: annotations file in a json format, all the necessary
            fields are listed in the example below
            [
                {
                    "pages": [
                        {
                            "file_path": "file_path",
                            "objects": [
                                {
                                    'id': 1,
                                    'parent_id': 1,
                                    'sibling_id': 1,
                                    'category_id': 3,
                                    'bbox': [159, 150, 276, 10],
                                    'content': "content",
                                    'page_id': 0,
                                },
                                ...
                            ]
                        },
                        ...
                    ]
                },
                ...
            ]
            To explain the entity:
            id: unique page object id inside one training sample
            parent_id: parent page object's id
            sibling_id: predecessor page object's id
            category: page object class or category like text, title etc
            bbox: page object's bounding box in [x1, y1, w, h] format
            content: semantic content of this page object
            page_id: page id
        :param preprocessor: dosa preprocessor
        """
        self._img_folder = img_folder
        with open(ann_file) as ifd:
            self._anns = json.load(ifd)
        self._preprocessor = preprocessor
        self._capacity = capacity

    def __getitem__(self, index: int):
        """
        :param index:
        :return: a tuple of (page objects, targets)
            page objects is a dict of Tensor containing
                visual embedding of shape n_sequence x CWH
                semantic embedding of shape n_sequence x d_embedding
                dimension embedding of shape n_sequence x 4
                position embedding of shape n_sequence x 5
                padding mask of shape n_sequence
            targets is a dict of Tensor containing 2 tensors:
                parent Tensor of shape n_sequence
                sibling Tensor of shape n_sequence
        """
        ann = self._anns[index]
        sample = generate_sample(ann, self._preprocessor, self._img_folder)
        target = generate_target(ann, self._capacity)

        return sample, target

    def __len__(self):
        return len(self._anns)
