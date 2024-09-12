from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
import json


class DOSADataset(Dataset):
    # TODO: Some optional data process steps
    #  1. for each document, we could chunk arbitrary continuous pages to
    #   augment training dataset
    #  2. shuffle page objects in same page, the model does not depend on order,
    #   but might worth a try optional
    #  3. miss classification certain page 5% to 10% page objects
    #  4. remove the leading chapter/section numbers in headers, change the
    #   format or font of certain headers
    #  5. filter out all the page footer, header and footnote

    def __init__(self, img_folder, ann_file, preprocessor, capacity=128):
        """
        Build a dataset for document structure detection training

        :param img_folder: root folder for data, it expects to contain:
            an image folder with sub-folder for each document, each sub-folder
            containing doc images with page number as name starting from 0
        :param ann_file: annotations file in a json format, the annotation
            should exclude all page metadata like header, footer and footnote.
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
                                    'continuation_id': 1
                                    'category': 3,
                                    'bbox': [159, 150, 276, 10],
                                    'content': ['EXPLORING THE “RUBIK’S'],
                                    'page_id': 0,
                                    'font': 50.1
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
            gid: global id across the whole dataset for a page object
            in_doc_id: object id in one doc, follows reading order
            in_page_id: object id in one page, follows reading order
            page_id: the page number starting from 0
            parent_id: parent object's in_doc_id
        :param preprocessor: dosa preprocessor
        """
        self._img_folder = img_folder
        with open(ann_file) as ifd:
            self._anns = json.load(ifd)
        self._preprocessor = preprocessor
        self._capacity = capacity

    def _generate_sample(self, ann):
        sample = []
        for page in ann["pages"]:
            image = Image.open(Path(self._img_folder, page["file_path"]))
            contents = [object_ann["content"] for object_ann in page["objects"]]
            boxes = [object_ann["bbox"] for object_ann in page["objects"]]
            labels = [object_ann["category"] for object_ann in page["objects"]]
            fonts = [object_ann["font"] for object_ann in page["objects"]]
            sample.append(
                {
                    "image": image,
                    "contents": contents,
                    "boxes": torch.Tensor(boxes),
                    "labels": torch.tensor(labels, dtype=torch.int8),
                    "fonts": torch.tensor(fonts, dtype=torch.int32),
                }
            )
        sample = self._preprocessor(sample)
        for key, value in sample.items():
            sample[key] = value.unsqueeze(0)
        return sample

    def _generate_target(self, ann):
        """
        :param ann: a chunk of annotations
        :return: a list of tensor, each tensor is of shape [capacity]
        """
        parent = []
        sibling = []
        continuation = []
        for page in ann["pages"]:
            parent.extend([po["parent_id"] for po in page["objects"]])
            sibling.extend([po["sibling_id"] for po in page["objects"]])
            continuation.extend([po["continuation_id"] for po in page["objects"]])

        target = {
            "parent": torch.tensor(parent, dtype=torch.long),
            "sibling": torch.tensor(sibling, dtype=torch.long),
            "continuation": torch.tensor(continuation, dtype=torch.long),
        }

        n_padding = self._capacity - len(parent)
        target = {
            key: torch.concat(
                [target, torch.zeros(n_padding, dtype=torch.long)]
            )
            for key, target in target.items()
        }
        return target

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
            targets is a dict of Tensor containing 3 tensors:
                parent Tensor of shape n_sequence
                sibling Tensor of shape n_sequence
                continuation Tensor of shape n_sequence
        """
        ann = self._anns[index]
        sample = self._generate_sample(ann)
        target = self._generate_target(ann)

        return sample, target

    def __len__(self):
        return len(self._anns)
