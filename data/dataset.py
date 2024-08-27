from pathlib import Path
from itertools import groupby, chain

import torch
from PIL import Image
from torch.utils.data import Dataset
import json


def build_context(anns, last):
    """
    Find the context skeleton from previous pages, the context skeleton refers
    to the last branch of the semantic structure tree for previous pages in
    current document.

    :param anns: a list of annotations for each page object in a document
    :param last: index of last page object(inclusive)
    :return: a list of annotations for the last branch
    """
    context_objects = {}
    # in_doc_id does not align with list index, build a lookup map
    idi2idx = {po["in_doc_id"]: idx for idx, po in enumerate(anns)}

    def find_branch(idx):
        # find the whole branch starting from idx
        cur = anns[idx]
        context_objects[cur["in_doc_id"]] = cur
        while cur["parent_id"] != -1:
            cur = anns[idi2idx[cur["parent_id"]]]
            context_objects[cur["in_doc_id"]] = cur

    # add the branch starting from last page object
    find_branch(last)

    # to avoid to confuse the model, find another branch ending with
    # non header object
    if anns[last]["category"] == 8:
        start = last
        while anns[start]["category"] == 8 and start > 0:
            start -= 1

        assert (
            anns[start]["category"] != 8
        ), "There should always be a non header object"

        find_branch(start)

    context_objects = [context_objects[k] for k in sorted(context_objects)]

    return context_objects


def split(anns, capacity):
    """
    Group one single doc annotation into chunks, a chunk cut between pages and
    contains up to n_sequence objects. If a chunk is not the first chunk for a
    document, we pre-append the last branch of the document hierarchy tree into
    this chunk.

    :param anns: annotation for a doc
    :param capacity: capacity of a chunk
    :return: a list of chunks, each chunk is a list of annotations
    """
    page_lengths = [
        len(list(page_anns))
        for _, page_anns in groupby(
            anns["in_doc_annotations"], key=lambda x: x["image_id"]
        )
    ]
    # guarantee, there is space left for context page objects
    assert all(
        x < capacity * 0.8 for x in page_lengths
    ), f"Some page in document {anns['doc_id']} has too many objects to handle"

    chunks = []
    chunk = []
    start = 0
    count = 0
    for pl in page_lengths:
        if count + pl > capacity:
            # wrap up a chunk
            chunks.append(chunk)
            chunk = build_context(anns["in_doc_annotations"], start - 1)
            count = len(chunk)
        chunk.extend(anns["in_doc_annotations"][start : start + pl])
        start += pl
        count += pl
    if chunk:
        chunks.append(chunk)

    return chunks


def generate_targets(chunk, capacity):
    """
    :param chunk: a list of page objects
    :param capacity: capacity of a chunk
    :return: a list of tensor, each tensor is of shape [capacity]
    """
    idi2idx = {po["in_doc_id"]: idx for idx, po in enumerate(chunk)}
    # generate parent child relation
    parent = torch.tensor(
        [
            idx if po["parent_id"] == -1 else idi2idx[po["parent_id"]]
            for idx, po in enumerate(chunk)
        ],
        dtype=torch.long,
    )
    continuation = torch.tensor(
        [idx for idx, _ in enumerate(chunk)], dtype=torch.long
    )
    sibling = torch.arange(len(chunk), dtype=torch.long)
    # parent to child dict
    p2c = {}
    for idx, po in enumerate(chunk):
        pid = po["parent_id"]
        if pid in p2c:
            # sibling relation
            sibling[p2c[pid]] = idx
        p2c[pid] = idx

    targets = {
        "parent": parent,
        "sibling": sibling,
        "continuation": continuation,
    }

    n_padding = capacity - len(chunk)
    targets = {
        key: torch.concat([target, torch.zeros(n_padding, dtype=torch.long)])
        for key, target in targets.items()
    }
    return targets


def prepare(docs, capacity):
    """
    Prepare annotations for DOSADataset, processing is fairly complex and
    explained as below:

    Split each document into chunks based on capacity, to maintain the relation
    between contiguous chunks, the last branch of the semantic tree from
    previous chunk is prepended to next chunk. For each chunk, annotations are
    grouped by page and image info like path and shape are denormalized into
    each page group. Finally, targets are generated accordingly for each chunk.

    :param docs: doc annotations
    :param capacity: chunk capacity
    :return: a list of samples
    """
    anns = docs["annotations"]
    images = docs["images"]
    chunks = list(chain.from_iterable([split(ann, capacity) for ann in anns]))

    samples = []
    targets = []
    for chunk in chunks:
        sample = []
        for image_id, page_anns in groupby(chunk, key=lambda x: x["image_id"]):
            image = images[image_id - 1]
            assert (
                image_id == image["id"]
            ), f"{image_id} not equal to {image['id']}"
            sample.append(
                {
                    "file_path": image["file_path"],
                    "shape": (image["width"], image["height"]),
                    "objects": list(page_anns),
                }
            )
        samples.append(sample)
        targets.append(generate_targets(chunk, capacity))
    return samples, targets


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
            {
                "annotations": [
                    {
                        "doc_id": 1,
                        "in_doc_annotations": [
                            {
                                'gid': 1,
                                'page_id': 0,
                                'parent_id': -1,
                                'in_doc_id': 0,
                                'in_page_id': 0,
                                'bbox': [159, 150, 276, 10],
                                'category': 3,
                                'content': ['EXPLORING THE “RUBIK’S'],
                                'image_id': 1
                            },
                            ...
                        ]
                    },
                    ...
                ],
                "categories": [
                    {
                        'id': 1,
                        'name': 'Caption',
                    },
                    ...
                ],
                "images": [
                    {
                        'id': 1,
                        'file_name': '1401.3699/0.png',
                        'height': 842,
                        'width': 596
                    },
                    ...
                ]
            }
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
            docs = json.load(ifd)
            self._samples, self._targets = prepare(docs, capacity)
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
            targets is a dict of Tensor containing 3 tensors:
                parent Tensor of shape n_sequence
                sibling Tensor of shape n_sequence
                continuation Tensor of shape n_sequence
        """
        sample = self._samples[index]
        target = self._targets[index]

        objects = []
        for page_anns in sample:
            image = Image.open(
                Path(self._img_folder, page_anns["file_path"])
            ).resize(page_anns["shape"])
            contents = [
                object_ann["content"] for object_ann in page_anns["objects"]
            ]
            boxes = [object_ann["bbox"] for object_ann in page_anns["objects"]]
            labels = [
                object_ann["category"] for object_ann in page_anns["objects"]
            ]
            fonts = [object_ann["font"] for object_ann in page_anns["objects"]]
            objects.append(
                {
                    "image": image,
                    "contents": contents,
                    "boxes": torch.Tensor(boxes),
                    "labels": torch.tensor(labels, dtype=torch.int8),
                    "fonts": torch.tensor(fonts, dtype=torch.int32),
                }
            )

        result = self._preprocessor(objects)
        for key, value in result.items():
            result[key] = value.unsqueeze(0)

        return result, target

    def __len__(self):
        return len(self._samples)
