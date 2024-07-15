import json

import torch

from data.dataset import (
    build_context,
    split,
    generate_targets,
    DOSADataset,
    prepare,
)
from models import (
    PreProcessor,
    FPNFeatureMapExtractor,
    SentenceTransformerEmbeder,
)
from test.config import TEST_DIR


def load_objects():
    # The doc already excludes footnote, header and footer
    with open(f"{TEST_DIR}/resource/hrdoc/annotations/train.json", "r") as ifd:
        doc = json.load(ifd)
    return doc


def test_build_context():
    doc = load_objects()
    in_doc_annotations = doc["annotations"][0]["in_doc_annotations"]
    context = build_context(in_doc_annotations, 67)
    assert len(context) == 3

    # Find one
    in_doc_annotations[66]["category"] = 8
    in_doc_annotations[67]["category"] = 8
    context = build_context(in_doc_annotations, 67)
    assert len(context) == 4


def test_split():
    doc = load_objects()
    chunks = split(doc["annotations"][0], 128)
    assert len(chunks) == 3
    assert chunks[1][0]["category"] == 8
    assert chunks[1][1]["parent_id"] == chunks[1][0]["in_doc_id"]
    assert chunks[2][0]["category"] == 8
    assert chunks[2][1]["parent_id"] == chunks[2][0]["in_doc_id"]


def test_generate_targets():
    doc = load_objects()["annotations"][0]
    doc["in_doc_annotations"] = doc["in_doc_annotations"][:48]
    chunks = split(doc, 20)
    targets = [generate_targets(chunk, 20) for chunk in chunks]

    assert torch.equal(
        targets[0]["parent"],
        torch.tensor(
            [0, 1, 2, 3, 3, 3, 3, 3, 8, 8, 10, 10, 3, 3, 14, 14, 14, 14, 0, 0],
            dtype=torch.long,
        ),
    )
    assert torch.equal(
        targets[0]["sibling"],
        torch.tensor(
            [
                1,
                2,
                3,
                8,
                5,
                6,
                7,
                12,
                10,
                9,
                14,
                11,
                13,
                13,
                14,
                16,
                17,
                17,
                0,
                0,
            ],
            dtype=torch.long,
        ),
    )
    assert torch.equal(
        targets[0]["continuation"],
        torch.tensor(
            [
                0,
                1,
                2,
                3,
                4,
                5,
                6,
                7,
                8,
                9,
                10,
                11,
                12,
                13,
                14,
                15,
                16,
                17,
                0,
                0,
            ],
            dtype=torch.long,
        ),
    )

    assert torch.equal(
        targets[1]["parent"],
        torch.tensor(
            [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            dtype=torch.long,
        ),
    )
    assert torch.equal(
        targets[1]["sibling"],
        torch.tensor(
            [0, 3, 2, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 14, 0, 0, 0, 0, 0],
            dtype=torch.long,
        ),
    )
    assert torch.equal(
        targets[1]["continuation"],
        torch.tensor(
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 0, 0, 0, 0, 0],
            dtype=torch.long,
        ),
    )

    assert torch.equal(
        targets[2]["parent"],
        torch.tensor(
            [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 14, 14, 14, 14, 14, 0],
            dtype=torch.long,
        ),
    )
    assert torch.equal(
        targets[2]["sibling"],
        torch.tensor(
            [
                14,
                3,
                2,
                4,
                5,
                6,
                7,
                8,
                9,
                10,
                11,
                12,
                13,
                13,
                14,
                16,
                17,
                18,
                18,
                0,
            ],
            dtype=torch.long,
        ),
    )
    assert torch.equal(
        targets[2]["continuation"],
        torch.tensor(
            [
                0,
                1,
                2,
                3,
                4,
                5,
                6,
                7,
                8,
                9,
                10,
                11,
                12,
                13,
                14,
                15,
                16,
                17,
                18,
                0,
            ],
            dtype=torch.long,
        ),
    )


def test_prepare():
    doc = load_objects()
    samples, targets = prepare(doc, 64)
    assert len(samples) == 5 and len(targets) == 5


def test_dataset():
    preprocessor = PreProcessor(
        128, FPNFeatureMapExtractor(), SentenceTransformerEmbeder()
    )
    dataset = DOSADataset(
        f"{TEST_DIR}/resource/hrdoc/images",
        f"{TEST_DIR}/resource/hrdoc/annotations/train.json",
        preprocessor,
    )

    assert dataset.__len__() == 3
    for _ in range(3):
        result, target = dataset.__getitem__(0)
        assert len(result.keys()) == 5 and len(target.keys()) == 3
