import pytest
import torch

from test.config import TEST_DIR
from test.loader import ObjectDetectionLoader
from models.preprocess import (
    FPNFeatureMapExtractor,
    SentenceTransformerEmbeder,
    PreProcessor,
)


@pytest.mark.parametrize("batch_size", [1, 2, 5, 6])
def test_fpn_feature_extractor(batch_size):
    fpn = FPNFeatureMapExtractor(batch_size=batch_size)
    images = []
    boxes = []
    for _ in range(0, 5):
        image = torch.rand(3, 1024, 1024)
        images.append(image)

        import random

        page_boxes = torch.rand(random.randint(1, 100), 4) * 256
        page_boxes[:, 2:] += page_boxes[:, :2]
        boxes.append(page_boxes)

    shapes = [(1024, 1024)] * 5
    features = fpn(images, shapes, boxes)

    assert len(features) == 5
    for idx, box in enumerate(boxes):
        assert features[idx].size() == (len(box), 12544)


@pytest.mark.parametrize("batch_size", [1, 2, 5])
def test_sentence_transform_embedder(batch_size):
    doc_objects = ObjectDetectionLoader.read_objects(
        f"{TEST_DIR}/resource/hai/HAI1.pdf",
        f"{TEST_DIR}/resource/hai/HAI1_predictions.json",
    )
    contents = [page_objects["contents"] for page_objects in doc_objects]
    embedder = SentenceTransformerEmbeder(batch_size=batch_size)
    embeddings = embedder(contents)
    for i, d in enumerate(doc_objects):
        assert len(d["labels"]) == len(embeddings[i])


def test_preprocessor():
    doc_objects = ObjectDetectionLoader.read_objects(
        f"{TEST_DIR}/resource/hai/HAI1.pdf",
        f"{TEST_DIR}/resource/hai/HAI1_predictions.json",
    )
    preprocessor = PreProcessor(
        128, FPNFeatureMapExtractor(), SentenceTransformerEmbeder()
    )
    result = preprocessor(doc_objects)
    assert result["visuals"].size() == (128, 12544)
    assert result["semantics"].shape[0] == 128
    assert result["dimensions"].size() == (128, 4)
    assert result["positions"].size() == (128, 5)
    assert result["masks"].shape == (128,)
