import random

import pytest
import torch
from transformers import AutoTokenizer

from dosa.models import FeatureMapExtractor, SemanticEmbeder
from dosa.test.config import TEST_DIR
from dosa.test.loader import ObjectDetectionLoader


@pytest.mark.parametrize("batch_size, image_number", [(1, 3), (2, 2)])
def test_fpn_feature_extractor(batch_size, image_number):
    n_sequence = 512
    fpn = FeatureMapExtractor(n_sequence=n_sequence)
    visuals = []
    box_count = []
    for _ in range(0, batch_size):
        batch_images = torch.rand(image_number, 3, 1024, 1024)
        batch_boxes = [
            torch.rand(random.randint(1, 100), 4) * 256
            for _ in range(0, image_number)
        ]
        box_count.append(sum(len(page_box) for page_box in batch_boxes))
        batch_shapes = [(1024, 1024)] * image_number
        visuals.append((batch_images, batch_boxes, batch_shapes))

    features = fpn(visuals)
    assert features.shape == (batch_size, n_sequence, 12544)


@pytest.mark.parametrize("batch_size", [1, 2])
def test_sentence_transform_embedder(batch_size):
    model_name = "sentence-transformers/distiluse-base-multilingual-cased-v2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    embedder = SemanticEmbeder(128, model_name)

    doc_objects = ObjectDetectionLoader.read_objects(
        f"{TEST_DIR}/resource/hai/HAI1.pdf",
        f"{TEST_DIR}/resource/hai/HAI1_predictions.json",
    )
    contents = [content for page in doc_objects for content in page["content"]]
    tokens = tokenizer(contents, padding=True, return_tensors="pt")
    embeddings = embedder([tokens] * batch_size)

    assert embeddings.shape == (batch_size, 128, 768)
