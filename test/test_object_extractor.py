from processing.object_extractor import (
    DeformableDETRDetector,
    PypdfiumExtractor,
    PromptGenerator,
    PageObjectExtractor,
)
from test.config import TEST_DIR


def test_object_extractor():
    detector = DeformableDETRDetector()
    pdf = PypdfiumExtractor(PromptGenerator())
    extractor = PageObjectExtractor(detector, pdf)
    path = f"{TEST_DIR}/resource/papers/Transformer.pdf"
    objects = extractor.extract(path)
    assert len(objects) == 11
