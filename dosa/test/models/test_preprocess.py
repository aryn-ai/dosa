from dosa.test.config import TEST_DIR
from dosa.test.loader import ObjectDetectionLoader
from dosa.models.preprocess import PreProcessor


def test_preprocessor():
    doc_objects = ObjectDetectionLoader.read_objects(
        f"{TEST_DIR}/resource/hai/HAI1.pdf",
        f"{TEST_DIR}/resource/hai/HAI1_predictions.json",
    )
    preprocessor = PreProcessor(128, "sentence-transformers/all-MiniLM-L6-v2")
    result = preprocessor.process(doc_objects)
    assert len(result.visuals[0]) == 3
    assert len(result.semantics[0]) == 3
    assert result.measurement.size() == (1, 128, 3)
    assert result.positions.size() == (1, 128, 5)
    assert result.masks.shape == (1, 128)
