from data.dataset import DOSADataset
from models import (
    PreProcessor,
    FPNFeatureMapExtractor,
    SentenceTransformerEmbeder,
)
from test.config import TEST_DIR


def test_dataset():
    preprocessor = PreProcessor(
        128, FPNFeatureMapExtractor(), SentenceTransformerEmbeder()
    )
    dataset = DOSADataset(
        f"{TEST_DIR}/resource/hrdoc/images",
        f"{TEST_DIR}/resource/hrdoc/annotations/train.json",
        preprocessor,
    )

    assert dataset.__len__() == 5
    for _ in range(5):
        result, target = dataset.__getitem__(0)
        assert len(result.keys()) == 5 and len(target.keys()) == 3
