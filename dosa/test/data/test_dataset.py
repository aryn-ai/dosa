from dosa.core.sample import Samples
from dosa.data.dataset import DOSADataset
from dosa.models import PreProcessor
from dosa.test.config import TEST_DIR


def test_dataset():
    preprocessor = PreProcessor(128, "sentence-transformers/all-MiniLM-L6-v2")
    dataset = DOSADataset(
        f"{TEST_DIR}/resource/dochienet/images",
        f"{TEST_DIR}/resource/dochienet/annotations/train.json",
        preprocessor,
    )

    assert dataset.__len__() == 2
    for idx in range(2):
        result, target = dataset.__getitem__(idx)
        assert isinstance(result, Samples) and len(target.keys()) == 3
