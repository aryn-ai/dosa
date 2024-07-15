import argparse

import pytest

from args import get_args_parser
from infer import init_model, infer
from dosa.test.config import TEST_DIR
from dosa.test.loader import ObjectDetectionLoader


MODEL_FILE = TEST_DIR / "resource/weights/checkpoint-dochienet.pth"


@pytest.mark.skipif(
    not MODEL_FILE.exists(),
    reason="resource/weights/checkpoint-dochienet.pth is required for this test",
)
def test_infer():
    parser = argparse.ArgumentParser("DOSA", parents=[get_args_parser()])
    args, unknown = parser.parse_known_args()
    args.device = "cpu"
    args.n_sequence = 256
    args.resume = (MODEL_FILE).as_posix()
    preprocessor, model, postprocessor = init_model(args)

    doc_objects = [
        ObjectDetectionLoader.read_objects(
            f"{TEST_DIR}/resource/hai/HAI1.pdf",
            f"{TEST_DIR}/resource/hai/HAI1_predictions.json",
        ),
        ObjectDetectionLoader.read_objects(
            f"{TEST_DIR}/resource/hai/HAI2.pdf",
            f"{TEST_DIR}/resource/hai/HAI2_predictions.json",
        ),
    ]

    result = infer(preprocessor, model, postprocessor, list(doc_objects))
    assert len(result[0]["parent"]["labels"]) == 69
    assert len(result[1]["parent"]["labels"]) == 118
