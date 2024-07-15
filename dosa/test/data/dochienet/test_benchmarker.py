import argparse

import pytest

from args import get_args_parser
from dosa.data.dochienet.benchmark import DocHieNetBenchmarker
from dosa.test.config import TEST_DIR


MODEL_FILE = TEST_DIR / "resource/weights/checkpoint-dochienet.pth"


@pytest.mark.skipif(
    not MODEL_FILE.exists(),
    reason="resource/weights/checkpoint-dochienet.pth is required for this test",
)
def test_benchmark():
    parser = argparse.ArgumentParser("DOSA", parents=[get_args_parser()])
    args, unknown = parser.parse_known_args()
    args.device = "cpu"
    args.n_sequence = 256
    args.data_path = f"{TEST_DIR}/resource/dochienet/"
    args.resume = (MODEL_FILE).as_posix()
    benchmarker = DocHieNetBenchmarker(args)
    benchmarker.evaluate("benchmark.json")
