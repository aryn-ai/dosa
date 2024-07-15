import argparse
import tempfile

from args import get_args_parser
from dosa.test.config import TEST_DIR
from train import train


def test_train():
    parser = argparse.ArgumentParser("DOSA", parents=[get_args_parser()])
    args, unknown = parser.parse_known_args()
    args.epochs = 2
    args.n_sequence = 128
    args.data_path = f"{TEST_DIR}/resource/dochienet"
    args.device = "cpu"
    args.batch_size = 1
    with tempfile.TemporaryDirectory() as output_dir:
        args.output_dir = output_dir
        train(args)
