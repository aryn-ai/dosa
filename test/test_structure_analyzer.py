import argparse
import json

from args import get_args_parser
from processing.structure_analyzer import Analyzer
from test.config import TEST_DIR
from processing.object_extractor import (
    DeformableDETRDetector,
    PypdfiumExtractor,
    PromptGenerator,
    PageObjectExtractor,
)


def test_build_batch_tree():
    # a normal tree with one root, no conflict between pids and sids
    pids = {
        1: 1,
        2: 1,
        3: 1,
        4: 1,
        5: 2,
        6: 3,
        7: 3,
        8: 2,
        9: 2,
        10: 1,
        11: 3,
        12: 2,
        13: 2,
        14: 1,
        15: 6,
        16: 6,
    }
    sids = {
        1: 1,
        2: 2,
        3: 14,
        4: 2,
        5: 5,
        6: 11,
        7: 7,
        8: 5,
        9: 8,
        10: 4,
        11: 7,
        12: 9,
        13: 12,
        14: 10,
        15: 16,
        16: 16,
    }
    tree = Analyzer.build_batch_tree(pids, sids)
    assert tree[0] == [1]
    assert tree[1] == [2, 4, 10, 14, 3]
    assert tree[2] == [5, 8, 9, 12, 13]
    assert tree[3] == [7, 11, 6]
    assert tree[6] == [16, 15]

    # a normal tree with multiple roots
    pids = {
        1: 1,
        2: 1,
        3: 3,
        4: 1,
        5: 2,
        6: 3,
        7: 3,
        8: 2,
        9: 2,
        10: 1,
        11: 3,
        12: 2,
        13: 2,
        14: 1,
        15: 6,
        16: 6,
    }
    sids = {
        1: 1,
        2: 2,
        3: 1,
        4: 2,
        5: 5,
        6: 11,
        7: 7,
        8: 5,
        9: 8,
        10: 4,
        11: 7,
        12: 9,
        13: 12,
        14: 10,
        15: 16,
        16: 16,
    }
    tree = Analyzer.build_batch_tree(pids, sids)
    assert tree[0] == [1, 3]
    assert tree[1] == [2, 4, 10, 14]
    assert tree[2] == [5, 8, 9, 12, 13]
    assert tree[3] == [7, 11, 6]
    assert tree[6] == [16, 15]


def test_analyze_tree():
    detector = DeformableDETRDetector()
    pdf = PypdfiumExtractor(PromptGenerator())
    extractor = PageObjectExtractor(detector, pdf)
    path = f"{TEST_DIR}/resource/papers/Transformer.pdf"
    pages = extractor.extract(path)

    parser = argparse.ArgumentParser("DOSA", parents=[get_args_parser()])
    args, unknown = parser.parse_known_args()
    args.device = "cpu"
    args.n_sequence = 64
    args.resume = (TEST_DIR / "resource/weights/checkpoint.pth").as_posix()
    analyzer = Analyzer(args)
    tree = analyzer.analyze(pages)
    with open("/Users/bohou/workspace/dosa/tree.json", "w") as ofd:
        json.dump(tree[0], ofd, indent=2)
