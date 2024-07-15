import json
from pathlib import Path

class2number = {
    "endnote": 0,
    "equation": 1,
    "figure": 2,
    "figure-caption": 3,
    "figure-title": 4,
    "footer": 5,
    "footnote": 6,
    "header": 7,
    "other": 8,
    "page-number": 9,
    "section-title": 10,
    "sidebar": 11,
    "sub-title": 12,
    "table": 13,
    "table-caption": 14,
    "table-of-content": 15,
    "table-title": 16,
    "text": 17,
    "title": 18,
    "toc-title": 19,
}

number2class = {
    0: "end note",
    1: "equation",
    2: "figure",
    3: "figure caption",
    4: "figure title",
    5: "footer",
    6: "footnote",
    7: "header",
    8: "unclassified object",
    9: "page number",
    10: "section title",
    11: "sidebar",
    12: "subtitle",
    13: "table",
    14: "table caption",
    15: "Table of Content",
    16: "table title",
    17: "paragraph",
    18: "title",
    19: "table of content title",
}


def get_train_test_prefix(train_test_path: Path):
    with open(train_test_path) as ifd:
        train_test = json.load(ifd)
        train = train_test["train"]
        test = train_test["test"]
    return train, test


def get_zh_en_prefix(zh_en_path: Path):
    with open(zh_en_path) as ifd:
        en_zh = json.load(ifd)
        en = en_zh["en"]
        zh = en_zh["zh"]
    return en, zh
