import json
import tempfile

import pdf2image
import torch

from dosa.test.extractor import (
    PageObjectExtractor,
    PypdfiumExtractor,
    DeformableDETRDetector,
)
from dosa.test.config import TEST_DIR


class ObjectDetectionLoader:
    @staticmethod
    def write_objects(input_path, output_path):
        extractor = PageObjectExtractor(
            DeformableDETRDetector(), PypdfiumExtractor()
        )
        pages = extractor.extract(input_path, (800, 800))
        results = []
        for page in pages:
            result = {
                "category": page["category"].tolist(),
                "bbox": page["bbox"].tolist(),
                "content": page["content"],
            }
            results.append(result)

        with open(output_path, "w") as ofd:
            json.dump(results, ofd, indent=2)

    @staticmethod
    def read_objects(doc_path, ann_path):
        """
        Read page object detection annotations for a document, return and
        organize them page by page
        """
        with open(ann_path, "r") as ifd:
            results = json.load(ifd)

        images = [
            image.resize((800, 800))
            for image in pdf2image.convert_from_path(doc_path)
        ]

        for image, result in zip(images, results):
            for k, v in result.items():
                if k == "bbox":
                    result[k] = torch.tensor(v)
                if k == "category":
                    result[k] = torch.tensor(v, dtype=torch.int8)
            result["image"] = image
        return results


def test_loader():
    with tempfile.NamedTemporaryFile() as output_path:
        doc_path = f"{TEST_DIR}/resource/hai/HAI2.pdf"
        ObjectDetectionLoader.write_objects(doc_path, output_path.name)
        ObjectDetectionLoader.read_objects(doc_path, output_path.name)
