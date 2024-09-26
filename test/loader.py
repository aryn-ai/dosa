import json
import tempfile

import pdf2image
import torch

from processing.object_extractor import (
    PageObjectExtractor,
    PromptGenerator,
    PypdfiumExtractor,
    DeformableDETRDetector,
)
from test.config import TEST_DIR


class ObjectDetectionLoader:
    @staticmethod
    def write_objects(input_path, output_path):
        extractor = PageObjectExtractor(
            DeformableDETRDetector(), PypdfiumExtractor(PromptGenerator())
        )
        pages = extractor.extract(input_path, (1024, 1024))
        results = []
        for page in pages:
            labels = [obj["label"] for obj in page.objects]
            boxes = [obj["box"] for obj in page.objects]
            fonts = [
                obj["font"] if "font" in obj else 0 for obj in page.objects
            ]
            contents = [
                obj["content"] if "content" in obj else "Unknown Content"
                for obj in page.objects
            ]
            result = {
                "labels": labels,
                "boxes": boxes,
                "fonts": fonts,
                "contents": contents,
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

        images = pdf2image.convert_from_path(doc_path)
        images = [image.resize((1024, 1024)) for image in images]

        for image, result in zip(images, results):
            for k, v in result.items():
                if k == "boxes":
                    result[k] = torch.tensor(v)
                if k == "labels":
                    result[k] = torch.tensor(v, dtype=torch.int8)
                if k == "fonts":
                    result[k] = torch.tensor(v, dtype=torch.float)
            result["image"] = image
        return results


def test_loader():
    with tempfile.NamedTemporaryFile() as output_path:
        doc_path = f"{TEST_DIR}/resource/hai/HAI1.pdf"
        ObjectDetectionLoader.write_objects(doc_path, output_path.name)
        ObjectDetectionLoader.read_objects(doc_path, output_path.name)
