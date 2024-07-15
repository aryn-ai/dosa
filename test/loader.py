import json
import tempfile

import pdf2image
import pypdfium2 as pdfium
import torch
from torch import Tensor

from test.config import TEST_DIR
from type.categories import Category


class PdfParser:
    @staticmethod
    def _convert_bbox_coordinates(
        rect: tuple[float, float, float, float],
        image_shape: tuple[float, float],
        page_shape: tuple[float, float],
    ) -> tuple[float, float, float, float]:
        """
        Convert a bbox of [x1, y1, x2, y2] format into pdf coordinates

        pdf coordinates are different, bottom left is origin, with
        [left, bottom, right, top] as output, refer
        https://www.leadtools.com/help/leadtools/v19/dh/to/pdf-topics-pdfcoordinatesystem.html
        """
        x1, y1, x2, y2 = rect

        x1 *= page_shape[0] / image_shape[0]
        x2 *= page_shape[0] / image_shape[0]

        y1 = page_shape[1] - y1 * page_shape[1] / image_shape[1]
        y2 = page_shape[1] - y2 * page_shape[1] / image_shape[1]
        return x1, y2, x2, y1

    @staticmethod
    def parse(
        path: str,
        shapes: list[tuple[int, int]],
        doc_objects: list[dict[str, Tensor]],
    ) -> list[list[str]]:
        pdf = pdfium.PdfDocument(path)
        doc_contents = []
        try:
            for page_idx, page_objects in enumerate(doc_objects):
                page = pdf[page_idx]
                image_width, image_height = shapes[page_idx]
                pdf_width, pdf_height = page.get_size()

                page_contents = []
                for label, box in zip(
                    page_objects["labels"], page_objects["boxes"]
                ):
                    if label == 7:
                        page_contents.append("A figure with content unknown")
                    elif label == 9:
                        page_contents.append("A table with content unknown")
                    else:
                        # text based, we do need to extract the content here
                        coordinates = PdfParser._convert_bbox_coordinates(
                            box.detach().numpy(),
                            (image_width, image_height),
                            (pdf_width, pdf_height),
                        )
                        # TODO, use nlp lib to fix the potential sentence issue
                        text_part = page.get_textpage().get_text_bounded(
                            *coordinates
                        )
                        page_contents.append(
                            f"A {Category(int(label)).name} with content: {text_part}"
                        )

                doc_contents.append(page_contents)
        finally:
            pdf.close()
        return doc_contents


class ObjectDetectionLoader:
    @staticmethod
    def write_objects(input_path, output_path):
        from transformers import (
            AutoImageProcessor,
            DeformableDetrForObjectDetection,
        )

        images = pdf2image.convert_from_path(input_path)
        images = [image.resize((1024, 1024)) for image in images]
        processor = AutoImageProcessor.from_pretrained(
            "Aryn/deformable-detr-DocLayNet"
        )
        model = DeformableDetrForObjectDetection.from_pretrained(
            "Aryn/deformable-detr-DocLayNet"
        )

        results = []
        for image in images:
            inputs = processor(images=image, return_tensors="pt")
            outputs = model(**inputs)
            target_sizes = torch.tensor([(1024, 1024)])
            result = processor.post_process_object_detection(
                outputs, target_sizes=target_sizes, threshold=0.5
            )
            results.extend(result)
        doc_contents = PdfParser.parse(
            input_path, [(1024, 1024)] * len(images), results
        )
        results = [
            {k: v.detach().numpy().tolist() for k, v in result.items()}
            for result in results
        ]
        for result, contents in zip(results, doc_contents):
            fonts = []
            for box, label, content in zip(
                result["boxes"], result["labels"], contents
            ):
                if label == 7 or label == 9:
                    fonts.append(0)
                else:
                    lines = content.split("\n")
                    cols = max([len(line) for line in lines])
                    rows = len(lines)
                    fonts.append(
                        ((box[3] - box[1]) / rows) * ((box[2] - box[0]) / cols)
                    )
            result["fonts"] = fonts
            result["contents"] = contents

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
