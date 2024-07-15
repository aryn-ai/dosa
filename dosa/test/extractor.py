from enum import Enum
from pathlib import PurePath
from typing import Union

import pdf2image
from PIL import Image
import pypdfium2 as pdfium
import torch
from transformers import AutoImageProcessor, DeformableDetrForObjectDetection


# The category compatible with DocLayNet
class Category(Enum):
    Caption = 1
    Footnote = 2
    Formula = 3
    ListItem = 4
    PageFooter = 5
    PageHeader = 6
    Picture = 7
    SectionHeader = 8
    Table = 9
    Text = 10
    Title = 11
    Section = 101
    Group = 102


class DeformableDETRDetector:
    def __init__(
        self,
        path="Aryn/deformable-detr-DocLayNet",
        device="cpu",
        batch_size=4,
        threshold=0.5,
    ):
        self._processor = AutoImageProcessor.from_pretrained(path)
        self._model = DeformableDetrForObjectDetection.from_pretrained(path)
        self._device = device
        self._batch_size = batch_size
        self._threshold = threshold

    def detect(self, images: list[Image]) -> list[dict]:
        results = []
        for i in range(0, len(images), self._batch_size):
            batch_images = images[i : i + self._batch_size]
            inputs = self._processor(images=batch_images, return_tensors="pt")
            outputs = self._model(**inputs)
            target_sizes = torch.tensor([image.size for image in batch_images])
            result = self._processor.post_process_object_detection(
                outputs, target_sizes=target_sizes, threshold=self._threshold
            )
            results.extend(result)

        pages = []
        base_id = 1
        for page_no, result in enumerate(results):
            page = {
                "image": images[page_no],
                "bbox": result["boxes"].cpu(),
                "category": result["labels"].cpu(),
            }
            ids = list(range(base_id, base_id + len(page["category"])))
            page["id"] = ids
            base_id += len(page["category"])
            pages.append(page)
        return pages


class PypdfiumExtractor:
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
    def _generate_content(label, text):
        if Category(label) == Category.ListItem:
            return f"A paragraph: {text}"
        elif Category(label) == Category.Text:
            return f"A paragraph: {text}"
        else:
            return f"{Category(label).name}:{text}"

    def extract(self, path: Union[str, PurePath], pages: list[dict]):
        doc = pdfium.PdfDocument(path)
        try:
            for page_no, page in enumerate(pages):
                pdf = doc[page_no]
                image_width, image_height = page["image"].size
                pdf_width, pdf_height = pdf.get_size()

                content = []
                for bbox, label in zip(
                    page["bbox"].tolist(), page["category"].tolist()
                ):
                    coordinates = self._convert_bbox_coordinates(
                        bbox,
                        (image_width, image_height),
                        (pdf_width, pdf_height),
                    )

                    text = pdf.get_textpage().get_text_bounded(*coordinates)
                    content.append(self._generate_content(label, text))
                page["content"] = content
        finally:
            doc.close()


class PageObjectExtractor:
    def __init__(
        self, detector: DeformableDETRDetector, extractor: PypdfiumExtractor
    ):
        self._detector = detector
        self._extractor = extractor

    def extract(
        self, path: Union[str, PurePath], shape: tuple[int, int] = (800, 800)
    ):
        """
        Extract page objects given a doc, this involves partition and content
        extractor, it should be inheritable, e.g. text extract could use
        pdfminer or an ocr tool; or a partitioner could be any model beyond of
        deformable detr
        """
        images = pdf2image.convert_from_path(path)
        images = [image.convert("RGB").resize(shape) for image in images]
        pages = self._detector.detect(images)
        self._extractor.extract(path, pages)
        for page in pages:
            bbox = page["bbox"]
            bbox[:, 2:] = bbox[:, 2:] - bbox[:, :2]
            page["bbox"] = bbox
        return pages


def test_object_extractor():
    from dosa.test.config import TEST_DIR

    detector = DeformableDETRDetector()
    pdf = PypdfiumExtractor()
    extractor = PageObjectExtractor(detector, pdf)
    path = f"{TEST_DIR}/resource/papers/Transformer.pdf"
    pages = extractor.extract(path)
    assert len(pages) == 11
