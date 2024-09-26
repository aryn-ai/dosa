from dataclasses import dataclass
from pathlib import PurePath
from typing import Union

import pdf2image
from PIL import Image
import pypdfium2 as pdfium
import torch
from transformers import (
    AutoImageProcessor,
    DeformableDetrForObjectDetection,
    AutoTokenizer,
)

from processing.categories import Category


@dataclass
class Page:
    image: Image = None
    page_no: int = None
    shape: tuple[int, int] = None
    objects: dict = None


class ObjectDetector:
    def detect(self, images: list[Image]) -> list[Page]:
        pass


class DeformableDETRDetector(ObjectDetector):
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

    def detect(self, images: list[Image]) -> list[Page]:
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
        for page_no, result in enumerate(results):
            page = Page()
            page.page_no = page_no
            page.image = images[page_no]
            labels = result["labels"].detach().cpu().numpy().tolist()
            boxes = result["boxes"].detach().cpu().numpy().tolist()
            page.objects = []
            for label, box in zip(labels, boxes):
                page.objects.append(
                    {
                        "label": label,
                        "box": box,
                        "page_no": page_no,
                        "children": [],
                    }
                )
            pages.append(page)
        return pages


class PromptGenerator:
    def __init__(self, model_path="sentence-transformers/all-MiniLM-L12-v2"):
        self._tokenizer = AutoTokenizer.from_pretrained(model_path)

    def token_count(self, text):
        # Tokenize the text
        encoded_input = self._tokenizer(text, return_tensors=None)
        return len(encoded_input["input_ids"])

    def generate_prompt(self, label, text):
        lines = [line.rstrip() for line in text.split("\n")]
        paragraph = ""
        for line in lines:
            if not paragraph:
                paragraph = line
            elif paragraph.endswith("-"):
                paragraph = paragraph.rstrip("-") + line
            else:
                paragraph = paragraph + " " + line

        if self.token_count(paragraph) <= 240:
            return (
                f"A {Category(label).name}:\n{paragraph}"
                if label != "Text"
                else f"A paragraph of text:\n{paragraph}"
            )
        else:
            return f"A paragraph of text with first line:\n{lines[0]}\nand last line:\n{lines[-1]}"


class ContentExtractor:
    def extract(self, path: Union[str, PurePath], pages: list[Page]) -> None:
        pass


class PypdfiumExtractor(ContentExtractor):
    def __init__(self, prompter: PromptGenerator):
        self._prompter = prompter

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
    def _font_size(text, box):
        lines = text.split("\n")
        height = box[3] - box[1]
        return height / len(lines)

    def extract(self, path: Union[str, PurePath], pages: list[Page]):
        pdf = pdfium.PdfDocument(path)
        try:
            for page in pages:
                pdf_page = pdf[page.page_no]
                image_width, image_height = page.image.size
                pdf_width, pdf_height = pdf_page.get_size()

                for obj in page.objects:
                    if obj["label"] != 7 and obj["label"] != 9:
                        # text based, we do need to extract the content here
                        coordinates = self._convert_bbox_coordinates(
                            obj["box"],
                            (image_width, image_height),
                            (pdf_width, pdf_height),
                        )
                        text = pdf_page.get_textpage().get_text_bounded(
                            *coordinates
                        )
                        content = self._prompter.generate_prompt(
                            obj["label"], text
                        )
                        font = self._font_size(text, obj["box"])
                        obj["text"] = text
                        obj["content"] = content
                        obj["font"] = font
        finally:
            pdf.close()


class GraphObjectExtractor(ContentExtractor):
    def extract(self, path: str, pages: list[Page]):
        pass


class PageObjectExtractor:
    def __init__(self, detector: ObjectDetector, extractor: ContentExtractor):
        self._detector = detector
        self._extractor = extractor

    def extract(
        self, path: Union[str, PurePath], shape: tuple[int, int] = (800, 800)
    ):
        """
        TODO, the content should return directly Tensor instead of text in case
            we have VLM aligned training backbone
        Extract page objects given a doc, this involves partition and content
        extractor, it should be inheritable, e.g. text extract could use
        pdfminer or an ocr tool; or a partitioner could be any model beyond of
        deformable detr
        """
        images = pdf2image.convert_from_path(path)
        images = [image.resize(shape) for image in images]
        pages = self._detector.detect(images)
        self._extractor.extract(path, pages)
        return pages
