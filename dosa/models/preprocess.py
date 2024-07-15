import torch
from torchvision.transforms import functional as F
from transformers import AutoTokenizer

from dosa.core.sample import Samples
from dosa.util.misc import pad


class PreProcessor:
    def __init__(
        self,
        n_sequence: int,
        semantic_model_name: str,
    ):
        super().__init__()
        self._n_sequence = n_sequence
        # BertTokenizerFast
        self._tokenizer = AutoTokenizer.from_pretrained(semantic_model_name)

    @staticmethod
    def _measurement(page_boxes):
        # compute measurement for page objects in a single page
        whs = page_boxes[:, 2:]
        area = whs[:, 0] * whs[:, 1]
        measurement = torch.concat([whs, area.unsqueeze(1)], dim=1)
        return measurement

    @staticmethod
    def _position(page_boxes, page_shape, orders):
        # compute position for page objects in a single page
        # [order, x1, y1, x2, y2], page and coordinates need
        # normalization separately,
        page_boxes[:, 2:] = page_boxes[:, 2:] + page_boxes[:, :2]
        width, height = page_shape
        page_boxes[:, 0] /= width
        page_boxes[:, 1] /= height
        page_boxes[:, 2] /= width
        page_boxes[:, 3] /= height
        position = torch.concat([orders.unsqueeze(1), page_boxes], dim=1)
        return position

    def process(self, sample) -> Samples:
        """
        :param sample: a single sample is a list of pages, each page is a
            dictionary containing:
            image: the PIL image containing page objects
            boxes: a tensor containing bounding boxes per page in [x1, y1, w, h]
            labels: a tensor containing labels per page
            contents: a list of content string for each page object, it could be
            the original content of a paragraph, the summary of an image, the
            description of a table etc., see detailed on how we process raw
            data for training and inferring.
        :return: a Samples containing only one preprocessed sample, this sample
            would be used by DOSA for training or inferring
        """
        images = [F.to_tensor(page["image"]) for page in sample]
        shapes = [page["image"].size for page in sample]
        bboxes = [page["bbox"] for page in sample]
        categories = [page["category"] for page in sample]
        orders = [page["order"] for page in sample]

        contents = [content for page in sample for content in page["content"]]
        semantics = self._tokenizer(
            contents,
            padding=True,
            truncation=True,
            max_length=256,
            return_tensors="pt",
        )

        measurement = []
        positions = []
        # relative page number order should be enough, this also restrict the
        # value under fixed sequence length
        for page_shape, page_bboxes, page_contents, page_orders, page in zip(
            shapes, bboxes, contents, orders, sample
        ):
            # compute the measurement info
            measurement.append(self._measurement(page_bboxes))

            # compute the position encoding:
            positions.append(
                self._position(page_bboxes, page_shape, page_orders)
            )
        measurement = torch.concat(measurement, dim=0)
        categories = torch.concat(categories, dim=0)
        positions = torch.concat(positions, dim=0)

        # padding tensors to sequence length and generate a mask
        masks = torch.arange(self._n_sequence) >= len(contents)
        measurement = pad(measurement, self._n_sequence)
        categories = pad(categories, self._n_sequence)
        positions = pad(positions, self._n_sequence)

        sample = Samples(
            [(torch.stack(images), bboxes, shapes)],
            [semantics],
            measurement.unsqueeze(0),
            categories.unsqueeze(-1).unsqueeze(0),
            positions.unsqueeze(0),
            masks.unsqueeze(0),
        )

        return sample
