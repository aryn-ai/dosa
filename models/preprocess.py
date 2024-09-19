from sentence_transformers import SentenceTransformer


import torch
from torch import Tensor
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.ops import MultiScaleRoIAlign
from torchvision.transforms import functional as F


class FeatureMapExtractor(torch.nn.Module):
    def forward(
        self,
        images: list[Tensor],
        shapes: list[tuple[int, int]],
        boxes: list[Tensor],
    ) -> list[Tensor]:
        """
        :param images: image tensors for all pages in a document
        :param shapes: (List[Tuple[height, width]]) the sizes of each
            image before they have been fed to a CNN to obtain feature maps.
            This allows us to infer the scale factor for each one of the levels
            to be pooled.
        :param boxes: boxes (List[Tensor[N, 4]]) to be used to perform the
            pooling operation, in (x1, y1, x2, y2) format and in the image
            reference size, not the feature map reference. The coordinate must
            satisfy ``0 <= x1 < x2`` and ``0 <= y1 < y2``.
        :return: a list of Tensor of shape object_count_per_image x dimension
        """
        pass


class FPNFeatureMapExtractor(FeatureMapExtractor):
    """
    Resnet50FPN backbone
    It first uses a Resnet50FPN to extract multiscale feature maps from
    layers (P2, P3, P4, P5, P6), then a MultiScaleROIAlign is used to
    extract page object feature maps.

    :param output_size (List[Tuple[int, int]] or List[int]): output size for
        the pooled region
    :param sampling_ratio: sampling ratio for ROIAlign
    """

    def __init__(
        self,
        output_size: int | tuple[int] | list[int] = 7,
        sampling_ratio: int = 2,
        batch_size: int = 2,
    ):

        super().__init__()
        self._body = resnet_fpn_backbone("resnet50", pretrained=True)
        self._roi = MultiScaleRoIAlign(
            ["0", "1", "2", "3", "pool"], output_size, sampling_ratio
        )
        self._batch_size = batch_size

    def to(self, device):
        self._body.to(device)
        self._roi.to(device)

    @torch.no_grad()
    def forward(
        self,
        images: list[Tensor],
        shapes: list[tuple[int, int]],
        boxes: list[Tensor],
    ) -> list[Tensor]:
        result = []
        for i in range(0, len(images), self._batch_size):
            batch_images = images[i : i + self._batch_size]
            batch_boxes = boxes[i : i + self._batch_size]
            batch_shapes = shapes[i : i + self._batch_size]

            batch_tensors = torch.stack(batch_images)
            features = self._body(batch_tensors)
            roi_features = self._roi(features, batch_boxes, batch_shapes)
            flattened = roi_features.flatten(start_dim=1, end_dim=-1)

            sizes = [len(box) for box in batch_boxes]
            roi_features_per_page = torch.split(flattened, sizes)
            result.extend(roi_features_per_page)

        return result


class SemanticEmbeder(torch.nn.Module):

    def forward(
        self,
        doc_contents: list[list[str]],
    ) -> list[Tensor]:
        """
        :param doc_contents: a list of content per page, contents in each page
            is a list of string
        :return: list of Tensor per page
        """
        pass


class SentenceTransformerEmbeder(SemanticEmbeder):
    def __init__(
        self, model_name_or_path: str = "all-MiniLM-L6-v2", batch_size: int = 2
    ):
        super().__init__()
        self._embedder = SentenceTransformer(model_name_or_path)
        self._batch_size = batch_size

    def to(self, device):
        self._embedder.to(device)

    @torch.no_grad()
    def forward(
        self,
        doc_contents: list[list[str]],
    ) -> list[Tensor]:
        # embedding
        results = []
        for page_contents in doc_contents:
            result = []
            for i in range(0, len(page_contents), self._batch_size):
                batch = page_contents[i : i + self._batch_size]
                batch_result = self._embedder.encode(
                    batch, convert_to_tensor=True
                )
                result.append(batch_result)
            results.append(torch.concat(result, 0))
        return results


class PreProcessor(torch.nn.Module):
    def __init__(
        self,
        n_sequence: int,
        visual_extractor: FeatureMapExtractor,
        semantic_embedder: SemanticEmbeder,
        device: str = "cpu",
    ):
        super().__init__()
        self._n_sequence = n_sequence
        self._device = device
        self._visual = visual_extractor
        self._semantic = semantic_embedder
        self._visual.to(device)
        self._semantic.to(device)

    def _dimension(self, page_boxes, page_fonts):
        # fonts are computed from the area of a character in a bounding box;
        # for figure and table, it's set to 0, therefore since rows/columns are
        # all require to set to inf
        whs = page_boxes[:, 2:]
        area = whs[:, 0] * whs[:, 1]
        dimension = torch.concat(
            [whs, page_fonts.unsqueeze(1), area.unsqueeze(1)], dim=1
        )
        return dimension

    def _position(self, page_boxes, page_shape, page_no, page_count):
        #  [page_no/page_count, x1, y1, x2, y2], page and coordinates need
        #  normalization separately,
        page_boxes[:, 2:] = page_boxes[:, 2:] + page_boxes[:, :2]
        width, height = page_shape
        page_boxes[:, 0] /= width
        page_boxes[:, 1] /= height
        page_boxes[:, 2] /= width
        page_boxes[:, 3] /= height
        page_idx = torch.full(
            (page_boxes.size(0), 1),
            page_no / page_count,
            device=self._device,
        )
        position = torch.concat([page_idx, page_boxes], dim=1)
        return position

    def _pad(self, visuals, semantics, dimensions, positions):
        # pad visuals and semantics to max sequence and generate masks
        length = visuals.size(0)
        padding = self._n_sequence - length
        assert 0 <= padding < 256, f"Invalid sequence length {padding}"

        visuals = torch.concat(
            [
                visuals,
                torch.zeros([padding, *visuals.shape[1:]], device=self._device),
            ],
            dim=0,
        )
        semantics = torch.concat(
            [
                semantics,
                torch.zeros(
                    [padding, *semantics.shape[1:]], device=self._device
                ),
            ],
            dim=0,
        )
        dimensions = torch.concat(
            [
                dimensions,
                torch.zeros(
                    [padding, *dimensions.shape[1:]], device=self._device
                ),
            ],
            dim=0,
        )
        positions = torch.concat(
            [
                positions,
                torch.zeros(
                    [padding, *positions.shape[1:]], device=self._device
                ),
            ],
            dim=0,
        )
        masks = torch.arange(self._n_sequence, device=self._device) >= length

        return visuals, semantics, dimensions, positions, masks

    @torch.no_grad()
    def forward(self, objects):
        """
        :param objects: page object info, including:
            a list of images containing related page object
            a list of content corresponding to those related page objects
            a list of bounding boxes tensor per page
            a list of labels tensor per page
            a list of fonts tensor per page
        :return: a dict of 4 tensors:
            visual embedding of shape n_sequence x CWH
            semantic embedding of shape n_sequence x d_embedding
            dimension embedding of shape n_sequence x 4
            position embedding of shape n_sequence x 5
            padding mask of shape n_sequence
        """
        images = [obj["image"] for obj in objects]
        contents = [obj["contents"] for obj in objects]
        shapes = [image.size for image in images]
        image_tensors = [
            F.to_tensor(image).to(device=self._device) for image in images
        ]
        visuals = self._visual(
            image_tensors,
            shapes,
            [
                page_objects["boxes"].to(device=self._device)
                for page_objects in objects
            ],
        )
        semantics = self._semantic(contents)
        assert len(visuals) == len(semantics)

        dimensions = []
        positions = []
        # relative page number order should be enough, this also restrict the
        # value under fixed sequence length
        page_count = len(shapes)
        for page_no, (page_shape, page_contents, page_objects) in enumerate(
            zip(shapes, contents, objects)
        ):
            page_boxes = page_objects["boxes"].to(device=self._device)
            page_fonts = page_objects["fonts"].to(device=self._device)
            # compute the dimension info
            dimensions.append(self._dimension(page_boxes, page_fonts))

            # compute the position encoding:
            positions.append(
                self._position(page_boxes, page_shape, page_no, page_count)
            )

        visuals = torch.concat(visuals, dim=0)
        semantics = torch.concat(semantics, dim=0)
        dimensions = torch.concat(dimensions, dim=0)
        positions = torch.concat(positions, dim=0)

        visuals, semantics, dimensions, positions, masks = self._pad(
            visuals, semantics, dimensions, positions
        )
        return {
            "visuals": visuals.flatten(start_dim=1, end_dim=-1),
            "semantics": semantics,
            "dimensions": dimensions,
            "positions": positions,
            "masks": masks,
        }
