import torch
from torch import Tensor
from torchvision.models import ResNet50_Weights
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.ops import MultiScaleRoIAlign
import torch.nn.functional as F
from transformers import AutoModel

from dosa.util.misc import pad


class FeatureMapExtractor(torch.nn.Module):
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
        n_sequence: int,
        output_size: int | tuple[int] | list[int] = 7,
        sampling_ratio: int = 2,
    ):

        super().__init__()
        self._n_sequence = n_sequence
        self._body = resnet_fpn_backbone(
            backbone_name="resnet50", weights=ResNet50_Weights.DEFAULT
        )
        self._roi = MultiScaleRoIAlign(
            ["0", "1", "2", "3", "pool"], output_size, sampling_ratio
        )

    @torch.no_grad()
    def forward(
        self, visuals: list[tuple[Tensor, list[Tensor], list[tuple[int, int]]]]
    ) -> Tensor:
        """
        We do this iteratively since the images in each batch might vary, we
        want to align the tensor on the number of page objects in each batch
        instead of the number of images in each batch, also each batch is
        already well parallel
        :param visuals a batch of visual input, each visual input contains
            images: an images tensor for all images in one sample
            shapes: a list of shapes for each image in one sample
            boxes: bounding boxes tensor for each image in one sample
        :return: visual embedding of shape batch_size x n_sequence x CWH, under
            current extractor it's 12544
        """
        result = []
        for sample_images, sample_boxes, sample_shapes in visuals:
            features = self._body(sample_images)
            roi_features = self._roi(features, sample_boxes, sample_shapes)
            flattened = roi_features.flatten(start_dim=1, end_dim=-1)
            padded = pad(flattened, self._n_sequence)
            result.append(padded)
        stacked = torch.stack(result)
        return stacked


class SemanticEmbeder(torch.nn.Module):
    def __init__(
        self,
        n_sequence: int,
        semantic_model_name: str,
    ):
        super().__init__()
        self._n_sequence = n_sequence
        self._embedder = AutoModel.from_pretrained(semantic_model_name)

    @staticmethod
    def mean_pooling(model_output, attention_mask):
        token_embeddings = model_output[0]
        # First element of model_output contains all token embeddings
        input_mask_expanded = (
            attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        )
        return torch.sum(
            token_embeddings * input_mask_expanded, 1
        ) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    @torch.no_grad()
    def forward(self, contents: list[dict[str, Tensor]]) -> Tensor:
        """
        Run bert model to get the embedding for semantic content, refer the code
        from https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2
        :param contents: each content is a batch of sample contents, each sample
            contents is a dict obtained from running PreProcessor's semantic
            tokenizer over all content objects over all images
        :return: semantic embedding of shape
            batch-size x n_sequence x d_embedding
        """
        result = []
        for sample_contents in contents:
            model_output = self._embedder(**sample_contents)
            embeddings = self.mean_pooling(
                model_output, sample_contents["attention_mask"]
            )
            # Normalize embeddings
            normalized = F.normalize(embeddings, p=2, dim=1)
            padded = pad(normalized, self._n_sequence)
            result.append(padded)
        return torch.stack(result)


class Backbone(torch.nn.Module):
    def __init__(
        self,
        visual_extractor: FeatureMapExtractor,
        semantic_embedder: SemanticEmbeder,
        train_visual: bool,
        train_semantic: bool,
    ):
        super().__init__()
        self._visual = visual_extractor
        self._semantic = semantic_embedder

        if not train_visual:
            for name, parameter in self._visual.named_parameters():
                parameter.requires_grad_(False)

        if not train_semantic:
            for name, parameter in self._semantic.named_parameters():
                parameter.requires_grad_(False)

    @torch.no_grad()
    def forward(
        self,
        visuals: list[tuple[Tensor, list[Tensor], list[tuple[int, int]]]],
        semantics: list[dict[str, Tensor]],
    ):
        visuals = self._visual(visuals)
        semantics = self._semantic(semantics)
        return visuals, semantics
