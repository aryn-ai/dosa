import torch
from torch import nn
from torch import Tensor

import dosa.util.misc as utils


class LinearProjection(nn.Module):
    def __init__(self, d_in: int, d_out: int, dropout: float = 0.1):
        super().__init__()
        self._fc = nn.Linear(d_in, d_out)
        # TODO, should remove the dropout here
        self._dropout = nn.Dropout(dropout)
        self._norm = nn.LayerNorm(d_out)

    def forward(self, src: Tensor):
        src = src.flatten(start_dim=2, end_dim=-1)
        src = self._dropout(self._fc(src))
        src = self._norm(src)

        return src


class Fuse(nn.Module):
    """
    Concatenate visual and semantic embedding together and use a two layer MLP
    to learn the fused representation of page objects.
    """

    def __init__(
        self,
        visual: LinearProjection,
        semantic: LinearProjection,
        category: LinearProjection,
        d_fused: int,
        d_hidden: int,
        d_out: int,
        dropout: float,
        activation: str = "relu",
    ):
        super().__init__()
        self._visual = visual
        self._semantic = semantic
        self._category = category

        self._linear1 = nn.Linear(d_fused, d_hidden)
        self._activation = utils.get_activation_fn(activation)
        self._dropout1 = nn.Dropout(dropout)
        self._linear2 = nn.Linear(d_hidden, d_out)
        self._dropout2 = nn.Dropout(dropout)
        self._norm = nn.LayerNorm(d_out)

    def forward(
        self,
        visual: Tensor,
        semantic: Tensor,
        category: Tensor,
    ) -> Tensor:
        """
        The forward expects visual appearance features and content semantic
        embedding for a batch of samples, each sample is a sequence of page
        objects
        :param visual: extracted from a ROI of shape
            [batch_size, n_sequence, 12544]
        :param semantic: of shape [batch_size, n_sequence, n_embedding], n_embedding is the
            dimension of a language embedder output
        :param measurement: of shape [batch_size, n_sequence, 3], 3 includes
            [w, h, area]
        :param category
        :return: a fused backbone representation for page objects of shape
            [batch_size, n_sequence, 1024]
        """
        visual_features = self._visual(visual)
        semantic_features = self._semantic(semantic)
        category_features = self._category(category)

        fused = torch.concat(
            [
                visual_features,
                semantic_features,
                category_features,
            ],
            dim=2,
        )

        out = self._linear2(
            self._dropout1(self._activation(self._linear1(fused)))
        )
        out = out + self._dropout2(out)
        out = self._norm(out)

        return out
