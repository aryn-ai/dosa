import torch
from torch import Tensor

import dosa.util.misc as utils


class RelationMLP(torch.nn.Module):

    def __init__(
        self,
        d_in: int,
        d_hidden: int,
        d_out: int,
        n_layers: int,
        activation: str = "relu",
    ):
        super().__init__()
        self._n_layers = n_layers
        h = [d_hidden] * (n_layers - 1)
        self._layers = torch.nn.ModuleList(
            torch.nn.Linear(n, k) for n, k in zip([d_in] + h, h + [d_out])
        )
        self._activation = activation

    def forward(self, enhanced: Tensor):
        """
        The forward expects a batch of enhanced representations, do pairwise
        concatenation to get the relation embedding. Finally, MLP is exploited
        to convert relation embedding into logits.

        :param enhanced: of shape [batch_size, n_sequence, d_model]
        :return: relation logits of shape [batch_size, n_sequence, n_sequence]
        """
        batch_size, n_sequence, d_model = enhanced.shape
        expanded2 = enhanced.unsqueeze(2).expand(
            batch_size, n_sequence, n_sequence, d_model
        )
        expanded1 = enhanced.unsqueeze(1).expand(
            batch_size, n_sequence, n_sequence, d_model
        )
        relation = torch.cat((expanded1, expanded2), dim=-1)

        activation = utils.get_activation_fn(self._activation)
        for i, layer in enumerate(self._layers):
            relation = (
                activation(layer(relation))
                if i < self._n_layers - 1
                else layer(relation)
            )
        output = relation.squeeze(dim=-1)
        return output
