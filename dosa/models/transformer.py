import copy

from torch import nn, Tensor

import dosa.util.misc as utils


class TransformerEncoderLayer(nn.Module):

    def __init__(
        self,
        d_model: int = 1024,
        n_head: int = 8,
        d_hidden: int = 2048,
        dropout: float = 0.1,
        activation: str = "relu",
    ):
        super().__init__()
        self._self_attn = nn.MultiheadAttention(
            d_model, n_head, dropout=dropout
        )
        # Implementation of Feedforward model
        self._linear1 = nn.Linear(d_model, d_hidden)
        self._dropout = nn.Dropout(dropout)
        self._linear2 = nn.Linear(d_hidden, d_model)

        self._norm1 = nn.LayerNorm(d_model)
        self._norm2 = nn.LayerNorm(d_model)
        self._dropout1 = nn.Dropout(dropout)
        self._dropout2 = nn.Dropout(dropout)

        self._activation = utils.get_activation_fn(activation)

    def forward(
        self,
        src: Tensor,
        pos: Tensor,
        padding_mask: Tensor,
    ):
        q = k = src + pos
        src2 = self._self_attn(q, k, value=src, key_padding_mask=padding_mask)[
            0
        ]
        src = src + self._dropout1(src2)
        src = self._norm1(src)
        src2 = self._linear2(
            self._dropout(self._activation(self._linear1(src)))
        )
        src = src + self._dropout2(src2)
        src = self._norm2(src)
        return src


class TransformerEncoder(nn.Module):
    def __init__(self, encoder_layer: TransformerEncoderLayer, n_layers: int):
        super().__init__()
        self._layers = _get_clones(encoder_layer, n_layers)
        self._n_layers = n_layers

    def forward(
        self,
        src: Tensor,
        padding_mask: Tensor = None,
        pos: Tensor = None,
    ):
        output = src

        for layer in self._layers:
            output = layer(output, pos=pos, padding_mask=padding_mask)

        return output


class Transformer(nn.Module):

    def __init__(
        self,
        d_model: int = 1024,
        n_heads: int = 8,
        n_enc_layers: int = 6,
        d_hidden: int = 2048,
        dropout: float = 0.1,
        activation: str = "relu",
    ):
        super().__init__()

        encoder_layer = TransformerEncoderLayer(
            d_model, n_heads, d_hidden, dropout, activation
        )
        self._encoder = TransformerEncoder(encoder_layer, n_enc_layers)

        self._reset_parameters()

        self._d_model = d_model
        self._n_head = n_heads

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src: Tensor, pos_embed: Tensor, padding_mask: Tensor):
        """
        forward expects three Tensors
        :param src: the fused page object representation of shape,
            [batch_size, n_sequence, d_model]
        :param pos_embed: of shape [batch_size, n_sequence, d_model]
        :param padding_mask: of shape [batch_size, n_sequence]
        :return: enhanced representation of shape
            [batch_size, n_sequence, d_model]
        """
        # transpose from batch_size x n_sequence x d_model into
        # n_sequence x batch_size x d_model
        src = src.permute(1, 0, 2)
        pos_embed = pos_embed.permute(1, 0, 2)

        output = self._encoder(src, pos=pos_embed, padding_mask=padding_mask)
        return output.permute(1, 0, 2)


def _get_clones(module: nn.Module, copies: int):
    return nn.ModuleList([copy.deepcopy(module) for i in range(copies)])
