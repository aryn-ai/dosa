from torch import nn, Tensor


class PositionEncoder(nn.Module):
    def __init__(self, d_in: int, d_out: int):
        super().__init__()
        self._fc = nn.Linear(d_in, d_out)

    def forward(self, positions: Tensor):
        """
        The forward expects a position Tensor
        :param positions: of shape [batch_size x n_sequence x 5], 5 is
            [page_no/page_count, x1/width, y1/height, x2/width, y2/height]
        :return: the position embedding of shape
            [batch_size, n_sequence, d_model], here d_model is same as
            transformer model dimension
        """
        position_encodings = self._fc(positions)
        return position_encodings
