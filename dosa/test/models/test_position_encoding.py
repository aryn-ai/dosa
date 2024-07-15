import torch

from dosa.models.position_encoding import PositionEncoder


def test_position_embedding():
    encoder = PositionEncoder(7, 1024)

    positions = torch.rand(3, 256, 7)
    results = encoder(positions)
    assert results.shape == (3, 256, 1024)
