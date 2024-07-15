import torch

from models.transformer import TransformerEncoderLayer, Transformer


def test_encoder_layer():
    layer = TransformerEncoderLayer()

    src = torch.rand(256, 3, 1024)
    pos_embed = torch.rand(256, 3, 1024)
    mask = torch.rand(3, 256) > 0.5

    results = layer(src, pos_embed, mask)
    assert results.shape == (256, 3, 1024)


def test_transformer():
    transformer = Transformer()

    src = torch.rand(3, 256, 1024)
    pos_embed = torch.rand(3, 256, 1024)
    mask = torch.rand(3, 256) > 0.5

    results = transformer(src, pos_embed, mask)
    assert results.shape == (3, 256, 1024)
