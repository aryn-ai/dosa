import torch

from models import LinearProjection, Fuse


def test_visual():
    visual = LinearProjection(12544, 1024)
    features = torch.rand(2, 128, 12544)
    result = visual(features)
    assert result.shape == (2, 128, 1024)


def test_semantic():
    semantic = LinearProjection(384, 256)
    embeddings = torch.rand(2, 128, 384)
    result = semantic(embeddings)
    assert result.shape == (2, 128, 256)


def test_dimension():
    dimension = LinearProjection(4, 256)
    embeddings = torch.rand(2, 128, 4)
    result = dimension(embeddings)
    assert result.shape == (2, 128, 256)


def test_fuse():
    visual = LinearProjection(12544, 1024)
    semantic = LinearProjection(384, 256)
    dimension = LinearProjection(4, 256)
    fuse = Fuse(
        visual,
        semantic,
        dimension,
        d_fused=1536,
        d_hidden=2048,
        d_out=1024,
        dropout=0.1,
        activation="relu",
    )

    e_visual = torch.rand(2, 64, 256, 7, 7)
    e_semantic = torch.rand(2, 64, 384)
    e_dimension = torch.rand(2, 64, 4)
    result = fuse(e_visual, e_semantic, e_dimension)
    assert result.shape == (2, 64, 1024)
