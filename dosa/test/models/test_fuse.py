import torch

from dosa.models import LinearProjection, Fuse


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


def test_measurement():
    measurement = LinearProjection(3, 256)
    embeddings = torch.rand(2, 128, 3)
    result = measurement(embeddings)
    assert result.shape == (2, 128, 256)


def test_category():
    category = LinearProjection(1, 128)
    embeddings = torch.rand(2, 128, 1)
    result = category(embeddings)
    assert result.shape == (2, 128, 128)


def test_fuse():
    visual = LinearProjection(12544, 512)
    semantic = LinearProjection(384, 256)
    measurement = LinearProjection(4, 128)
    category = LinearProjection(1, 128)
    fuse = Fuse(
        visual,
        semantic,
        measurement,
        category,
        d_fused=1024,
        d_hidden=2048,
        d_out=512,
        dropout=0.1,
        activation="relu",
    )

    e_visual = torch.rand(2, 64, 256, 7, 7)
    e_semantic = torch.rand(2, 64, 384)
    e_measurement = torch.rand(2, 64, 4)
    e_category = torch.rand(2, 64, 1)
    result = fuse(e_visual, e_semantic, e_measurement, e_category)
    assert result.shape == (2, 64, 512)
