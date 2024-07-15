import torch

from models.fuse import LinearProjection, Fuse
from models.dosa import DOSA, DOSACriterion, PostProcess
from models.position_encoding import PositionEncoder
from models.relation import RelationMLP
from models.transformer import Transformer


def test_dosa():
    backbone = Fuse(
        visual=LinearProjection(12544, 1024),
        semantic=LinearProjection(384, 256),
        dimension=LinearProjection(4, 256),
        d_fused=1536,
        d_hidden=2048,
        d_out=1024,
        dropout=0.1,
        activation="relu",
    )
    position = PositionEncoder(5, 1024)
    transformer = Transformer()
    relations = [RelationMLP(2048, 2048, 1, 3) for _ in range(3)]
    dosa = DOSA(backbone, position, transformer, *relations)
    samples = {
        "visuals": torch.rand(4, 256, 12544),
        "semantics": torch.rand(4, 256, 384),
        "dimensions": torch.rand(4, 256, 4),
        "positions": torch.rand(4, 256, 5),
        "masks": torch.rand(4, 256),
    }
    result = dosa(samples)
    assert len(result) == 3
    for k, r in result.items():
        assert r.shape == (4, 256, 256)


def test_postprocess():
    logits = {
        "parent": torch.rand(3, 16, 16),
        "sibling": torch.rand(3, 16, 16),
        "continuation": torch.rand(3, 16, 16),
    }
    lengths = torch.randint(1, 16, (3,))
    masks = torch.stack(
        [
            torch.arange(16) >= lengths[0],
            torch.arange(16) >= lengths[1],
            torch.arange(16) >= lengths[2],
        ]
    )
    postprocess = PostProcess()
    results = postprocess(logits, masks)
    for result, length in zip(results, lengths):
        assert result["parent"]["labels"].shape[0] == length
        assert result["parent"]["scores"].shape[0] == length
        assert result["sibling"]["labels"].shape[0] == length
        assert result["sibling"]["scores"].shape[0] == length
        assert result["continuation"]["labels"].shape[0] == length
        assert result["continuation"]["scores"].shape[0] == length


def test_dosa_criterion():
    logits = {
        "parent": torch.rand(3, 256, 256, requires_grad=True),
        "sibling": torch.rand(3, 256, 256, requires_grad=True),
        "continuation": torch.rand(3, 256, 256, requires_grad=True),
    }
    targets = {
        "parent": torch.randint(0, 200, (3, 256)),
        "sibling": torch.randint(0, 200, (3, 256)),
        "continuation": torch.randint(0, 200, (3, 256)),
    }
    # construct variable length of sequences
    masks = torch.stack(
        [
            torch.arange(256) > 200,
            torch.arange(256) > 200,
            torch.arange(256) > 200,
        ]
    )

    criterion = DOSACriterion(
        256, {"parent": 1, "sibling": 1, "continuation": 1}
    )
    loss = criterion(logits, targets, masks)
    total = torch.stack(list(loss.values())).sum()
    total.backward()
