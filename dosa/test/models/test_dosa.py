import math

import torch
from transformers import AutoTokenizer

from dosa.core.sample import Samples
from dosa.models import Backbone, FeatureMapExtractor, SemanticEmbeder
from dosa.models.fuse import LinearProjection, Fuse
from dosa.models.dosa import DOSA, DOSACriterion, PostProcess
from dosa.models.position_encoding import PositionEncoder
from dosa.models.relation import RelationMLP
from dosa.models.transformer import Transformer


def test_dosa():
    backbone = Backbone(
        FeatureMapExtractor(256),
        SemanticEmbeder(256, "sentence-transformers/all-MiniLM-L6-v2"),
        False,
        False,
    )
    fuse = Fuse(
        visual=LinearProjection(12544, 512),
        semantic=LinearProjection(384, 256),
        measurement=LinearProjection(3, 128),
        category=LinearProjection(1, 128),
        d_fused=1024,
        d_hidden=2048,
        d_out=512,
        dropout=0.1,
        activation="relu",
    )
    position = PositionEncoder(5, 512)
    transformer = Transformer(d_model=512)
    relations = [RelationMLP(1024, 2048, 1, 3) for _ in range(2)]
    dosa = DOSA(backbone, fuse, position, transformer, *relations)

    tokenizer = AutoTokenizer.from_pretrained(
        "sentence-transformers/all-MiniLM-L6-v2"
    )
    semantic = tokenizer(["hello, world!"], padding=True, return_tensors="pt")
    semantics = [
        {key: value.repeat(100, 1) for key, value in semantic.items()}
    ] * 4
    visuals = [
        (
            torch.rand(5, 3, 1024, 1024),
            [torch.rand(20, 4) * 512] * 5,
            [(1024, 1024)] * 5,
        )
        for _ in range(4)
    ]
    samples = Samples(
        visuals,
        semantics,
        torch.rand(4, 256, 3),
        torch.rand(4, 256, 1),
        torch.rand(4, 256, 5),
        torch.rand(4, 256),
    )
    result = dosa(samples)
    assert len(result) == 3
    for k, r in result.items():
        if k != "masks":
            assert r.shape == (4, 256, 256)


def test_postprocess():
    lengths = torch.tensor([8, 3, 16])
    masks = torch.stack(
        [
            torch.arange(16) >= lengths[0],
            torch.arange(16) >= lengths[1],
            torch.arange(16) >= lengths[2],
        ]
    )
    logits = {
        "parent": torch.rand(3, 16, 16),
        "sibling": torch.rand(3, 16, 16),
        "continuation": torch.rand(3, 16, 16),
        "masks": masks,
    }
    postprocess = PostProcess()
    results = postprocess(logits)
    for result, length in zip(results, lengths):
        assert result["parent"]["labels"].shape[0] == length
        assert result["parent"]["scores"].shape[0] == length
        assert result["sibling"]["labels"].shape[0] == length
        assert result["sibling"]["scores"].shape[0] == length
        assert result["continuation"]["labels"].shape[0] == length
        assert result["continuation"]["scores"].shape[0] == length


def test_relation_loss():
    torch.manual_seed(0)
    logits = torch.rand(3, 16, 16, requires_grad=True)
    targets = torch.stack(
        [
            torch.randint(0, 8, (16,)),
            torch.randint(0, 12, (16,)),
            torch.randint(0, 16, (16,)),
        ]
    )
    masks = torch.stack(
        [
            torch.arange(16) >= 8,
            torch.arange(16) >= 12,
            torch.arange(16) >= 16,
        ]
    )
    criterion = DOSACriterion(
        16, {"parent": 1, "sibling": 1, "continuation": 1}, 0.25
    )
    loss = criterion.focal_relation_loss(logits, targets, masks)
    assert math.isclose(loss.item(), 0.2788, rel_tol=1e-03)
    loss.backward()


def test_dosa_criterion():
    torch.manual_seed(0)
    # construct variable length of sequences
    masks = torch.stack(
        [
            torch.arange(16) >= 8,
            torch.arange(16) >= 12,
            torch.arange(16) >= 16,
        ]
    )

    logits = {
        "parent": torch.rand(3, 16, 16, requires_grad=True),
        "sibling": torch.rand(3, 16, 16, requires_grad=True),
        "masks": masks,
    }
    targets = {
        "parent": torch.stack(
            [
                torch.randint(0, 8, (16,)),
                torch.randint(0, 12, (16,)),
                torch.randint(0, 16, (16,)),
            ]
        ),
        "sibling": torch.stack(
            [
                torch.randint(0, 8, (16,)),
                torch.randint(0, 12, (16,)),
                torch.randint(0, 16, (16,)),
            ]
        ),
    }

    criterion = DOSACriterion(16, {"parent": 1, "sibling": 1}, 0.1)
    loss = criterion(logits, targets)
    total = torch.stack(list(loss.values())).sum()
    assert math.isclose(total.item(), 0.6567, rel_tol=1e-03)
    total.backward()
