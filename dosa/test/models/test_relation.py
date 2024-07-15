import torch

from dosa.models.relation import RelationMLP


def test_relation():
    relation = RelationMLP(1024, 2048, 1, 3)
    enhanced = torch.rand(5, 256, 512)
    result = relation(enhanced)
    assert result.shape == (5, 256, 256)
