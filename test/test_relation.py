import torch

from models.relation import RelationMLP


def test_relation():
    relation = RelationMLP(2048, 2048, 1, 3)
    enhanced = torch.rand(5, 256, 1024)
    result = relation(enhanced)
    assert result.shape == (5, 256, 256)
