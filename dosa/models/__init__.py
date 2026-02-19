from dosa.models.backbone import Backbone, FeatureMapExtractor, SemanticEmbeder
from dosa.models.dosa import DOSA, DOSACriterion, PostProcess
from dosa.models.fuse import LinearProjection, Fuse
from dosa.models.preprocess import PreProcessor
from dosa.models.position_encoding import PositionEncoder
from dosa.models.relation import RelationMLP
from dosa.models.transformer import (
    TransformerEncoderLayer,
    TransformerEncoder,
    Transformer,
)


def build_preprocessor(args):
    preprocessor = PreProcessor(args.n_sequence, args.semantic_model_name)
    return preprocessor


def build_dosa(args):
    backbone = Backbone(
        FeatureMapExtractor(args.n_sequence),
        SemanticEmbeder(args.n_sequence, args.semantic_model_name),
        args.lr_visual > 0,
        args.lr_semantic > 0,
    )
    visual = LinearProjection(
        d_in=args.d_visual_in,
        d_out=args.d_visual_out,
        dropout=args.dropout,
    )
    semantic = LinearProjection(
        d_in=args.d_semantic_in, d_out=args.d_semantic_out
    )
    category = LinearProjection(
        d_in=args.d_category_in, d_out=args.d_category_out
    )
    d_fused = (
        args.d_visual_out
        + args.d_semantic_out
        + args.d_category_out
    )
    fuse = Fuse(
        visual=visual,
        semantic=semantic,
        category=category,
        d_fused=d_fused,
        d_hidden=args.d_fuse_hidden,
        d_out=args.d_model,
        dropout=args.dropout,
        activation=args.fuse_activation,
    )
    position_encoding = PositionEncoder(args.d_position_in, args.d_model)
    transformer = Transformer(
        d_model=args.d_model,
        dropout=args.dropout,
        n_heads=args.n_heads,
        d_hidden=args.d_hidden,
        n_enc_layers=args.n_enc_layers,
        activation=args.transformer_activation,
    )
    relations = [
        RelationMLP(
            args.d_model * 2,
            args.d_relation_hidden,
            1,
            2,
            args.relation_activation,
        )
        for _ in range(2)
    ]
    return DOSA(backbone, fuse, position_encoding, transformer, *relations)


def build_criterion(args):
    loss_weight = {
        "parent": args.parent_loss,
        "sibling": args.sibling_loss,
        "continuation": args.continuation_loss,
    }
    criterion = DOSACriterion(args.n_sequence, loss_weight, args.focal_alpha)
    return criterion


def build_postprocess(args):
    postprocessor = PostProcess()
    return postprocessor


__all__ = [
    "build_preprocessor",
    "build_dosa",
    "build_criterion",
    "build_postprocess",
    "Backbone",
    "FeatureMapExtractor",
    "SemanticEmbeder",
    "PreProcessor",
    "LinearProjection",
    "Fuse",
    "PositionEncoder",
    "RelationMLP",
    "TransformerEncoderLayer",
    "TransformerEncoder",
    "Transformer",
    "DOSA",
    "DOSACriterion",
    "PostProcess",
]
