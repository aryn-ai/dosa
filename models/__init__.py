from models.fuse import LinearProjection, Fuse
from models.preprocess import (
    FPNFeatureMapExtractor,
    SentenceTransformerEmbeder,
    PreProcessor,
)
from models.position_encoding import PositionEncoder
from models.relation import RelationMLP
from models.transformer import (
    TransformerEncoderLayer,
    TransformerEncoder,
    Transformer,
)
from models.dosa import DOSA, DOSACriterion, PostProcess


def build_preprocessor(args):
    preprocessor = PreProcessor(
        args.n_sequence,
        FPNFeatureMapExtractor(batch_size=args.visual_batch_size),
        SentenceTransformerEmbeder(batch_size=args.semantic_batch_size),
        args.device,
    )
    return preprocessor


def build_dosa(args):
    visual = LinearProjection(
        d_in=args.d_visual_in,
        d_out=args.d_visual_out,
        dropout=args.dropout,
    )
    semantic = LinearProjection(
        d_in=args.d_semantic_in, d_out=args.d_semantic_out
    )
    dimension = LinearProjection(
        d_in=args.d_dimension_in, d_out=args.d_dimension_out
    )
    backbone = Fuse(
        visual=visual,
        semantic=semantic,
        dimension=dimension,
        d_fused=args.d_visual_out + args.d_semantic_out + args.d_dimension_out,
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
            3,
            args.relation_activation,
        )
        for _ in range(3)
    ]
    return DOSA(backbone, position_encoding, transformer, *relations)


def build_criterion(args):
    loss_weight = {
        "parent": args.parent_loss,
        "sibling": args.sibling_loss,
        "continuation": args.continuation_loss,
    }
    criterion = DOSACriterion(args.n_sequence, loss_weight, args.sco)
    return criterion


def build_postprocess(args):
    postprocessor = PostProcess()
    return postprocessor


__all__ = [
    "build_preprocessor",
    "build_dosa",
    "build_criterion",
    "build_postprocess",
    "FPNFeatureMapExtractor",
    "SentenceTransformerEmbeder",
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
