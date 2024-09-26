import argparse


def get_args_parser():
    # Hyper
    parser = argparse.ArgumentParser("Set transformer detector", add_help=False)
    # TODO, potentially learning rate for different layers
    parser.add_argument("--lr", default=2e-4, type=float)
    parser.add_argument("--batch_size", default=2, type=int)
    parser.add_argument("--weight_decay", default=1e-4, type=float)
    parser.add_argument("--epochs", default=50, type=int)
    parser.add_argument("--lr_drop", default=40, type=int)
    parser.add_argument(
        "--clip_max_norm",
        default=0.1,
        type=float,
        help="gradient clipping max norm",
    )

    # PreProcessor
    parser.add_argument(
        "--visual_batch_size",
        default=2,
        type=int,
        help="batch size for visual feature extraction and ROI",
    )
    parser.add_argument(
        "--semantic_batch_size",
        default=2,
        type=int,
        help="batch size for semantic embedding",
    )

    # Fuse
    parser.add_argument(
        "--d_semantic_in",
        default=384,
        type=int,
        help="Dimension of semantic input embedding",
    )
    parser.add_argument(
        "--d_semantic_out",
        default=256,
        type=int,
        help="Dimension of semantic output embedding",
    )
    parser.add_argument(
        "--d_visual_in",
        default=12544,
        type=int,
        help="Dimension of visual input embedding",
    )
    parser.add_argument(
        "--d_visual_out",
        default=768,
        type=int,
        help="Dimension of visual output embedding",
    )
    parser.add_argument(
        "--d_dimension_in",
        default=4,
        type=int,
        help="Dimension of visual input embedding",
    )
    parser.add_argument(
        "--d_dimension_out",
        default=256,
        type=int,
        help="Dimension of visual output embedding",
    )
    parser.add_argument(
        "--d_fuse_hidden",
        default=2048,
        type=int,
        help="Dimension of visual hidden embedding",
    )
    parser.add_argument(
        "--fuse_activation",
        default="relu",
        type=str,
    )

    # Position
    parser.add_argument(
        "--d_position_in",
        default=5,
        type=int,
        help="Dimension of position",
    )

    # Transformer
    parser.add_argument(
        "--n_enc_layers",
        default=6,
        type=int,
        help="Number of encoding layers in the transformer",
    )
    parser.add_argument(
        "--n_sequence",
        default=256,
        type=int,
        help="Sequence length in transformer",
    )
    parser.add_argument(
        "--d_hidden",
        default=2048,
        type=int,
        help="Embedding size of the hidden layer in the transformer blocks",
    )
    parser.add_argument(
        "--d_model",
        default=256,
        type=int,
        help="Size of the embeddings (dimension of the transformer)",
    )
    parser.add_argument(
        "--dropout",
        default=0.1,
        type=float,
        help="Dropout applied in the transformer",
    )
    parser.add_argument(
        "--n_heads",
        default=8,
        type=int,
        help="Number of attention heads inside the transformer's attentions",
    )
    parser.add_argument(
        "--transformer_activation",
        default="relu",
        type=str,
    )

    # RelationMLP
    parser.add_argument(
        "--d_relation_hidden",
        default=2048,
        type=int,
        help="Embedding size of the hidden layer in the transformer blocks",
    )
    parser.add_argument(
        "--relation_activation",
        default="relu",
        type=str,
    )

    # Loss coefficients
    parser.add_argument("--parent_loss", default=1, type=float)
    parser.add_argument("--sibling_loss", default=1, type=float)
    parser.add_argument("--continuation_loss", default=1, type=float)
    parser.add_argument(
        "--sco",
        default=0.1,
        type=float,
        help="Relative classification weight for self continuation object",
    )

    # Train
    parser.add_argument("--data_path", default="./dataset/hrdoc/", type=str)
    parser.add_argument(
        "--output_dir",
        default="",
        help="path where to save, empty for no saving",
    )
    parser.add_argument(
        "--device", default="cuda", help="device to use for training / testing"
    )
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--resume", default="", help="resume from checkpoint")
    parser.add_argument(
        "--start_epoch", default=0, type=int, metavar="N", help="start epoch"
    )
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--num_workers", default=2, type=int)
    parser.add_argument(
        "--world_size",
        default=1,
        type=int,
        help="number of distributed processes",
    )
    parser.add_argument(
        "--dist_url",
        default="env://",
        help="url used to set up distributed training",
    )
    parser.add_argument(
        "--model",
        default="structure",
        type=str,
        help="DOSA sub-model name: structure or relation",
    )

    return parser
