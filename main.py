import argparse
import json
import datetime
import math
import sys
from collections import defaultdict
from pathlib import Path
import time
from typing import Iterable
import random

import numpy as np
import torch
from torch.utils.data import (
    RandomSampler,
    SequentialSampler,
    BatchSampler,
    DataLoader,
    DistributedSampler,
)

from data.evaluator import DosaEvaluator
from models import (
    LinearProjection,
    Fuse,
    PositionEncoder,
    Transformer,
    RelationMLP,
    DOSA,
    DOSACriterion,
    PostProcess,
    PreProcessor,
    FPNFeatureMapExtractor,
    SentenceTransformerEmbeder,
)
from data import build_dataset
import util.misc as utils
from util import MetricLogger, SmoothedValue


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


def init_model(args):
    device = torch.device(args.device)

    preprocessor = PreProcessor(
        args.n_sequence,
        FPNFeatureMapExtractor(batch_size=args.visual_batch_size),
        SentenceTransformerEmbeder(batch_size=args.semantic_batch_size),
        args.device,
    )

    model = build_dosa(args)
    model.to(device)
    model.eval()
    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu")
        model.load_state_dict(checkpoint["model"])

    postprocessor = PostProcess()
    postprocessor.to(device)

    return preprocessor, model, postprocessor


def train_one_epoch(
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    data_loader: Iterable,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    max_norm: float = 0,
):
    model.train()
    criterion.train()
    metric_logger = MetricLogger(delimiter="  ")
    metric_logger.add_meter(
        "lr", SmoothedValue(window_size=1, fmt="{value:.6f}")
    )
    header = "Epoch: [{}]".format(epoch)
    print_freq = 10

    for samples, targets in metric_logger.log_every(
        data_loader, print_freq, header
    ):
        samples = {k: v.to(device) for k, v in samples.items()}
        targets = {k: v.to(device) for k, v in targets.items()}

        outputs = model(samples)
        loss_dict = criterion(outputs, targets, samples["masks"])
        losses = sum(
            loss_dict[k] * criterion.loss_weight[k]
            for k in loss_dict.keys()
            if k in criterion.loss_weight
        )

        # reduce losses over all GPUs for logging purposes
        loss_reduced = utils.reduce(loss_dict)
        loss_reduced_scaled = {
            k: v * criterion.loss_weight[k]
            for k, v in loss_reduced.items()
            if k in criterion.loss_weight
        }
        loss_value = sum(loss_reduced_scaled.values()).item()
        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        losses.backward()
        if max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        optimizer.step()

        metric_logger.update(loss=loss_value, **loss_reduced_scaled)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate_one_epoch(model, criterion, postprocessor, data_loader, device):
    model.eval()
    criterion.eval()

    metric_logger = MetricLogger(delimiter="  ")
    header = "Test:"

    evaluator = DosaEvaluator(device)

    for samples, targets in metric_logger.log_every(data_loader, 10, header):
        samples = {k: v.to(device) for k, v in samples.items()}
        targets = {k: v.to(device) for k, v in targets.items()}

        outputs = model(samples)
        loss_dict = criterion(outputs, targets, samples["masks"])

        # reduce losses over all GPUs for logging purposes
        loss_reduced = utils.reduce(loss_dict)
        loss_reduced_scaled = {
            k: v * criterion.loss_weight[k]
            for k, v in loss_reduced.items()
            if k in criterion.loss_weight
        }
        loss_value = sum(loss_reduced_scaled.values()).item()
        metric_logger.update(loss=loss_value, **loss_reduced_scaled)

        results = postprocessor(outputs, samples["masks"])
        evaluator.update(results, targets, samples["masks"])

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    evaluator.synchronize_between_processes()
    evaluator.summarize()

    print("Averaged stats:", metric_logger)
    stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    return stats


def infer(preprocessor, model, postprocessor, samples):
    """
    Process document given all utilities.
    :param preprocessor: preprocess features from multi-feature extractor
    :param model: model
    :param postprocessor: generate output from relation logits
    :param samples: a list of samples, each sample is a list of dict containing:
        image in PIL.Image
        contents per page object, organized per page, list[str]
        bboxes per page object, organized per page, Tensor
        labels per page object, organized per page, Tensor
        fonts per page object, organized per page, Tensor
    :return: results
    """
    preprocessed = defaultdict(list)
    for sample in samples:
        output = preprocessor(sample)
        for key, value in output.items():
            preprocessed[key].append(value)

    batched = {}
    for key, values in preprocessed.items():
        batched[key] = torch.stack(values)

    logits = model(batched)

    return postprocessor(logits, batched["masks"])


def train(args):
    utils.init_distributed_mode(args)

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # load model
    model = build_dosa(args)
    criterion = build_criterion(args)
    postprocessor = PostProcess()
    model.to(device)
    criterion.to(device)
    postprocessor.to(device)

    model_without_ddp = model
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("number of params:", n_parameters)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.gpu]
        )
        model_without_ddp = model.module
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("number of params:", n_parameters)

    dataset_train = build_dataset(image_set="train", args=args)
    dataset_val = build_dataset(image_set="val", args=args)

    if args.distributed:
        # TODO, share data for all processes with one copy
        sampler_train = DistributedSampler(dataset_train)
        sampler_val = DistributedSampler(dataset_val, shuffle=False)
    else:
        sampler_train = RandomSampler(dataset_train)
        sampler_val = SequentialSampler(dataset_val)

    batch_sampler_train = BatchSampler(
        sampler_train, args.batch_size, drop_last=True
    )

    data_loader_train = DataLoader(
        dataset_train,
        batch_sampler=batch_sampler_train,
        collate_fn=utils.collate_fn,
        num_workers=args.num_workers,
    )
    data_loader_val = DataLoader(
        dataset_val,
        args.batch_size,
        sampler=sampler_val,
        drop_last=False,
        collate_fn=utils.collate_fn,
        num_workers=args.num_workers,
    )

    param_dicts = [
        {
            "params": [
                p
                for n, p in model_without_ddp.named_parameters()
                if p.requires_grad
            ]
        },
    ]

    optimizer = torch.optim.AdamW(
        param_dicts, lr=args.lr, weight_decay=args.weight_decay
    )
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.gpu]
        )
        model_without_ddp = model.module

    output_dir = Path(args.output_dir)
    if args.resume:
        if args.resume.startswith("https"):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.resume, map_location="cpu", check_hash=True
            )
        else:
            checkpoint = torch.load(args.resume, map_location="cpu")
        model_without_ddp.load_state_dict(checkpoint["model"])
        if (
            "optimizer" in checkpoint
            and "lr_scheduler" in checkpoint
            and "epoch" in checkpoint
        ):
            optimizer.load_state_dict(checkpoint["optimizer"])
            lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
            args.start_epoch = checkpoint["epoch"] + 1

    print("Start training")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            sampler_train.set_epoch(epoch)
        train_stats = train_one_epoch(
            model,
            criterion,
            data_loader_train,
            optimizer,
            device,
            epoch,
            args.clip_max_norm,
        )
        lr_scheduler.step()
        if args.output_dir:
            checkpoint_paths = [output_dir / "checkpoint.pth"]
            # extra checkpoint before LR drop and every 5 epochs
            if (epoch + 1) % args.lr_drop == 0 or (epoch + 1) % 5 == 0:
                checkpoint_paths.append(
                    output_dir / f"checkpoint{epoch:04}.pth"
                )
            for checkpoint_path in checkpoint_paths:
                utils.save_on_master(
                    {
                        "model": model_without_ddp.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "lr_scheduler": lr_scheduler.state_dict(),
                        "epoch": epoch,
                        "args": args,
                    },
                    checkpoint_path,
                )

        test_stats = evaluate_one_epoch(
            model, criterion, postprocessor, data_loader_val, device
        )

        log_stats = {
            **{f"train_{k}": v for k, v in train_stats.items()},
            **{f"test_{k}": v for k, v in test_stats.items()},
            "epoch": epoch,
            "n_parameters": n_parameters,
        }

        if args.output_dir and utils.is_main_process():
            with (output_dir / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")

        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print("Training time {}".format(total_time_str))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "DOSA training script", parents=[get_args_parser()]
    )
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    else:
        raise ValueError("Missing argument output_dir in training")
    train(args)
