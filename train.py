import argparse
import datetime
import json
import math
import random
import sys
import time
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from torch.utils.data import (
    DistributedSampler,
    RandomSampler,
    SequentialSampler,
    BatchSampler,
    DataLoader,
)

import util.misc as utils
from args import get_args_parser
from data import build_dataset
from data.evaluator import DosaEvaluator
from models import build_dosa, build_criterion, build_postprocess


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
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter(
        "lr", utils.SmoothedValue(window_size=1, fmt="{value:.6f}")
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

    metric_logger = utils.MetricLogger(delimiter="  ")
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
    postprocessor = build_postprocess(args)
    model.to(device)
    criterion.to(device)
    postprocessor.to(device)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.gpu]
        )
        model_without_ddp = model.module
    else:
        model_without_ddp = model
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
