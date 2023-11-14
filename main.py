# ------------------------------------------------------------------------
# Plain-DETR
# Copyright (c) 2023 Xi'an Jiaotong University & Microsoft Research Asia.
# Licensed under The MIT License [see LICENSE for details]
# ------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------


import argparse
import datetime
import json
import random
import time
from pathlib import Path
import os

import numpy as np
import torch
from torch.utils.data import DataLoader
import datasets
import util.misc as utils
import datasets.samplers as samplers
from datasets import build_dataset, get_coco_api_from_dataset
from engine import evaluate, train_one_epoch
from models import build_model
from torch import distributed as dist
import wandb


def get_args_parser():
    parser = argparse.ArgumentParser("Deformable DETR Detector", add_help=False)
    parser.add_argument("--lr", default=2e-4, type=float)
    parser.add_argument(
        "--lr_backbone_names", default=["backbone.0"], type=str, nargs="+"
    )
    parser.add_argument("--lr_backbone", default=2e-5, type=float)
    parser.add_argument(
        "--lr_linear_proj_names",
        default=["reference_points", "sampling_offsets"],
        type=str,
        nargs="+",
    )
    parser.add_argument("--lr_linear_proj_mult", default=0.1, type=float)
    parser.add_argument("--batch_size", default=1, type=int)
    parser.add_argument("--weight_decay", default=1e-4, type=float)
    parser.add_argument("--epochs", default=50, type=int)
    parser.add_argument("--lr_drop", default=40, type=int)
    parser.add_argument("--lr_drop_epochs", default=None, type=int, nargs="+")
    parser.add_argument(
        "--clip_max_norm", default=0.1, type=float, help="gradient clipping max norm"
    )
    parser.add_argument("--warmup", default=0, type=int)

    parser.add_argument("--sgd", action="store_true")

    # * Modern DETR tricks
    # Deformable DETR tricks
    parser.add_argument("--with_box_refine", default=False, action="store_true")
    parser.add_argument("--two_stage", default=False, action="store_true")
    # DINO DETR tricks
    parser.add_argument("--mixed_selection", action="store_true", default=False)
    parser.add_argument("--look_forward_twice", action="store_true", default=False)
    # Hybrid Matching tricks
    parser.add_argument("--k_one2many", default=5, type=int)
    parser.add_argument("--lambda_one2many", default=1.0, type=float)
    parser.add_argument(
        "--num_queries_one2one",
        default=300,
        type=int,
        help="Number of query slots for one-to-one matching",
    )
    parser.add_argument(
        "--num_queries_one2many",
        default=0,
        type=int,
        help="Number of query slots for one-to-many matchining",
    )
    # Absolute coordinates & box regression reparameterization
    parser.add_argument(
        "--reparam",
        action="store_true",
        help="If true, we use absolute coordindates & reparameterization for bounding boxes",
    )

    # Model parameters
    parser.add_argument(
        "--frozen_weights",
        type=str,
        default=None,
        help="Path to the pretrained model. If set, only the mask head will be trained",
    )

    # * Backbone
    parser.add_argument(
        "--backbone",
        default="resnet50",
        type=str,
        help="Name of the convolutional backbone to use",
    )
    parser.add_argument(
        "--dilation",
        action="store_true",
        help="If true, we replace stride with dilation in the last convolutional block (DC5)",
    )
    parser.add_argument(
        "--position_embedding",
        default="sine",
        type=str,
        choices=("sine", "learned", "sine_unnorm"),
        help="Type of positional embedding to use on top of the image features",
    )
    parser.add_argument(
        "--position_embedding_scale",
        default=2 * np.pi,
        type=float,
        help="position / size * scale",
    )
    parser.add_argument(
        "--num_feature_levels", default=1, type=int, help="number of feature levels"
    )
    # swin backbone
    parser.add_argument(
        "--pretrained_backbone_path",
        default="./swin_tiny_patch4_window7_224.pkl",
        type=str,
    )
    parser.add_argument("--drop_path_rate", default=0.1, type=float)
    # upsample backbone output features
    parser.add_argument(
        "--upsample_backbone_output",
        action="store_true",
        help="If true, we upsample the backbone output feature to the target stride"
    )
    parser.add_argument(
        "--upsample_stride",
        default=16,
        type=int,
        help="Target stride for upsampling backbone output feature"
    )

    # * Transformer
    parser.add_argument(
        "--dec_layers",
        default=6,
        type=int,
        help="Number of decoding layers in the transformer",
    )
    parser.add_argument(
        "--dim_feedforward",
        default=2048,
        type=int,
        help="Intermediate size of the feedforward layers in the transformer blocks",
    )
    parser.add_argument(
        "--hidden_dim",
        default=256,
        type=int,
        help="Size of the embeddings (dimension of the transformer)",
    )
    parser.add_argument(
        "--dropout", default=0.1, type=float, help="Dropout applied in the transformer"
    )
    parser.add_argument(
        "--nheads",
        default=8,
        type=int,
        help="Number of attention heads inside the transformer's attentions",
    )
    parser.add_argument("--norm_type", default="pre_norm", type=str)
    parser.add_argument("--auto_resume", action="store_true", default=True)
    parser.add_argument("--not_auto_resume", action="store_false", dest="auto_resume")
    # * dev: proposals
    parser.add_argument("--proposal_feature_levels", default=1, type=int)
    parser.add_argument("--proposal_in_stride", default=8, type=int)
    parser.add_argument("--proposal_tgt_strides", default=[8, 16, 32, 64], type=int, nargs="+")
    # * dev decoder: global decoder
    parser.add_argument("--decoder_type", default="deform", type=str)
    parser.add_argument("--decoder_use_checkpoint", default=False, action="store_true")
    parser.add_argument("--decoder_rpe_hidden_dim", default=512, type=int)
    parser.add_argument("--decoder_rpe_type", default="linear", type=str)
    # weight decay mult
    parser.add_argument(
        "--wd_norm_names",
        default=["norm", "bias", "rpb_mlp", "cpb_mlp", "logit_scale", "relative_position_bias_table",
                 "level_embed", "reference_points", "sampling_offsets", "rel_pos"],
        type=str,
        nargs="+"
    )
    parser.add_argument("--wd_norm_mult", default=1.0, type=float)
    parser.add_argument("--use_layerwise_decay", action="store_true", default=False)
    parser.add_argument("--lr_decay_rate", default=1.0, type=float)

    # * Segmentation
    parser.add_argument(
        "--masks",
        action="store_true",
        help="Train segmentation head if the flag is provided",
    )

    # Loss
    parser.add_argument(
        "--no_aux_loss",
        dest="aux_loss",
        action="store_false",
        help="Disables auxiliary decoding losses (loss at each layer)",
    )

    # * Matcher
    parser.add_argument(
        "--set_cost_class",
        default=2,
        type=float,
        help="Class coefficient in the matching cost",
    )
    parser.add_argument(
        "--set_cost_bbox",
        default=5,
        type=float,
        help="L1 box coefficient in the matching cost",
    )
    parser.add_argument(
        "--set_cost_giou",
        default=2,
        type=float,
        help="giou box coefficient in the matching cost",
    )

    # * Loss coefficients
    parser.add_argument("--mask_loss_coef", default=1, type=float)
    parser.add_argument("--dice_loss_coef", default=1, type=float)
    parser.add_argument("--cls_loss_coef", default=2, type=float)
    parser.add_argument("--bbox_loss_coef", default=5, type=float)
    parser.add_argument("--giou_loss_coef", default=2, type=float)
    parser.add_argument("--focal_alpha", default=0.25, type=float)

    # dataset parameters
    parser.add_argument("--dataset_file", default="coco")
    parser.add_argument("--coco_path", default="./data/coco", type=str)
    parser.add_argument("--coco_panoptic_path", type=str)
    parser.add_argument("--remove_difficult", action="store_true")

    parser.add_argument(
        "--output_dir", default="", help="path where to save, empty for no saving"
    )
    parser.add_argument(
        "--device", default="cuda", help="device to use for training / testing"
    )
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--resume", default="", help="resume from checkpoint")
    parser.add_argument(
        "--start_epoch", default=0, type=int, metavar="N", help="start epoch"
    )
    parser.add_argument("--num_workers", default=2, type=int)
    parser.add_argument(
        "--cache_mode",
        default=False,
        action="store_true",
        help="whether to cache images on memory",
    )

    # * eval technologies
    parser.add_argument("--eval", action="store_true")
    # topk for eval
    parser.add_argument("--topk", default=100, type=int)

    # * training technologies
    parser.add_argument("--use_fp16", default=False, action="store_true")
    parser.add_argument("--use_checkpoint", default=False, action="store_true")

    # * logging technologies
    parser.add_argument("--use_wandb", action="store_true", default=False)
    parser.add_argument("--wandb_entity", type=str)
    parser.add_argument("--wandb_name", type=str)
    return parser


def main(args):
    utils.init_distributed_mode(args)
    print("git:\n  {}\n".format(utils.get_sha()))

    if args.frozen_weights is not None:
        assert args.masks, "Frozen training is meant for segmentation only"
    print(args)

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    model, criterion, postprocessors = build_model(args)
    model.to(device)

    model_without_ddp = model
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # print(model_without_ddp)
    print("number of params:", n_parameters)

    if args.use_fp16:
        scaler = torch.cuda.amp.GradScaler()

    dataset_train = build_dataset(image_set="train", args=args)
    dataset_val = build_dataset(image_set="val", args=args)

    if args.distributed:
        if args.cache_mode:
            sampler_train = samplers.NodeDistributedSampler(dataset_train)
            sampler_val = samplers.NodeDistributedSampler(dataset_val, shuffle=False)
        else:
            sampler_train = samplers.DistributedSampler(dataset_train)
            sampler_val = samplers.DistributedSampler(dataset_val, shuffle=False)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    batch_sampler_train = torch.utils.data.BatchSampler(
        sampler_train, args.batch_size, drop_last=True
    )

    data_loader_train = DataLoader(
        dataset_train,
        batch_sampler=batch_sampler_train,
        collate_fn=utils.collate_fn,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    data_loader_val = DataLoader(
        dataset_val,
        args.batch_size,
        sampler=sampler_val,
        drop_last=False,
        collate_fn=utils.collate_fn,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    # lr_backbone_names = ["backbone.0", "backbone.neck", "input_proj", "transformer.encoder"]
    param_dicts = utils.get_param_dict(model_without_ddp, args, use_layerwise_decay=args.use_layerwise_decay)
    if args.sgd:
        optimizer = torch.optim.SGD(
            param_dicts, lr=args.lr, momentum=0.9, weight_decay=args.weight_decay
        )
    else:
        optimizer = torch.optim.AdamW(
            param_dicts, lr=args.lr, weight_decay=args.weight_decay
        )

    # TODO: is there any more elegant way to print the param groups?
    name_dicts = utils.get_param_dict(model_without_ddp, args, return_name=True, use_layerwise_decay=args.use_layerwise_decay)
    if args.use_layerwise_decay:
        for i, name_dict in enumerate(name_dicts):
            print(f"Group-{i} {json.dumps(name_dict, indent=2)}")
    else:
        for i, name_dict in enumerate(name_dicts):
            print(f"Group-{i} lr: {name_dict['lr']} wd: {name_dict['weight_decay']}")
            print(json.dumps(name_dict["params"], indent=2))
    # lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop)
    epoch_iter = len(data_loader_train)
    if args.warmup:
        lambda0 = lambda cur_iter: cur_iter / args.warmup if cur_iter < args.warmup else (0.1 if cur_iter > args.lr_drop * epoch_iter else 1)
    else:
        lambda0 = lambda cur_iter: 0.1 if cur_iter > args.lr_drop * epoch_iter else 1
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda0)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module

    if args.dataset_file == "coco_panoptic":
        # We also evaluate AP during panoptic training, on original coco DS
        coco_val = datasets.coco.build("val", args)
        base_ds = get_coco_api_from_dataset(coco_val)
    else:
        base_ds = get_coco_api_from_dataset(dataset_val)

    if args.frozen_weights is not None:
        checkpoint = torch.load(args.frozen_weights, map_location="cpu")
        model_without_ddp.detr.load_state_dict(checkpoint["model"])

    if args.use_wandb and dist.get_rank() == 0:
        wandb.init(
            entity=args.wandb_entity,
            project='Plain-DETR',
            id=args.wandb_name,  # set id as wandb_name for resume
            name=args.wandb_name,
        )

    output_dir = Path(args.output_dir)
    if args.auto_resume:
        resume_from = utils.find_latest_checkpoint(output_dir)
        if resume_from is not None:
            print(f'Use autoresume, overwrite args.resume with {resume_from}')
            args.resume = resume_from
        else:
            print(f'Use autoresume, but can not find checkpoint in {output_dir}')
    if args.resume and os.path.exists(args.resume):
        if args.resume.startswith("https"):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.resume, map_location="cpu", check_hash=True
            )
        else:
            checkpoint = torch.load(args.resume, map_location="cpu")
        missing_keys, unexpected_keys = model_without_ddp.load_state_dict(
            checkpoint["model"], strict=False
        )
        if len(missing_keys) > 0:
            print("Missing Keys: {}".format(missing_keys))
        if len(unexpected_keys) > 0:
            print("Unexpected Keys: {}".format(unexpected_keys))
        if (
            not args.eval
            and "optimizer" in checkpoint
            and "lr_scheduler" in checkpoint
            and "epoch" in checkpoint
        ):
            import copy

            p_groups = copy.deepcopy(optimizer.param_groups)
            optimizer.load_state_dict(checkpoint["optimizer"])
            for pg, pg_old in zip(optimizer.param_groups, p_groups):
                pg["lr"] = pg_old["lr"]
                pg["initial_lr"] = pg_old["initial_lr"]

            lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
            print(
                # For LambdaLR, the lambda funcs are not been stored in state_dict, see
                # https://pytorch.org/docs/stable/_modules/torch/optim/lr_scheduler.html#LambdaLR.state_dict
                "Warning: lr scheduler has been resumed from checkpoint, but the lambda funcs are not been stored in state_dict. \n"
                "So the new lr schedule would override the resumed lr schedule."
            )
            lr_scheduler.step(lr_scheduler.last_epoch)
            args.start_epoch = checkpoint["epoch"] + 1

            if args.use_fp16 and "scaler" in checkpoint:
                scaler.load_state_dict(checkpoint["scaler"])
        # check the resumed model
        if not args.eval:
            test_stats, coco_evaluator = evaluate(
                model,
                criterion,
                postprocessors,
                data_loader_val,
                base_ds,
                device,
                args.output_dir,
                step=args.start_epoch * len(data_loader_train),
                use_wandb=args.use_wandb,
                reparam=args.reparam,
            )

    if args.eval:
        test_stats, coco_evaluator = evaluate(
            model,
            criterion,
            postprocessors,
            data_loader_val,
            base_ds,
            device,
            args.output_dir,
            step=args.start_epoch * len(data_loader_train),
            use_wandb=args.use_wandb,
            reparam=args.reparam,
        )
        if args.output_dir:
            utils.save_on_master(
                coco_evaluator.coco_eval["bbox"].eval, output_dir / "eval.pth"
            )
        if utils.is_main_process():
            areaRngLbl = ['', '50', '75', 's', 'm', 'l']
            msg = "copypaste: "
            for label in areaRngLbl:
                msg += f"AP{label} "
            for ap in coco_evaluator.coco_eval["bbox"].stats[:len(areaRngLbl)]:
                msg += "{:.3f} ".format(ap)
            print(msg)
        return

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
            lr_scheduler,
            args.clip_max_norm,
            k_one2many=args.k_one2many,
            lambda_one2many=args.lambda_one2many,
            use_wandb=args.use_wandb,
            use_fp16=args.use_fp16,
            scaler=scaler if args.use_fp16 else None,
        )
        if args.output_dir:
            checkpoint_paths = [output_dir / "checkpoint.pth"]
            # extra checkpoint before LR drop and every 5 epochs
            checkpoint_paths.append(output_dir / f"checkpoint{epoch:04}.pth")
            for checkpoint_path in checkpoint_paths:
                save_dict = {
                    "model": model_without_ddp.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "lr_scheduler": lr_scheduler.state_dict(),
                    "epoch": epoch,
                    "args": args,
                }
                if args.use_fp16:
                    save_dict["scaler"] = scaler.state_dict()
                utils.save_on_master(
                    save_dict,
                    checkpoint_path,
                )

        test_stats, coco_evaluator = evaluate(
            model,
            criterion,
            postprocessors,
            data_loader_val,
            base_ds,
            device,
            args.output_dir,
            step=(epoch + 1) * len(data_loader_train),
            use_wandb=args.use_wandb,
            reparam=args.reparam,
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

            # for evaluation logs
            if coco_evaluator is not None:
                (output_dir / "eval").mkdir(exist_ok=True)
                if "bbox" in coco_evaluator.coco_eval:
                    filenames = ["latest.pth"]
                    if epoch % 50 == 0:
                        filenames.append(f"{epoch:03}.pth")
                    for name in filenames:
                        torch.save(
                            coco_evaluator.coco_eval["bbox"].eval,
                            output_dir / "eval" / name,
                        )

                areaRngLbl = ['', '50', '75', 's', 'm', 'l']
                msg = "copypaste: "
                for label in areaRngLbl:
                    msg += f"AP{label} "
                for ap in coco_evaluator.coco_eval["bbox"].stats[:len(areaRngLbl)]:
                    msg += "{:.3f} ".format(ap)
                print(msg)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print("Training time {}".format(total_time_str))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "Deformable DETR training and evaluation script", parents=[get_args_parser()]
    )
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
