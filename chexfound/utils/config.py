# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

import math
import logging
import os

import torch
from omegaconf import OmegaConf

import chexfound.distributed as distributed
from chexfound.logging import setup_logging
from chexfound.utils import utils
from chexfound.configs import dinov2_default_config


logger = logging.getLogger("chexfound")


def apply_scaling_rules_to_cfg(cfg):  # to fix
    if cfg.optim.scaling_rule == "sqrt_wrt_1024":
        user_lr = cfg.optim.lr
        if user_lr > 0:
            base_lr = user_lr
            global_size = distributed.get_global_size()
            cfg.optim.lr = base_lr * math.sqrt(
                cfg.train.batch_size_per_gpu * global_size / 1024.0
            )
            logger.info(
                f"sqrt scaling (user-specified base): "
                f"base_lr={base_lr:.6f}, world_size={global_size}, "
                f"batch_per_gpu={cfg.train.batch_size_per_gpu}, new_lr={cfg.optim.lr:.6f}"
            )
        else:
            base_lr = cfg.optim.base_lr
            global_size = distributed.get_global_size()
            cfg.optim.lr = base_lr * math.sqrt(
                cfg.train.batch_size_per_gpu * global_size / 1024.0
            )
            logger.info(
                f"sqrt scaling (default base): "
                f"base_lr={base_lr:.6f}, world_size={global_size}, "
                f"batch_per_gpu={cfg.train.batch_size_per_gpu}, new_lr={cfg.optim.lr:.6f}"
            )
    elif cfg.optim.scaling_rule == "none":
        global_size = distributed.get_global_size()
        logger.info(
            f"LR scaling DISABLED (scaling_rule=none): "
            f"lr={cfg.optim.lr:.6f}, world_size={global_size}, "
            f"batch_per_gpu={cfg.train.batch_size_per_gpu}, "
            f"effective_batch={cfg.train.batch_size_per_gpu * global_size}"
        )
    else:
        raise NotImplementedError
    return cfg


def write_config(cfg, output_dir, name="config.yaml"):
    logger.info(OmegaConf.to_yaml(cfg))
    saved_cfg_path = os.path.join(output_dir, name)
    with open(saved_cfg_path, "w") as f:
        OmegaConf.save(config=cfg, f=f)
    return saved_cfg_path


def get_cfg_from_args(args):
    args.output_dir = os.path.abspath(args.output_dir)
    args.opts = list(args.opts or [])
    args.opts += [f"train.output_dir={args.output_dir}"]
    default_cfg = OmegaConf.create(dinov2_default_config)
    cfg = OmegaConf.load(args.config_file)
    cfg = OmegaConf.merge(default_cfg, cfg, OmegaConf.from_cli(args.opts))
    return cfg


def default_setup(args):
    if not distributed.is_enabled():
        distributed.enable(overwrite=True)
    seed = getattr(args, "seed", 0)
    rank = distributed.get_global_rank()

    global logger
    setup_logging(output=args.output_dir, level=logging.INFO)
    logger = logging.getLogger("chexfound")

    utils.fix_random_seeds(seed + rank)
    logger.info("git:\n  {}\n".format(utils.get_sha()))
    logger.info("\n".join("%s: %s" % (k, str(v)) for k, v in sorted(dict(vars(args)).items())))


def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg_from_args(args)
    os.makedirs(args.output_dir, exist_ok=True)
    default_setup(args)
    apply_scaling_rules_to_cfg(cfg)
    if distributed.is_main_process():
        write_config(cfg, args.output_dir)
    if distributed.is_enabled():
        torch.distributed.barrier()
    return cfg
