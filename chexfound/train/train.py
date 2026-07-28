# chexfound/train/train.py
#
# ============================================================
# 改动说明（SAM 引导混合掩码版）：
#
#   新增：
#     (A) 导入 DataAugmentationDINOWithSAM / SAMGuidedMaskGenerator /
#              build_lumbar_sam_dataset
#     (B) 初始化 SAMGuidedMaskGenerator（grid=40×40，SAM 比例 60%）
#     (C) make_dataset 时先以 transform=None 获取裸数据集，
#         再用 build_lumbar_sam_dataset 包装（注入 SAM 掩码加载 +
#         DataAugmentationDINOWithSAM 同步增强）
#     (D) collate_fn 额外传入 sam_guided_mask_generator
#     (E) 训练循环：每个 epoch 切换时调用
#              sam_mask_generator.set_epoch(current_epoch)
#         以触发动态难度调整（线性升概率）
#
#   保留：
#     · 原有 MaskingGenerator 实例（随机块回退 / 40% 迭代步使用）
#     · DataAugmentationDINO 不删除（代码保留，但 do_train 切换为 WithSAM 版本）
#     · 向后兼容：若 cfg.train.sam_masks_dir 未配置，
#       SAMDatasetWrapper 的 sam_masks_dir=None，等效原始训练行为
#
# 配置新增（YAML 可选字段）：
#   train:
#     sam_masks_dir: "/path/to/sam_masks"   # 预计算 SAM 分割图目录
#                                           # 省略或空字符串 → 原始随机掩码
#   ibot:
#     sam_ratio: 0.6         # SAM 策略迭代步占比（默认 0.6）
#     sam_warmup_ratio: 0.7  # 动态难度阶段占比（默认 0.7）
# ============================================================

import argparse
import logging
import math
import os
from functools import partial

from fvcore.common.checkpoint import Checkpointer, PeriodicCheckpointer
import torch

from chexfound.data import SamplerType, make_data_loader, make_dataset
from chexfound.data import collate_data_and_cast, DataAugmentationDINO, MaskingGenerator
from chexfound.data.augmentations import DataAugmentationDINOWithSAM       # ← (A) 新增
from chexfound.data.sam_masking import SAMGuidedMaskGenerator              # ← (A) 新增
from chexfound.data.lumbar_dataset import (
    build_lumbar_sam_dataset,
    build_lumbar_sam_dataset_from_images,
)
import chexfound.distributed as distributed
from chexfound.fsdp import FSDPCheckpointer
from chexfound.logging import MetricLogger
from chexfound.utils.config import setup
from chexfound.utils.utils import CosineScheduler
from chexfound.train.ssl_meta_arch import SSLMetaArch
from chexfound import utils

from torch.distributed.elastic.multiprocessing.errors import record
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.benchmark = True
logger = logging.getLogger("chexfound")


# ─────────────────────────────────────────────────────────────────────────────
# CLI 参数（与原版完全相同）
# ─────────────────────────────────────────────────────────────────────────────

def get_args_parser(add_help: bool = True):
    parser = argparse.ArgumentParser("DINOv2 training", add_help=add_help)
    parser.add_argument("--config-file", default="", metavar="FILE",
                        help="path to config file")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--eval", type=str, default="")
    parser.add_argument(
        "opts", default=None, nargs=argparse.REMAINDER,
        help='Modify config options (e.g. "path.key=value")',
    )
    parser.add_argument("--output-dir", "--output_dir", default="", type=str)
    return parser


# ─────────────────────────────────────────────────────────────────────────────
# 优化器 & 调度器（与原版完全相同）
# ─────────────────────────────────────────────────────────────────────────────

def build_optimizer(cfg, params_groups):
    return torch.optim.AdamW(
        params_groups,
        betas=(cfg.optim.adamw_beta1, cfg.optim.adamw_beta2),
    )


def build_schedulers(cfg):
    OFFICIAL_EPOCH_LENGTH = cfg.train.OFFICIAL_EPOCH_LENGTH
    print(f"Debug build_schedulers - epochs: {cfg.optim.epochs}, "
          f"OFFICIAL_EPOCH_LENGTH: {OFFICIAL_EPOCH_LENGTH}")
    print(f"Debug - warmup_epochs: {cfg.optim.warmup_epochs}")

    lr = dict(
        base_value         = cfg.optim["lr"],
        final_value        = cfg.optim["min_lr"],
        total_iters        = cfg.optim["epochs"] * OFFICIAL_EPOCH_LENGTH,
        warmup_iters       = cfg.optim["warmup_epochs"] * OFFICIAL_EPOCH_LENGTH,
        start_warmup_value = 0,
    )
    wd = dict(
        base_value  = cfg.optim["weight_decay"],
        final_value = cfg.optim["weight_decay_end"],
        total_iters = cfg.optim["epochs"] * OFFICIAL_EPOCH_LENGTH,
    )
    momentum = dict(
        base_value  = cfg.teacher["momentum_teacher"],
        final_value = cfg.teacher["final_momentum_teacher"],
        total_iters = cfg.optim["epochs"] * OFFICIAL_EPOCH_LENGTH,
    )
    teacher_temp = dict(
        base_value         = cfg.teacher["teacher_temp"],
        final_value        = cfg.teacher["teacher_temp"],
        total_iters        = cfg.teacher["warmup_teacher_temp_epochs"] * OFFICIAL_EPOCH_LENGTH,
        warmup_iters       = cfg.teacher["warmup_teacher_temp_epochs"] * OFFICIAL_EPOCH_LENGTH,
        start_warmup_value = cfg.teacher["warmup_teacher_temp"],
    )

    lr_schedule            = CosineScheduler(**lr)
    wd_schedule            = CosineScheduler(**wd)
    momentum_schedule      = CosineScheduler(**momentum)
    teacher_temp_schedule  = CosineScheduler(**teacher_temp)
    last_layer_lr_schedule = CosineScheduler(**lr)

    last_layer_lr_schedule.schedule[
        : cfg.optim["freeze_last_layer_epochs"] * OFFICIAL_EPOCH_LENGTH
    ] = 0

    logger.info("Schedulers ready.")
    return (
        lr_schedule,
        wd_schedule,
        momentum_schedule,
        teacher_temp_schedule,
        last_layer_lr_schedule,
    )


def apply_optim_scheduler(optimizer, lr, wd, last_layer_lr):
    for param_group in optimizer.param_groups:
        is_last_layer  = param_group["is_last_layer"]
        lr_multiplier  = param_group["lr_multiplier"]
        wd_multiplier  = param_group["wd_multiplier"]
        param_group["weight_decay"] = wd * wd_multiplier
        param_group["lr"] = (last_layer_lr if is_last_layer else lr) * lr_multiplier


def clip_model_gradients(model, max_norm):
    modules = list(model.student.values())
    if getattr(model, "is_ddn", False):
        modules.extend(model.teacher.values())
        modules.append(model.decoder)
    for module in modules:
        if hasattr(module, "clip_grad_norm_"):
            module.clip_grad_norm_(max_norm)
        else:
            torch.nn.utils.clip_grad_norm_(module.parameters(), max_norm)


# ─────────────────────────────────────────────────────────────────────────────
# 评估 & checkpoint（与原版完全相同）
# ─────────────────────────────────────────────────────────────────────────────

def do_test(cfg, model, iteration):
    new_state_dict  = model.teacher.state_dict()
    new_state_dict2 = model.state_dict()

    if distributed.is_main_process():
        iterstring       = str(iteration)
        eval_dir         = os.path.join(cfg.train.output_dir, "eval", iterstring)
        os.makedirs(eval_dir, exist_ok=True)
        teacher_ckp_path = os.path.join(eval_dir, "teacher_checkpoint.pth")
        torch.save({"teacher": new_state_dict}, teacher_ckp_path)
        model_ckp_path   = os.path.join(eval_dir, "model_checkpoint.pth")
        torch.save({"model": new_state_dict2}, model_ckp_path)


# ─────────────────────────────────────────────────────────────────────────────
# 训练主函数
# ─────────────────────────────────────────────────────────────────────────────

def do_train(cfg, model, resume=False):
    model.train()
    inputs_dtype = torch.half
    fp16_scaler  = model.fp16_scaler

    optimizer = build_optimizer(cfg, model.get_params_groups())
    (
        lr_schedule,
        wd_schedule,
        momentum_schedule,
        teacher_temp_schedule,
        last_layer_lr_schedule,
    ) = build_schedulers(cfg)

    checkpointer = FSDPCheckpointer(
        model, cfg.train.output_dir, optimizer=optimizer, save_to_disk=True
    )
    start_iter = (
        checkpointer.resume_or_load('', resume=resume).get("iteration", -1) + 1
    )

    OFFICIAL_EPOCH_LENGTH = cfg.train.OFFICIAL_EPOCH_LENGTH
    max_iter              = cfg.optim.epochs * OFFICIAL_EPOCH_LENGTH

    periodic_checkpointer = PeriodicCheckpointer(
        checkpointer,
        period     = 3 * OFFICIAL_EPOCH_LENGTH,
        max_iter   = max_iter,
        max_to_keep= 3,
    )

    # ── (6) 图像尺寸 & Patch Token 参数 ──────────────────────────────────────
    img_size   = cfg.crops.global_crops_size    # 640
    patch_size = cfg.student.patch_size         # 16
    n_tokens   = (img_size // patch_size) ** 2  # 1600
    grid_size  = (img_size // patch_size, img_size // patch_size)   # (40, 40)

    # ── 原生随机块掩码生成器（始终保留，用于 40% 迭代步 + 兼容回退）───────────
    mask_generator = MaskingGenerator(
        input_size      = grid_size,                         # (40, 40)
        max_num_patches = int(0.5 * n_tokens),               # 800
    )

    # ── (B) SAM 引导混合掩码生成器 ───────────────────────────────────────────
    # 读取可选配置字段（YAML 中 ibot 节点下，若未配置使用合理默认值）
    sam_ratio       = float(getattr(cfg.ibot, "sam_ratio",       0.6))   # 60%
    sam_warmup_ratio= float(getattr(cfg.ibot, "sam_warmup_ratio",0.7))   # 前 70% epoch

    sam_mask_generator = SAMGuidedMaskGenerator(
        grid_size              = grid_size,             # (40, 40)
        max_mask_ratio         = 0.5,                   # 最大 50% 掩码
        sam_ratio              = sam_ratio,             # 60% 迭代步用 SAM
        total_epochs           = cfg.optim.epochs,      # 总 epoch 数
        warmup_ratio           = sam_warmup_ratio,      # 前 70% 线性升概率
        random_mask_generator  = mask_generator,        # 随机块回退生成器
    )

    # 从起始 epoch 恢复动态难度进度
    start_epoch = start_iter // OFFICIAL_EPOCH_LENGTH
    sam_mask_generator.set_epoch(start_epoch)
    logger.info(f"[SAM Masking] 恢复至 epoch={start_epoch}，{sam_mask_generator}")

    # ── DDN warmup: 前 warmup_epochs 仅训练教师重建 ────────────────────────────
    ddn_cfg = getattr(cfg, "ddn", None)
    ddn_warmup_epochs = int(getattr(ddn_cfg, "warmup_teacher_recon_epochs", 5)) if ddn_cfg else 5
    logger.info(f"[DDN Warmup] warmup_teacher_recon_epochs={ddn_warmup_epochs}")

    # ── (C) 数据增强（SAM 同步版） ────────────────────────────────────────────
    data_transform = DataAugmentationDINOWithSAM(
        cfg.crops.global_crops_scale,
        cfg.crops.local_crops_scale,
        cfg.crops.local_crops_number,
        global_crops_size = img_size,    # 640
        rotate_p          = 0.5,
        flip_p            = 0.5,
        max_angle         = 6.0,
    )

    # ── (C) 数据集（SAM 掩码包装）────────────────────────────────────────────
    # 读取 SAM 掩码目录（可选配置，若未配置则 sam_masks_dir=None → 等效原始行为）
    sam_masks_dir = getattr(cfg.train, "sam_masks_dir", None) or None

    sam_images_dir = getattr(cfg.train, "sam_images_dir", None) or None
    skip_missing_sam = bool(getattr(cfg.train, "skip_missing_sam", True))
    if sam_images_dir:
        if sam_masks_dir is None:
            raise ValueError(
                "train.sam_masks_dir is required when train.sam_images_dir is set"
            )
        logger.info(
            "[SAM Pairing] Using original long-name images from %s",
            sam_images_dir,
        )
        dataset = build_lumbar_sam_dataset_from_images(
            image_root=sam_images_dir,
            sam_masks_dir=sam_masks_dir,
            data_transform=data_transform,
            skip_missing_sam=skip_missing_sam,
        )
    else:
        base_dataset = make_dataset(
            dataset_str=cfg.train.dataset_path,
            transform=None,
            target_transform=lambda _: (),
        )
        dataset = build_lumbar_sam_dataset(
            base_dataset=base_dataset,
            sam_masks_dir=sam_masks_dir,
            data_transform=data_transform,
            skip_missing_sam=skip_missing_sam,
        )

    logger.info(
        f"[SAM Masking] Dataset ready: {len(dataset)} samples, "
        f"sam_masks_dir={sam_masks_dir or 'None (random masking fallback)'}"
    )

    # ── (D) Collate（传入 SAM 生成器）────────────────────────────────────────
    collate_fn = partial(
        collate_data_and_cast,
        mask_ratio_tuple          = cfg.ibot.mask_ratio_min_max,
        mask_probability          = cfg.ibot.mask_sample_probability,
        n_tokens                  = n_tokens,
        mask_generator            = mask_generator,        # 随机块回退
        sam_guided_mask_generator = sam_mask_generator,   # SAM 引导生成器
        dtype                     = inputs_dtype,
    )

    sampler_type = SamplerType.SHARDED_INFINITE
    data_loader  = make_data_loader(
        dataset       = dataset,
        batch_size    = cfg.train.batch_size_per_gpu,
        num_workers   = cfg.train.num_workers,
        shuffle       = True,
        seed          = start_iter,
        sampler_type  = sampler_type,
        sampler_advance= 0,
        drop_last     = True,
        collate_fn    = collate_fn,
    )

    # ── 训练主循环 ────────────────────────────────────────────────────────────
    iteration    = start_iter
    metrics_file = os.path.join(cfg.train.output_dir, "training_metrics.json")
    metric_logger= MetricLogger(delimiter="  ", output_file=metrics_file)
    header       = "Training"

    logger.info(f"Starting training from iteration {start_iter}")

    # 追踪上一个 epoch（用于触发动态难度调整）
    _last_epoch = start_epoch

    for data in metric_logger.log_every(
        data_loader, 10, header, max_iter, start_iter
    ):
        current_batch_size = data["collated_global_crops"].shape[0] / 2
        if iteration > max_iter:
            return

        # ── (E) 动态难度调整：epoch 切换时更新 SAM 掩码概率 ─────────────────
        current_epoch = iteration // OFFICIAL_EPOCH_LENGTH
        if current_epoch != _last_epoch:
            sam_mask_generator.set_epoch(current_epoch)
            _last_epoch = current_epoch
            logger.info(
                f"[SAM Masking] epoch {current_epoch}: "
                f"mask_probs={sam_mask_generator._current_probs()}"
            )
            if current_epoch == ddn_warmup_epochs:
                logger.info(f"[DDN Warmup] epoch {current_epoch}: alignment + student_recon ACTIVATED")

        # ── DDN warmup alpha ──────────────────────────────────────────────
        if current_epoch < ddn_warmup_epochs:
            warmup_alpha = 0.0
        else:
            warmup_alpha = 1.0

        # 调度器更新
        lr             = lr_schedule[iteration]
        wd             = wd_schedule[iteration]
        mom            = momentum_schedule[iteration]
        teacher_temp   = teacher_temp_schedule[iteration]
        last_layer_lr  = last_layer_lr_schedule[iteration]
        apply_optim_scheduler(optimizer, lr, wd, last_layer_lr)

        # 前向 + 反向
        optimizer.zero_grad(set_to_none=True)
        loss_dict = model.forward_backward(data, teacher_temp=teacher_temp,
                                           warmup_alpha=warmup_alpha)

        # 梯度裁剪 & 参数更新
        if fp16_scaler is not None:
            if cfg.optim.clip_grad:
                fp16_scaler.unscale_(optimizer)
                clip_model_gradients(model, cfg.optim.clip_grad)
            fp16_scaler.step(optimizer)
            fp16_scaler.update()
        else:
            if cfg.optim.clip_grad:
                clip_model_gradients(model, cfg.optim.clip_grad)
            optimizer.step()

        # ── EMA 更新已移除 ────────────────────────────────────────────────────
        # model.update_teacher(mom)   ← 已删除

        # 日志
        if distributed.get_global_size() > 1:
            for v in loss_dict.values():
                torch.distributed.all_reduce(v)
        loss_dict_reduced = {
            k: v.item() / distributed.get_global_size()
            for k, v in loss_dict.items()
        }

        if math.isnan(sum(loss_dict_reduced.values())):
            logger.info("NaN detected")
            raise AssertionError

        losses_reduced = sum(loss for loss in loss_dict_reduced.values())

        metric_logger.update(lr=lr)
        metric_logger.update(wd=wd)
        metric_logger.update(mom=mom)
        metric_logger.update(last_layer_lr=last_layer_lr)
        metric_logger.update(current_batch_size=current_batch_size)
        metric_logger.update(**loss_dict_reduced)

        # 记录当前 SAM 掩码策略（每 epoch 记录一次）
        if iteration % OFFICIAL_EPOCH_LENGTH == 0:
            metric_logger.update(
                sam_epoch     = current_epoch,
                sam_disc_prob = sam_mask_generator._current_probs().get(2, 0.0),
                sam_vert_prob = sam_mask_generator._current_probs().get(1, 0.0),
            )

        # checkpoint & eval
        if (cfg.evaluation.eval_period_iterations > 0
                and (iteration + 1) % cfg.evaluation.eval_period_iterations == 0):
            do_test(cfg, model, f"training_{iteration}")
            torch.cuda.synchronize()
        periodic_checkpointer.step(iteration)

        iteration += 1

    metric_logger.synchronize_between_processes()
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


# ─────────────────────────────────────────────────────────────────────────────
# 入口（与原版完全相同）
# ─────────────────────────────────────────────────────────────────────────────
@record
def main(args):
    cfg   = setup(args)
    device = torch.device("cuda", distributed.get_local_rank())
    model = SSLMetaArch(cfg).to(device)

    if cfg.MODEL.WEIGHTS:
        utils.utils.load_pretrained_weights_train(model, cfg.MODEL.WEIGHTS)

    model.prepare_for_distributed_training()
    logger.info("Model:\n{}".format(model))

    if args.eval_only:
        iteration = (
            FSDPCheckpointer(model, save_dir=cfg.train.output_dir)
            .resume_or_load(cfg.MODEL.WEIGHTS, resume=not args.no_resume)
            .get("iteration", -1) + 1
        )
        return do_test(cfg, model, f"manual_{iteration}")

    do_train(cfg, model, resume=not args.no_resume)


if __name__ == "__main__":
    args = get_args_parser(add_help=True).parse_args()
    main(args)
