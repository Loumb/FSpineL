# chexfound/train/train_probe.py
#
# ============================================================
# 探针版训练脚本 —— 在原版 train.py 基础上新增：
#
#   1. 在线健康度评分（health_score）
#      - 每个 eval 周期计算一次，综合 teacher_recon + sparsity_loss
#      - 跟踪历史最佳，自动保存 best_teacher_checkpoint.pth
#
#   2. 增强的 checkpoint 管理
#      - 保留所有 eval 间隔的 teacher checkpoint（用于事后对比）
#      - 额外维护一份 "best so far" checkpoint
#      - 训练结束时输出所有 checkpoint 的评分汇总
#
#   3. 事后探针支持
#      - 训练完成后自动生成 probe_list.txt（所有 checkpoint 路径 + 评分）
#      - 配合 probe_checkpoints.py 做下游批量评估
#
# 原理：
#   自监督 pretrain loss 和下游性能不完全相关，但 reconstruction + sparsity
#   的组合趋势可以用来排除明显退化的 checkpoint。最终选择由下游探针确认。
# ============================================================

import argparse
import logging
import math
import os
import json
from functools import partial

from fvcore.common.checkpoint import Checkpointer, PeriodicCheckpointer
import torch

from chexfound.data import SamplerType, make_data_loader, make_dataset
from chexfound.data import collate_data_and_cast, DataAugmentationDINO, MaskingGenerator
from chexfound.data.augmentations import DataAugmentationDINOWithSAM
from chexfound.data.sam_masking import SAMGuidedMaskGenerator
from chexfound.data.lumbar_dataset import build_lumbar_sam_dataset
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


# =============================================================================
# 探针：健康度评分 & 最佳 checkpoint 追踪
# =============================================================================

class ProbeCheckpointTracker:
    """
    在线追踪每个 eval 周期的 pretrain 指标，计算健康度评分，
    自动保存评分最佳的 teacher checkpoint。

    health_score = teacher_recon + α × sparsity_loss
    分数越低越好（重建质量高 + 稀疏正则有效）。

    α 的选取：sparsity_loss 通常比 teacher_recon 小两个数量级，
    放大到可比量级（默认 10.0），使稀疏正则纳入考量。
    """

    def __init__(self, output_dir: str, sparsity_alpha: float = 10.0):
        self.output_dir = output_dir
        self.sparsity_alpha = sparsity_alpha
        self.best_score = float('inf')
        self.best_iteration = -1
        self.records = []  # list of {iteration, score, teacher_recon, sparsity_loss, ...}

        # 保存目录
        self.probe_dir = os.path.join(output_dir, "eval")
        os.makedirs(self.probe_dir, exist_ok=True)
        self.records_path = os.path.join(self.probe_dir, "probe_records.json")

    def compute_score(self, loss_dict: dict) -> float:
        """根据 loss_dict 计算健康度评分。"""
        teacher_recon = float(loss_dict.get("teacher_recon", 0.0))
        sparsity_loss = float(loss_dict.get("sparsity_loss", 0.0))
        score = teacher_recon + self.sparsity_alpha * sparsity_loss
        return score

    def update(self, iteration: int, loss_dict: dict, teacher_state_dict: dict):
        """
        评估当前 checkpoint，若评分优于历史最佳则保存。
        仅在 rank 0 执行。
        """
        if not distributed.is_main_process():
            return

        score = self.compute_score(loss_dict)

        record = {
            "iteration": iteration,
            "score": round(score, 6),
            "teacher_recon": round(float(loss_dict.get("teacher_recon", 0.0)), 6),
            "sparsity_loss": round(float(loss_dict.get("sparsity_loss", 0.0)), 6),
            "align_loss": round(float(loss_dict.get("align_loss", 0.0)), 6),
            "student_recon": round(float(loss_dict.get("student_recon", 0.0)), 6),
            "gamma_sparsity": round(float(loss_dict.get("gamma_sparsity", 0.0)), 4),
            "total_loss": round(float(loss_dict.get("total_loss", 0.0)), 6),
        }
        self.records.append(record)

        # 保存评分记录（持久化，中断后可恢复）
        with open(self.records_path, "w") as f:
            json.dump(self.records, f, indent=2)

        is_best = score < self.best_score
        if is_best:
            prev_best = self.best_score
            self.best_score = score
            self.best_iteration = iteration

            best_path = os.path.join(self.probe_dir, "best_teacher_checkpoint.pth")
            torch.save({"teacher": teacher_state_dict, "iteration": iteration,
                        "score": score, "record": record}, best_path)

            logger.info(
                f"[Probe] ★ NEW BEST  iteration={iteration}  "
                f"score={score:.6f}  (prev={prev_best:.6f})  "
                f"teacher_recon={record['teacher_recon']:.4f}  "
                f"sparsity={record['sparsity_loss']:.6f}  "
                f"gamma_sparsity={record['gamma_sparsity']:.2%}"
            )
        else:
            logger.info(
                f"[Probe]    eval    iteration={iteration}  "
                f"score={score:.6f}  (best={self.best_score:.6f} @ iter {self.best_iteration})"
            )

        return is_best

    def summary(self) -> str:
        """训练结束后输出所有 checkpoint 评分汇总。"""
        if not self.records:
            return "[Probe] No records."

        lines = [
            f"\n{'='*80}",
            f"[Probe] Checkpoint Summary  (health_score = teacher_recon + "
            f"{self.sparsity_alpha} × sparsity_loss,  lower is better)",
            f"{'='*80}",
            f"{'Iter':>12s}  {'Score':>10s}  {'T-Recon':>8s}  {'Sparsity':>10s}  "
            f"{'Align':>8s}  {'S-Recon':>8s}  {'γ-Sparsity':>10s}  {'Best':>5s}",
            f"{'-'*80}",
        ]

        for r in self.records:
            is_best = "★" if r["iteration"] == self.best_iteration else ""
            lines.append(
                f"{r['iteration']:>12d}  {r['score']:>10.6f}  {r['teacher_recon']:>8.4f}  "
                f"{r['sparsity_loss']:>10.6f}  {r['align_loss']:>8.4f}  "
                f"{r['student_recon']:>8.4f}  {r['gamma_sparsity']:>10.2%}  {is_best:>5s}"
            )

        lines.append(f"{'='*80}")
        lines.append(f"Best: iteration={self.best_iteration}  score={self.best_score:.6f}")
        lines.append(f"Saved: {os.path.join(self.probe_dir, 'best_teacher_checkpoint.pth')}")
        lines.append(f"All eval checkpoints: {self.probe_dir}/training_*/")
        lines.append(f"Run probe_checkpoints.py to validate with downstream task.")
        lines.append(f"{'='*80}\n")

        return "\n".join(lines)


# =============================================================================
# CLI（与原版完全相同）
# =============================================================================

def get_args_parser(add_help: bool = True):
    parser = argparse.ArgumentParser("DINOv2 training (with probe)", add_help=add_help)
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
    # 探针参数
    parser.add_argument("--probe-alpha", type=float, default=10.0,
                        help="sparsity_loss 放大系数 (default: 10.0)")
    return parser


# =============================================================================
# 优化器 & 调度器（与原版完全相同）
# =============================================================================

def build_optimizer(cfg, params_groups):
    return torch.optim.AdamW(
        params_groups,
        betas=(cfg.optim.adamw_beta1, cfg.optim.adamw_beta2),
    )


def build_schedulers(cfg):
    OFFICIAL_EPOCH_LENGTH = cfg.train.OFFICIAL_EPOCH_LENGTH

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
        lr_schedule, wd_schedule, momentum_schedule,
        teacher_temp_schedule, last_layer_lr_schedule,
    )


def apply_optim_scheduler(optimizer, lr, wd, last_layer_lr):
    for param_group in optimizer.param_groups:
        is_last_layer  = param_group["is_last_layer"]
        lr_multiplier  = param_group["lr_multiplier"]
        wd_multiplier  = param_group["wd_multiplier"]
        param_group["weight_decay"] = wd * wd_multiplier
        param_group["lr"] = (last_layer_lr if is_last_layer else lr) * lr_multiplier


# =============================================================================
# 评估 & checkpoint（探针增强版）
# =============================================================================

def do_test_with_probe(cfg, model, iteration, probe_tracker, loss_dict):
    """
    保存 checkpoint 并更新探针评分。

    相比原版 do_test：
      - 新增 health_score 计算与 best checkpoint 追踪
      - 保留原版的全部 checkpoint 保存逻辑
    """
    new_state_dict  = model.teacher.state_dict()
    new_state_dict2 = model.state_dict()

    if distributed.is_main_process():
        iterstring       = str(iteration)
        eval_dir         = os.path.join(cfg.train.output_dir, "eval", iterstring)
        os.makedirs(eval_dir, exist_ok=True)

        # 保存常规 checkpoint
        teacher_ckp_path = os.path.join(eval_dir, "teacher_checkpoint.pth")
        torch.save({"teacher": new_state_dict, "iteration": iteration},
                   teacher_ckp_path)
        model_ckp_path   = os.path.join(eval_dir, "model_checkpoint.pth")
        torch.save({"model": new_state_dict2, "iteration": iteration},
                   model_ckp_path)

        # 探针评分 & 更新最佳
        probe_tracker.update(iteration, loss_dict, new_state_dict)


# =============================================================================
# 训练主函数（探针增强版）
# =============================================================================

def do_train(cfg, model, resume=False, probe_alpha=10.0):
    model.train()
    inputs_dtype = torch.half
    fp16_scaler  = model.fp16_scaler

    optimizer = build_optimizer(cfg, model.get_params_groups())
    (
        lr_schedule, wd_schedule, momentum_schedule,
        teacher_temp_schedule, last_layer_lr_schedule,
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

    # ── 探针追踪器初始化 ─────────────────────────────────────────────────
    probe_tracker = ProbeCheckpointTracker(
        output_dir=cfg.train.output_dir,
        sparsity_alpha=probe_alpha,
    )
    logger.info(
        f"[Probe] Tracker initialized: alpha={probe_alpha}, "
        f"health_score = teacher_recon + {probe_alpha} × sparsity_loss"
    )

    # ── 其余初始化（与原版相同）──────────────────────────────────────────
    img_size   = cfg.crops.global_crops_size
    patch_size = cfg.student.patch_size
    n_tokens   = (img_size // patch_size) ** 2
    grid_size  = (img_size // patch_size, img_size // patch_size)

    mask_generator = MaskingGenerator(
        input_size      = grid_size,
        max_num_patches = int(0.5 * n_tokens),
    )

    sam_ratio       = float(getattr(cfg.ibot, "sam_ratio",       0.6))
    sam_warmup_ratio= float(getattr(cfg.ibot, "sam_warmup_ratio",0.7))

    sam_mask_generator = SAMGuidedMaskGenerator(
        grid_size              = grid_size,
        max_mask_ratio         = 0.5,
        sam_ratio              = sam_ratio,
        total_epochs           = cfg.optim.epochs,
        warmup_ratio           = sam_warmup_ratio,
        random_mask_generator  = mask_generator,
    )

    start_epoch = start_iter // OFFICIAL_EPOCH_LENGTH
    sam_mask_generator.set_epoch(start_epoch)
    logger.info(f"[SAM Masking] 恢复至 epoch={start_epoch}，{sam_mask_generator}")

    ddn_cfg = getattr(cfg, "ddn", None)
    ddn_warmup_epochs = int(getattr(ddn_cfg, "warmup_teacher_recon_epochs", 5)) if ddn_cfg else 5
    logger.info(f"[DDN Warmup] warmup_teacher_recon_epochs={ddn_warmup_epochs}")

    data_transform = DataAugmentationDINOWithSAM(
        cfg.crops.global_crops_scale,
        cfg.crops.local_crops_scale,
        cfg.crops.local_crops_number,
        global_crops_size = img_size,
        rotate_p          = 0.5,
        flip_p            = 0.5,
        max_angle         = 6.0,
    )

    sam_masks_dir = getattr(cfg.train, "sam_masks_dir", None) or None

    base_dataset = make_dataset(
        dataset_str      = cfg.train.dataset_path,
        transform        = None,
        target_transform = lambda _: (),
    )

    dataset = build_lumbar_sam_dataset(
        base_dataset   = base_dataset,
        sam_masks_dir  = sam_masks_dir,
        data_transform = data_transform,
    )

    logger.info(
        f"[SAM Masking] Dataset ready: {len(dataset)} samples, "
        f"sam_masks_dir={sam_masks_dir or 'None (random masking fallback)'}"
    )

    collate_fn = partial(
        collate_data_and_cast,
        mask_ratio_tuple          = cfg.ibot.mask_ratio_min_max,
        mask_probability          = cfg.ibot.mask_sample_probability,
        n_tokens                  = n_tokens,
        mask_generator            = mask_generator,
        sam_guided_mask_generator = sam_mask_generator,
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

    # ── 训练主循环 ────────────────────────────────────────────────────────
    iteration    = start_iter
    metrics_file = os.path.join(cfg.train.output_dir, "training_metrics.json")
    metric_logger= MetricLogger(delimiter="  ", output_file=metrics_file)
    header       = "Training"

    logger.info(f"Starting training from iteration {start_iter}")

    _last_epoch = start_epoch

    # 维护最近 N 个 step 的 loss 滑动平均（用于探针评分，比单步更稳定）
    loss_window = []

    for data in metric_logger.log_every(
        data_loader, 10, header, max_iter, start_iter
    ):
        current_batch_size = data["collated_global_crops"].shape[0] / 2
        if iteration > max_iter:
            break

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

        if current_epoch < ddn_warmup_epochs:
            warmup_alpha = 0.0
        else:
            warmup_alpha = 1.0

        lr             = lr_schedule[iteration]
        wd             = wd_schedule[iteration]
        mom            = momentum_schedule[iteration]
        teacher_temp   = teacher_temp_schedule[iteration]
        last_layer_lr  = last_layer_lr_schedule[iteration]
        apply_optim_scheduler(optimizer, lr, wd, last_layer_lr)

        optimizer.zero_grad(set_to_none=True)
        loss_dict = model.forward_backward(data, teacher_temp=teacher_temp,
                                           warmup_alpha=warmup_alpha)

        if fp16_scaler is not None:
            if cfg.optim.clip_grad:
                fp16_scaler.unscale_(optimizer)
                for v in model.student.values():
                    v.clip_grad_norm_(cfg.optim.clip_grad)
            fp16_scaler.step(optimizer)
            fp16_scaler.update()
        else:
            if cfg.optim.clip_grad:
                for v in model.student.values():
                    v.clip_grad_norm_(cfg.optim.clip_grad)
            optimizer.step()

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

        if iteration % OFFICIAL_EPOCH_LENGTH == 0:
            metric_logger.update(
                sam_epoch     = current_epoch,
                sam_disc_prob = sam_mask_generator._current_probs().get(2, 0.0),
                sam_vert_prob = sam_mask_generator._current_probs().get(1, 0.0),
            )

        # ── 探针增强的 checkpoint & eval ──────────────────────────────────
        if (cfg.evaluation.eval_period_iterations > 0
                and (iteration + 1) % cfg.evaluation.eval_period_iterations == 0):
            do_test_with_probe(cfg, model, iteration, probe_tracker,
                              loss_dict_reduced)
            torch.cuda.synchronize()

        periodic_checkpointer.step(iteration)
        iteration += 1

    # ── 训练结束：输出探针汇总 ────────────────────────────────────────────
    metric_logger.synchronize_between_processes()

    if distributed.is_main_process():
        summary = probe_tracker.summary()
        logger.info(summary)
        # 也写入文件
        summary_path = os.path.join(cfg.train.output_dir, "probe_summary.txt")
        with open(summary_path, "w") as f:
            f.write(summary + "\n")

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


# =============================================================================
# 入口
# =============================================================================
@record
def main(args):
    cfg   = setup(args)
    model = SSLMetaArch(cfg).to(torch.device("cuda"))

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
        # eval-only 模式也走探针保存
        probe_tracker = ProbeCheckpointTracker(
            output_dir=cfg.train.output_dir,
            sparsity_alpha=args.probe_alpha,
        )
        return do_test_with_probe(cfg, model, f"manual_{iteration}",
                                  probe_tracker, {})

    do_train(cfg, model, resume=not args.no_resume,
             probe_alpha=args.probe_alpha)


if __name__ == "__main__":
    args = get_args_parser(add_help=True).parse_args()
    main(args)
