#!/usr/bin/env python3
# chexfound/train/probe_checkpoints.py  (v2)
#
# ============================================================
# 事后探针评估脚本 —— 对 SSL 训练保存的所有 teacher checkpoint
# 批量运行下游评估，选出最优 checkpoint。
#
# v2 新增:
#   --iter-list  精确指定要测的 iteration 列表，覆盖 --top-k
#
# 使用方式：
#
#   # 方式 A: 按 pretrain 评分测 top-K
#   python -m chexfound.train.probe_checkpoints \
#       --eval-dir ./outputs/xxx/eval \
#       --mode seg_probe \
#       --seg-train-dir /data/train \
#       --seg-val-dir /data/val \
#       --probe-epochs 10 --top-k 5 --probe-size 320
#
#   # 方式 B: 精确指定要测哪些 iteration
#   python -m chexfound.train.probe_checkpoints \
#       --eval-dir ./outputs/xxx/eval \
#       --mode seg_probe \
#       --seg-train-dir /data/train \
#       --seg-val-dir /data/val \
#       --iter-list 249999,199999,149999 \
#       --probe-epochs 10 --probe-size 320
#
#   # 方式 C: 仅排序 pretrain 指标（无需标签）
#   python -m chexfound.train.probe_checkpoints \
#       --eval-dir ./outputs/xxx/eval \
#       --mode pretrain_only
#
# 输出：
#   probe_results.json  — 每个 checkpoint 的评分排名
#   自动复制最优 checkpoint 到 eval_dir/best_by_probe/
# ============================================================

import argparse
import json
import os
import sys
import time
import traceback
import logging
from collections import defaultdict

import torch
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("probe_checkpoints")


# =============================================================================
# 工具：扫描 eval 目录下所有 teacher checkpoint
# =============================================================================

def _parse_probe_summary(eval_dir: str) -> dict:
    """
    解析 probe_summary.txt，返回 {iteration: {"score": ..., "t_recon": ..., ...}} 字典。
    """
    # 尝试多个可能的位置
    candidates = [
        os.path.join(eval_dir, "probe_summary.txt"),
        os.path.join(os.path.dirname(eval_dir), "probe_summary.txt"),
        os.path.join(os.path.dirname(os.path.dirname(eval_dir)), "probe_summary.txt"),
    ]

    summary_path = None
    for p in candidates:
        p = os.path.abspath(p)
        if os.path.exists(p):
            summary_path = p
            break

    if summary_path is None:
        return {}

    records = {}
    try:
        with open(summary_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("=") or line.startswith("-") or line.startswith("Iter") or line.startswith("["):
                    continue
                parts = line.split()
                if len(parts) >= 2:
                    try:
                        iteration = int(parts[0])
                        score = float(parts[1])
                        t_recon = float(parts[2]) if len(parts) > 2 else None
                        sparsity = float(parts[3]) if len(parts) > 3 else None
                        records[iteration] = {
                            "score": score,
                            "t_recon": t_recon,
                            "sparsity_loss": sparsity,
                        }
                    except (ValueError, IndexError):
                        continue
        if records:
            logger.info(f"  Parsed {len(records)} records from probe_summary.txt")
    except Exception as e:
        logger.warning(f"  Failed to parse probe_summary.txt: {e}")
    return records


def scan_checkpoints(eval_dir: str) -> list[dict]:
    """
    扫描 eval_dir 下所有 training_*/teacher_checkpoint.pth。

    Returns:
        [{path, iteration, pretrain_score (if records exist)}, ...]
    """
    checkpoints = []

    # 读取 probe_records.json（若存在）
    records_path = os.path.join(eval_dir, "probe_records.json")
    records_by_iter = {}
    if os.path.exists(records_path):
        with open(records_path) as f:
            records = json.load(f)
        records_by_iter = {r["iteration"]: r for r in records}
        logger.info(f"  Loaded {len(records_by_iter)} records from probe_records.json")

    # 如果 probe_records.json 不存在，尝试解析 probe_summary.txt
    if not records_by_iter:
        logger.info("  probe_records.json not found, trying probe_summary.txt...")
        summary_records = _parse_probe_summary(eval_dir)
        if summary_records:
            records_by_iter = summary_records
            logger.info(f"  ✓ Using pretrain scores from probe_summary.txt ({len(summary_records)} entries)")
        else:
            logger.warning("  ✗ No pretrain scores found from either source")

    # 扫描子目录
    if not os.path.isdir(eval_dir):
        logger.error(f"Eval directory not found: {eval_dir}")
        return []

    for subdir in sorted(os.listdir(eval_dir)):
        subdir_path = os.path.join(eval_dir, subdir)
        if not os.path.isdir(subdir_path):
            continue

        ckpt_path = os.path.join(subdir_path, "teacher_checkpoint.pth")
        if not os.path.exists(ckpt_path):
            continue

        # 解析 iteration（支持多种目录命名格式）
        iteration = None
        if subdir.startswith("training_"):
            try:
                iteration = int(subdir.split("_")[1])
            except (ValueError, IndexError):
                pass
        elif subdir.isdigit():
            # 目录名直接是数字（如 "104999"）
            iteration = int(subdir)

        info = {
            "path": ckpt_path,
            "subdir": subdir,
            "iteration": iteration,
            "pretrain_score": records_by_iter.get(iteration, {}).get("score"),
            "pretrain_record": records_by_iter.get(iteration),
        }
        checkpoints.append(info)

    logger.info(f"Found {len(checkpoints)} teacher checkpoints in {eval_dir}")
    return checkpoints


# =============================================================================
# 模式 A: 基于 pretrain 指标排序（无需标签数据）
# =============================================================================

def rank_by_pretrain(checkpoints: list[dict]) -> list[dict]:
    """按 pretrain health_score 排序（低→高）。"""
    scored = [c for c in checkpoints if c["pretrain_score"] is not None]
    unscored = [c for c in checkpoints if c["pretrain_score"] is None]

    scored.sort(key=lambda c: c["pretrain_score"])

    if unscored:
        logger.warning(f"{len(unscored)} checkpoints have no pretrain_score "
                       f"(probe_records.json not found or incomplete). "
                       f"Placed at end of ranking.")

    return scored + unscored


# =============================================================================
# 模式 B: 快速分割探针（需要下游 DDN-UNet + 标签数据）
# =============================================================================

def run_segmentation_probe(
    teacher_ckpt_path: str,
    seg_train_dir: str,
    seg_val_dir: str,
    num_filters: int = 32,
    probe_epochs: int = 10,
    num_classes: int = 13,
    device: str = "cuda:0",
    probe_size: int = 320,
) -> dict:
    """
    用单个 teacher checkpoint 初始化 DDN-UNet V2 编码器，
    快速 finetune probe_epochs 轮，返回验证集 Dice。
    """
    try:
        # 动态导入下游模块（避免对 SSL 训练环境产生硬依赖）
        # 尝试多个候选路径（支持不同工作目录和跨项目结构）
        _probe_dir = os.path.dirname(os.path.abspath(__file__))
        _candidates = [
            os.path.join(_probe_dir, "..", "..", "..", "D3_DST_SEG"),                # 从 SSL 项目内
            os.path.join(_probe_dir, "..", "..", "..", "..", "D3_DST_SEG"),          # 备选
            os.path.join(os.path.expanduser("~"), "projects", "D3_SEG_15shot_ddp"),
        ]
        _seg_path = os.environ.get("D3_DST_SEG_PATH", "")
        if _seg_path:
            _candidates.insert(0, _seg_path)  # 环境变量优先

        _imported = False
        for _p in _candidates:
            _p = os.path.abspath(_p)
            if os.path.isdir(_p):
                sys.path.insert(0, _p)
                try:
                    from DDN_UNet_Wrapper_LargeKernel_User_LLN_cld_v2 import (
                        DDNUNetWrapper_LargeKernel_User_V2, CombinedLoss)
                    from dataloader import SagittalDataset
                    from torch.utils.data import DataLoader
                    import torch.optim as optim
                    import torch.nn.functional as F
                    _imported = True
                    break
                except ImportError as _e:
                    # 打印详细错误（import 链中的真实报错）
                    import traceback as _tb
                    logger.warning(f"  Import from {_p} failed:\n    {_e}")
                    _tb.print_exc()
                    sys.path.pop(0)  # 移除失败的路径
                    continue

        if not _imported:
            raise ImportError(
                f"Could not find D3_DST_SEG under any candidate path.\n"
                f"  Candidates tried:\n    " + "\n    ".join(_candidates) + "\n"
                f"  Set env var D3_DST_SEG_PATH or pass --seg-root to specify."
            )
    except ImportError as e:
        logger.error(f"Failed to import downstream modules: {e}")
        logger.error("Make sure D3_DST_SEG is accessible. Skipping this checkpoint.")
        return {"dice": 0.0, "error": str(e)}

    # ── 数据集 ─────────────────────────────────────────────────────────
    try:
        train_dataset = SagittalDataset(data_dir=seg_train_dir, num_classes=num_classes)
        val_dataset = SagittalDataset(data_dir=seg_val_dir, num_classes=num_classes)
    except Exception as e:
        logger.error(f"Failed to load datasets: {e}")
        return {"dice": 0.0, "error": str(e)}

    # WSL 下 num_workers>0 容易死锁，默认用 0；可通过环境变量覆盖
    _nw = int(os.environ.get("PROBE_NUM_WORKERS", "0"))
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True,
                              num_workers=_nw, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False,
                            num_workers=_nw, pin_memory=True)

    # ── 模型初始化 ─────────────────────────────────────────────────────
    model = DDNUNetWrapper_LargeKernel_User_V2(
        in_chans=1, num_filters=num_filters, num_classes=num_classes,
        kernel_size=3, unfoldings=3, do_bg=True,
        use_checkpoint=False, use_tddc=False, use_aspp=True,
        use_deep_supervision=False, use_boundary_aware=False,
        use_instance_norm=True, use_feature_consistency=False,
    ).to(device)

    # 加载 Teacher 权重到编码器
    try:
        ckpt_size = os.path.getsize(teacher_ckpt_path)
        checkpoint = torch.load(teacher_ckpt_path, map_location="cpu")
        teacher_sd = checkpoint.get("teacher", checkpoint)
    except Exception as e:
        logger.error(f"  Failed to load checkpoint: {e}")
        traceback.print_exc()
        return {"dice": 0.0, "error": f"Checkpoint load failed: {e}"}

    def clean_key(k):
        k = k.replace("module.", "").replace("backbone.", "")
        return k

    pretrained_dict = {}
    for k, v in teacher_sd.items():
        ck = clean_key(k)
        if ck.startswith(("ddn", "down")):
            pretrained_dict[ck] = v

    try:
        missing, unexpected = model.load_state_dict(pretrained_dict, strict=False)
    except Exception as e:
        logger.error(f"  Failed to load state_dict: {e}")
        traceback.print_exc()
        return {"dice": 0.0, "error": f"State dict load failed: {e}"}

    encoder_loaded = len(pretrained_dict)
    logger.info(f"  Loaded {encoder_loaded} encoder params from {os.path.basename(teacher_ckpt_path)}")

    # ── resize 辅助函数 ───────────────────────────────────────────────
    def _resize_batch(imgs, labels, size):
        """将图像和标签 resize 到 (size, size)，标签用 nearest 插值。"""
        imgs_r = F.interpolate(imgs, size=(size, size), mode="bilinear", align_corners=False)
        labels_r = F.interpolate(
            labels.unsqueeze(1).float(), size=(size, size), mode="nearest"
        ).squeeze(1).long()
        return imgs_r, labels_r

    # ── 快速 finetune ──────────────────────────────────────────────────
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    logger.info(f"  Starting training ({len(train_loader)} batches/epoch, size={probe_size})...")

    GRAD_CLIP_NORM = 1.0

    best_dice = 0.0
    for epoch in range(probe_epochs):
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        t_epoch = time.time()
        for batch in train_loader:
            imgs, labels = batch[0].to(device), batch[1].to(device)
            if probe_size and imgs.shape[-1] != probe_size:
                imgs, labels = _resize_batch(imgs, labels, probe_size)

            optimizer.zero_grad()
            loss, logits = model(x=imgs, seg_target=labels, is_training=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1
            if n_batches == 1:
                logger.info(f"  Epoch {epoch+1}: first batch done in {time.time()-t_epoch:.1f}s")

        # 验证
        model.eval()
        val_dice = 0.0
        with torch.no_grad():
            for batch in val_loader:
                imgs, labels = batch[0].to(device), batch[1].to(device)
                if probe_size and imgs.shape[-1] != probe_size:
                    imgs, labels = _resize_batch(imgs, labels, probe_size)
                logits = model(x=imgs, seg_target=None, is_training=False)
                pred = F.softmax(logits, dim=1).argmax(dim=1)
                pred_oh = F.one_hot(pred, num_classes).permute(0, 3, 1, 2).float()
                target_oh = F.one_hot(labels, num_classes).permute(0, 3, 1, 2).float()
                # 排除背景
                intersection = (pred_oh[:, 1:] * target_oh[:, 1:]).sum(dim=(0, 2, 3))
                union = pred_oh[:, 1:].sum(dim=(0, 2, 3)) + target_oh[:, 1:].sum(dim=(0, 2, 3))
                dice = (2. * intersection + 1e-5) / (union + 1e-5)
                val_dice += dice.mean().item()

        val_dice /= max(len(val_loader), 1)
        avg_loss = epoch_loss / max(n_batches, 1)
        improved = "★" if val_dice > best_dice else ""
        logger.info(f"  Epoch {epoch+1}/{probe_epochs}  loss={avg_loss:.4f}  val_dice={val_dice:.4f} {improved}")
        if val_dice > best_dice:
            best_dice = val_dice

    logger.info(f"  Best Dice after {probe_epochs} epochs: {best_dice:.4f}")

    del model, optimizer
    torch.cuda.empty_cache()

    return {"dice": best_dice}


# =============================================================================
# 主流程
# =============================================================================

def main():
    print("\n" + "=" * 60)
    print(">>> SSL Teacher Checkpoint Probe <<<")
    print("=" * 60 + "\n")

    parser = argparse.ArgumentParser("SSL Teacher Checkpoint Probe")
    parser.add_argument("--eval-dir", type=str, required=True,
                        help="Path to eval directory with training_*/ subdirs")
    parser.add_argument("--mode", type=str, default="pretrain_only",
                        choices=["pretrain_only", "knn", "seg_probe"],
                        help="Probe mode (default: pretrain_only)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output JSON path (default: eval_dir/probe_results.json)")

    # seg_probe 参数
    parser.add_argument("--seg-train-dir", type=str, default="",
                        help="Segmentation training data dir for seg_probe mode")
    parser.add_argument("--seg-val-dir", type=str, default="",
                        help="Segmentation validation data dir for seg_probe mode")
    parser.add_argument("--probe-epochs", type=int, default=10,
                        help="Quick finetune epochs for seg_probe (default: 10)")
    parser.add_argument("--num-filters", type=int, default=32,
                        help="Number of filters (must match SSL config)")
    parser.add_argument("--num-classes", type=int, default=13,
                        help="Number of segmentation classes")
    parser.add_argument("--top-k", type=int, default=5,
                        help="Number of top checkpoints to probe with seg_probe "
                             "(use pretrain ranking to filter, default: 5)")
    parser.add_argument("--seg-root", type=str, default="",
                        help="Path to D3_DST_SEG project root (or set env D3_DST_SEG_PATH)")
    parser.add_argument("--probe-size", type=int, default=320,
                        help="Resize input to this size for probe (default: 320). "
                             "Set to 0 to use original size.")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--iter-list", type=str, default=None,
                        help="Comma-separated iteration numbers to probe, e.g. '249999,199999,149999'. "
                             "Overrides --top-k when specified.")

    args = parser.parse_args()

    # 如果 --seg-root 指定了，注入环境变量（run_segmentation_probe 会读取）
    if args.seg_root:
        os.environ["D3_DST_SEG_PATH"] = os.path.abspath(args.seg_root)

    eval_dir = args.eval_dir
    output_path = args.output or os.path.join(eval_dir, "probe_results.json")

    # ── Step 1: 扫描所有 checkpoint ────────────────────────────────────
    checkpoints = scan_checkpoints(eval_dir)
    if not checkpoints:
        logger.error("No checkpoints found. Exiting.")
        return

    # ── Step 2: 根据模式评估 ────────────────────────────────────────────
    results = []

    if args.mode == "pretrain_only":
        # 仅按 pretrain 指标排序
        ranked = rank_by_pretrain(checkpoints)
        for rank, ckpt in enumerate(ranked, 1):
            results.append({
                "rank": rank,
                "iteration": ckpt["iteration"],
                "path": ckpt["path"],
                "pretrain_score": ckpt["pretrain_score"],
                "teacher_recon": (ckpt.get("pretrain_record") or {}).get("teacher_recon"),
                "sparsity_loss": (ckpt.get("pretrain_record") or {}).get("sparsity_loss"),
                "gamma_sparsity": (ckpt.get("pretrain_record") or {}).get("gamma_sparsity"),
                "probe_dice": None,
            })
        logger.info("Ranked by pretrain health_score (lower = better).")
        logger.info("Run with --mode seg_probe for downstream validation.")

    elif args.mode == "seg_probe":
        if not args.seg_train_dir or not args.seg_val_dir:
            logger.error("seg_probe mode requires --seg-train-dir and --seg-val-dir")
            return

        # 决定要 probe 哪些 checkpoint
        if args.iter_list:
            # 用户指定了具体的 iteration 列表
            target_iters = [int(x.strip()) for x in args.iter_list.split(",") if x.strip()]
            ckpt_by_iter = {c["iteration"]: c for c in checkpoints if c["iteration"] is not None}
            candidates = []
            for it in target_iters:
                if it in ckpt_by_iter:
                    candidates.append(ckpt_by_iter[it])
                else:
                    logger.warning(f"  Iteration {it} not found in eval dir, skipping")
            if not candidates:
                logger.error("None of the specified --iter-list checkpoints were found.")
                return
            logger.info(f"Using --iter-list with {len(candidates)} checkpoints: {target_iters}")
        else:
            # 默认：按 pretrain 评分筛选 top-K
            ranked = rank_by_pretrain(checkpoints)
            candidates = ranked[:min(args.top_k, len(ranked))]
            logger.info(f"Running seg_probe on top {len(candidates)} checkpoints "
                         f"(out of {len(checkpoints)} total, probe_epochs={args.probe_epochs})")
            logger.info("  Selected checkpoints (by pretrain score, lower=better):")
            for i, ckpt in enumerate(candidates, 1):
                logger.info(f"    #{i}: iter={ckpt['iteration']}, score={ckpt['pretrain_score']}, dir={ckpt['subdir']}")

        for ckpt in candidates:
            logger.info(f"Probing: {ckpt['subdir']} (iter={ckpt['iteration']})")

            # 跨 checkpoint 的 CUDA 清理（避免 GPU 状态累积导致崩溃）
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.empty_cache()

            try:
                probe_result = run_segmentation_probe(
                    teacher_ckpt_path=ckpt["path"],
                    seg_train_dir=args.seg_train_dir,
                    seg_val_dir=args.seg_val_dir,
                    num_filters=args.num_filters,
                    probe_epochs=args.probe_epochs,
                    num_classes=args.num_classes,
                    device=args.device,
                    probe_size=args.probe_size if args.probe_size > 0 else None,
                )
            except Exception as e:
                logger.error(f"Probe FAILED for {ckpt['subdir']}: {e}")
                traceback.print_exc()
                # 把失败的 checkpoint 记入结果（Dice=0），继续下一个
                probe_result = {"dice": 0.0, "error": str(e)}
            results.append({
                "iteration": ckpt["iteration"],
                "path": ckpt["path"],
                "pretrain_score": ckpt["pretrain_score"],
                "teacher_recon": (ckpt.get("pretrain_record") or {}).get("teacher_recon"),
                "sparsity_loss": (ckpt.get("pretrain_record") or {}).get("sparsity_loss"),
                "gamma_sparsity": (ckpt.get("pretrain_record") or {}).get("gamma_sparsity"),
                "probe_dice": probe_result.get("dice", 0.0),
                "probe_error": probe_result.get("error"),
            })

        # 按 Dice 排序
        results.sort(key=lambda r: r["probe_dice"], reverse=True)
        for rank, r in enumerate(results, 1):
            r["rank"] = rank

        # 最佳 checkpoint 复制到独立目录
        if results and results[0]["probe_dice"] > 0:
            best = results[0]
            best_dir = os.path.join(eval_dir, "best_by_probe")
            os.makedirs(best_dir, exist_ok=True)
            best_dst = os.path.join(best_dir, "best_teacher_checkpoint.pth")
            import shutil
            shutil.copy2(best["path"], best_dst)
            logger.info(f"★ Best checkpoint (Dice={best['probe_dice']:.4f}) "
                        f"copied to {best_dst}")

    # ── Step 3: 保存结果 ───────────────────────────────────────────────
    output = {
        "eval_dir": eval_dir,
        "mode": args.mode,
        "results": results,
    }
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    # ── Step 4: 终端输出汇总 ────────────────────────────────────────────
    print(f"\n{'='*90}")
    print(f"Probe Results  (mode={args.mode})")
    print(f"{'='*90}")
    if args.mode == "pretrain_only":
        print(f"{'Rank':<6s} {'Iter':<10s} {'Score':<10s} {'T-Recon':<8s} "
              f"{'Sparsity':<10s} {'γ-Sparsity':<10s}")
        print(f"{'-'*90}")
        for r in results[:20]:  # top 20
            print(f"{r['rank']:<6d} {str(r['iteration']):<10s} "
                  f"{str(r['pretrain_score']):<10s} {str(r['teacher_recon']):<8s} "
                  f"{str(r['sparsity_loss']):<10s} {str(r['gamma_sparsity']):<10s}")
    else:
        print(f"{'Rank':<6s} {'Iter':<10s} {'Probe Dice':<12s} "
              f"{'Pretrain Score':<14s} {'γ-Sparsity':<10s}")
        print(f"{'-'*90}")
        for r in results:
            print(f"{r['rank']:<6d} {str(r['iteration']):<10s} "
                  f"{r['probe_dice']:<12.4f} {str(r['pretrain_score']):<14s} "
                  f"{str(r['gamma_sparsity']):<10s}")

    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
