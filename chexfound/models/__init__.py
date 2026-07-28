# chexfound/models/__init__.py
#
# ============================================================
# 改动说明（稀疏系数对齐版）：
#   · ddn 分支：
#     教师 → DDNBackboneWrapper（use_aspp=False，保持稀疏性）
#     学生 → DSTNetBackboneWrapper（ARNet 三尺度并行稀疏编码器）
#   · 不再向 model_dict 添加 dino_head / ibot_head
#   · use_aspp 默认 False（ASPP 破坏稀疏性）
# ============================================================

import logging
import warnings

from chexfound.models.DDN_Wrapper import DDNBackboneWrapper, DSTNetBackboneWrapper

logger = logging.getLogger("chexfound")


def _get_vits():
    """Lazy import of vision_transformer — stored in module cache after first call."""
    if not hasattr(_get_vits, "_cached"):
        try:
            from . import vision_transformer as vits
            _get_vits._cached = vits
        except ImportError:
            warnings.warn("vision_transformer module not available; DDN mode only")
            _get_vits._cached = None
    return _get_vits._cached


def build_model(args, only_teacher=False, img_size=640):
    args.arch = args.arch.removesuffix("_memeff")

    if args.arch == "ddn":
        num_filters = getattr(args, "num_filters", 16)
        in_chans    = getattr(args, "in_chans", 1)
        embed_dim   = getattr(args, "embed_dim", 512)

        common_kw = dict(
            img_size    = img_size,
            patch_size  = args.patch_size,
            in_chans    = in_chans,
            embed_dim   = embed_dim,
            num_filters = num_filters,
        )

        # 教师：DDN 多尺度纯编码器（无 ASPP，保持稀疏性）
        use_ckpt = getattr(args, "use_checkpoint", False)  # 默认关闭加速训练
        teacher = DDNBackboneWrapper(
            **common_kw,
            use_checkpoint = use_ckpt,
        )
        logger.info(
            f"Teacher: DDNBackboneWrapper  "
            f"num_filters={num_filters}  use_aspp=False  "
            f"γ=[2B,{num_filters},{img_size//8},{img_size//8}]  冻结"
        )

        if only_teacher:
            return teacher, embed_dim

        # 学生：ARNet 三尺度并行稀疏编码器（含 SE Block 融合）
        student = DSTNetBackboneWrapper(
            **common_kw,
            num_iter = getattr(args, "num_iter", 3),
        )
        logger.info(
            f"Student: DSTNetBackboneWrapper  "
            f"num_filters={num_filters}  SE Block 融合  "
            f"γ=[2B,{num_filters},{img_size//8},{img_size//8}]  可训练"
        )

        return student, teacher, embed_dim

    # ── ViT 系列（懒加载）───────────────────────────────────────────────────
    vits = _get_vits()
    if vits is None:
        raise ImportError(
            "vision_transformer module is required for ViT architectures but is missing. "
            "Restore chexfound/models/vision_transformer.py or use arch=ddn."
        )
    if "vit" in args.arch:
        vit_kwargs = dict(
            img_size              = img_size,
            patch_size            = args.patch_size,
            init_values           = args.layerscale,
            ffn_layer             = args.ffn_layer,
            block_chunks          = args.block_chunks,
            qkv_bias              = args.qkv_bias,
            proj_bias             = args.proj_bias,
            ffn_bias              = args.ffn_bias,
            num_register_tokens   = args.num_register_tokens,
            interpolate_offset    = args.interpolate_offset,
            interpolate_antialias = args.interpolate_antialias,
        )
        teacher = vits.__dict__[args.arch](**vit_kwargs)
        if only_teacher:
            return teacher, teacher.embed_dim
        student = vits.__dict__[args.arch](
            **vit_kwargs,
            drop_path_rate    = args.drop_path_rate,
            drop_path_uniform = args.drop_path_uniform,
        )
        return student, teacher, student.embed_dim


def build_model_from_cfg(cfg, only_teacher=False):
    return build_model(
        cfg.student,
        only_teacher=only_teacher,
        img_size=cfg.crops.global_crops_size,
    )