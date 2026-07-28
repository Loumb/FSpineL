# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

from functools import partial
import logging
import warnings

import torch
from torch import nn
import torch.nn.functional as F

from chexfound.loss import DINOLoss, iBOTPatchLoss, KoLeoLoss
from chexfound.models import build_model_from_cfg
from chexfound.layers import DINOHead
from chexfound.utils.utils import has_batchnorms
from chexfound.utils.param_groups import get_params_groups_with_decay, fuse_params_groups
from chexfound.fsdp import get_fsdp_wrapper, ShardedGradScaler, get_fsdp_modules, reshard_fsdp_model

logger = logging.getLogger("chexfound")


# Lazy imports for ViT-specific modules (not needed by DDN)
def _get_xformers_fmha():
    if not hasattr(_get_xformers_fmha, "_cached"):
        try:
            from xformers.ops import fmha
            _get_xformers_fmha._cached = fmha
        except ImportError:
            _get_xformers_fmha._cached = None
    return _get_xformers_fmha._cached


def _get_block_chunk():
    if not hasattr(_get_block_chunk, "_cached"):
        try:
            from chexfound.models.vision_transformer import BlockChunk
            _get_block_chunk._cached = BlockChunk
        except ImportError:
            _get_block_chunk._cached = None
    return _get_block_chunk._cached


class SSLMetaArch(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.fp16_scaler = ShardedGradScaler() if cfg.compute_precision.grad_scaler else None

        student_model_dict = dict()
        teacher_model_dict = dict()

        student_backbone, teacher_backbone, embed_dim = build_model_from_cfg(cfg)
        student_model_dict["backbone"] = student_backbone
        teacher_model_dict["backbone"] = teacher_backbone
        logger.info(f"OPTIONS -- architecture : embed_dim: {embed_dim}")

        if cfg.student.pretrained_weights:
            chkpt = torch.load(cfg.student.pretrained_weights)
            logger.info(f"OPTIONS -- pretrained weights: loading from {cfg.student.pretrained_weights}")
            student_backbone.load_state_dict(chkpt["model"], strict=False)

        self.embed_dim = embed_dim

        # ── Detect DDN (sparse coefficient alignment) mode ─────────────────
        self.is_ddn = cfg.student.arch == "ddn"

        if self.is_ddn:
            logger.info("OPTIONS -- DDN dual reconstruction + sparse alignment mode")
            logger.info(f"OPTIONS -- DDN -- student: {type(student_backbone).__name__}")
            logger.info(f"OPTIONS -- DDN -- teacher: {type(teacher_backbone).__name__}")

            from chexfound.models.decoder import SparseDecoder, SobelEdgeExtractor

            # ── 读取可配置权重（缺失时使用默认值，向后兼容旧 YAML）─────────
            ddn_cfg = getattr(cfg, "ddn", None)
            _f = lambda key, default: getattr(ddn_cfg, key, default) if ddn_cfg else default

            self.teacher_recon_weight  = _f("teacher_recon_weight",   1.0)
            self.student_recon_weight  = _f("student_recon_weight",   0.3)
            self.align_weight          = _f("align_weight",           1.0)
            self.edge_loss_weight      = _f("edge_loss_weight",       0.05)
            self.use_multiscale_recon  = _f("use_multiscale_recon",   True)
            self.aux_320_weight        = _f("aux_recon_320_weight",   0.3)
            self.aux_160_weight        = _f("aux_recon_160_weight",   0.1)
            self.use_anatomy_weight    = _f("use_anatomy_weight",     True)
            self.anatomy_w_vertebra    = _f("anatomy_weight_vertebra", 2.0)
            self.anatomy_w_disc        = _f("anatomy_weight_disc",     3.0)
            self.anatomy_w_canal       = _f("anatomy_weight_canal",    1.5)
            self.warmup_epochs         = _f("warmup_teacher_recon_epochs", 5)
            self.l1_sparsity_weight    = _f("l1_sparsity_weight",     0.01)
            self.l2_sparsity_weight    = _f("l2_sparsity_weight",     0.01)

            self.decoder = SparseDecoder(
                sparse_ch=cfg.student.num_filters,
                out_ch=cfg.student.in_chans,
                use_multiscale=self.use_multiscale_recon,
            )
            self.sobel = SobelEdgeExtractor()
            logger.info(f"OPTIONS -- DDN -- decoder: {type(self.decoder).__name__} "
                        f"(multiscale={self.use_multiscale_recon})")
            logger.info(f"OPTIONS -- DDN -- weights: "
                        f"teacher_recon={self.teacher_recon_weight} "
                        f"align={self.align_weight} "
                        f"student_recon={self.student_recon_weight} "
                        f"edge={self.edge_loss_weight} "
                        f"l1_sparsity={self.l1_sparsity_weight} "
                        f"l2_sparsity={self.l2_sparsity_weight}")
            if self.use_anatomy_weight:
                logger.info(f"OPTIONS -- DDN -- anatomy weights: "
                            f"vertebra={self.anatomy_w_vertebra} "
                            f"disc={self.anatomy_w_disc} "
                            f"canal={self.anatomy_w_canal}")
            # DDN doesn't need DINO/iBOT/KoLeo loss heads
        else:
            # ── DINOv2 teacher-student knowledge distillation ──────────────
            self.do_dino = cfg.dino.loss_weight > 0
            self.do_koleo = cfg.dino.koleo_loss_weight > 0
            self.do_ibot = cfg.ibot.loss_weight > 0
            self.ibot_separate_head = cfg.ibot.separate_head

            logger.info("OPTIONS -- DINO")
            if self.do_dino:
                logger.info(f"OPTIONS -- DINO -- loss_weight: {cfg.dino.loss_weight}")
                logger.info(f"OPTIONS -- DINO -- head_n_prototypes: {cfg.dino.head_n_prototypes}")
                logger.info(f"OPTIONS -- DINO -- head_bottleneck_dim: {cfg.dino.head_bottleneck_dim}")
                logger.info(f"OPTIONS -- DINO -- head_hidden_dim: {cfg.dino.head_hidden_dim}")
                self.dino_loss_weight = cfg.dino.loss_weight
                dino_head = partial(
                    DINOHead,
                    in_dim=embed_dim,
                    out_dim=cfg.dino.head_n_prototypes,
                    hidden_dim=cfg.dino.head_hidden_dim,
                    bottleneck_dim=cfg.dino.head_bottleneck_dim,
                    nlayers=cfg.dino.head_nlayers,
                )
                self.dino_loss = DINOLoss(cfg.dino.head_n_prototypes)
                if self.do_koleo:
                    logger.info("OPTIONS -- DINO -- applying KOLEO regularization")
                    self.koleo_loss = KoLeoLoss()
            else:
                logger.info("OPTIONS -- DINO -- not using DINO")

            if self.do_dino or self.do_ibot:
                student_model_dict["dino_head"] = dino_head()
                teacher_model_dict["dino_head"] = dino_head()

            logger.info("OPTIONS -- IBOT")
            logger.info(f"OPTIONS -- IBOT -- loss_weight: {cfg.ibot.loss_weight}")
            if self.do_ibot:
                self.ibot_loss_weight = cfg.ibot.loss_weight
                assert max(cfg.ibot.mask_ratio_min_max) > 0, "please provide a positive mask ratio tuple for ibot"
                assert cfg.ibot.mask_sample_probability > 0, "please provide a positive mask probability for ibot"
                self.ibot_out_dim = cfg.ibot.head_n_prototypes if self.ibot_separate_head else cfg.dino.head_n_prototypes
                self.ibot_patch_loss = iBOTPatchLoss(self.ibot_out_dim)
                if self.ibot_separate_head:
                    logger.info(f"OPTIONS -- IBOT -- loss_weight: {cfg.ibot.loss_weight}")
                    logger.info(f"OPTIONS -- IBOT -- head_n_prototypes: {cfg.ibot.head_n_prototypes}")
                    ibot_head = partial(
                        DINOHead,
                        in_dim=embed_dim,
                        out_dim=cfg.ibot.head_n_prototypes,
                        hidden_dim=cfg.ibot.head_hidden_dim,
                        bottleneck_dim=cfg.ibot.head_bottleneck_dim,
                        nlayers=cfg.ibot.head_nlayers,
                    )
                    student_model_dict["ibot_head"] = ibot_head()
                    teacher_model_dict["ibot_head"] = ibot_head()
                else:
                    logger.info("OPTIONS -- IBOT -- head shared with DINO")

        self.need_to_synchronize_fsdp_streams = True

        self.student = nn.ModuleDict(student_model_dict)
        self.teacher = nn.ModuleDict(teacher_model_dict)

        # ── 教师梯度策略 ───────────────────────────────────────────────
        if self.is_ddn:
            for p in self.teacher.parameters():
                p.requires_grad = True
            logger.info("DDN mode: teacher UNFROZEN (trained via reconstruction loss)")
        else:
            for p in self.teacher.parameters():
                p.requires_grad = False
        logger.info(f"Student and Teacher are built: they are both {cfg.student.arch} network.")

    def forward(self, inputs):
        raise NotImplementedError

    def backprop_loss(self, loss):
        if self.fp16_scaler is not None:
            self.fp16_scaler.scale(loss).backward()
        else:
            loss.backward()

    def forward_backward(self, images, teacher_temp, warmup_alpha=1.0):
        self._warmup_alpha = warmup_alpha
        if getattr(self, 'is_ddn', False):
            return self._forward_backward_ddn(images, teacher_temp)
        return self._forward_backward_vit(images, teacher_temp)

    # ═════════════════════════════════════════════════════════════════════
    # DDN: sparse coefficient MSE alignment
    # ═════════════════════════════════════════════════════════════════════

    # ═════════════════════════════════════════════════════════════════════
    # DDN loss helpers
    # ═════════════════════════════════════════════════════════════════════

    def _build_anatomy_weight(self, sam_seg, device, dtype):
        if sam_seg is None:
            return None
        w = torch.ones_like(sam_seg, dtype=dtype, device=device)
        w[sam_seg == 1] = self.anatomy_w_vertebra
        w[sam_seg == 2] = self.anatomy_w_disc
        w[sam_seg == 3] = self.anatomy_w_canal
        return w.unsqueeze(1)

    def _reconstruction_loss(self, recon, target, sam_segs=None,
                             aux_320=None, aux_160=None):
        # ── 主重建损失（640×640）─────────────────────────────────────────
        if sam_segs is not None and self.use_anatomy_weight:
            anatomy_w = self._build_anatomy_weight(
                sam_segs, target.device, target.dtype)
            L = (anatomy_w * (recon - target) ** 2).mean()
        else:
            L = F.mse_loss(recon, target)

        # ── 边缘感知损失 ─────────────────────────────────────────────────
        if self.edge_loss_weight > 0:
            edge_target = self.sobel(target)
            edge_recon  = self.sobel(recon)
            L = L + self.edge_loss_weight * F.mse_loss(edge_recon, edge_target)

        # ── 多尺度辅助重建 ──────────────────────────────────────────────
        if aux_320 is not None:
            target_320 = F.interpolate(target, size=(320, 320),
                                       mode='bilinear', align_corners=False)
            L = L + self.aux_320_weight * F.mse_loss(aux_320, target_320)
        if aux_160 is not None:
            target_160 = F.interpolate(target, size=(160, 160),
                                       mode='bilinear', align_corners=False)
            L = L + self.aux_160_weight * F.mse_loss(aux_160, target_160)

        return L

    # ═════════════════════════════════════════════════════════════════════
    # DDN: dual reconstruction + sparse alignment (enhanced)
    # ═════════════════════════════════════════════════════════════════════

    def _forward_backward_ddn(self, images, teacher_temp=None):
        """
        DDN enhanced forward-backward.

        Loss = teacher_recon_weight × L_teacher_recon
             + warmup_alpha × align_weight × L_align
             + warmup_alpha × student_recon_weight × L_student_recon
             + λ_l1·||gamma||₁ + λ_l2·||gamma||₂²

        L_recon includes: anatomy-weighted MSE + edge-aware loss + multi-scale aux
        """
        collated = images["collated_global_crops"].cuda(non_blocking=True)
        B = collated.shape[0] // 2

        teacher_inputs = collated[:B]
        student_inputs = collated[B:]

        masks = None
        if "collated_masks" in images:
            masks = images["collated_masks"][B:].cuda(non_blocking=True)

        sam_segs = images.get("collated_sam_segs", None)
        sam_segs_T = sam_segs[:B] if sam_segs is not None else None
        sam_segs_S = sam_segs[B:] if sam_segs is not None else None

        warmup_alpha = getattr(self, '_warmup_alpha', 1.0)

        # ── Teacher forward ──────────────────────────────────────────────
        teacher_out = self.teacher.backbone(teacher_inputs)
        gamma = teacher_out["gamma"]

        # ── Student forward ──────────────────────────────────────────────
        local_sizes = [320, 213, 160]
        local_crops_list = [
            F.interpolate(student_inputs, size=(s, s),
                          mode='bilinear', align_corners=False)
            for s in local_sizes
        ]

        student_out = self.student.backbone(
            [student_inputs, local_crops_list], masks=masks,
        )
        z_global = student_out[0]["sparse_coeff"]
        z_local  = student_out[1]["sparse_coeff"]

        # ── Shared decoder: 单次调用 (避免 FSDP 多卡同一模块调用两次) ──
        need_student_recon = (
            self.student_recon_weight > 0 and warmup_alpha > 0
        )
        if need_student_recon:
            combined = torch.cat([gamma, z_global], dim=0)  # [2B, 16, 80, 80]
            recon_out = self.decoder(combined,
                                     return_multiscale=self.use_multiscale_recon)
            if self.use_multiscale_recon:
                recon_all, aux_320_all, aux_160_all = recon_out
                recon_T, recon_S = recon_all.chunk(2, dim=0)
                aux_T_320, aux_S_320 = aux_320_all.chunk(2, dim=0)
                aux_T_160, aux_S_160 = aux_160_all.chunk(2, dim=0)
            else:
                recon_T, recon_S = recon_out.chunk(2, dim=0)
                aux_T_320 = aux_T_160 = aux_S_320 = aux_S_160 = None
        else:
            recon_out = self.decoder(gamma,
                                     return_multiscale=self.use_multiscale_recon)
            if self.use_multiscale_recon:
                recon_T, aux_T_320, aux_T_160 = recon_out
            else:
                recon_T = recon_out
                aux_T_320 = aux_T_160 = None
            recon_S = aux_S_320 = aux_S_160 = None

        # ── Teacher reconstruction ────────────────────────────────────────
        L_teacher_recon = self._reconstruction_loss(
            recon_T, teacher_inputs, sam_segs_T,
            aux_320=aux_T_320, aux_160=aux_T_160,
        )
        L_teacher_recon = self.teacher_recon_weight * L_teacher_recon

        # ── Sparsity regularization (L1 + L2 on gamma) ──────────────────
        L_sparsity = torch.tensor(0.0, device=gamma.device)
        if self.l1_sparsity_weight > 0:
            L_sparsity = L_sparsity + self.l1_sparsity_weight * gamma.abs().mean()
        if self.l2_sparsity_weight > 0:
            L_sparsity = L_sparsity + self.l2_sparsity_weight * gamma.square().mean()

        # ── Student reconstruction (warmup-gated) ────────────────────────
        L_student_recon = torch.tensor(0.0, device=z_global.device)
        if self.student_recon_weight > 0 and warmup_alpha > 0:
            L_student_recon = self._reconstruction_loss(
                recon_S, student_inputs, sam_segs_S,
                aux_320=aux_S_320, aux_160=aux_S_160,
            )
            L_student_recon = (self.student_recon_weight *
                               warmup_alpha * L_student_recon)

        # ── Sparse coefficient alignment (warmup-gated, gamma detached) ──
        L_align = torch.tensor(0.0, device=z_global.device)
        if self.align_weight > 0 and warmup_alpha > 0:
            gamma_target = gamma.detach()
            L_align = F.mse_loss(z_global, gamma_target) + \
                      F.mse_loss(z_local, gamma_target)
            L_align = self.align_weight * warmup_alpha * L_align

        # ── Total ────────────────────────────────────────────────────────
        loss = L_teacher_recon + L_align + L_student_recon + L_sparsity

        self.backprop_loss(loss)

        loss_dict = {
            "total_loss":     loss.detach(),
            "teacher_recon":  L_teacher_recon.detach(),
            "align_loss":     L_align.detach(),
            "student_recon":  L_student_recon.detach(),
            "sparsity_loss":  L_sparsity.detach(),
            "warmup_alpha":   torch.tensor(warmup_alpha, device=gamma.device),
            "gamma_sparsity": (gamma.abs() < 1e-6).float().mean().detach(),
        }
        return loss_dict

    # ═════════════════════════════════════════════════════════════════════
    # ViT DINOv2: original forward/backward (unchanged)
    # ═════════════════════════════════════════════════════════════════════

    def _forward_backward_vit(self, images, teacher_temp):
        fmha = _get_xformers_fmha()
        if fmha is None:
            raise ImportError("xformers is required for ViT training")

        n_global_crops = 2
        assert n_global_crops == 2
        n_local_crops = self.cfg.crops.local_crops_number

        global_crops = images["collated_global_crops"].cuda(non_blocking=True)
        local_crops = images.get("collated_local_crops", None)
        if local_crops is not None:
            local_crops = local_crops.cuda(non_blocking=True)
        masks = images["collated_masks"].cuda(non_blocking=True)
        mask_indices_list = images["mask_indices_list"].cuda(non_blocking=True)
        n_masked_patches_tensor = images["n_masked_patches"].cuda(non_blocking=True)
        n_masked_patches = mask_indices_list.shape[0]
        upperbound = images["upperbound"]
        masks_weight = images["masks_weight"].cuda(non_blocking=True)

        n_local_crops_loss_terms = max(n_local_crops * n_global_crops, 1)
        n_global_crops_loss_terms = (n_global_crops - 1) * n_global_crops

        do_dino = self.do_dino
        do_ibot = self.do_ibot
        ibot_loss_scale = 1.0 / n_global_crops

        @torch.no_grad()
        def get_teacher_output():
            x, n_global_crops_teacher = global_crops, n_global_crops
            teacher_backbone_output_dict = self.teacher.backbone(x, is_training=True)
            teacher_cls_tokens = teacher_backbone_output_dict["x_norm_clstoken"]
            teacher_cls_tokens = teacher_cls_tokens.chunk(n_global_crops_teacher)
            teacher_cls_tokens = torch.cat((teacher_cls_tokens[1], teacher_cls_tokens[0]))
            ibot_teacher_patch_tokens = teacher_backbone_output_dict["x_norm_patchtokens"]
            _dim = ibot_teacher_patch_tokens.shape[-1]
            n_cls_tokens = teacher_cls_tokens.shape[0]

            if do_ibot and not self.ibot_separate_head:
                buffer_tensor_teacher = ibot_teacher_patch_tokens.new_zeros(upperbound + n_cls_tokens, _dim)
                buffer_tensor_teacher[:n_cls_tokens].copy_(teacher_cls_tokens)
                torch.index_select(
                    ibot_teacher_patch_tokens.flatten(0, 1),
                    dim=0,
                    index=mask_indices_list,
                    out=buffer_tensor_teacher[n_cls_tokens : n_cls_tokens + n_masked_patches],
                )
                tokens_after_head = self.teacher.dino_head(buffer_tensor_teacher)
                teacher_cls_tokens_after_head = tokens_after_head[:n_cls_tokens]
                masked_teacher_patch_tokens_after_head = tokens_after_head[
                    n_cls_tokens : n_cls_tokens + n_masked_patches
                ]
            elif do_ibot and self.ibot_separate_head:
                buffer_tensor_teacher = ibot_teacher_patch_tokens.new_zeros(upperbound, _dim)
                torch.index_select(
                    ibot_teacher_patch_tokens.flatten(0, 1),
                    dim=0,
                    index=mask_indices_list,
                    out=buffer_tensor_teacher[:n_masked_patches],
                )
                teacher_cls_tokens_after_head = self.teacher.dino_head(teacher_cls_tokens)
                masked_teacher_patch_tokens_after_head = self.teacher.ibot_head(buffer_tensor_teacher)[
                    :n_masked_patches
                ]
            else:
                teacher_cls_tokens_after_head = self.teacher.dino_head(teacher_cls_tokens)
                masked_teacher_ibot_softmaxed_centered = None

            if self.cfg.train.centering == "centering":
                teacher_dino_softmaxed_centered_list = self.dino_loss.softmax_center_teacher(
                    teacher_cls_tokens_after_head, teacher_temp=teacher_temp
                ).view(n_global_crops_teacher, -1, *teacher_cls_tokens_after_head.shape[1:])
                self.dino_loss.update_center(teacher_cls_tokens_after_head)
                if do_ibot:
                    masked_teacher_patch_tokens_after_head = masked_teacher_patch_tokens_after_head.unsqueeze(0)
                    masked_teacher_ibot_softmaxed_centered = self.ibot_patch_loss.softmax_center_teacher(
                        masked_teacher_patch_tokens_after_head[:, :n_masked_patches], teacher_temp=teacher_temp
                    )
                    masked_teacher_ibot_softmaxed_centered = masked_teacher_ibot_softmaxed_centered.squeeze(0)
                    self.ibot_patch_loss.update_center(masked_teacher_patch_tokens_after_head[:n_masked_patches])

            elif self.cfg.train.centering == "sinkhorn_knopp":
                teacher_dino_softmaxed_centered_list = self.dino_loss.sinkhorn_knopp_teacher(
                    teacher_cls_tokens_after_head, teacher_temp=teacher_temp
                ).view(n_global_crops_teacher, -1, *teacher_cls_tokens_after_head.shape[1:])

                if do_ibot:
                    masked_teacher_ibot_softmaxed_centered = self.ibot_patch_loss.sinkhorn_knopp_teacher(
                        masked_teacher_patch_tokens_after_head,
                        teacher_temp=teacher_temp,
                        n_masked_patches_tensor=n_masked_patches_tensor,
                    )
            else:
                raise NotImplementedError

            return teacher_dino_softmaxed_centered_list, masked_teacher_ibot_softmaxed_centered

        teacher_dino_softmaxed_centered_list, masked_teacher_ibot_softmaxed_centered = get_teacher_output()
        reshard_fsdp_model(self.teacher)

        loss_dict = {}
        loss_accumulator = 0

        student_global_backbone_output_dict, student_local_backbone_output_dict = self.student.backbone(
            [global_crops, local_crops], masks=[masks, None], is_training=True
        )

        inputs_for_student_head_list = []

        # 1a: local crops cls tokens
        if n_local_crops > 0 and student_local_backbone_output_dict is not None:
            student_local_cls_tokens = student_local_backbone_output_dict["x_norm_clstoken"]
            inputs_for_student_head_list.append(student_local_cls_tokens.unsqueeze(0))

        # 1b: global crops cls tokens
        student_global_cls_tokens = student_global_backbone_output_dict["x_norm_clstoken"]
        inputs_for_student_head_list.append(student_global_cls_tokens.unsqueeze(0))

        # 1c: global crops patch tokens
        if do_ibot:
            _dim = student_global_backbone_output_dict["x_norm_clstoken"].shape[-1]
            ibot_student_patch_tokens = student_global_backbone_output_dict["x_norm_patchtokens"]
            buffer_tensor_patch_tokens = ibot_student_patch_tokens.new_zeros(upperbound, _dim)
            buffer_tensor_patch_tokens[:n_masked_patches].copy_(
                torch.index_select(ibot_student_patch_tokens.flatten(0, 1), dim=0, index=mask_indices_list)
            )
            if not self.ibot_separate_head:
                inputs_for_student_head_list.append(buffer_tensor_patch_tokens.unsqueeze(0))
            else:
                student_global_masked_patch_tokens_after_head = self.student.ibot_head(buffer_tensor_patch_tokens)[
                    :n_masked_patches
                ]

        # 2: run head with block-diagonal attention mask
        _attn_bias, cat_inputs = fmha.BlockDiagonalMask.from_tensor_list(inputs_for_student_head_list)
        outputs_list = _attn_bias.split(self.student.dino_head(cat_inputs))

        # 3a: local crops cls tokens
        if n_local_crops > 0:
            student_local_cls_tokens_after_head = outputs_list.pop(0).squeeze(0)

        # 3b: global crops cls tokens
        student_global_cls_tokens_after_head = outputs_list.pop(0).squeeze(0)

        # 3c: global crops patch tokens
        if do_ibot and not self.ibot_separate_head:
            student_global_masked_patch_tokens_after_head = outputs_list.pop(0).squeeze(0)[:n_masked_patches]

        if n_local_crops > 0:
            dino_local_crops_loss = self.dino_loss(
                student_output_list=student_local_cls_tokens_after_head.chunk(n_local_crops),
                teacher_out_softmaxed_centered_list=teacher_dino_softmaxed_centered_list,
            ) / (n_global_crops_loss_terms + n_local_crops_loss_terms)
            loss_dict["dino_local_crops_loss"] = dino_local_crops_loss
            loss_accumulator += self.dino_loss_weight * dino_local_crops_loss

        if do_dino:
            dino_global_crops_loss = (
                self.dino_loss(
                    student_output_list=[student_global_cls_tokens_after_head],
                    teacher_out_softmaxed_centered_list=[
                        teacher_dino_softmaxed_centered_list.flatten(0, 1)
                    ],
                )
                * (2 / max(n_global_crops_loss_terms + n_local_crops_loss_terms, 1))
            )
            loss_dict["dino_global_crops_loss"] = dino_global_crops_loss
            loss_accumulator += self.dino_loss_weight * dino_global_crops_loss

            student_cls_tokens = student_global_cls_tokens

            if self.do_koleo:
                koleo_loss = self.cfg.dino.koleo_loss_weight * sum(
                    self.koleo_loss(p) for p in student_cls_tokens.chunk(2)
                )
                loss_accumulator += koleo_loss
                loss_dict["koleo_loss"] = koleo_loss / 2

        if do_ibot:
            ibot_patch_loss = (
                self.ibot_patch_loss.forward_masked(
                    student_global_masked_patch_tokens_after_head,
                    masked_teacher_ibot_softmaxed_centered,
                    student_masks_flat=masks,
                    n_masked_patches=n_masked_patches,
                    masks_weight=masks_weight,
                )
                * 2 * ibot_loss_scale
            )
            loss_dict["ibot_loss"] = ibot_patch_loss / 2
            loss_accumulator += self.ibot_loss_weight * ibot_patch_loss

        self.backprop_loss(loss_accumulator)
        return loss_dict

    # ═════════════════════════════════════════════════════════════════════
    # Distributed training & param groups
    # ═════════════════════════════════════════════════════════════════════

    def prepare_for_distributed_training(self):
        logger.info("DISTRIBUTED FSDP -- preparing model for distributed training")
        if has_batchnorms(self.student):
            raise NotImplementedError

        if getattr(self, 'is_ddn', False):
            # DDN: teacher & student have different architectures,
            # so skip teacher-from-student weight copy.
            # Teacher + Student + Decoder all trainable.
            for k, v in self.student.items():
                student_model_cfg = self.cfg.compute_precision.student[k]
                self.student[k] = get_fsdp_wrapper(student_model_cfg, modules_to_wrap={})(self.student[k])
                teacher_model_cfg = self.cfg.compute_precision.teacher[k]
                self.teacher[k] = get_fsdp_wrapper(teacher_model_cfg, modules_to_wrap={})(self.teacher[k])
            # FSDP-wrap shared decoder (use student backbone cfg as reference)
            decoder_cfg = self.cfg.compute_precision.student["backbone"]
            self.decoder = get_fsdp_wrapper(decoder_cfg, modules_to_wrap={})(self.decoder)
        else:
            BlockChunk = _get_block_chunk()
            modules_to_wrap = {BlockChunk} if BlockChunk is not None else {}
            for k, v in self.student.items():
                self.teacher[k].load_state_dict(self.student[k].state_dict())
                student_model_cfg = self.cfg.compute_precision.student[k]
                self.student[k] = get_fsdp_wrapper(student_model_cfg, modules_to_wrap=modules_to_wrap)(self.student[k])
                teacher_model_cfg = self.cfg.compute_precision.teacher[k]
                self.teacher[k] = get_fsdp_wrapper(teacher_model_cfg, modules_to_wrap=modules_to_wrap)(self.teacher[k])

    def get_params_groups(self):
        if getattr(self, 'is_ddn', False):
            # DDN: student + teacher + decoder all trainable
            all_params = []
            for name, model in [("student", self.student), ("teacher", self.teacher)]:
                for p in model.parameters():
                    if p.requires_grad:
                        all_params.append(p)
            for p in self.decoder.parameters():
                if p.requires_grad:
                    all_params.append(p)
            return [{
                "params": all_params,
                "lr_multiplier": 1.0,
                "wd_multiplier": 1.0,
                "is_last_layer": False,
            }]
        # ViT: layer-wise decay param groups
        all_params_groups = []
        for m in self.student.values():
            all_params_groups += self.get_maybe_fused_params_for_submodel(m)
        return all_params_groups

    def get_maybe_fused_params_for_submodel(self, m):
        params_groups = get_params_groups_with_decay(
            model=m,
            lr_decay_rate=self.cfg.optim.layerwise_decay,
            patch_embed_lr_mult=self.cfg.optim.patch_embed_lr_mult,
        )
        fused_params_groups = fuse_params_groups(params_groups)
        logger.info("fusing param groups")
        for g in fused_params_groups:
            g["foreach"] = True
        return fused_params_groups

    def train(self):
        super().train()
        if not getattr(self, 'is_ddn', False):
            self.teacher.eval()
