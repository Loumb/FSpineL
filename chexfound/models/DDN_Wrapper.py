# chexfound/models/DDN_Wrapper.py
#
# ============================================================
# 改动说明（稀疏系数对齐版 v5 — 根据论文原始代码修正）：
#
#  分析论文 SingleTFCnet_Dconv 的实际结构后修正：
#
#    论文: input[28] → DyConv(WA)[49] → SoftThreshold[49] → channel_mapper[28]
#    学生: z[49]      → DyConv(WA)[49] → SoftThreshold[49] → z'[49]
#    无 WS→re-encode 循环（v4 错误添加了这一步）
#
#  v4 的错误：
#    SynthesisDictLayer 中加了 z→WS(49→32)→feat→re_analysis(32→49)→z_delta
#    论文没有这一步！WS 是 1×1 Conv 49→16（在 DictBlock 末尾），不在迭代内部。
#
#  v5 修正：
#    ① DictLayer 恢复论文 SingleTFCnet_Dconv 模式: DyConv(WA) → Soft
#    ② 去掉 SynthesisDictLayer 中的 WS→re-encode
#    ③ 1×1 Conv 49→16 (to_gamma) = WS（真正的综合字典，在外层）
#    ④ 完整流程: 图像 → WA(SparseCoeffGen) → [WA→Soft]×4 → WS(1×1Conv 49→16) → gamma
#
#  接口完全兼容（无任何外部修改）：
#    teacher.backbone(x)     → {"gamma": [2B, 16, 80, 80]}
#    student.backbone([g,lc], masks) → (global_dict, local_dict)
# ============================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from chexfound.models.DDN import ConvLista_T
from chexfound.models.DDN import SoftThreshold
from chexfound.models.DDN import DyConv
from chexfound.models.DDN import AFG

# 延迟导入以避免循环依赖
# chexfound.models.DDN 定义在 DDN.py 中


# =============================================================================
# §0  教师编码器公用子模块（未改动）
# =============================================================================

class _DownsampleBlock(nn.Module):
    def __init__(self, channels: int, num_groups: int = 8):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, stride=2, padding=1, bias=False)
        self.norm = nn.GroupNorm(num_groups=num_groups, num_channels=channels)
        self.act  = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(self.conv.weight.dtype)
        return self.act(self.norm(self.conv(x)))


# =============================================================================
# §1  教师主干（未改动）
# =============================================================================

class DDNBackboneWrapper(nn.Module):
    def __init__(
        self,
        img_size: int        = 640,
        patch_size: int      = 16,
        in_chans: int        = 1,
        embed_dim: int       = 512,
        num_filters: int     = 16,
        use_checkpoint: bool = True,
    ):
        super().__init__()
        self.img_size    = img_size
        self.patch_size  = patch_size
        self.num_filters = num_filters
        _num_groups = 8
        _ddn_kw = dict(kernel_size=3, stride=1, unfoldings=3, use_checkpoint=use_checkpoint)
        self.ddn1  = ConvLista_T(in_channels=in_chans,    num_filters=num_filters, **_ddn_kw)
        self.down1 = _DownsampleBlock(num_filters, _num_groups)
        self.ddn2  = ConvLista_T(in_channels=num_filters, num_filters=num_filters, **_ddn_kw)
        self.down2 = _DownsampleBlock(num_filters, _num_groups)
        self.ddn3  = ConvLista_T(in_channels=num_filters, num_filters=num_filters, **_ddn_kw)
        self.down3 = _DownsampleBlock(num_filters, _num_groups)
        self.ddn4  = ConvLista_T(in_channels=num_filters, num_filters=num_filters, **_ddn_kw)

    def _ddn_forward(self, ddn_module: nn.Module, x: torch.Tensor) -> torch.Tensor:
        x = x.to(ddn_module.channel_proj.weight.dtype)
        return ddn_module((x, None, None))

    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        f1  = self._ddn_forward(self.ddn1, x); f1d = self.down1(f1);  del f1
        f2  = self._ddn_forward(self.ddn2, f1d); f2d = self.down2(f2);  del f1d, f2
        f3  = self._ddn_forward(self.ddn3, f2d); f3d = self.down3(f3);  del f2d, f3
        bn  = self._ddn_forward(self.ddn4, f3d); del f3d
        return bn

    def prepare_tokens_with_masks(self, x, masks=None):
        if x.shape[1] != 1:
            x = x.mean(dim=1, keepdim=True)
        # 统一模型计算精度（FSDP mixed-precision 可能改变输入 dtype）
        x = x.to(self.ddn1.channel_proj.weight.dtype)
        gamma = self._encode(x)                          # [B, 16, 80, 80] 已稀疏
        return {"gamma": gamma}

    def forward_features(self, x, masks=None, **kwargs):
        return self.prepare_tokens_with_masks(x, masks)
    def forward(self, x, masks=None, **kwargs):
        return self.prepare_tokens_with_masks(x, masks)


# =============================================================================
# §2  学生主干子模块（v5 — 根据论文原始代码修正）
# =============================================================================
#
# 与论文 SingleTFCnet_Dconv 的对应关系：
#
#   论文 SingleTFCnet_Dconv:
#     input [B,28,H,W] → DyConv(WA)[B,49,H,W] → SoftThreshold[B,49,H,W]
#       → channel_mapper(1×1,49→28)[B,28,H,W]
#
#   v5 学生模型 DictLayer（在稀疏系数域 49ch 中迭代）:
#     z [B,49,H,W] → DyConv(WA)[B,49,H,W] → SoftThreshold[B,49,H,W] → z'
#                                                                           ↑残差连接 z' = z + Δz_hat
#
#   v5 学生模型 DictBlock（综合字典 = 多次 WA→Soft 迭代 + 外层 WS 投影）:
#     z₀ → [DictLayer × 4 (WA→Soft)] → 1×1 Conv(49→16) = WS → γ [B,16,H,W]
#                                        ↑
#                         综合字典在这里，不在迭代内部
#
#   v4 的错误：
#     把 WS→re-encode 放到了每轮迭代内部 (SynthesisDictLayer)
#     → 冗余的 49→32→49 映射，论文没有这样做
#
#   v5 的正确映射：
#     ① DictLayer 内的 DyConv = WA（分析字典，图像→稀疏），对应论文 conv_w
#     ② DictLayer 内的 SoftThreshold = 稀疏化，对应论文 soft_threshold
#     ③ DictBlock 末尾的 1×1 Conv 49→16 = WS（综合字典，稀疏→特征域）
#       对应论文中 WS（将稀疏系数映射到目标输出空间）
#     ④ 整个流程：WA→Soft 迭代 ×4（纯稀疏域）→ WS 投影（跨域）→ gamma
#


# ── BCNet：自适应阈值 ─────────────────────────────────────────────────────────

class BCNet(nn.Module):
    """自适应阈值生成：DyConv 特征 → Softplus 阈值图"""
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, max(in_channels, out_channels), 3, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(max(in_channels, out_channels), max(in_channels, out_channels),
                      3, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(max(in_channels, out_channels), out_channels, 1, bias=False),
            nn.Softplus(),
        )
    def forward(self, x):
        return self.net(x)


# ── SparseCoeffGenerator v2：图像 → 稀疏系数 z₀（分析字典 WA 角色）────────────

class SparseCoeffGenerator(nn.Module):
    """
    ★ 分析字典（论文 WA 角色）

    图像 → 8× 编码器 → DyConv → BCNet(阈值) → SoftThreshold → z₀ [B, 49, H/8, W/8]

    对应论文 SingleTFCnet_Dconv 的前半部分：
      论文: input_feat[28] → DyConv(WA)[49] → BCNet → SoftThreshold → z[49]
      这里: image → encoder[28] → DyConv → BCNet → SoftThreshold → z₀[49]
    """
    def __init__(self, in_chans: int = 1, internal_ch: int = 28, sparse_ch: int = 16):
        super().__init__()
        self.sparse_ch = sparse_ch
        self.encoder = nn.Sequential(
            nn.Conv2d(in_chans, internal_ch // 2, 3, stride=2, padding=1, bias=False),
            nn.GroupNorm(max(1, (internal_ch // 2) // 7), internal_ch // 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(internal_ch // 2, internal_ch, 3, stride=2, padding=1, bias=False),
            nn.GroupNorm(max(1, internal_ch // 4), internal_ch),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(internal_ch, internal_ch, 3, stride=2, padding=1, bias=False),
            nn.GroupNorm(max(1, internal_ch // 4), internal_ch),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(internal_ch, internal_ch, 3, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
        )
        # WA: 28→49 的稀疏编码（论文中 kernel_size=7→49ch，这里用 3→49ch 通过 proj_sparse 扩展）
        self.dy_conv = DyConv(internal_ch, kernel_size=3)
        self.threshold_net = BCNet(internal_ch, sparse_ch)
        self.proj_sparse = nn.Conv2d(internal_ch, sparse_ch, 1, bias=False)
        self.soft_threshold = SoftThreshold()

    def forward(self, x):
        x = x.to(self.encoder[0].weight.dtype)
        feat = self.encoder(x)                           # [B, 28, H/8, W/8]
        encoded = self.dy_conv(feat)                      # [B, 28, H/8, W/8]
        threshold = self.threshold_net(encoded)           # [B, 16, H/8, W/8]
        z0 = self.proj_sparse(encoded)                    # [B, 16, H/8, W/8]
        z0 = self.soft_threshold(z0, threshold)           # [B, 16, H/8, W/8]
        return z0


# ── DictLayer：稀疏编码层（论文 SingleTFCnet_Dconv 模式，WA + SoftThreshold）─

class DictLayer(nn.Module):
    """
    ★ 稀疏编码层 = WA + SoftThreshold（对应论文 SingleTFCnet_Dconv 的核心）

    数据流：
        z [B, 16, H, W]
          → DyConv(WA): [B, 16, H, W]        （分析与字典变换）
          → + z (残差): [B, 16, H, W]          （增强训练稳定性）
          → 自适应阈值: scale * scale_factor * Softplus(z_res) → threshold [1]
          → SoftThreshold: [B, 16, H, W]       （稀疏化）
          → z' [B, 16, H, W]

    论文对照:
      论文 SingleTFCnet_Dconv: input → DyConv(WA)[49] → SoftThreshold → channel_mapper[28]
      这里 DictLayer:          z     → DyConv(WA)[16] → +residual → SoftThreshold → z'
      区别: 通道数 16 而非 49（与教师 gamma 对齐）；多了残差连接 (稳定训练)
    """
    def __init__(self, sparse_ch: int = 16, scale_factor: float = 1.0):
        super().__init__()
        # WA: 动态分析稀疏变换（论文 conv_w）
        self.dict_conv = DyConv(sparse_ch, kernel_size=3)

        # 自适应阈值生成（类似论文 BCNet + sigma 的简化版）
        self.threshold_head = nn.Sequential(
            nn.Conv2d(sparse_ch, sparse_ch, 3, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(sparse_ch, 1, 1, bias=False),
            nn.Softplus(),
        )
        self.scale = nn.Parameter(torch.tensor(0.05))
        self.scale_factor = scale_factor  # 细尺度↓/粗尺度↑
        self.soft_threshold = SoftThreshold()

    def forward(self, z):
        """
        Args:
            z: 稀疏系数 [B, 16, H, W]

        Returns:
            z_out: 稀疏编码后的系数 [B, 16, H, W]
        """
        z = z.to(self.dict_conv.afg.conv_feat.weight.dtype)
        # WA 分析字典（同论文 conv_w）
        delta = self.dict_conv(z)                          # [B, 49, H, W]

        # 残差连接
        z_res = z + delta

        # 生成阈值（同论文 BCNet + sigma）
        threshold = self.scale.abs() * self.scale_factor * self.threshold_head(z_res)

        # 软阈值稀疏化（同论文 soft_threshold）
        z_out = self.soft_threshold(z_res, threshold)      # [B, 49, H, W]

        return z_out


# ── DictBlock：多次 DictLayer 迭代 + WS 投影 ──────────────────────────────────

class DictBlock(nn.Module):
    """
    ★ 字典块 = 多次 WA→Soft 迭代，输出 16ch 稀疏系数（无 WS 投影）

    数据流：
        z₀ [B, 16, H, W]
          → [DictLayer × 4]                    每轮: WA→residual→Soft → 稀疏系数迭代
          → z [B, 16, H, W]                    直接与教师 gamma 对齐

    与教师对齐方式：
        DictBlock 输出的 z [B, 16, H, W] 与教师 gamma [B, 16, H, W] 直接算 MSE
    """
    def __init__(self, sparse_ch: int = 16, num_iters: int = 4,
                 scale_factor: float = 1.0):
        super().__init__()
        assert num_iters >= 2, "需要至少 2 轮字典迭代"
        self.layers = nn.ModuleList([
            DictLayer(sparse_ch, scale_factor) for _ in range(num_iters)
        ])

    def forward(self, z0):
        """
        Args:
            z0: 初始稀疏系数 [B, 16, H, W]

        Returns:
            z: 迭代后的稀疏系数 [B, 16, H, W]
        """
        z = z0
        for layer in self.layers:
            z = layer(z)
        return z


# ═════════════════════════════════════════════════════════════════════════════
# §3  MultiScaleSparseFusion — 多尺度注意力融合（保持 v3 设计）
# ═════════════════════════════════════════════════════════════════════════════
#
# 融合局部分支 3 个尺度的 gamma，输出统一 80×80
# =============================================================================

class MultiScaleSparseFusion(nn.Module):
    """
    多尺度稀疏系数注意力融合（49ch 版）

    Input:  list of sparse_coeff [B, 49, H_i, W_i] × S
    Output: fused sparse_coeff [B, 49, target, target]

    数据流：
        z_i → upsample to target → stack [B, S, 49, H, W]
        → concat for spatial attention [B, S*49, H, W]
        → 2× conv + softmax → spatial_weights [B, S, H, W]
        → channel_gating [S, 49] × spatial_weights
        → weighted_sum → [B, 49, H, W]
        → soft_threshold → output
    """
    def __init__(self, sparse_ch: int = 16, num_scales: int = 3):
        super().__init__()
        self.num_scales = num_scales
        self.sparse_ch  = sparse_ch

        # ── 空间注意力 ──────────────────────────────────────────
        self.spatial_attn = nn.Sequential(
            nn.Conv2d(sparse_ch * num_scales, sparse_ch * num_scales,
                      3, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(sparse_ch * num_scales, sparse_ch, 3, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(sparse_ch, num_scales, 1, bias=False),
        )

        # ── 通道门控 ────────────────────────────────────────────
        self.channel_gate = nn.Parameter(
            torch.ones(num_scales, sparse_ch)
        )

        # ── 医学边缘增强 ─────────────────────────────────────────
        sobel_k = torch.tensor([[[[-1., 0., 1.],
                                  [-2., 0., 2.],
                                  [-1., 0., 1.]]]])
        self.register_buffer("sobel_x", sobel_k)
        self.register_buffer("sobel_y", sobel_k.transpose(-1, -2))

        # ── 融合后稀疏性保持 ─────────────────────────────────────
        self.post_threshold = nn.Parameter(torch.tensor(0.01))
        self.soft_threshold = SoftThreshold()

    def _edge_strength(self, x: torch.Tensor) -> torch.Tensor:
        # edge_x = F.conv2d(x.mean(dim=1, keepdim=True), self.sobel_x, padding=1)
        # edge_y = F.conv2d(x.mean(dim=1, keepdim=True), self.sobel_y, padding=1)

        x_in = x.mean(dim=1, keepdim=True)
        edge_x = F.conv2d(x_in, self.sobel_x.to(x.dtype), padding=1)
        edge_y = F.conv2d(x_in, self.sobel_y.to(x.dtype), padding=1)

        return (edge_x.abs() + edge_y.abs()).detach()

    def forward(self, z_list: list, target_size: int = 80):
        """
        Args:
            z_list: [z_1, z_2, z_3] 各 [B, 49, H_i, W_i]
            target_size: 融合目标尺寸

        Returns:
            fused: [B, 49, target_size, target_size]
        """
        S = len(z_list)

        # ── ① 统一分辨率 ──────────────────────────────────────────
        param_dtype = self.spatial_attn[0].weight.dtype  # 模型计算精度
        aligned = []
        for z in z_list:
            if z.shape[-1] != target_size:
                z = F.interpolate(z.float(), size=(target_size, target_size),
                                  # mode='bilinear', align_corners=False).to(z.dtype)
                                mode = 'bilinear', align_corners = False).to(param_dtype)
            # aligned.append(z)
            aligned.append(z.to(param_dtype))

        stack = torch.stack(aligned, dim=1)          # [B, S, 49, H, W]
        B, _, C, H, W = stack.shape

        # ── ② 空间注意力 ──────────────────────────────────────────
        concat = stack.reshape(B, S * C, H, W)
        logits = self.spatial_attn(concat)            # [B, S, H, W]

        # ── ③ 边缘增强 ────────────────────────────────────────────
        edge_map = self._edge_strength(aligned[0])
        edge_bias = torch.zeros_like(logits)
        edge_bias[:, 0, :, :] = edge_map.squeeze(1) * 3.0
        logits = logits + edge_bias

        spatial_weights = F.softmax(logits, dim=1)   # [B, S, H, W]

        # ── ④ 通道门控 ────────────────────────────────────────────
        cg = torch.sigmoid(self.channel_gate).view(1, S, C, 1, 1)

        # ── ⑤ 加权融合 ────────────────────────────────────────────
        fused = (stack * cg * spatial_weights.unsqueeze(2)).sum(dim=1)

        # ── ⑥ 稀疏性保持 ──────────────────────────────────────────
        fused = self.soft_threshold(fused, self.post_threshold.abs())

        return fused


# =============================================================================
# §4  学生主干 DSTNetBackboneWrapper（16ch 稀疏系数对齐版）
# =============================================================================
#
# 设计：学生与教师都在 16ch 稀疏系数空间直接对齐
#
#   教师:
#     DDNBackboneWrapper → ConvLista_T(T+SoftThreshold) → gamma [16ch]  已稀疏
#
#   学生:
#     全局: 掩码图像 → SparseCoeffGen(WA) → [DictLayer×4] → z [16ch]    稀疏
#     局部: 下采样图像 → SparseCoeffGen(WA) → [DictLayer×4] → z_i → 融合 → z [16ch]
#
#   对齐: MSE(global_z, teacher_gamma) + MSE(local_z, teacher_gamma)
#         ↑ 同一空间，无需任何投影/MLP
#
#   注意:
#     - DictBlock 无 WS 投影，直接输出 16ch 稀疏系数
#     - teacher gamma 已由 ConvLista_T 内部 SoftThreshold 稀疏化
# =============================================================================
#
# 与论文 DSTNet / SingleTFCnet_Dconv 的对应关系：
#
#   论文 DSTNet 公式(11):  x_{k+0.5} = WS · Soft(WA · x_k, ι)
#   论文 SingleTFCnet_Dconv: input → DyConv(WA) → SoftThreshold → channel_mapper
#
#   v5 学生模型映射:
#
#     SparseCoeffGenerator  = 图像级 WA（图像 → 稀疏系数 z₀）
#       图像 → 8×编码器 → DyConv → SoftThreshold → z₀ [B, 49, 80, 80]
#
#     DictLayer             = 稀疏域 WA→Soft 迭代
#       z → DyConv(WA) → +residual → SoftThreshold → z'
#       对应论文每轮 SingleTFCnet_Dconv 的 WA→Soft 部分
#
#     DictBlock → to_gamma  = WS 综合字典（在末尾，不在迭代内）
#       [DictLayer × 4] → 1×1 Conv(49→16) → γ [B, 16, 80, 80]
#       ★ 1×1 Conv 映射到 gamma 空间 = 综合字典（论文中 WS 映射到图像域）
#
#   v4 错误（已修正）:
#     SynthesisDictLayer 在每轮内部做了 WS(49→32)→re_analysis(32→49)
#     → 论文没有这个，v5 已去掉
#
#   v5 完整数据流:
#     图像 → SparseCoeffGen(WA₀) → [DyConv(WA)→Soft]×4 → 1×1Conv(WS)(49→16) → γ
#                                  ↑分析字典迭代×4  ↑综合字典（只1次，在最外层）
# =============================================================================

class DSTNetBackboneWrapper(nn.Module):
    """
    学生模型主干（稀疏系数对齐版）

    架构：

        ┌─ SparseCoeffGenerator（图像级 WA）──────────────┐
        │    图像 → 8× 编码器 → DyConv → SoftThreshold → z₀│
        └──────────────────────────────────────────────────┘

        ┌─ DictBlock（稀疏域 WA 迭代，无 WS 投影）────────┐
        │    z₀ → [DictLayer × 4: DyConv→residual→Soft]   │
        │    → z [B, 16, 80, 80]    ← 直接输出稀疏系数     │
        └──────────────────────────────────────────────────┘

        ┌─ 全局分支 ───────────────────────────────────────┐
        │  掩码 [640] → SparseCoeffGen → DictBlock → z     │
        └──────────────────────────────────────────────────┘

        ┌─ 局部分支 ───────────────────────────────────────┐
        │  预下采样 [320,213,160] → DictBlock_i → z_i      │
        │  → MultiScaleSparseFusion → z_fused [B, 16, 80]  │
        └──────────────────────────────────────────────────┘

    对齐方式：
        教师 gamma [B, 16, 80, 80] (ConvLista_T 已稀疏化)
        ← MSE → 学生 global z / local z_fused [B, 16, 80, 80]
    """

    def __init__(
        self,
        img_size: int    = 640,
        patch_size: int  = 16,
        in_chans: int    = 1,
        embed_dim: int   = 512,
        num_filters: int = 16,
        num_iter: int    = 3,
    ):
        super().__init__()
        self.img_size    = img_size
        self.patch_size  = patch_size
        self.num_filters = num_filters
        self.in_chans    = in_chans

        # ── 共享稀疏编码器（图像级 WA）─────────────────────────────────
        self.sparse_encoder = SparseCoeffGenerator(
            in_chans=in_chans, internal_ch=28, sparse_ch=num_filters,
        )

        # ── 全局分支字典块（DictLayer×4，输出 16ch 稀疏系数）───────────
        self.dict_global = DictBlock(
            sparse_ch=num_filters, num_iters=4, scale_factor=1.0,
        )

        # ── 局部分支三尺度字典（不同 scale_factor）────────────────────
        self.dict_local_scales = nn.ModuleList([
            DictBlock(sparse_ch=num_filters, num_iters=4, scale_factor=0.8),
            DictBlock(sparse_ch=num_filters, num_iters=4, scale_factor=1.0),
            DictBlock(sparse_ch=num_filters, num_iters=4, scale_factor=1.2),
        ])

        # ── 局部分支多尺度注意力融合（16ch 稀疏系数）─────────────────
        self.fusion = MultiScaleSparseFusion(
            sparse_ch=num_filters, num_scales=3,
        )

    def _apply_pixel_mask(self, x, masks):
        B, C, H, W = x.shape
        p = self.patch_size
        gh, gw = H // p, W // p
        mask_sp = masks.view(B, gh, gw)[:, None, :, :]
        mask_sp = mask_sp.repeat_interleave(p, 2).repeat_interleave(p, 3)
        x = x.clone()
        x[mask_sp.bool()] = 0.0
        return x

    # ── 全局分支 ──────────────────────────────────────────────────────

    def _global_forward(self, x, masks=None):
        """
        全局分支：
        掩码图像 → SparseCoeffGen(图像级WA) → [WA→Soft]×4 → 稀疏系数 z

        Returns: {"sparse_coeff": [2B, 16, 80, 80], "masks": masks}
        """
        if x.shape[1] != 1:
            x = x.mean(dim=1, keepdim=True)
        x = x.to(self.sparse_encoder.encoder[0].weight.dtype)
        if masks is not None:
            x = self._apply_pixel_mask(x, masks)

        z0    = self.sparse_encoder(x)                     # [2B, 16, 80, 80]
        z     = self.dict_global(z0)                        # [2B, 16, 80, 80]
        return {"sparse_coeff": z, "masks": masks}

    # ── 局部分支（三尺度 DictBlock + 注意力融合）──────────────────────

    def _local_forward(self, local_crops_list):
        """
        局部分支：3 个预下采样图 → DictBlock_i → z_i → 融合

        local_crops_list: [2B,1,320], [2B,1,213], [2B,1,160]

        每个尺度独立经过 WA→Soft 迭代：
          320 → dict_local_scales[0] (scale_factor=0.8)
          213 → dict_local_scales[1] (scale_factor=1.0)
          160 → dict_local_scales[2] (scale_factor=1.2)
        """
        sizes = [c.shape[-1] for c in local_crops_list]
        scale_order = sorted(range(len(sizes)), key=lambda i: sizes[i], reverse=True)

        z_list = []
        for idx, crop in enumerate(local_crops_list):
            z0  = self.sparse_encoder(crop)                     # [B,16,H/8,W/8]
            pos = scale_order.index(idx)
            z   = self.dict_local_scales[pos](z0)                # [B,16,H/8,W/8]
            z_list.append(z)

        z_fused = self.fusion(z_list, target_size=80)           # [2B, 16, 80, 80]
        return {"sparse_coeff": z_fused}

    # ── 对外主接口 ───────────────────────────────────────────────────

    def forward(self, x, masks=None, local_crops_list=None,
                local_offsets=None, local_image_indices=None,
                is_local_flags=None, is_training=False):
        if isinstance(x, list):
            global_crops = x[0]
            lc_list      = x[1]
            global_out = self._global_forward(global_crops, masks)
            local_out  = self._local_forward(lc_list)
            return global_out, local_out
        return self._global_forward(x, masks)
