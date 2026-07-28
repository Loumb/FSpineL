# chexfound/models/DDN.py
#
# ============================================================
# 改动说明（稀疏系数对齐版）：
#
#   · ConvLista_T_Flex.forward 最后返回 gamma_k（原始稀疏系数）
#     而非 self.act(gamma_k)（稠密激活特征）
#     原因：在稀疏系数层面验证字典学习对自监督的提升；
#     soft_threshold 已保证大量位置严格为 0（稀疏性），去掉激活
#     以保留符号信息并方便 MSE 直接对齐
#
#   · self.act 保留实例但不调用
#     切回稠密模式：将 forward 最后一行改为 return self.act(gamma_k)
#
#   · 其余逻辑（AFG / DyConv / AdaptiveThresholding）完全不变
# ============================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as ckpt

try:
    from chexfound.networks.fshadowformer import ShadowFormer
except ImportError:
    import warnings
    warnings.warn("ShadowFormer not found, using 1x1 conv fallback for non-local branch")
    class ShadowFormer(nn.Module):
        """Minimal fallback when chexfound/networks/fshadowformer.py is missing."""
        def __init__(self, img_size=None, img_siwin_size=4, in_chans=32):
            super().__init__()
            self.proj = nn.Conv2d(in_chans, in_chans, 1)
        def forward(self, x):
            return self.proj(x)


# =============================================================================
# §1  工具函数
# =============================================================================

def get_act(name: str) -> nn.Module:
    name = name.lower()
    if name == "leakyrelu":
        return nn.LeakyReLU(negative_slope=0.2, inplace=True)
    elif name == "relu":
        return nn.ReLU(inplace=True)
    elif name == "prelu":
        return nn.PReLU()
    raise ValueError(f"Unsupported activation: {name}")


# =============================================================================
# §2  基础算子
# =============================================================================

class SoftThreshold(nn.Module):
    """
    mask 式软阈值
    输出在 (-threshold, threshold) 区间内严格为 0 ——保证稀疏性
    相比 sign()*relu() 在 |x|==threshold 处梯度无跳变
    """
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor, threshold: torch.Tensor) -> torch.Tensor:
        mask_pos = (x >  threshold).float()
        mask_neg = (x < -threshold).float()
        return mask_pos * (x - threshold) + mask_neg * (x + threshold)


class AFG(nn.Module):
    """
    Adaptive Filter Generator
    为每个空间位置动态生成 softmax 归一化卷积核权重 [B, C, K*K, H, W]
    """
    def __init__(self, in_channels: int = 32, kernel_size: int = 3,
                 mid_channel: int = 32):
        super().__init__()
        self.kernel_size = kernel_size
        self.conv_feat   = nn.Conv2d(in_channels, mid_channel,
                                     kernel_size=3, padding=1, bias=False)
        self.act         = nn.LeakyReLU(0.2, inplace=False)
        self.conv_weight = nn.Conv2d(mid_channel,
                                     in_channels * kernel_size * kernel_size,
                                     kernel_size=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(self.conv_feat.weight.dtype)
        b, c, h, w = x.shape
        w_map = self.act(self.conv_feat(x))
        w_map = self.conv_weight(w_map)
        return F.softmax(
            w_map.view(b, c, self.kernel_size ** 2, h, w), dim=2
        )


class DyConv(nn.Module):
    """动态卷积：AFG 权重 × Unfold 局部窗口，每位置独立卷积核"""
    def __init__(self, in_channels: int = 32, kernel_size: int = 3):
        super().__init__()
        self.kernel_size = kernel_size
        self.in_channels = in_channels
        self.afg    = AFG(in_channels, kernel_size, mid_channel=max(in_channels, 32))
        self.unfold = nn.Unfold(kernel_size, dilation=1,
                                padding=kernel_size // 2, stride=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(self.afg.conv_feat.weight.dtype)
        b, c, h, w = x.shape
        filter_w = self.afg(x).contiguous()
        unf = self.unfold(x).view(b, c, self.kernel_size ** 2, h, w).contiguous()
        return (unf * filter_w).sum(dim=2).contiguous()


# =============================================================================
# §3  自适应阈值估计（三分支：Local + Large + NonLocal）
# =============================================================================

class AdaptiveThresholding(nn.Module):
    """
    三分支自适应阈值估计

    ┌─ Local    3×3 × 3 层    —— 像素级细粒度纹理（无检查点）
    ├─ Large    41×41 × 3 层  —— 跨结构长程分布（走检查点）
    └─ NonLocal ShadowFormer  —— 非局部结构相关性（走检查点）

    三分支相加 → proj_out → sigmoid → 软阈值图 ∈ (0, 1)
    """
    def __init__(self, in_channel: int, mid_channel: int = 32,
                 nonlocal_win_size: int = 4, use_checkpoint: bool = True):
        super().__init__()
        self.use_checkpoint = use_checkpoint

        self.proj_in  = nn.Conv2d(in_channel, mid_channel, 1, bias=False)
        self.proj_out = nn.Conv2d(mid_channel, in_channel, 1, bias=False)

        self.information_Local = nn.Sequential(
            nn.Conv2d(mid_channel, mid_channel, 3, 1, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(mid_channel, mid_channel, 3, 1, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(mid_channel, mid_channel, 3, 1, 1, bias=False),
        )
        self.information_Large = nn.Sequential(
            nn.Conv2d(mid_channel, mid_channel, 41, 1, padding="same", bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(mid_channel, mid_channel, 41, 1, padding="same", bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(mid_channel, mid_channel, 41, 1, padding="same", bias=False),
        )
        self.information_NonLocal = ShadowFormer(
            img_size=None, img_siwin_size=nonlocal_win_size, in_chans=mid_channel)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x    = x.to(self.proj_in.weight.dtype)
        feat = self.proj_in(x)

        local_out = self.information_Local(feat)

        if self.use_checkpoint and self.training:
            large_out = ckpt.checkpoint(self.information_Large, feat, use_reentrant=False)
        else:
            large_out = self.information_Large(feat)
        large_out = torch.clamp(large_out, -3.0, 3.0)
        large_out = large_out / (large_out.norm(p=2, dim=(1, 2, 3), keepdim=True) + 1e-6)

        if self.use_checkpoint and self.training:
            nonlocal_out = ckpt.checkpoint(self.information_NonLocal, feat, use_reentrant=False)
        else:
            nonlocal_out = self.information_NonLocal(feat)

        return torch.sigmoid(self.proj_out(local_out + large_out + nonlocal_out))


# =============================================================================
# §4  ConvLista_T_Flex — 动态 DDN 展开模块（返回稀疏系数 gamma_k）
# =============================================================================

class ConvLista_T_Flex(nn.Module):
    """
    融合版动态 DDN 展开模块

    【核心改动】forward 返回 gamma_k（ISTA 迭代后原始稀疏系数），不再调用 self.act：
      · soft_threshold 保证大量位置严格为 0（稀疏性）
      · 保留负值，方便与学生稀疏系数做无偏 MSE 对齐
      · 切回稠密模式只需将最后一行改为 return self.act(gamma_k)

    接口兼容原 ConvLista_T：
        forward((x, prompt, prompt_text)) → Tensor [B, num_filters, H, W]
                                            — 原始稀疏系数，大量位置严格为 0
    """
    def __init__(
        self,
        in_channels: int       = 1,
        num_filters: int       = 16,
        kernel_size: int       = 3,
        stride: int            = 1,
        unfoldings: int        = 3,
        act: str               = "leakyrelu",
        num_groups: int        = 8,
        nonlocal_win_size: int = 4,
        use_checkpoint: bool   = True,
    ):
        super().__init__()
        self.unfoldings     = unfoldings
        self.num_filters    = num_filters
        self.use_checkpoint = use_checkpoint

        self.channel_proj = nn.Conv2d(
            in_channels, num_filters, kernel_size=3, stride=1, padding=1, bias=False
        )

        # 9 个独立动态字典算子，4 组迭代
        self.apply_B  = DyConv(num_filters, kernel_size)
        self.apply_A1 = DyConv(num_filters, kernel_size)
        self.apply_B1 = DyConv(num_filters, kernel_size)
        self.apply_A2 = DyConv(num_filters, kernel_size)
        self.apply_B2 = DyConv(num_filters, kernel_size)
        self.apply_A3 = DyConv(num_filters, kernel_size)
        self.apply_B3 = DyConv(num_filters, kernel_size)
        self.apply_A4 = DyConv(num_filters, kernel_size)
        self.apply_B4 = DyConv(num_filters, kernel_size)

        self.thresholding   = AdaptiveThresholding(
            in_channel=num_filters,
            mid_channel=max(num_filters, 32),
            nonlocal_win_size=nonlocal_win_size,
            use_checkpoint=use_checkpoint,
        )
        self.soft_threshold = SoftThreshold()
        self.act            = get_act(act)   # 保留实例，当前不调用

    def _make_group_fn(self, apply_A: nn.Module, apply_B: nn.Module):
        unfoldings     = self.unfoldings
        soft_threshold = self.soft_threshold

        def _group(gamma_k, I_feat, threshold):
            for _ in range(unfoldings - 1):
                x_k     = apply_A(gamma_k)
                r_k     = apply_B(x_k - I_feat)
                gamma_k = soft_threshold(gamma_k - r_k, threshold)
            return gamma_k

        return _group

    def _run_group(self, apply_A, apply_B, gamma_k, I_feat, threshold):
        fn = self._make_group_fn(apply_A, apply_B)
        if self.use_checkpoint and self.training:
            return ckpt.checkpoint(fn, gamma_k, I_feat, threshold, use_reentrant=False)
        return fn(gamma_k, I_feat, threshold)

    def forward(self, input_tuple: tuple) -> torch.Tensor:
        """
        Returns:
          gamma_k [B, num_filters, H, W] — 原始稀疏系数
            · 大量位置严格为 0（soft_threshold 保证）
            · 非零位置保留符号信息，可直接做 MSE
        """
        x, _, _ = input_tuple
        x = x.to(self.channel_proj.weight.dtype)

        I_feat     = self.channel_proj(x)
        conv_input = self.apply_B(I_feat)
        threshold  = self.thresholding(conv_input)
        gamma_k    = self.soft_threshold(conv_input, threshold)

        gamma_k = self._run_group(self.apply_A1, self.apply_B1, gamma_k, I_feat, threshold)
        gamma_k = self._run_group(self.apply_A2, self.apply_B2, gamma_k, I_feat, threshold)
        gamma_k = self._run_group(self.apply_A3, self.apply_B3, gamma_k, I_feat, threshold)
        gamma_k = self._run_group(self.apply_A4, self.apply_B4, gamma_k, I_feat, threshold)

        # ★ 返回原始稀疏系数（不经过激活）
        # 切回稠密模式：return self.act(gamma_k)
        return gamma_k

# =============================================================================
# §5  向后兼容别名
# =============================================================================
ConvLista_T = ConvLista_T_Flex