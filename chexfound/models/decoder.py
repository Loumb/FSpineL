# chexfound/models/decoder.py
#
# 共享稀疏系数解码器 + Sobel 边缘提取器
#
# SparseDecoder: 16ch 稀疏系数 → 多尺度重建图像
#   主输出 640×640 + 辅助输出 320×320 / 160×160
#   教师和学生共享权重，确保稀疏系数的"字典语言"一致
#
# SobelEdgeExtractor: 图像 → 边缘强度图 (用于边缘感知损失)

import torch
import torch.nn as nn
import torch.nn.functional as F


class SobelEdgeExtractor(nn.Module):
    """Sobel 边缘检测器：提取图像的梯度幅值作为边缘强度图"""

    def __init__(self):
        super().__init__()
        kx = torch.tensor([[[[-1., 0., 1.],
                             [-2., 0., 2.],
                             [-1., 0., 1.]]]])
        ky = torch.tensor([[[[-1., -2., -1.],
                             [ 0.,  0.,  0.],
                             [ 1.,  2.,  1.]]]])
        self.register_buffer("sobel_x", kx)
        self.register_buffer("sobel_y", ky)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        edges = []
        for c in range(C):
            x_c = x[:, c:c + 1, :, :]
            gx = F.conv2d(x_c, self.sobel_x.to(x.dtype), padding=1)
            gy = F.conv2d(x_c, self.sobel_y.to(x.dtype), padding=1)
            edges.append(torch.sqrt(gx ** 2 + gy ** 2 + 1e-6))
        return torch.cat(edges, dim=1)


class SparseDecoder(nn.Module):
    """
    共享稀疏系数解码器（含多尺度辅助重建头）

    输入:  [B, 16, 80, 80]  稀疏系数
    输出:
      train mode (return_multiscale=True):
        (recon_640 [B,out_ch,640,640],
         recon_320 [B,out_ch,320,320],
         recon_160 [B,out_ch,160,160])
      eval mode (return_multiscale=False):
        recon_640 [B,out_ch,640,640]
    """

    def __init__(self, sparse_ch: int = 16, out_ch: int = 1,
                 use_multiscale: bool = True):
        super().__init__()
        self.use_multiscale = use_multiscale

        # 80 → 160
        self.up1 = nn.Sequential(
            nn.Conv2d(sparse_ch, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
        )
        # 160 → 320
        self.up2 = nn.Sequential(
            nn.Conv2d(64, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
        )
        # 320 → 640
        self.up3 = nn.Sequential(
            nn.Conv2d(32, 16, 3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
        )
        self.out_conv = nn.Conv2d(16, out_ch, 1)

        # 多尺度辅助重建头（从中间特征直接输出）
        if use_multiscale:
            self.aux_head_160 = nn.Conv2d(64, out_ch, 1)
            self.aux_head_320 = nn.Conv2d(32, out_ch, 1)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out',
                                        nonlinearity='leaky_relu')

    def forward(self, x: torch.Tensor, return_multiscale: bool = False):
        f1 = self.up1(x)          # [B, 64, 160, 160]
        f2 = self.up2(f1)         # [B, 32, 320, 320]
        f3 = self.up3(f2)         # [B, 16, 640, 640]
        out = self.out_conv(f3)   # [B, out_ch, 640, 640]

        if return_multiscale and self.use_multiscale:
            aux_160 = self.aux_head_160(f1)
            aux_320 = self.aux_head_320(f2)
            return out, aux_320, aux_160
        return out
