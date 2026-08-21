# FewSpineL: Optimization-Inspired Self-Supervised Pretraining for Lumbar MRI

[中文](#中文说明) | [English](#english-summary)

Repository: https://github.com/g809180109-code/FSpineL

## 中文说明

这是一个面向腰椎 MRI 的研究型自监督预训练代码仓库。项目在
[CheXFound](https://github.com/RPIDIAL/CheXFound) 视觉自监督框架基础上，
加入分析–综合字典协同、稀疏系数对齐、多尺度重建、边缘约束、SAM 解剖先验
以及 FSDP 多卡训练流程，用于学习可迁移的腰椎解剖结构表征。

### 项目定位

- 任务：无标注腰椎 MRI 自监督预训练与下游迁移。
- 数据规模：项目内部使用四中心约 50 万张脱敏腰椎 MRI 切片。
- 训练：PyTorch、FP16、FSDP、单机多卡 `torchrun`。
- 论文：`Optimization-Inspired Self-Supervised Learning for Few-Shot Lumbar Medical Image Analysis`，投稿中。
- 公开范围：仅源代码、配置和启动脚本，不包含医学影像、标签、患者元数据、模型权重、训练日志或内部服务器路径。

本仓库是研究代码，不是医疗器械，也不能用于临床诊断。

## 核心方法

训练链路由以下模块组成：

1. DICOM/NIfTI 数据在授权环境中完成重采样、强度标准化和切片构建。
2. SAM-Med2D 产生解剖先验掩码。
3. Student/Teacher 自监督框架学习全局与局部一致表征。
4. 分析–综合字典模块在稀疏系数空间进行对齐和重建。
5. Sobel 边缘与 SAM 掩码约束强化边界和解剖结构。
6. FSDP 对 Student、Teacher 和共享解码器进行分片训练。

## 仓库结构

```text
chexfound/                  核心模型、数据适配、损失与训练流程
chexfound/configs/train/    训练配置
scripts/                    多卡训练和评测启动脚本
precompute_sam_masks.py     SAM 解剖先验预计算
docs/                       公开说明、脱敏报告与打印版材料
```

## 安装

```bash
conda env create -f conda-extras.yaml
conda activate dinov2-extras
export PYTHONPATH=.
```

根据实际 CUDA 和 GPU 环境调整 PyTorch 与 xFormers 版本。

## 数据接口

数据不随仓库发布。请在获得合法授权并完成脱敏后，按照以下占位结构准备：

```text
<DATA_ROOT>/
  train/
  val/
  test/
<EXTRA_ROOT>/
  entries-TRAIN.npy
  entries-VAL.npy
  class-ids-TRAIN.npy
  class-names-TRAIN.npy
```

需要从标签 CSV 生成额外索引时，通过环境变量提供路径：

```bash
export LDH_LABEL_CSV=/path/to/deidentified_labels.csv
```

## SAM 先验

```bash
export SAM_MED2D_ROOT=/path/to/SAM-Med2D
python precompute_sam_masks.py \
  --checkpoint /path/to/sam_checkpoint.pth \
  --image_dir /path/to/deidentified_images \
  --mask_dir ./outputs/sam_masks
```

SAM 权重不包含在本仓库中。

## 多卡训练

```bash
bash scripts/run_train_ddn_multi_gpu.sh \
  chexfound/configs/train/lista16_ibot333_highres640.yaml \
  4 \
  ./outputs/ddn_train \
  'train.dataset_path=CXRDatabase:split=TRAIN:root=/path/to/images:extra=/path/to/extra' \
  'train.sam_masks_dir=/path/to/sam_masks'
```

## 隐私与安全

- 不要提交 DICOM、NIfTI、NRRD、PNG 病例图像或患者标签表。
- 不要提交 `.pth`、`.pt`、`.ckpt`、`.onnx` 或大规模 `.npy` 元数据。
- 不要在代码中写入医院名、患者编号、服务器地址、用户名或本机绝对路径。
- 提交前运行 `scripts/public_release_check.ps1`。

详细范围见 [公开发布与脱敏说明](docs/PUBLIC_RELEASE_AND_PRIVACY.md)。

## 上游项目与许可证

本仓库包含对 CheXFound 的研究性修改。CheXFound 源码采用 MIT License，
原始版权与许可证见 [LICENSE-CheXFound](LICENSE-CheXFound)。

上游项目：

- https://github.com/RPIDIAL/CheXFound
- Yang et al., *Chest X-ray Foundation Model with Global and Local Representations Integration*.

## English Summary

This repository contains a privacy-safe research snapshot for optimization-inspired
self-supervised pretraining on lumbar MRI. It extends the MIT-licensed CheXFound
codebase with sparse analysis–synthesis dictionary learning, SAM anatomical priors,
edge-aware objectives, and multi-GPU FSDP training. Medical images, metadata,
checkpoints, logs, and machine-specific paths are intentionally excluded.

## Disclaimer

For research and engineering demonstration only. This software is not a medical
device and must not be used for autonomous diagnosis or treatment decisions.
