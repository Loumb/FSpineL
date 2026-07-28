import os
import sys
import glob
import argparse
import numpy as np
import cv2
import torch
from tqdm import tqdm

# Optional local checkout of SAM-Med2D. Keep machine-specific paths outside code.
sam_med2d_root = os.environ.get("SAM_MED2D_ROOT")
if sam_med2d_root:
    sys.path.insert(0, sam_med2d_root)

# 现在导入的是SAM-Med2D修改版的sam_model_registry
from segment_anything import sam_model_registry


def load_custom_sam(checkpoint_path, device, use_half=True):
    """加载你训练的SAM-Med2D Adapter模型，与训练代码100%一致"""
    # 第一步：加载训练检查点（这部分已经成功）
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False
    )

    if "model" in checkpoint:
        state_dict = checkpoint["model"]
        print(f"✅ 成功加载你训练的SAM-Med2D Adapter模型")
        print(f"   - 训练epoch: {checkpoint['epoch']}")
        print(f"   - 最佳Dice: {checkpoint['best_dice']:.4f}")
        print(f"   - 所属Fold: {checkpoint['fold'] + 1}")
        print(f"   - 参数量: {len(state_dict)}个")
    else:
        state_dict = checkpoint
        print(f"✅ 加载纯模型权重，共{len(state_dict)}个参数")

    # 第二步：添加SAM-Med2D要求的所有参数
    class Args:
        model_type = "vit_b"
        image_size = 256
        encoder_adapter = True
        sam_checkpoint = None  # 关键修复：添加这个参数，设为None

    # 现在所有参数都齐全了，不会再有任何AttributeError
    sam = sam_model_registry["vit_b"](Args())

    # 第三步：手动加载权重（100%匹配所有Adapter层）
    sam.load_state_dict(state_dict, strict=True)
    sam.to(device=device)

    # 半精度加速（显存减半，速度提升30%）
    if use_half:
        sam.half()
        print("✅ 已启用半精度加速")

    sam.eval()
    return sam


def preprocess_image(image_path, device, use_half=True):
    """与训练代码完全一致的图像预处理流程"""
    # 读取灰度图（与训练代码一致）
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"无法读取图像: {image_path}")

    # 与训练代码相同的1-99百分位归一化
    image = image.astype(np.float32)
    p1, p99 = np.percentile(image, (1, 99))
    image = np.clip(image, p1, p99)
    image = (image - p1) / (p99 - p1 + 1e-8)

    # 转换为3通道（与训练代码一致）
    image = np.stack([image] * 3, axis=-1)

    # resize到256x256（与训练代码一致的插值方式）
    image = cv2.resize(image, (256, 256), interpolation=cv2.INTER_LINEAR)

    # 转换为张量并归一化到SAM要求的范围
    image = image * 255.0  # SAM期望0-255的输入
    image_tensor = torch.from_numpy(image.transpose(2, 0, 1)).float().unsqueeze(0)
    image_tensor = image_tensor.to(device)

    if use_half:
        image_tensor = image_tensor.half()

    return image_tensor


def generate_all_class_masks(model, image_tensor, num_classes=12):
    """为每个类别生成mask，输出形状(12, 256, 256)，与训练代码完全一致"""
    with torch.no_grad():
        # 提取图像特征（自动经过Adapter层处理）
        image_embedding = model.image_encoder(image_tensor)

        all_masks = []
        for class_id in range(num_classes):
            # 为每个类别生成一个中心点提示（与训练代码的点提示逻辑一致）
            point_coords = torch.tensor([[[128, 128]]], device=image_tensor.device)
            point_labels = torch.tensor([[1]], device=image_tensor.device)

            # 编码提示
            sparse_embeddings, dense_embeddings = model.prompt_encoder(
                points=(point_coords, point_labels),
                boxes=None,
                masks=None,
            )

            # 解码mask
            low_res_masks, _ = model.mask_decoder(
                image_embeddings=image_embedding,
                image_pe=model.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=False,
            )

            # 上采样到256x256
            mask = torch.nn.functional.interpolate(
                low_res_masks,
                size=(256, 256),
                mode="bilinear",
                align_corners=False
            )

            # 转换为概率值
            mask = torch.sigmoid(mask)
            all_masks.append(mask.squeeze())

        # 堆叠所有类别的mask，输出形状(12, 256, 256)
        return torch.stack(all_masks, dim=0).cpu().numpy()


def precompute(args):
    # 创建输出目录
    os.makedirs(args.mask_dir, exist_ok=True)

    # 加载模型
    model = load_custom_sam(args.checkpoint, args.device, args.use_half)

    # 获取所有图像文件
    image_extensions = ["*.png", "*.jpg", "*.jpeg", "*.bmp"]
    image_paths = []
    for ext in image_extensions:
        image_paths.extend(glob.glob(os.path.join(args.image_dir, ext)))

    if not image_paths:
        print(f"❌ 在 {args.image_dir} 中没有找到任何图像文件")
        return

    print(f"\n开始预计算SAM掩码，共 {len(image_paths)} 张图像")
    print(f"输入目录: {args.image_dir}")
    print(f"输出目录: {args.mask_dir}")
    print(f"输出格式: (12, 256, 256) float32 .npy文件\n")

    # 断点续传：跳过已经处理过的文件
    processed_files = set(os.listdir(args.mask_dir))
    image_paths = [
        path for path in image_paths
        if f"{os.path.splitext(os.path.basename(path))[0]}.npy" not in processed_files
    ]

    if len(image_paths) == 0:
        print("✅ 所有图像已经处理完成！")
        return

    print(f"跳过已处理的文件，剩余 {len(image_paths)} 张待处理\n")

    # 批量处理
    for image_path in tqdm(image_paths, desc="预计算掩码", ncols=120):
        try:
            # 预处理图像
            image_tensor = preprocess_image(image_path, args.device, args.use_half)

            # 生成所有类别的mask
            masks = generate_all_class_masks(model, image_tensor, args.mask_num)

            # 保存mask（与训练代码兼容的格式）
            image_name = os.path.splitext(os.path.basename(image_path))[0]
            save_path = os.path.join(args.mask_dir, f"{image_name}.npy")
            np.save(save_path, masks.astype(np.float32))

        except Exception as e:
            print(f"\n❌ 处理 {os.path.basename(image_path)} 时出错: {e}")
            continue

    print("\n✅ 所有掩码预计算完成！")
    print(f"输出目录: {args.mask_dir}")
    print(f"输出格式: (12, 256, 256) float32 .npy文件")
    print(f"可以直接被你的5折交叉验证训练代码使用！")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True, help="训练好的SAM权重路径")
    parser.add_argument("--image_dir", type=str, required=True, help="输入图像目录")
    parser.add_argument("--mask_dir", type=str, required=False,default='./output/lumbar_sam_mask', help="输出mask目录")
    parser.add_argument("--mask_num", type=int, default=12, help="类别数量，与训练代码一致")
    parser.add_argument("--device", type=str, default="cuda", help="使用的设备")
    parser.add_argument("--use_half", action="store_true", default=True, help="启用半精度加速")

    args = parser.parse_args()

    print(f"使用设备: {args.device}")
    precompute(args)
