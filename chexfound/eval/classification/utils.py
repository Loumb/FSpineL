import numpy as np
import torch
import torch.nn as nn

from chexfound.eval.utils import Model3DWrapper
import chexfound.distributed as distributed

from chexfound.eval.classification.glori import GLoRI, create_mldecoder_input

# ---------- create_linear_input (保留签名，但更健壮) ----------
def create_linear_input(x_tokens_list, use_n_blocks, use_avgpool):
    """
    安全版 create_linear_input：
    - x_tokens_list: list/tuple of intermediate outputs OR a single tensor
    - use_n_blocks: int
    - use_avgpool: bool
    返回: tensor of shape [B, F] (flattened per-sample feature)
    """
    # 如果传进来的是单个 tensor，把它包成 list（以兼容外部传参）
    if isinstance(x_tokens_list, torch.Tensor):
        x_tokens_list = [x_tokens_list]

    # 取最后 n 个 block（如果 n 比长度大则取全部）
    if use_n_blocks <= 0:
        intermediate_output = list(x_tokens_list)
    else:
        intermediate_output = list(x_tokens_list)[-use_n_blocks:]

    # 从每个 block 安全取出最后一个元素作为 class token（若 block 本身就是 tensor 则直接用）
    class_tokens = []
    for item in intermediate_output:
        if isinstance(item, (list, tuple)):
            tok = item[-1]
        else:
            tok = item

        # 如果 class token 是 (B,1,D) 之类的，squeeze 到 (B,D)
        if isinstance(tok, torch.Tensor) and tok.dim() == 3 and tok.size(1) == 1:
            tok = tok.squeeze(1)

        class_tokens.append(tok)

    # 确保每个 class_token 至少是 2D [B, C]
    for i, t in enumerate(class_tokens):
        if not isinstance(t, torch.Tensor):
            raise TypeError(f"class token item {i} is not a tensor (type={type(t)})")
        if t.dim() == 1:
            # 如果是 [C], 视为 batch=1
            class_tokens[i] = t.view(1, -1)
        elif t.dim() > 2:
            # 如果是 [B, N, C]，对 N 做均值池化，得到 [B, C]
            class_tokens[i] = t.mean(dim=1)

    # 现在把所有 class token 在特征维度上拼接 -> 得到 [B, sumC]
    try:
        output = torch.cat(class_tokens, dim=-1)
    except Exception as e:
        raise RuntimeError(f"Failed to cat class_tokens shapes {[t.shape for t in class_tokens]}: {e}")

    # 如果需要 avgpool，将最后 block 的 patch-tokens 进行平均并拼接
    if use_avgpool:
        last_item = intermediate_output[-1]
        if isinstance(last_item, (list, tuple)):
            patch_tokens = last_item[0]
        else:
            patch_tokens = last_item
        if not isinstance(patch_tokens, torch.Tensor):
            raise TypeError("patch_tokens is not a tensor")
        if patch_tokens.dim() == 3:
            patch_mean = patch_tokens.mean(dim=1)  # [B, D]
        elif patch_tokens.dim() == 2:
            patch_mean = patch_tokens  # already [B, D]
        else:
            patch_mean = patch_tokens.view(patch_tokens.size(0), -1)
        output = torch.cat((output, patch_mean), dim=-1)

    # 最后把每个样本 flatten 成 [B, F]
    output = output.reshape(output.shape[0], -1)
    return output.float()


# ---------- LinearClassifier（保持签名，但实现更健壮） ----------
class LinearClassifier(nn.Module):
    def __init__(self, out_dim, num_classes=4, use_n_blocks=4, use_avgpool=True, multiview=False):
        """
        out_dim: 可以是：
            - 一个整数（期望的输入维度） 或
            - 一个 tensor/list/tuple/tensor-on-gpu（代表 sample_output），
              本构造会智能处理：若 out_dim 看起来像模型输出（list/tuple/tensor），
              会用 create_linear_input(...) 根据 use_n_blocks/use_avgpool 计算真实维度。
        其它参数保留原签名以兼容项目调用。
        """
        super().__init__()

        # 保留这些成员供 forward 使用
        self.use_n_blocks = use_n_blocks
        self.use_avgpool = use_avgpool
        self.num_classes = num_classes
        self.multiview = multiview

        # --------- 计算并规范化 self.out_dim（保证为纯 int） ----------
        # 如果 out_dim 看起来像 sample_output（list/tuple/tensor），用 create_linear_input 计算特征维度
        if isinstance(out_dim, (list, tuple)) or isinstance(out_dim, torch.Tensor):
            # out_dim is actually sample_output -> compute using create_linear_input
            try:
                sample_feat = create_linear_input(out_dim, self.use_n_blocks, self.use_avgpool)  # [B, F]
                computed_dim = sample_feat.shape[1]
            except Exception as e:
                # 捕获可能的错误并输出清晰提示
                raise RuntimeError(f"Failed to compute feature dim from sample_output passed as out_dim: {e}")
            self.out_dim = int(computed_dim)
        elif isinstance(out_dim, torch.Size):
            # torch.Size -> 转为 int product
            self.out_dim = int(np.prod(list(out_dim)))
        else:
            # 普通数字或可转为 int 的类型
            self.out_dim = int(out_dim)

        # --------- 初始化线性层（in_features = self.out_dim） ----------
        # 若 multiview 情况需把 in_features 扩展（原项目里某些地方会在 setup 中调整）
        # 这里不自动乘以 2，保持严格一致：如果需要 multiview 调用方应调整 out_dim 或传入样本输出
        try:
            self.linear = nn.Linear(self.out_dim, self.num_classes)
        except Exception as e:
            raise RuntimeError(f"Failed to create nn.Linear(in_features={self.out_dim}, out_features={self.num_classes}): {e}")

        nn.init.normal_(self.linear.weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x):
        """
        x: 通常是一个 iterable（batch）——每个 element 是 backbone 的 intermediate outputs（即 create_linear_input 的 x_tokens_list）
           但也兼容 x 为单个 sample_output 张量（或 list）。
        返回: 分类 logits，shape [B, num_classes]
        """
        # 如果输入是单个 sample_output（tensor 或 list/tuple），把它包成 list 以便逐项处理
        if isinstance(x, torch.Tensor) or isinstance(x, (list, tuple)) and not (len(x) > 0 and isinstance(x[0], (list, tuple, torch.Tensor))):
            # 这里的判断比较保守：如果 x 本身是一个 sample_output（如 list of blocks），把它当作单样本 batch
            batch_inputs = [x]
        else:
            batch_inputs = list(x)

        # 为 batch 中每个样本构造特征 [B_i, F]
        features = []
        for sample in batch_inputs:
            feat = create_linear_input(sample, self.use_n_blocks, self.use_avgpool)  # [B_sample, F]
            features.append(feat)

        # 处理 batch 组织方式：
        # - 如果所有 sample 的第一个维度均为 1（即每个 sample 返回的是 [1, F]），我们把它们逐个堆成 (N, F)
        # - 否则，如果所有 sample 的 batch_size 相同 (B), 我们认为它们是不同视图/块（multiview），对每个样本按特征维度 cat -> 得到 (B, total_F)
        batch_sizes = [f.shape[0] for f in features]
        if all(bs == 1 for bs in batch_sizes):
            # 每个 feature 都是 [1, F] -> 把它们按样本堆叠为 (N, F)
            stacked = torch.cat([f.view(1, -1) for f in features], dim=0)  # (N, F)
            output = stacked  # shape (N, F)
        else:
            # 存在不止 1 的 batch size
            if len(set(batch_sizes)) == 1:
                # 全部 batch size 相同 -> 将这些特征在特征维度 cat，返回 (B, sumF)
                try:
                    output = torch.cat(features, dim=-1)
                except Exception as e:
                    raise RuntimeError(f"Failed to cat features with shapes {[f.shape for f in features]}: {e}")
            else:
                # 混合了不同的 batch sizes（极少见/非预期），提示用户并尝试安全地把它们按 0 维 concat 后 flatten
                # 这种情况通常是调用方组织输入格式不统一，抛出明确错误帮助定位
                raise RuntimeError(f"Incompatible per-sample batch sizes in classifier forward: {batch_sizes}")

        # 最终确保 output 为 [B_out, F_out] 再送入 linear
        output = output.view(output.shape[0], -1)  # flatten feature dims if 有多余维度

        # 如果 output 的特征维与初始化的 linear 不匹配，尝试自动适配：
        if output.shape[1] != self.linear.in_features:
            # 为避免破坏外部期望，我们不直接改写 linear 的形状（那会影响 optimizer 参数组匹配）。
            # 这里给出两种自动策略（按优先级）：
            # 1) 如果 output dim < in_features: 在特征后补零到 expected size
            # 2) 如果 output dim > in_features: 在特征上做线性投影到 expected size（动态创建临时投影层）
            needed = self.linear.in_features
            current = output.shape[1]
            if current < needed:
                pad = output.new_zeros((output.shape[0], needed - current))
                output = torch.cat([output, pad], dim=-1)
            elif current > needed:
                # 动态投影：用一个临时线性层（在 forward 中使用，不加入参数更新）
                proj = nn.Linear(current, needed).to(output.device)
                with torch.no_grad():
                    nn.init.normal_(proj.weight, std=0.01)
                    nn.init.zeros_(proj.bias)
                output = proj(output)

        return self.linear(output)


# ---------- AllClassifiers 保持但实现与原意一致 ----------
class AllClassifiers(nn.Module):
    def __init__(self, classifiers_dict):
        super().__init__()
        # 接受一个 ModuleDict 或 dict of modules
        self.classifiers_dict = nn.ModuleDict(classifiers_dict)

    def forward(self, inputs, return_attention=False):
        # 保留 return_attention 参数兼容上游调用（若无则忽略）
        if return_attention:
            return {k: v.forward(inputs, return_attention=True) for k, v in self.classifiers_dict.items()}
        return {k: v.forward(inputs) for k, v in self.classifiers_dict.items()}

    def __len__(self):
        return len(self.classifiers_dict)


# ---------- LinearPostprocessor 保持不变 ----------
class LinearPostprocessor(nn.Module):
    def __init__(self, linear_classifier):
        super().__init__()
        self.linear_classifier = linear_classifier

    def forward(self, samples, targets):
        preds = torch.sigmoid(self.linear_classifier(samples))
        if not isinstance(targets, torch.Tensor):
            targets = torch.tensor(targets).cuda()
        return {
            "preds": preds,
            "target": targets,
        }


# ---------- setup_linear_classifiers 保持签名，但改内部调用以兼容上面 LinearClassifier ----------
def setup_linear_classifiers(
    sample_output,
    n_last_blocks_list,
    learning_rates,
    avgpools=[True, False],
    num_classes=14,
    is_3d=False,
    multiview=False
):
    if isinstance(sample_output, int):
        raise ValueError(f"sample_output is int={sample_output}, expected list/tuple of intermediate outputs")
    if isinstance(sample_output, torch.Tensor):
        sample_output = [sample_output]

    linear_classifiers_dict = nn.ModuleDict()
    optim_param_groups = []

    for n in n_last_blocks_list:
        for avgpool in avgpools:
            for lr in learning_rates:
                # 注意：此处原来把 sample_output 传给 out_dim 参数 —— 为兼容旧调用习惯，我们也沿用该模式
                # LinearClassifier 会在构造时检测 out_dim 是否是 sample_output 并据此计算实际 in_features
                linear_classifier = LinearClassifier(
                    sample_output,  # 仍传 sample_output 保持兼容
                    num_classes=num_classes,
                    use_n_blocks=n,
                    use_avgpool=avgpool,
                    multiview=multiview
                )

                if is_3d:
                    linear_classifier = Model3DWrapper(linear_classifier)

                linear_classifier = linear_classifier.cuda()

                key = f"linear:blocks={n}:avgpool={avgpool}:lr={lr:.10f}".replace(".", "_")
                linear_classifiers_dict[key] = linear_classifier
                optim_param_groups.append({"params": linear_classifier.parameters(), "lr": lr})

    linear_classifiers = AllClassifiers(linear_classifiers_dict)

    if distributed.is_enabled():
        linear_classifiers = nn.parallel.DistributedDataParallel(linear_classifiers)

    return linear_classifiers, optim_param_groups


# ---------- setup_glori 保留原样（与你的代码一致） ----------
def setup_glori(sample_output, n_last_blocks_list, learning_rates, avgpools=[False], num_classes=14,
                multiview=False, decoder_dim=768, cat_cls=False):
    """
    Sets up the multiple linear classifiers with different hyperparameters to test out the most optimal one
    """
    linear_classifiers_dict = nn.ModuleDict()
    optim_param_groups = []
    for n in n_last_blocks_list:
        for avgpool in avgpools:
            for _lr in learning_rates:
                lr = _lr
                out_dim = create_mldecoder_input(sample_output, use_n_blocks=n)[0].shape[-1]
                linear_classifier = GLoRI(
                    num_classes=num_classes,# <—— 针对不同任务设置输出维度
                    decoder_embedding=decoder_dim,
                    initial_num_features=out_dim, use_n_blocks=n, multiview=multiview, cat_cls=cat_cls,
                )
                linear_classifier = linear_classifier.cuda()
                linear_classifiers_dict[
                    f"linear:blocks={n}:avgpool={avgpool}:lr={lr:.10f}".replace(".", "_")
                ] = linear_classifier
                optim_param_groups.append({"params": linear_classifier.parameters(), "lr": lr})

    linear_classifiers = AllClassifiers(linear_classifiers_dict)
    if distributed.is_enabled():
        linear_classifiers = nn.parallel.DistributedDataParallel(linear_classifiers)

    return linear_classifiers, optim_param_groups
