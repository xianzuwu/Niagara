import torch
import torch.nn as nn
from torchmetrics import Metric
from typing import Any, Literal, List, Dict

# 引入我们之前定义的 LPIPS 网络
from torchmetrics.functional.image.lpips import Vgg16, Alexnet, SqueezeNet, ScalingLayer, _normalize_tensor, _spatial_average, _upsample


class NetLinLayer(nn.Module):
    """A single linear layer which does a 1x1 conv."""
    def __init__(self, chn_in: int, chn_out: int = 1, use_dropout: bool = False) -> None:
        super().__init__()

        layers = [nn.Dropout()] if use_dropout else []
        layers += [nn.Conv2d(chn_in, chn_out, 1, stride=1, padding=0, bias=False)]
        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Process input."""
        return self.model(x)


class LPIPSLayer(Metric):
    def __init__(
        self,
        net_type: Literal["vgg", "alex", "squeeze"] = "alex",
        reduction: Literal["sum", "mean"] = "mean",
        normalize: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)

        self.net_type = net_type
        self.normalize = normalize

        # 选择网络类型
        if net_type == "vgg":
            self.net = Vgg16(pretrained=True)
            self.chns = [64, 128, 256, 512, 512]
        elif net_type == "alex":
            self.net = Alexnet(pretrained=True)
            self.chns = [64, 192, 384, 256, 256]
        elif net_type == "squeeze":
            self.net = SqueezeNet(pretrained=True)
            self.chns = [64, 128, 256, 384, 384, 512, 512]
        else:
            raise ValueError(f"Invalid network type: {net_type}")

        self.reduction = reduction

        # 创建线性层
        self.lins = nn.ModuleList([NetLinLayer(chn, use_dropout=True) for chn in self.chns])

        # 初始化 LPIPS 总损失和逐层损失
        self.per_layer_losses: Dict[str, List[torch.Tensor]] = {}
        self.add_state("sum_scores", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0.0), dist_reduce_fx="sum")

        # Scaling layer to preprocess images
        self.scaling_layer = ScalingLayer()

    def update(self, img_pred: torch.Tensor, img_gt: torch.Tensor) -> None:
        # 标准化图像到 [-1, 1]
        if self.normalize:
            img_pred = 2 * img_pred - 1
            img_gt = 2 * img_gt - 1

        # 使用 scaling layer
        img_pred_input = self.scaling_layer(img_pred)
        img_gt_input = self.scaling_layer(img_gt)

        # 前向传播获取每一层的特征
        feats_pred = self.net.forward(img_pred_input)
        feats_gt = self.net.forward(img_gt_input)

        # 计算逐层损失
        total_loss = 0.0
        for i, (feat_pred, feat_gt) in enumerate(zip(feats_pred, feats_gt)):
            feat_pred_norm = _normalize_tensor(feat_pred)
            feat_gt_norm = _normalize_tensor(feat_gt)

            diff = (feat_pred_norm - feat_gt_norm) ** 2

            if self.reduction == "mean":
                diff_loss = _spatial_average(self.lins[i](diff), keep_dim=False)
            elif self.reduction == "sum":
                diff_loss = _spatial_average(self.lins[i](diff), keep_dim=False).sum()

            # 将逐层损失存储在 per_layer_losses 中
            layer_name = f"layer_{i+1}"
            if layer_name not in self.per_layer_losses:
                self.per_layer_losses[layer_name] = []
            self.per_layer_losses[layer_name].append(diff_loss.detach().cpu())

            total_loss += diff_loss

        # 更新总损失
        self.sum_scores += total_loss.item()
        self.total += 1

    def compute(self):
        """计算最终的 LPIPS 损失和每一层的平均损失"""
        per_layer_avg_losses = {layer: torch.stack(losses).mean() for layer, losses in self.per_layer_losses.items()}
        # print('per_layer_avg_losses')
        # print(per_layer_avg_losses)

        # 返回总损失和逐层损失
        return {
            "lpips_total": self.sum_scores / self.total,
            "lpips_per_layer": per_layer_avg_losses
        }
