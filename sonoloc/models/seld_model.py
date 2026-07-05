"""端到端 SELD 模型：CRNN 骨干 + ACCDOA 头。"""

from __future__ import annotations

from sonoloc.models import HAS_TORCH
from sonoloc.models.accdoa import ACCDOAHead
from sonoloc.models.crnn import CRNN

if HAS_TORCH:
    import torch
    from torch import nn

    class SeldModel(nn.Module):
        """把多通道特征映射为逐帧 ACCDOA 预测。"""

        def __init__(
            self,
            in_channels: int,
            n_classes: int,
            n_mels: int = 64,
            gru_hidden: int = 128,
        ) -> None:
            super().__init__()
            self.backbone = CRNN(in_channels, n_mels=n_mels, gru_hidden=gru_hidden)
            self.head = ACCDOAHead(self.backbone.out_dim, n_classes)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.head(self.backbone(x))


def accdoa_loss(pred, target):
    """ACCDOA 的均方误差损失（薄封装，便于统一调用）。"""
    if not HAS_TORCH:  # pragma: no cover
        raise ImportError("accdoa_loss 需要 PyTorch")
    return nn.functional.mse_loss(pred, target)
