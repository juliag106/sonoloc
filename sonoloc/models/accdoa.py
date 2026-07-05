"""ACCDOA 输出头。

把逐帧特征映射到 ``(batch, frames, n_classes * 3)`` 的 ACCDOA 输出，
再 reshape 成 ``(batch, frames, n_classes, 3)``；向量方向即 DOA、模长即活跃度。
"""

from __future__ import annotations

from sonoloc.models import HAS_TORCH

if HAS_TORCH:
    import torch
    from torch import nn

    class ACCDOAHead(nn.Module):
        """线性 ACCDOA 回归头，输出经 ``tanh`` 约束到单位立方体内。"""

        def __init__(self, in_dim: int, n_classes: int) -> None:
            super().__init__()
            self.n_classes = n_classes
            self.fc = nn.Linear(in_dim, n_classes * 3)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            out = torch.tanh(self.fc(x))
            b, t, _ = out.shape
            return out.reshape(b, t, self.n_classes, 3)
