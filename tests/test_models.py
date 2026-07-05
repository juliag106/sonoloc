"""可选 torch 模型的前向形状测试；未安装 torch 时自动跳过。"""

import pytest

from sonoloc.models import HAS_TORCH

pytestmark = pytest.mark.skipif(not HAS_TORCH, reason="需要安装 torch")


def test_seld_model_forward_shape() -> None:
    import torch

    from sonoloc.models.seld_model import SeldModel

    model = SeldModel(in_channels=7, n_classes=13, n_mels=64)
    x = torch.randn(2, 7, 50, 64)  # (batch, channels, frames, mels)
    out = model(x)
    assert out.shape[0] == 2
    assert out.shape[2] == 13
    assert out.shape[3] == 3


def test_accdoa_head_range() -> None:
    import torch

    from sonoloc.models.accdoa import ACCDOAHead

    head = ACCDOAHead(in_dim=32, n_classes=4)
    out = head(torch.randn(1, 10, 32))
    assert out.abs().max() <= 1.0
    assert out.shape == (1, 10, 4, 3)
