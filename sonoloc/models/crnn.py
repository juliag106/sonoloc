"""SELD 用的卷积-循环神经网络（CRNN）骨干。

结构参考经典 SELDnet：若干 2D 卷积块在频率维下采样后，把时间维交给
双向 GRU 建模上下文，输出逐帧的时间特征序列。仅在安装了 ``torch`` 时可用。
"""

from __future__ import annotations

from sonoloc.models import HAS_TORCH

if HAS_TORCH:
    import torch
    from torch import nn

    class ConvBlock(nn.Module):
        """Conv2d + BatchNorm + ReLU + 频率维最大池化。"""

        def __init__(self, in_ch: int, out_ch: int, pool_freq: int = 2) -> None:
            super().__init__()
            self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)
            self.bn = nn.BatchNorm2d(out_ch)
            self.act = nn.ReLU(inplace=True)
            self.pool = nn.MaxPool2d(kernel_size=(1, pool_freq))

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.pool(self.act(self.bn(self.conv(x))))

    class CRNN(nn.Module):
        """把 ``(batch, channels, frames, freq)`` 编码为逐帧特征序列。"""

        def __init__(
            self,
            in_channels: int,
            n_mels: int = 64,
            conv_channels: tuple[int, ...] = (64, 128, 128),
            gru_hidden: int = 128,
        ) -> None:
            super().__init__()
            blocks = []
            prev = in_channels
            freq = n_mels
            for ch in conv_channels:
                blocks.append(ConvBlock(prev, ch, pool_freq=2))
                prev = ch
                freq //= 2
            self.conv = nn.Sequential(*blocks)
            self.feature_dim = prev * max(freq, 1)
            self.gru = nn.GRU(
                self.feature_dim,
                gru_hidden,
                batch_first=True,
                bidirectional=True,
            )
            self.out_dim = gru_hidden * 2

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x = self.conv(x)  # (B, C, T, F')
            b, c, t, f = x.shape
            x = x.permute(0, 2, 1, 3).reshape(b, t, c * f)
            out, _ = self.gru(x)
            return out
