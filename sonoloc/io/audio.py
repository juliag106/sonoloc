"""多通道音频读写。

先放占位实现，真正的读取 / 重采样在后续提交中补上。
"""

from __future__ import annotations

import numpy as np


def load_audio(path: str, sample_rate: int | None = None) -> tuple[np.ndarray, int]:
    """读取多通道音频，返回 ``(signal, sample_rate)``。

    ``signal`` 形状为 ``(n_channels, n_samples)``。
    """
    # TODO: 用 soundfile 实现，并支持可选重采样
    raise NotImplementedError
