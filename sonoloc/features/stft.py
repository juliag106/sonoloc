"""多通道短时傅里叶变换（STFT）。"""

from __future__ import annotations

import numpy as np


def stft(signal: np.ndarray, n_fft: int, hop_length: int) -> np.ndarray:
    """对多通道信号做 STFT。占位实现，后续补上。"""
    # TODO: 基于 scipy 的窗函数实现，返回 (n_channels, n_freq, n_frames)
    raise NotImplementedError
