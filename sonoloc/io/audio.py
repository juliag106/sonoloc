"""多通道音频读写。

统一使用 ``(n_channels, n_samples)`` 的通道优先布局，float64 采样。
"""

from __future__ import annotations

import numpy as np
import soundfile as sf


def load_audio(path: str, sample_rate: int | None = None) -> tuple[np.ndarray, int]:
    """读取多通道音频。

    Parameters
    ----------
    path:
        音频文件路径（任何 libsndfile 支持的格式）。
    sample_rate:
        目标采样率；为 ``None`` 时保持原始采样率。

    Returns
    -------
    signal:
        形状 ``(n_channels, n_samples)`` 的 float64 数组。
    sr:
        实际采样率。
    """
    data, sr = sf.read(path, always_2d=True, dtype="float64")
    signal = np.ascontiguousarray(data.T)
    if sample_rate is not None and sample_rate != sr:
        # 重采样将在后续实现
        raise ValueError(f"文件采样率 {sr} 与目标 {sample_rate} 不一致")
    return signal, sr


def save_audio(path: str, signal: np.ndarray, sample_rate: int) -> None:
    """写出多通道音频，输入按通道优先布局。"""
    signal = np.asarray(signal, dtype=np.float64)
    if signal.ndim == 1:
        signal = signal[np.newaxis, :]
    sf.write(path, signal.T, sample_rate)
