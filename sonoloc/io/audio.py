"""多通道音频读写。

统一使用 ``(n_channels, n_samples)`` 的通道优先布局，float64 采样。
"""

from __future__ import annotations

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly


def resample_signal(signal: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """将多通道信号从 ``orig_sr`` 重采样到 ``target_sr``。

    使用多相滤波（``resample_poly``），比朴素 FFT 重采样更稳、无边界振铃。
    """
    if orig_sr == target_sr:
        return np.asarray(signal, dtype=np.float64)
    gcd = np.gcd(int(orig_sr), int(target_sr))
    up = target_sr // gcd
    down = orig_sr // gcd
    return resample_poly(np.asarray(signal, dtype=np.float64), up, down, axis=-1)



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
        signal = resample_signal(signal, sr, sample_rate)
        sr = sample_rate
    return signal, sr


def save_audio(path: str, signal: np.ndarray, sample_rate: int) -> None:
    """写出多通道音频，输入按通道优先布局。"""
    signal = np.asarray(signal, dtype=np.float64)
    if signal.ndim == 1:
        signal = signal[np.newaxis, :]
    sf.write(path, signal.T, sample_rate)
