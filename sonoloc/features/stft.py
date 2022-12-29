"""多通道短时傅里叶变换（STFT）。

约定输出布局为 ``(n_channels, n_freq, n_frames)``，其中
``n_freq = n_fft // 2 + 1``（只保留非负频率）。
"""

from __future__ import annotations

import numpy as np
from scipy.signal import get_window


def _make_window(window: str, win_length: int, n_fft: int) -> np.ndarray:
    win = get_window(window, win_length, fftbins=True).astype(np.float64)
    if win_length < n_fft:  # 居中零填充到 n_fft
        total = n_fft - win_length
        left = total // 2
        win = np.pad(win, (left, total - left))
    elif win_length > n_fft:
        raise ValueError("win_length 不能大于 n_fft")
    return win


def stft(
    signal: np.ndarray,
    n_fft: int,
    hop_length: int,
    win_length: int | None = None,
    window: str = "hann",
    center: bool = True,
) -> np.ndarray:
    """计算多通道 STFT。

    Parameters
    ----------
    signal:
        ``(n_channels, n_samples)`` 或 ``(n_samples,)``。
    center:
        为 ``True`` 时在两端做 ``reflect`` 填充，使第 ``t`` 帧以
        ``t * hop_length`` 为中心。
    """
    signal = np.atleast_2d(np.asarray(signal, dtype=np.float64))
    win_length = win_length or n_fft
    win = _make_window(window, win_length, n_fft)

    if center:
        pad = n_fft // 2
        signal = np.pad(signal, ((0, 0), (pad, pad)), mode="reflect")

    n_channels, n_samples = signal.shape
    if n_samples < n_fft:
        raise ValueError("信号长度不足以容纳一帧")
    n_frames = 1 + (n_samples - n_fft) // hop_length

    starts = np.arange(n_frames) * hop_length
    frames = np.stack([signal[:, s : s + n_fft] for s in starts], axis=1)
    frames = frames * win
    spec = np.fft.rfft(frames, n=n_fft, axis=-1)
    return np.transpose(spec, (0, 2, 1))


def stft_frequencies(sample_rate: int, n_fft: int) -> np.ndarray:
    """返回 STFT 各频点对应的物理频率（Hz）。"""
    return np.fft.rfftfreq(n_fft, d=1.0 / sample_rate)
