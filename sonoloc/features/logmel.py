"""log-mel 频谱特征。

对每个通道的功率谱应用 mel 滤波器组，再取对数，得到
``(n_channels, n_mels, n_frames)`` 的特征张量。
"""

from __future__ import annotations

import numpy as np


def _hz_to_mel(freq: np.ndarray) -> np.ndarray:
    return 2595.0 * np.log10(1.0 + freq / 700.0)


def _mel_to_hz(mel: np.ndarray) -> np.ndarray:
    return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)


class LogMelExtractor:
    """把（多通道）功率谱转换为 log-mel 特征的可复用提取器。"""

    def __init__(
        self,
        sample_rate: int,
        n_fft: int,
        n_mels: int = 64,
        fmin: float = 50.0,
        fmax: float | None = None,
        eps: float = 1e-10,
    ) -> None:
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.n_mels = n_mels
        self.fmin = fmin
        self.fmax = fmax if fmax is not None else sample_rate / 2
        self.eps = eps

        # 在构造时一次性建立三角 mel 滤波器组。
        n_freq = n_fft // 2 + 1
        fft_freqs = np.fft.rfftfreq(n_fft, d=1.0 / sample_rate)
        lo_mel = _hz_to_mel(np.array(self.fmin))
        hi_mel = _hz_to_mel(np.array(self.fmax))
        mel_points = np.linspace(lo_mel, hi_mel, n_mels + 2)
        hz_points = _mel_to_hz(mel_points)
        fb = np.zeros((n_mels, n_freq), dtype=np.float64)
        for m in range(n_mels):
            lo, ctr, hi = hz_points[m], hz_points[m + 1], hz_points[m + 2]
            left = (fft_freqs - lo) / max(ctr - lo, 1e-12)
            right = (hi - fft_freqs) / max(hi - ctr, 1e-12)
            fb[m] = np.clip(np.minimum(left, right), 0.0, None)
        self.filterbank = fb

    def __call__(self, spec: np.ndarray) -> np.ndarray:
        """输入复数 STFT ``(..., n_freq, n_frames)``，输出 log-mel。"""
        power = np.abs(spec) ** 2
        mel = np.tensordot(power, self.filterbank, axes=([-2], [1]))
        # tensordot 把 mel 维放到末尾，移回频率维的位置
        mel = np.moveaxis(mel, -1, -2)
        return np.log(mel + self.eps)
