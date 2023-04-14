"""特征拼接流水线。

把多通道 log-mel 与麦克风对之间的 GCC-PHAT 拼成统一的输入张量
``(n_feature_maps, n_mels, n_frames)``，供后续模型或分析使用。
"""

from __future__ import annotations

import numpy as np

from sonoloc.config import SonolocConfig
from sonoloc.features.logmel import LogMelExtractor
from sonoloc.features.stft import stft
from sonoloc.io.arrays import MicArray, get_array


class FeaturePipeline:
    """按配置从多通道信号提取 log-mel + GCC-PHAT 特征。"""

    def __init__(self, config: SonolocConfig | None = None, array: MicArray | None = None) -> None:
        self.config = config or SonolocConfig()
        self.array = array or get_array(self.config.array)
        self.logmel = LogMelExtractor(
            self.config.sample_rate,
            self.config.n_fft,
            n_mels=self.config.n_mels,
            fmin=self.config.fmin,
            fmax=self.config.fmax,
        )

    def __call__(self, signal: np.ndarray) -> np.ndarray:
        spec = stft(
            signal,
            n_fft=self.config.n_fft,
            hop_length=self.config.hop_length,
            window=self.config.window,
        )
        logmel = self.logmel(spec)
        gcc = self._gcc_features(spec)
        return np.concatenate([logmel, gcc], axis=0)

    def _gcc_features(self, spec: np.ndarray) -> np.ndarray:
        """对每个麦克风对计算逐帧 GCC-PHAT，并截取中心若干时延。"""
        n_mels = self.config.n_mels
        n_fft = self.config.n_fft
        feats = []
        for i, j in self.array.pairs():
            cross = spec[i] * np.conj(spec[j])
            denom = np.abs(cross)
            denom[denom < 1e-12] = 1e-12
            phat = cross / denom
            cc = np.fft.irfft(phat, n=n_fft, axis=0)
            cc = np.fft.fftshift(cc, axes=0)
            mid = cc.shape[0] // 2
            half = n_mels // 2
            feats.append(cc[mid - half : mid - half + n_mels])
        if not feats:
            return np.empty((0, n_mels, spec.shape[2]))
        return np.stack(feats, axis=0)
