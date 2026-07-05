"""log-mel 提取器测试。"""

import numpy as np

from sonoloc.features.logmel import LogMelExtractor
from sonoloc.features.stft import stft


def test_logmel_shape() -> None:
    sr, n_fft, hop, n_mels = 24000, 1024, 480, 64
    rng = np.random.default_rng(0)
    signal = rng.standard_normal((4, sr))  # 1 秒，4 通道
    spec = stft(signal, n_fft=n_fft, hop_length=hop)
    extractor = LogMelExtractor(sr, n_fft, n_mels=n_mels)
    logmel = extractor(spec)
    assert logmel.shape[0] == 4
    assert logmel.shape[1] == n_mels
    assert logmel.shape[2] == spec.shape[2]


def test_logmel_is_finite() -> None:
    sr, n_fft, hop = 16000, 512, 256
    signal = np.zeros((2, 8000))  # 全零输入不应产生 -inf / nan
    spec = stft(signal, n_fft=n_fft, hop_length=hop)
    logmel = LogMelExtractor(sr, n_fft)(spec)
    assert np.all(np.isfinite(logmel))


def test_filterbank_rows_nonnegative() -> None:
    fb = LogMelExtractor(24000, 1024, n_mels=40).filterbank
    assert fb.shape[0] == 40
    assert np.all(fb >= 0.0)
