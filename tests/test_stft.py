"""STFT 测试。"""

import numpy as np

from sonoloc.features.stft import stft, stft_frequencies


def test_stft_output_shape() -> None:
    n_fft, hop = 512, 256
    signal = np.random.default_rng(0).standard_normal((3, 8000))
    spec = stft(signal, n_fft=n_fft, hop_length=hop)
    assert spec.shape[0] == 3
    assert spec.shape[1] == n_fft // 2 + 1
    assert spec.dtype == np.complex128


def test_stft_accepts_mono() -> None:
    spec = stft(np.zeros(4096), n_fft=512, hop_length=256)
    assert spec.shape[0] == 1


def test_stft_frequencies_range() -> None:
    freqs = stft_frequencies(24000, 1024)
    assert freqs[0] == 0.0
    assert np.isclose(freqs[-1], 12000.0)


def test_stft_rejects_oversized_window() -> None:
    try:
        stft(np.zeros(4096), n_fft=512, hop_length=256, win_length=1024)
    except ValueError:
        return
    raise AssertionError("win_length > n_fft 应报错")
