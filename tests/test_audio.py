"""音频读写往返测试。"""

import numpy as np

from sonoloc.io.audio import load_audio, resample_signal, save_audio


def test_audio_round_trip(tmp_path) -> None:
    sr = 24000
    signal = np.random.default_rng(0).standard_normal((4, sr)) * 0.1
    path = tmp_path / "x.wav"
    save_audio(str(path), signal, sr)
    loaded, loaded_sr = load_audio(str(path))
    assert loaded_sr == sr
    assert loaded.shape == signal.shape
    assert np.allclose(loaded, signal, atol=1e-4)


def test_load_with_resample(tmp_path) -> None:
    signal = np.random.default_rng(1).standard_normal((2, 24000)) * 0.1
    path = tmp_path / "y.wav"
    save_audio(str(path), signal, 24000)
    loaded, sr = load_audio(str(path), sample_rate=16000)
    assert sr == 16000
    assert loaded.shape[0] == 2
    assert loaded.shape[1] == 16000


def test_resample_identity() -> None:
    signal = np.random.default_rng(2).standard_normal((2, 1000))
    out = resample_signal(signal, 16000, 16000)
    np.testing.assert_allclose(out, signal)
