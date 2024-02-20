"""声强向量测试。"""

import numpy as np

from sonoloc.features.intensity import intensity_doa, intensity_vectors
from sonoloc.features.stft import stft


def _foa_plane_wave(az_deg, el_deg, n=24000, sr=24000, seed=0):
    """用理想 FOA 编码合成单声源：W=s, X=cos(el)cos(az)*s, ..."""
    rng = np.random.default_rng(seed)
    src = rng.standard_normal(n)
    az, el = np.deg2rad(az_deg), np.deg2rad(el_deg)
    w = src
    x = np.cos(el) * np.cos(az) * src
    y = np.cos(el) * np.sin(az) * src
    z = np.sin(el) * src
    return np.stack([w, x, y, z], axis=0)


def test_intensity_vectors_shape() -> None:
    foa = _foa_plane_wave(30.0, 10.0)
    spec = stft(foa, n_fft=1024, hop_length=480)
    iv = intensity_vectors(spec)
    assert iv.shape[0] == 3
    assert iv.shape[1:] == spec.shape[1:]


def test_intensity_doa_recovers_azimuth() -> None:
    foa = _foa_plane_wave(60.0, 0.0)
    spec = stft(foa, n_fft=1024, hop_length=480)
    az, _el = intensity_doa(spec)
    assert abs(np.rad2deg(az) - 60.0) <= 5.0


def test_intensity_vectors_requires_four_channels() -> None:
    spec = np.zeros((2, 10, 5), dtype=complex)
    try:
        intensity_vectors(spec)
    except ValueError:
        return
    raise AssertionError("非 4 通道输入应抛出 ValueError")
