"""MUSIC 定位测试。"""

import numpy as np

from sonoloc.config import SonolocConfig
from sonoloc.io.geometry import sph2cart, tetrahedral_array
from sonoloc.localization.music import music, music_map


def _synthesize(array, az_deg, el_deg, n=24000, sr=24000, c=343.0, seed=0):
    rng = np.random.default_rng(seed)
    src = rng.standard_normal(n)
    u = sph2cart(np.deg2rad(az_deg), np.deg2rad(el_deg), 1.0)
    delays = -(array.positions @ u) / c
    freqs = np.fft.rfftfreq(n, 1.0 / sr)
    spectrum = np.fft.rfft(src) * ((freqs > 300.0) & (freqs < 1600.0))
    channels = [np.fft.irfft(spectrum * np.exp(-2j * np.pi * freqs * d), n=n) for d in delays]
    return np.stack(channels, axis=0)


def test_music_peak_near_source() -> None:
    array = tetrahedral_array()
    config = SonolocConfig(sample_rate=24000)
    signal = _synthesize(array, az_deg=30.0, el_deg=0.0)
    az, _el = music(signal, array, config)
    assert abs(az - 30.0) <= 20.0


def test_music_map_is_positive() -> None:
    array = tetrahedral_array()
    config = SonolocConfig(sample_rate=24000)
    signal = _synthesize(array, az_deg=0.0, el_deg=0.0, seed=1)
    pseudo, grid = music_map(signal, array, config)
    assert pseudo.shape[0] == len(grid)
    assert np.all(pseudo > 0.0)
