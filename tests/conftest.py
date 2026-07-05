"""共享测试夹具。"""

import numpy as np
import pytest

from sonoloc.config import SonolocConfig
from sonoloc.io.arrays import MicArray, sph2cart, tetrahedral_array


@pytest.fixture
def config() -> SonolocConfig:
    return SonolocConfig(sample_rate=24000)


@pytest.fixture
def array() -> MicArray:
    return tetrahedral_array()


@pytest.fixture
def plane_wave():
    """返回一个按方位/仰角合成带限远场平面波的工厂函数。"""

    def _make(mic_array, az_deg, el_deg, n=24000, sr=24000, c=343.0, band=(300.0, 1600.0), seed=0):
        rng = np.random.default_rng(seed)
        src = rng.standard_normal(n)
        u = sph2cart(np.deg2rad(az_deg), np.deg2rad(el_deg), 1.0)
        delays = -(mic_array.positions @ u) / c
        freqs = np.fft.rfftfreq(n, 1.0 / sr)
        spectrum = np.fft.rfft(src) * ((freqs > band[0]) & (freqs < band[1]))
        channels = [np.fft.irfft(spectrum * np.exp(-2j * np.pi * freqs * d), n=n) for d in delays]
        return np.stack(channels, axis=0)

    return _make
