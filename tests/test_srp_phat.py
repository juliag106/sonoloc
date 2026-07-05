"""SRP-PHAT 定位测试。

用频域相移合成远场平面波多通道信号，验证 SRP-PHAT 能恢复方位角。
信号带限在空间混叠频率以下，避免网格搜索出现镜像峰。
"""

import numpy as np

from sonoloc.config import SonolocConfig
from sonoloc.io.geometry import sph2cart, tetrahedral_array
from sonoloc.localization.srp_phat import srp_phat


def _synthesize(array, az_deg, el_deg, n=24000, sr=24000, c=343.0, seed=0):
    rng = np.random.default_rng(seed)
    src = rng.standard_normal(n)
    u = sph2cart(np.deg2rad(az_deg), np.deg2rad(el_deg), 1.0)
    delays = -(array.positions @ u) / c  # 每个麦克风的传播时延（秒）
    freqs = np.fft.rfftfreq(n, 1.0 / sr)
    spectrum = np.fft.rfft(src)
    band = (freqs > 200.0) & (freqs < 1800.0)  # 带限到空间混叠以下
    spectrum = spectrum * band
    channels = [np.fft.irfft(spectrum * np.exp(-2j * np.pi * freqs * d), n=n) for d in delays]
    return np.stack(channels, axis=0)


def test_srp_phat_recovers_azimuth() -> None:
    array = tetrahedral_array()
    config = SonolocConfig(sample_rate=24000)
    signal = _synthesize(array, az_deg=60.0, el_deg=0.0)
    az, _el = srp_phat(signal, array, config)
    assert abs(az - 60.0) <= 15.0


def test_srp_phat_recovers_opposite_side() -> None:
    array = tetrahedral_array()
    config = SonolocConfig(sample_rate=24000)
    signal = _synthesize(array, az_deg=-120.0, el_deg=0.0, seed=3)
    az, _el = srp_phat(signal, array, config)
    # 方位角是环形量，比较时归一化到 [-180, 180)
    err = (az - (-120.0) + 180.0) % 360.0 - 180.0
    assert abs(err) <= 15.0
