"""SRP-PHAT 声源定位。

对候选方向做导向对齐后，累加各麦克风对的 GCC-PHAT 互谱，得到
可控响应功率（Steered Response Power）。功率最大的方向即为 DOA 估计。
"""

from __future__ import annotations

import numpy as np

from sonoloc.config import SonolocConfig
from sonoloc.features.stft import stft, stft_frequencies
from sonoloc.io.arrays import MicArray
from sonoloc.localization.grid import SphereGrid, make_grid
from sonoloc.localization.steering import steering_delays


def srp_phat_map(
    signal: np.ndarray,
    array: MicArray,
    config: SonolocConfig | None = None,
    grid: SphereGrid | None = None,
) -> tuple[np.ndarray, SphereGrid]:
    """计算整张 SRP-PHAT 功率图。"""
    config = config or SonolocConfig()
    grid = grid or make_grid()

    spec = stft(signal, n_fft=config.n_fft, hop_length=config.hop_length, window=config.window)
    freqs = stft_frequencies(config.sample_rate, config.n_fft)
    # 归一化候选方向，避免自定义网格传入非单位向量时产生偏差
    directions = grid.directions / np.linalg.norm(grid.directions, axis=1, keepdims=True)
    delays = steering_delays(array.positions, directions, config.sound_speed)

    srp = np.zeros(len(grid), dtype=np.float64)
    for i, j in array.pairs():
        cross = spec[i] * np.conj(spec[j])
        denom = np.abs(cross)
        denom[denom < 1e-12] = 1e-12
        phat = (cross / denom).sum(axis=1)  # 在帧维累加 -> (n_freq,)
        tdoa = delays[:, i] - delays[:, j]  # (n_dirs,)
        steer = np.exp(2j * np.pi * freqs[None, :] * tdoa[:, None])
        srp += np.real(steer @ phat)
    return srp, grid


def srp_phat(
    signal: np.ndarray,
    array: MicArray,
    config: SonolocConfig | None = None,
    grid: SphereGrid | None = None,
) -> tuple[float, float]:
    """返回 SRP-PHAT 估计的 ``(azimuth_deg, elevation_deg)``。"""
    srp, grid = srp_phat_map(signal, array, config, grid)
    best = int(np.argmax(srp))
    return float(np.rad2deg(grid.azimuths[best])), float(np.rad2deg(grid.elevations[best]))
