"""窄带 MUSIC 声源定位（宽带频率平均）。

对每个频点估计空间协方差矩阵并做特征分解，用噪声子空间构造 MUSIC
谱：导向向量越接近信号子空间，噪声投影越小、谱值越大。多个频点上的
谱取平均得到宽带估计。
"""

from __future__ import annotations

import numpy as np

from sonoloc.config import SonolocConfig
from sonoloc.features.stft import stft, stft_frequencies
from sonoloc.io.geometry import MicArray
from sonoloc.localization.grid import SphereGrid, make_grid
from sonoloc.localization.steering import steering_vector


def music_map(
    signal: np.ndarray,
    array: MicArray,
    config: SonolocConfig | None = None,
    grid: SphereGrid | None = None,
    n_sources: int = 1,
    freq_range: tuple[float, float] = (500.0, 3500.0),
) -> tuple[np.ndarray, SphereGrid]:
    """计算 MUSIC 伪谱。"""
    config = config or SonolocConfig()
    grid = grid or make_grid()

    spec = stft(signal, n_fft=config.n_fft, hop_length=config.hop_length, window=config.window)
    freqs = stft_frequencies(config.sample_rate, config.n_fft)
    steer = steering_vector(array.positions, grid.directions, freqs, config.sound_speed)

    n_mics = array.n_mics
    band = np.where((freqs >= freq_range[0]) & (freqs <= freq_range[1]))[0]
    pseudo = np.zeros(len(grid), dtype=np.float64)
    for f in band:
        snapshots = spec[:, f, :]  # (n_mics, n_frames)
        cov = snapshots @ snapshots.conj().T / snapshots.shape[1]
        _eigvals, eigvecs = np.linalg.eigh(cov)
        noise = eigvecs[:, : n_mics - n_sources]  # 最小特征值对应噪声子空间
        proj = steer[:, f, :].conj() @ noise  # a^H · E_noise，(n_dirs, n_noise)
        denom = np.sum(np.abs(proj) ** 2, axis=1)
        pseudo += 1.0 / (denom + 1e-12)
    return pseudo, grid


def music(
    signal: np.ndarray,
    array: MicArray,
    config: SonolocConfig | None = None,
    grid: SphereGrid | None = None,
    n_sources: int = 1,
) -> tuple[float, float]:
    """返回 MUSIC 估计的 ``(azimuth_deg, elevation_deg)``。"""
    pseudo, grid = music_map(signal, array, config, grid, n_sources=n_sources)
    best = int(np.argmax(pseudo))
    return float(np.rad2deg(grid.azimuths[best])), float(np.rad2deg(grid.elevations[best]))
