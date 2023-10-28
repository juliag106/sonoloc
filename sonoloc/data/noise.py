"""扩散噪声与信噪比（SNR）混合。

用于构造鲁棒性评测的噪声场景：既支持逐通道独立的白噪声，也支持
所有通道共享的“空间扩散”噪声近似，并按目标 SNR 缩放后叠加到信号上。
"""

from __future__ import annotations

import numpy as np


def signal_power(signal: np.ndarray) -> float:
    """信号的平均功率（所有通道、所有样本）。"""
    signal = np.asarray(signal, dtype=np.float64)
    return float(np.mean(signal**2))


def white_noise(shape: tuple[int, ...], rng: np.random.Generator) -> np.ndarray:
    """逐通道独立的高斯白噪声。"""
    return rng.standard_normal(shape)


def diffuse_noise(shape: tuple[int, int], rng: np.random.Generator) -> np.ndarray:
    """近似空间扩散噪声：共享分量与独立分量的加权和。"""
    n_channels, n_samples = shape
    common = rng.standard_normal(n_samples)
    independent = rng.standard_normal((n_channels, n_samples))
    return (common[None, :] + independent) / np.sqrt(2.0)


def add_noise_at_snr(
    signal: np.ndarray,
    noise: np.ndarray,
    snr_db: float,
) -> np.ndarray:
    """按目标 SNR（dB）把噪声叠加到信号上。"""
    signal = np.asarray(signal, dtype=np.float64)
    noise = np.asarray(noise, dtype=np.float64)
    sig_power = signal_power(signal)
    noise_power = signal_power(noise)
    if noise_power < 1e-20:
        return signal
    target_noise_power = sig_power / (10.0 ** (snr_db / 10.0))
    scale = np.sqrt(target_noise_power / noise_power)
    return signal + scale * noise


def apply_diffuse_noise(signal: np.ndarray, snr_db: float, seed: int = 0) -> np.ndarray:
    """向多通道信号叠加指定 SNR 的扩散噪声。"""
    rng = np.random.default_rng(seed)
    noise = diffuse_noise(signal.shape, rng)
    return add_noise_at_snr(signal, noise, snr_db)
