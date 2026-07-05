"""平面波远场导向向量与时延。

对来自单位方向 ``u`` 的远场平面波，第 ``m`` 个麦克风相对坐标原点的
传播时延为 ``tau_m = -(p_m · u) / c``；朝声源方向偏移的麦克风更早接收，
时延为负。
"""

from __future__ import annotations

import numpy as np


def steering_delays(
    positions: np.ndarray, directions: np.ndarray, sound_speed: float = 343.0
) -> np.ndarray:
    """计算每个方向、每个麦克风的传播时延（秒）。

    Parameters
    ----------
    positions:
        麦克风坐标 ``(n_mics, 3)``（米）。
    directions:
        单位方向向量 ``(n_dirs, 3)``。

    Returns
    -------
    ndarray
        形状 ``(n_dirs, n_mics)`` 的时延矩阵。
    """
    positions = np.asarray(positions, dtype=np.float64)
    directions = np.asarray(directions, dtype=np.float64)
    return -(directions @ positions.T) / sound_speed


def steering_vector(
    positions: np.ndarray,
    directions: np.ndarray,
    freqs: np.ndarray,
    sound_speed: float = 343.0,
) -> np.ndarray:
    """构造导向向量 ``(n_dirs, n_freq, n_mics)``。"""
    delays = steering_delays(positions, directions, sound_speed)
    freqs = np.asarray(freqs, dtype=np.float64)
    phase = -2j * np.pi * freqs[None, :, None] * delays[:, None, :]
    return np.exp(phase)
