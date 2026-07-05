"""ACCDOA 标签编解码。

ACCDOA（Activity-Coupled Cartesian DOA）把事件活跃度与方向合并为
一个 3 维笛卡尔向量：向量方向表示 DOA，模长表示该类事件的活跃程度。
每类事件对应一个 3 维向量，因此 ``C`` 类事件共 ``3C`` 维输出。
"""

from __future__ import annotations

import numpy as np

from sonoloc.io.arrays import cart2sph, sph2cart


def encode_accdoa(activity: np.ndarray, azimuth: np.ndarray, elevation: np.ndarray) -> np.ndarray:
    """把逐帧的活跃度与方向编码为 ACCDOA 向量。

    Parameters
    ----------
    activity:
        ``(n_frames, n_classes)``，取值 0/1 或概率。
    azimuth, elevation:
        ``(n_frames, n_classes)``，单位为弧度。

    Returns
    -------
    ndarray
        ``(n_frames, n_classes, 3)`` 的 ACCDOA 张量。
    """
    unit = sph2cart(azimuth, elevation, 1.0)
    return unit * activity[..., np.newaxis]


def decode_accdoa(accdoa: np.ndarray, threshold: float = 0.5) -> dict[str, np.ndarray]:
    """从 ACCDOA 向量解码活跃度与方向。

    Returns
    -------
    dict
        含 ``active``（布尔）、``azimuth``、``elevation``（弧度）与
        ``magnitude``（向量模长）四个 ``(n_frames, n_classes)`` 数组。
    """
    magnitude = np.linalg.norm(accdoa, axis=-1)
    active = magnitude >= threshold
    azimuth, elevation, _ = cart2sph(accdoa)
    return {
        "active": active,
        "azimuth": azimuth,
        "elevation": elevation,
        "magnitude": magnitude,
    }


def encode_multi_accdoa(
    activity: np.ndarray, azimuth: np.ndarray, elevation: np.ndarray
) -> np.ndarray:
    """Multi-ACCDOA 编码，支持每类事件的多个同时轨迹。

    输入形状为 ``(n_frames, n_tracks, n_classes)``，输出
    ``(n_frames, n_tracks, n_classes, 3)``。轨迹维用于表示同类事件的
    重叠声源（复调），配合置换不变训练使用。
    """
    if activity.ndim != 3:
        raise ValueError("multi-ACCDOA 需要 (n_frames, n_tracks, n_classes) 形状")
    unit = sph2cart(azimuth, elevation, 1.0)
    return unit * activity[..., np.newaxis]


def decode_multi_accdoa(accdoa: np.ndarray, threshold: float = 0.5) -> dict[str, np.ndarray]:
    """解码 multi-ACCDOA 张量，返回逐轨迹的活跃度与方向。"""
    if accdoa.ndim != 4:
        raise ValueError("multi-ACCDOA 解码需要 4 维张量")
    return decode_accdoa(accdoa, threshold=threshold)
