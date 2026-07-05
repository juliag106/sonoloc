"""一阶 Ambisonics（FOA）声强向量特征。

FOA 四通道按 ACN/SN3D 约定排列为 ``[W, X, Y, Z]``。有源声强
``I = Re{ W* · [X, Y, Z] }`` 指向声能流动方向，其反方向即声源方位。
归一化后的声强向量是 SELD 中常用的空间特征。
"""

from __future__ import annotations

import numpy as np


def intensity_vectors(foa_spec: np.ndarray, eps: float = 1e-10) -> np.ndarray:
    """由 FOA STFT 计算归一化有源声强向量。

    Parameters
    ----------
    foa_spec:
        形状 ``(4, n_freq, n_frames)`` 的复数 STFT，通道顺序 ``[W, X, Y, Z]``。

    Returns
    -------
    ndarray
        形状 ``(3, n_freq, n_frames)`` 的归一化声强特征，取值在 ``[-1, 1]``。
    """
    if foa_spec.shape[0] != 4:
        raise ValueError("声强向量需要 4 通道 FOA 输入 [W, X, Y, Z]")
    w = foa_spec[0]
    xyz = foa_spec[1:4]
    intensity = np.real(np.conj(w)[None, ...] * xyz)
    denom = np.abs(w) ** 2 + np.sum(np.abs(xyz) ** 2, axis=0)
    return intensity / (denom[None, ...] + eps)


def intensity_doa(foa_spec: np.ndarray) -> tuple[float, float]:
    """基于全局有源声强估计单声源的 ``(azimuth, elevation)``（弧度）。

    采用常见的 Ambisonics 声源编码约定 ``[X, Y, Z] = u · W``，此时有源声强
    ``Re{W* · [X, Y, Z]}`` 指向声源方向，因此 DOA 与声强方向一致。
    """
    w = foa_spec[0]
    xyz = foa_spec[1:4]
    intensity = np.real(np.conj(w)[None, ...] * xyz)
    mean_i = intensity.reshape(3, -1).mean(axis=1)
    norm = np.linalg.norm(mean_i)
    if norm < 1e-12:
        return 0.0, 0.0
    direction = mean_i / norm
    azimuth = float(np.arctan2(direction[1], direction[0]))
    elevation = float(np.arctan2(direction[2], np.hypot(direction[0], direction[1])))
    return azimuth, elevation
