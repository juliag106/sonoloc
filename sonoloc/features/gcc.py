"""广义互相关相位变换（GCC-PHAT）。

GCC-PHAT 通过对互功率谱做幅度归一化来突出时延信息，
对混响和有色噪声更鲁棒，是麦克风对时延估计的经典方法。
"""

from __future__ import annotations

import numpy as np


def gcc_phat(
    sig: np.ndarray,
    ref: np.ndarray,
    n_fft: int | None = None,
    max_tau: int | None = None,
    interp: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """估计 ``sig`` 相对 ``ref`` 的到达时延。

    Returns
    -------
    lags:
        时延（样本数）坐标轴。
    cc:
        对应的 GCC-PHAT 相关序列，峰值位置即时延估计。
    """
    sig = np.asarray(sig, dtype=np.float64)
    ref = np.asarray(ref, dtype=np.float64)
    n = sig.shape[-1] + ref.shape[-1]
    if n_fft is None:
        n_fft = 1 << (int(n - 1).bit_length())  # 向上取到 2 的幂

    x = np.fft.rfft(sig, n=n_fft)
    y = np.fft.rfft(ref, n=n_fft)
    cross = x * np.conj(y)
    denom = np.abs(cross)
    denom[denom < 1e-12] = 1e-12  # 避免除零
    cross /= denom

    cc = np.fft.irfft(cross, n=interp * n_fft)
    max_shift = interp * n_fft // 2
    if max_tau is not None:
        max_shift = min(interp * max_tau, max_shift)

    cc = np.concatenate((cc[-max_shift:], cc[: max_shift + 1]))
    lags = np.arange(-max_shift, max_shift + 1) / interp
    return lags, cc


def estimate_tdoa(sig: np.ndarray, ref: np.ndarray, max_tau: int | None = None) -> float:
    """返回使 GCC-PHAT 取最大值的时延（样本数，可为负）。"""
    lags, cc = gcc_phat(sig, ref, max_tau=max_tau)
    return float(lags[int(np.argmax(cc))])
