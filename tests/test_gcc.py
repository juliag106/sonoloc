"""GCC-PHAT 时延估计测试。"""

import numpy as np

from sonoloc.features.gcc import estimate_tdoa, gcc_phat


def test_gcc_phat_recovers_known_delay() -> None:
    rng = np.random.default_rng(42)
    ref = rng.standard_normal(2048)
    delay = 7
    sig = np.roll(ref, delay)  # sig 相对 ref 延后 7 个样本
    tau = estimate_tdoa(sig, ref, max_tau=64)
    assert abs(tau - delay) <= 1


def test_gcc_phat_zero_delay() -> None:
    rng = np.random.default_rng(1)
    x = rng.standard_normal(1024)
    tau = estimate_tdoa(x, x, max_tau=32)
    assert tau == 0


def test_gcc_phat_axis_shapes() -> None:
    rng = np.random.default_rng(2)
    a = rng.standard_normal(512)
    b = rng.standard_normal(512)
    lags, cc = gcc_phat(a, b, max_tau=40)
    assert lags.shape == cc.shape
    assert lags[0] == -40 and lags[-1] == 40
