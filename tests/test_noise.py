"""噪声混合测试。"""

import numpy as np

from sonoloc.data.noise import add_noise_at_snr, diffuse_noise, signal_power


def test_add_noise_hits_target_snr() -> None:
    rng = np.random.default_rng(0)
    signal = rng.standard_normal((4, 24000))
    noise = rng.standard_normal((4, 24000))
    for target in (0.0, 10.0, 20.0):
        noisy = add_noise_at_snr(signal, noise, snr_db=target)
        residual = noisy - signal
        measured = 10 * np.log10(signal_power(signal) / signal_power(residual))
        assert abs(measured - target) < 0.5


def test_zero_noise_is_noop() -> None:
    signal = np.ones((2, 100))
    out = add_noise_at_snr(signal, np.zeros((2, 100)), snr_db=10.0)
    np.testing.assert_allclose(out, signal)


def test_diffuse_noise_shape() -> None:
    rng = np.random.default_rng(1)
    noise = diffuse_noise((4, 1000), rng)
    assert noise.shape == (4, 1000)
