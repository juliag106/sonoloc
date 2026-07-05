"""ACCDOA 编解码测试。"""

import numpy as np

from sonoloc.labels.accdoa import decode_accdoa, encode_accdoa


def test_accdoa_round_trip() -> None:
    n_frames, n_classes = 5, 3
    rng = np.random.default_rng(0)
    activity = (rng.random((n_frames, n_classes)) > 0.5).astype(float)
    azimuth = rng.uniform(-np.pi, np.pi, (n_frames, n_classes))
    elevation = rng.uniform(-np.pi / 4, np.pi / 4, (n_frames, n_classes))

    accdoa = encode_accdoa(activity, azimuth, elevation)
    assert accdoa.shape == (n_frames, n_classes, 3)

    decoded = decode_accdoa(accdoa, threshold=0.5)
    np.testing.assert_array_equal(decoded["active"], activity.astype(bool))

    # 仅在活跃处比较方向
    mask = activity.astype(bool)
    assert np.allclose(decoded["azimuth"][mask], azimuth[mask], atol=1e-6)
    assert np.allclose(decoded["elevation"][mask], elevation[mask], atol=1e-6)


def test_accdoa_inactive_has_small_magnitude() -> None:
    activity = np.zeros((2, 2))
    az = np.zeros((2, 2))
    el = np.zeros((2, 2))
    accdoa = encode_accdoa(activity, az, el)
    decoded = decode_accdoa(accdoa)
    assert not decoded["active"].any()
