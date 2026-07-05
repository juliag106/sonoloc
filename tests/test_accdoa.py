"""ACCDOA 编解码测试。"""

import numpy as np

from sonoloc.labels.accdoa import (
    decode_accdoa,
    decode_multi_accdoa,
    encode_accdoa,
    encode_multi_accdoa,
)


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


def test_multi_accdoa_round_trip() -> None:
    n_frames, n_tracks, n_classes = 4, 2, 3
    rng = np.random.default_rng(5)
    activity = (rng.random((n_frames, n_tracks, n_classes)) > 0.5).astype(float)
    az = rng.uniform(-np.pi, np.pi, (n_frames, n_tracks, n_classes))
    el = rng.uniform(-np.pi / 4, np.pi / 4, (n_frames, n_tracks, n_classes))

    accdoa = encode_multi_accdoa(activity, az, el)
    assert accdoa.shape == (n_frames, n_tracks, n_classes, 3)

    decoded = decode_multi_accdoa(accdoa)
    np.testing.assert_array_equal(decoded["active"], activity.astype(bool))


def test_multi_accdoa_requires_three_dims() -> None:
    try:
        encode_multi_accdoa(np.zeros((4, 3)), np.zeros((4, 3)), np.zeros((4, 3)))
    except ValueError:
        return
    raise AssertionError("2 维输入应触发 ValueError")
