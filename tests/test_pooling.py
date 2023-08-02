"""MIL 池化函数测试。"""

import numpy as np

from sonoloc.detection.pooling import (
    exp_softmax_pool,
    linear_softmax_pool,
    max_pool,
    mean_pool,
    pool,
)


def test_pool_output_shape() -> None:
    frame_probs = np.random.default_rng(0).random((50, 6))
    for method in ("mean", "max", "linear_softmax", "exp_softmax"):
        clip = pool(frame_probs, method=method)
        assert clip.shape == (6,)


def test_pooling_between_mean_and_max() -> None:
    # 自加权池化应落在 mean 与 max 之间。
    frame_probs = np.array([[0.1], [0.9], [0.2], [0.0]])
    m = mean_pool(frame_probs)[0]
    mx = max_pool(frame_probs)[0]
    for value in (linear_softmax_pool(frame_probs)[0], exp_softmax_pool(frame_probs)[0]):
        assert m <= value <= mx


def test_single_active_frame_dominates_max() -> None:
    frame_probs = np.zeros((10, 3))
    frame_probs[4, 1] = 1.0
    clip = max_pool(frame_probs)
    assert clip[1] == 1.0
    assert clip[0] == 0.0


def test_unknown_method_raises() -> None:
    try:
        pool(np.zeros((3, 2)), method="nope")
    except KeyError:
        return
    raise AssertionError("未知方法应抛出 KeyError")
