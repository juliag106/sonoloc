"""弱标注下的多示例学习（MIL）池化。

只有片段级（clip-level）标签时，需要把逐帧的类别预测聚合为片段级预测。
不同的池化函数在“定位能力”和“分类能力”之间做出不同折中：

* ``max`` / ``mean``：最简单，但 max 只回传单帧梯度，mean 会低估短事件；
* ``linear_softmax`` / ``exp_softmax``：用预测值本身做自加权，兼顾两者；
* ``attention``：由单独的注意力打分决定各帧权重。

约定输入 ``frame_probs`` 形状为 ``(n_frames, n_classes)``，沿帧维聚合。
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

EPS = 1e-7


def mean_pool(frame_probs: np.ndarray, axis: int = 0) -> np.ndarray:
    return np.asarray(frame_probs).mean(axis=axis)


def max_pool(frame_probs: np.ndarray, axis: int = 0) -> np.ndarray:
    return np.asarray(frame_probs).max(axis=axis)


def linear_softmax_pool(frame_probs: np.ndarray, axis: int = 0) -> np.ndarray:
    """以预测值为权重的线性自加权池化。"""
    frame_probs = np.asarray(frame_probs, dtype=np.float64)
    weights = frame_probs / (frame_probs.sum(axis=axis, keepdims=True) + EPS)
    return (weights * frame_probs).sum(axis=axis)


def exp_softmax_pool(frame_probs: np.ndarray, axis: int = 0) -> np.ndarray:
    """以 ``exp(prob)`` 为权重的软最大池化。"""
    frame_probs = np.asarray(frame_probs, dtype=np.float64)
    weights = np.exp(frame_probs)
    weights = weights / (weights.sum(axis=axis, keepdims=True) + EPS)
    return (weights * frame_probs).sum(axis=axis)


def attention_pool(frame_probs: np.ndarray, attention: np.ndarray, axis: int = 0) -> np.ndarray:
    """由外部注意力打分加权的池化。"""
    frame_probs = np.asarray(frame_probs, dtype=np.float64)
    attention = np.asarray(attention, dtype=np.float64)
    weights = attention / (attention.sum(axis=axis, keepdims=True) + EPS)
    return (weights * frame_probs).sum(axis=axis)


POOLING_FUNCTIONS: dict[str, Callable[..., np.ndarray]] = {
    "mean": mean_pool,
    "max": max_pool,
    "linear_softmax": linear_softmax_pool,
    "exp_softmax": exp_softmax_pool,
}


def pool(frame_probs: np.ndarray, method: str = "linear_softmax", axis: int = 0) -> np.ndarray:
    """按名称选择池化函数。"""
    try:
        func = POOLING_FUNCTIONS[method]
    except KeyError as exc:
        raise KeyError(f"未知池化方法: {method!r}") from exc
    return func(frame_probs, axis=axis)
