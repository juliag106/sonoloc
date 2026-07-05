"""弱标注聚合与帧级后处理。

把逐帧的类别概率聚合成片段级标签（用于弱监督训练的目标），
以及从帧级概率恢复事件区间的简单后处理工具。
"""

from __future__ import annotations

import numpy as np

from sonoloc.detection.pooling import pool


def clip_probabilities(frame_probs: np.ndarray, method: str = "linear_softmax") -> np.ndarray:
    """把 ``(n_frames, n_classes)`` 概率聚合为 ``(n_classes,)`` 片段概率。"""
    return pool(frame_probs, method=method, axis=0)


def clip_labels(
    frame_probs: np.ndarray, method: str = "linear_softmax", threshold: float = 0.5
) -> np.ndarray:
    """聚合并阈值化，得到片段级二值标签。"""
    return clip_probabilities(frame_probs, method=method) >= threshold


def median_filter(frame_probs: np.ndarray, size: int = 5) -> np.ndarray:
    """对每一类做时间维中值滤波，平滑抖动的帧级预测。"""
    frame_probs = np.asarray(frame_probs, dtype=np.float64)
    if size <= 1:
        return frame_probs
    pad = size // 2
    padded = np.pad(frame_probs, ((pad, pad), (0, 0)), mode="edge")
    out = np.empty_like(frame_probs)
    for t in range(frame_probs.shape[0]):
        out[t] = np.median(padded[t : t + size], axis=0)
    return out


def frames_to_events(
    frame_active: np.ndarray, hop_seconds: float
) -> list[tuple[int, float, float]]:
    """把逐帧的布尔活跃度转换为 ``(class, onset, offset)`` 事件列表。"""
    frame_active = np.asarray(frame_active, dtype=bool)
    events: list[tuple[int, float, float]] = []
    n_frames, n_classes = frame_active.shape
    for c in range(n_classes):
        active = frame_active[:, c]
        start: int | None = None
        for t in range(n_frames):
            if active[t] and start is None:
                start = t
            elif not active[t] and start is not None:
                events.append((c, start * hop_seconds, t * hop_seconds))
                start = None
        if start is not None:
            events.append((c, start * hop_seconds, n_frames * hop_seconds))
    return events
