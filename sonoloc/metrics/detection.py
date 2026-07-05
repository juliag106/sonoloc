"""基于分段（segment-based）的检测指标：错误率 ER 与 F 分数。

按 DCASE 的做法，把逐帧的多标签活跃度聚合到固定长度的时间段（默认 1 秒），
再在段级别统计替换 / 删除 / 插入，得到错误率与 F 分数。
"""

from __future__ import annotations

import numpy as np


def _segmentize(activity: np.ndarray, frames_per_segment: int) -> np.ndarray:
    """把 ``(n_frames, n_classes)`` 聚合为 ``(n_segments, n_classes)`` 段级活跃度。"""
    activity = np.asarray(activity, dtype=bool)
    n_frames, n_classes = activity.shape
    n_segments = int(np.ceil(n_frames / frames_per_segment))
    out = np.zeros((n_segments, n_classes), dtype=bool)
    for s in range(n_segments):
        chunk = activity[s * frames_per_segment : (s + 1) * frames_per_segment]
        out[s] = chunk.any(axis=0)
    return out


def segment_detection_scores(
    reference: np.ndarray, prediction: np.ndarray, frames_per_segment: int = 10
) -> dict[str, float]:
    """计算段级 ER、F 分数、precision 与 recall。"""
    ref = _segmentize(reference, frames_per_segment)
    sys = _segmentize(prediction, frames_per_segment)
    if ref.shape != sys.shape:
        raise ValueError("参考与预测的形状不一致")

    n_ref = ref.sum(axis=1).astype(np.int64)
    n_sys = sys.sum(axis=1).astype(np.int64)
    tp = np.logical_and(ref, sys).sum(axis=1).astype(np.int64)
    fn = n_ref - tp
    fp = n_sys - tp

    substitutions = np.minimum(fn, fp)
    deletions = np.maximum(0, fn - fp)
    insertions = np.maximum(0, fp - fn)

    total_ref = float(n_ref.sum())
    error_rate = float((substitutions + deletions + insertions).sum()) / max(total_ref, 1.0)

    tp_sum = float(tp.sum())
    fp_sum = float(fp.sum())
    fn_sum = float(fn.sum())
    precision = tp_sum / max(tp_sum + fp_sum, 1e-12)
    recall = tp_sum / max(tp_sum + fn_sum, 1e-12)
    f_score = 2 * precision * recall / max(precision + recall, 1e-12)

    return {
        "error_rate": error_rate,
        "f_score": f_score,
        "precision": precision,
        "recall": recall,
    }
