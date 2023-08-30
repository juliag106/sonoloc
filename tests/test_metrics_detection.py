"""段级检测指标测试。"""

import numpy as np

from sonoloc.metrics.detection import segment_detection_scores


def test_perfect_prediction() -> None:
    rng = np.random.default_rng(0)
    ref = rng.random((40, 5)) > 0.7
    scores = segment_detection_scores(ref, ref, frames_per_segment=10)
    assert scores["error_rate"] == 0.0
    assert scores["f_score"] == 1.0


def test_all_insertions() -> None:
    ref = np.zeros((20, 3), dtype=bool)
    pred = np.ones((20, 3), dtype=bool)
    scores = segment_detection_scores(ref, pred, frames_per_segment=10)
    # 没有任何参考事件却全部预测为正 -> 纯插入，F 分数为 0
    assert scores["f_score"] == 0.0
    assert scores["error_rate"] > 0.0


def test_partial_recall() -> None:
    ref = np.zeros((10, 2), dtype=bool)
    ref[:, 0] = True
    ref[:, 1] = True
    pred = np.zeros((10, 2), dtype=bool)
    pred[:, 0] = True  # 只命中一类
    scores = segment_detection_scores(ref, pred, frames_per_segment=10)
    assert 0.0 < scores["recall"] < 1.0
    assert scores["precision"] == 1.0
