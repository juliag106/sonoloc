"""定位指标测试。"""

import numpy as np

from sonoloc.metrics.localization import angular_distance, localization_scores


def test_angular_distance_zero() -> None:
    d = angular_distance(np.array(0.3), np.array(0.1), np.array(0.3), np.array(0.1))
    assert abs(float(d)) < 1e-9


def test_angular_distance_ninety_degrees() -> None:
    # 方位角相差 90°、仰角为 0 时，夹角应为 90°
    d = angular_distance(np.array(0.0), np.array(0.0), np.array(np.pi / 2), np.array(0.0))
    assert abs(float(d) - 90.0) < 1e-6


def test_localization_perfect() -> None:
    active = np.ones((4, 2), dtype=bool)
    az = np.zeros((4, 2))
    el = np.zeros((4, 2))
    scores = localization_scores(active, az, el, active, az, el)
    assert scores["localization_error"] < 1e-9
    assert scores["localization_recall"] == 1.0


def test_localization_partial_recall() -> None:
    ref_active = np.ones((2, 1), dtype=bool)
    pred_active = np.array([[True], [False]])
    az = np.zeros((2, 1))
    el = np.zeros((2, 1))
    scores = localization_scores(ref_active, az, el, pred_active, az, el)
    assert scores["localization_recall"] == 0.5
