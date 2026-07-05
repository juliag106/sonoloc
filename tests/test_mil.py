"""弱标注聚合与后处理测试。"""

import numpy as np

from sonoloc.detection.mil import clip_labels, frames_to_events, median_filter


def test_clip_labels_detect_active_class() -> None:
    frame_probs = np.zeros((20, 3))
    frame_probs[5:10, 1] = 0.9  # 仅第 1 类在若干帧活跃
    labels = clip_labels(frame_probs, method="max", threshold=0.5)
    assert labels[1]
    assert not labels[0]


def test_median_filter_removes_single_spike() -> None:
    probs = np.zeros((11, 1))
    probs[5, 0] = 1.0  # 孤立尖峰
    smoothed = median_filter(probs, size=3)
    assert smoothed[5, 0] == 0.0


def test_frames_to_events_segments() -> None:
    active = np.zeros((10, 1), dtype=bool)
    active[2:5, 0] = True
    events = frames_to_events(active, hop_seconds=0.1)
    assert len(events) == 1
    cls, onset, offset = events[0]
    assert cls == 0
    assert np.isclose(onset, 0.2)
    assert np.isclose(offset, 0.5)
