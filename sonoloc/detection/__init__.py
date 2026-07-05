"""声学事件检测（SED）：弱标注下的 MIL 池化与后处理。"""

from sonoloc.detection.mil import (
    clip_labels,
    clip_probabilities,
    frames_to_events,
    median_filter,
)
from sonoloc.detection.pooling import POOLING_FUNCTIONS, pool

__all__ = [
    "POOLING_FUNCTIONS",
    "clip_labels",
    "clip_probabilities",
    "frames_to_events",
    "median_filter",
    "pool",
]
