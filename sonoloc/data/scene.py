"""合成声学场景的数据结构。

一个 ``Scene`` 由若干带方向和时间区间的 ``SoundEvent`` 组成，
可据此渲染多通道信号并生成逐帧的 SELD 标签。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class SoundEvent:
    """带空间方向与起止时间的单个事件。"""

    class_index: int
    azimuth: float  # 弧度
    elevation: float  # 弧度
    onset: float  # 秒
    offset: float  # 秒
    signal: np.ndarray | None = None  # 可选的单声道源波形

    def duration(self) -> float:
        return max(self.offset - self.onset, 0.0)


@dataclass
class Scene:
    """一段时长内的事件集合。"""

    duration: float
    sample_rate: int
    n_classes: int
    events: list[SoundEvent] = field(default_factory=list)

    def label_frames(self, label_rate: float) -> dict[str, np.ndarray]:
        """按标签帧率生成 ``activity`` / ``azimuth`` / ``elevation`` 数组。"""
        n_frames = int(round(self.duration * label_rate))
        activity = np.zeros((n_frames, self.n_classes), dtype=np.float64)
        azimuth = np.zeros((n_frames, self.n_classes), dtype=np.float64)
        elevation = np.zeros((n_frames, self.n_classes), dtype=np.float64)
        for event in self.events:
            start = int(round(event.onset * label_rate))
            end = int(round(event.offset * label_rate))
            start = max(start, 0)
            end = min(end, n_frames)
            activity[start:end, event.class_index] = 1.0
            azimuth[start:end, event.class_index] = event.azimuth
            elevation[start:end, event.class_index] = event.elevation
        return {"activity": activity, "azimuth": azimuth, "elevation": elevation}
