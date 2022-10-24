"""麦克风阵列几何。

坐标系约定（与 DCASE / 常见 SELD 数据一致）：

* x 轴指向正前方（方位角 0°）；
* y 轴指向正左方（方位角 +90°）；
* z 轴指向正上方（仰角 +90°）；
* 方位角 azimuth 在水平面内逆时针为正，仰角 elevation 向上为正。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class MicArray:
    """一个麦克风阵列，用米制笛卡尔坐标描述每个麦克风的位置。"""

    name: str
    positions: np.ndarray  # 形状 (n_mics, 3)

    def __post_init__(self) -> None:
        pos = np.asarray(self.positions, dtype=float)
        if pos.ndim != 2 or pos.shape[1] != 3:
            raise ValueError("positions 必须是形状 (n_mics, 3) 的数组")
        object.__setattr__(self, "positions", pos)

    @property
    def n_mics(self) -> int:
        return int(self.positions.shape[0])

    def pairs(self) -> list[tuple[int, int]]:
        """返回所有无序麦克风对 (i, j)，i < j。"""
        n = self.n_mics
        return [(i, j) for i in range(n) for j in range(i + 1, n)]


def tetrahedral_array(radius: float = 0.042) -> MicArray:
    """半径约 4.2 cm 的四面体阵列（FOA 麦克风格式常用几何）。"""
    az = np.deg2rad([45.0, -45.0, 135.0, -135.0])
    el = np.deg2rad([35.0, -35.0, -35.0, 35.0])
    x = radius * np.cos(el) * np.cos(az)
    y = radius * np.cos(el) * np.sin(az)
    z = radius * np.sin(el)
    return MicArray("tetra", np.stack([x, y, z], axis=1))


def get_array(name: str) -> MicArray:
    """按名称获取内置阵列预设。"""
    if name == "tetra":
        return tetrahedral_array()
    raise KeyError(f"未知阵列预设: {name!r}")
