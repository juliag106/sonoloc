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


def sph2cart(
    azimuth: np.ndarray, elevation: np.ndarray, radius: np.ndarray | float = 1.0
) -> np.ndarray:
    """球坐标（弧度）转笛卡尔坐标，返回形状 ``(..., 3)``。"""
    az = np.asarray(azimuth, dtype=float)
    el = np.asarray(elevation, dtype=float)
    r = np.asarray(radius, dtype=float)
    x = r * np.cos(el) * np.cos(az)
    y = r * np.cos(el) * np.sin(az)
    z = r * np.sin(el)
    return np.stack([x, y, z], axis=-1)


def cart2sph(xyz: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """笛卡尔坐标转球坐标，返回 ``(azimuth, elevation, radius)``（弧度）。"""
    xyz = np.asarray(xyz, dtype=float)
    x, y, z = xyz[..., 0], xyz[..., 1], xyz[..., 2]
    radius = np.sqrt(x**2 + y**2 + z**2)
    azimuth = np.arctan2(y, x)
    elevation = np.arctan2(z, np.sqrt(x**2 + y**2))
    return azimuth, elevation, radius


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

    @classmethod
    def from_spherical(
        cls,
        name: str,
        azimuth_deg: list[float],
        elevation_deg: list[float],
        radius: float,
    ) -> MicArray:
        """由每个麦克风的方位角 / 仰角（度）与半径构造阵列。"""
        az = np.deg2rad(azimuth_deg)
        el = np.deg2rad(elevation_deg)
        return cls(name, sph2cart(az, el, radius))

    def pairs(self) -> list[tuple[int, int]]:
        """返回所有无序麦克风对 (i, j)，i < j。"""
        n = self.n_mics
        return [(i, j) for i in range(n) for j in range(i + 1, n)]


def tetrahedral_array(radius: float = 0.042) -> MicArray:
    """半径约 4.2 cm 的四面体阵列（FOA 麦克风格式常用几何）。"""
    return MicArray.from_spherical(
        "tetra",
        azimuth_deg=[45.0, -45.0, 135.0, -135.0],
        elevation_deg=[35.0, -35.0, -35.0, 35.0],
        radius=radius,
    )


def get_array(name: str) -> MicArray:
    """按名称获取内置阵列预设。"""
    if name == "tetra":
        return tetrahedral_array()
    raise KeyError(f"未知阵列预设: {name!r}")
