"""方位角 / 仰角搜索网格。"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from sonoloc.io.arrays import sph2cart


@dataclass
class SphereGrid:
    """离散化的球面方向集合。"""

    directions: np.ndarray  # (n_dirs, 3) 单位向量
    azimuths: np.ndarray  # (n_dirs,) 弧度
    elevations: np.ndarray  # (n_dirs,) 弧度

    def __len__(self) -> int:
        return int(self.directions.shape[0])

    def nearest(self, azimuth: float, elevation: float) -> int:
        """返回与给定方向夹角最小的网格点索引。"""
        target = sph2cart(np.array(azimuth), np.array(elevation), 1.0)
        cos_angle = self.directions @ target
        return int(np.argmax(cos_angle))


def make_grid(
    az_step: float = 10.0,
    el_step: float = 10.0,
    el_min: float = -40.0,
    el_max: float = 40.0,
) -> SphereGrid:
    """构造覆盖整个方位、有限仰角范围的搜索网格。"""
    az = np.deg2rad(np.arange(-180.0, 180.0, az_step))
    el = np.deg2rad(np.arange(el_min, el_max + el_step / 2.0, el_step))
    grid_az, grid_el = np.meshgrid(az, el)
    az_flat = grid_az.ravel()
    el_flat = grid_el.ravel()
    directions = sph2cart(az_flat, el_flat, 1.0)
    return SphereGrid(directions, az_flat, el_flat)
