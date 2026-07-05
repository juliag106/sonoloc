"""搜索网格测试。"""

import numpy as np

from sonoloc.localization.grid import make_grid


def test_grid_covers_azimuth() -> None:
    grid = make_grid(az_step=10.0, el_step=10.0)
    assert len(grid) > 0
    assert grid.directions.shape[1] == 3
    # 方向向量应为单位向量
    norms = np.linalg.norm(grid.directions, axis=1)
    assert np.allclose(norms, 1.0)


def test_nearest_index_matches_query() -> None:
    grid = make_grid(az_step=10.0, el_step=10.0)
    idx = grid.nearest(np.deg2rad(90.0), np.deg2rad(0.0))
    assert abs(np.rad2deg(grid.azimuths[idx]) - 90.0) <= 5.0
    assert abs(np.rad2deg(grid.elevations[idx])) <= 5.0


def test_elevation_bounds() -> None:
    grid = make_grid(el_min=-40.0, el_max=40.0)
    el_deg = np.rad2deg(grid.elevations)
    assert el_deg.min() >= -40.0 - 1e-6
    assert el_deg.max() <= 40.0 + 1e-6
