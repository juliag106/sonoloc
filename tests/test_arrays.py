"""麦克风阵列几何测试。"""

import numpy as np

from sonoloc.io.arrays import cart2sph, get_array, sph2cart, tetrahedral_array


def test_tetra_has_four_mics() -> None:
    array = tetrahedral_array()
    assert array.n_mics == 4
    assert array.positions.shape == (4, 3)


def test_tetra_radius_matches() -> None:
    array = tetrahedral_array(radius=0.042)
    radii = np.linalg.norm(array.positions, axis=1)
    assert np.allclose(radii, 0.042, atol=1e-9)


def test_pairs_count() -> None:
    array = tetrahedral_array()
    # C(4, 2) = 6 个麦克风对
    assert len(array.pairs()) == 6


def test_sph_cart_round_trip() -> None:
    az = np.deg2rad([0.0, 45.0, -120.0])
    el = np.deg2rad([0.0, 30.0, -15.0])
    xyz = sph2cart(az, el, 1.0)
    az2, el2, r = cart2sph(xyz)
    assert np.allclose(az, az2, atol=1e-9)
    assert np.allclose(el, el2, atol=1e-9)
    assert np.allclose(r, 1.0, atol=1e-9)


def test_get_array_unknown() -> None:
    try:
        get_array("does-not-exist")
    except KeyError:
        return
    raise AssertionError("应对未知阵列抛出 KeyError")
