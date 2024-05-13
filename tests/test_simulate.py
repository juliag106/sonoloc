"""端到端测试：先仿真场景，再用 SRP-PHAT 找回声源方位。"""

import numpy as np

from sonoloc.config import SonolocConfig
from sonoloc.data.scene import Scene, SoundEvent
from sonoloc.data.simulate import simulate_scene
from sonoloc.io.geometry import tetrahedral_array
from sonoloc.localization.srp_phat import srp_phat


def test_simulate_then_localize() -> None:
    config = SonolocConfig(sample_rate=24000)
    array = tetrahedral_array()
    event = SoundEvent(
        class_index=2, azimuth=np.deg2rad(45.0), elevation=0.0, onset=0.05, offset=0.95
    )
    scene = Scene(duration=1.0, sample_rate=24000, n_classes=13, events=[event])

    signal, labels = simulate_scene(scene, array, config, seed=7)
    assert signal.shape[0] == array.n_mics
    assert labels["activity"].shape == (10, 13)

    az, _el = srp_phat(signal, array, config)
    assert abs(az - 45.0) <= 15.0


def test_simulate_with_noise_changes_power() -> None:
    config = SonolocConfig(sample_rate=16000)
    array = tetrahedral_array()
    event = SoundEvent(class_index=0, azimuth=0.0, elevation=0.0, onset=0.0, offset=1.0)
    scene = Scene(duration=1.0, sample_rate=16000, n_classes=13, events=[event])

    clean, _ = simulate_scene(scene, array, config, seed=1)
    noisy, _ = simulate_scene(scene, array, config, snr_db=0.0, seed=1)
    assert np.mean(noisy**2) > np.mean(clean**2)
