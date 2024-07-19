"""示例：仿真一个已知方位的声源，比较 SRP-PHAT 与 MUSIC 的定位结果。

运行：
    python examples/01_doa_estimation/run.py
"""

import numpy as np

from sonoloc.config import SonolocConfig
from sonoloc.data.scene import Scene, SoundEvent
from sonoloc.data.simulate import simulate_scene
from sonoloc.io.geometry import tetrahedral_array
from sonoloc.localization import music, srp_phat


def main() -> None:
    config = SonolocConfig(sample_rate=24000)
    array = tetrahedral_array()

    true_az, true_el = 55.0, 10.0
    event = SoundEvent(
        class_index=0,
        azimuth=np.deg2rad(true_az),
        elevation=np.deg2rad(true_el),
        onset=0.0,
        offset=1.0,
    )
    scene = Scene(duration=1.0, sample_rate=24000, n_classes=13, events=[event])

    for snr in (None, 10.0, 0.0):
        signal, _ = simulate_scene(scene, array, config, snr_db=snr, seed=0)
        srp_az, srp_el = srp_phat(signal, array, config)
        mus_az, mus_el = music(signal, array, config)
        tag = "clean" if snr is None else f"{snr:.0f} dB"
        print(
            f"[{tag:>6}] true=({true_az:.0f},{true_el:.0f})  "
            f"srp=({srp_az:.0f},{srp_el:.0f})  music=({mus_az:.0f},{mus_el:.0f})"
        )


if __name__ == "__main__":
    main()
