# sonoloc

多通道声学事件检测与声源定位（SELD）工具箱。

- 特征：多通道 log-mel、GCC-PHAT、FOA 声强向量
- 定位：SRP-PHAT、MUSIC 等经典 DOA 方法
- 检测：面向弱标注的多示例学习（MIL）池化

特征流水线 `FeaturePipeline` 会把逐通道 log-mel 与麦克风对之间的
GCC-PHAT 拼接成统一张量，作为下游模型或经典定位方法的输入。

## 快速上手

仿真一个已知方位的场景，再用 SRP-PHAT 找回方向：

```python
import numpy as np
from sonoloc.config import SonolocConfig
from sonoloc.data.scene import Scene, SoundEvent
from sonoloc.data.simulate import simulate_scene
from sonoloc.io.arrays import tetrahedral_array
from sonoloc.localization import srp_phat

config = SonolocConfig(sample_rate=24000)
array = tetrahedral_array()
event = SoundEvent(class_index=0, azimuth=np.deg2rad(45), elevation=0.0, onset=0.0, offset=1.0)
scene = Scene(duration=1.0, sample_rate=24000, n_classes=13, events=[event])

signal, labels = simulate_scene(scene, array, config, snr_db=10.0)
azimuth, elevation = srp_phat(signal, array, config)
print(azimuth, elevation)  # 约 45°, 0°
```

命令行：

```bash
sonoloc simulate scene.wav --azimuth 45 --snr 10
sonoloc doa scene.wav --method srp-phat
```

> 仍在早期开发阶段。
