# sonoloc

[![CI](https://github.com/juliag106/sonoloc/actions/workflows/ci.yml/badge.svg)](https://github.com/juliag106/sonoloc/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

**sonoloc** 是一个面向**多通道声学事件检测与声源定位（SELD）**的 Python 工具箱，
围绕“弱标注”和“鲁棒噪声场景”两个实际难点组织：既提供可解释的经典信号处理
基线，也给出与 DCASE 任务三对齐的特征、标签和评价指标。

核心以 **NumPy / SciPy** 实现，安装即用、便于复现；神经网络部分依赖
**PyTorch**，作为可选组件（`pip install "sonoloc[torch]"`），不装也不影响其余功能。

## 特性

- **多通道特征**：逐通道 log-mel、麦克风对 GCC-PHAT、FOA 有源声强向量，
  以及把它们拼接成统一张量的 `FeaturePipeline`。
- **经典 DOA 定位**：SRP-PHAT、MUSIC，以及可复用的导向向量 / 球面搜索网格。
- **弱标注检测**：线性/指数 softmax、注意力等 MIL 池化，把帧级预测聚合成片段级标签。
- **ACCDOA 标签**：单/多轨 ACCDOA 编解码，方向与活跃度联合表示。
- **鲁棒性数据**：自由场平面波渲染 + 扩散噪声按目标 SNR 混合，生成可控的评测场景。
- **SELD 指标**：段级 ER / F 分数、定位误差 LE、定位召回 LR，及综合 SELD 分数。
- **可选神经模型**：CRNN 骨干 + ACCDOA 头（PyTorch）。
- **命令行**：`sonoloc doa / features / simulate / info`。

## 安装

```bash
pip install sonoloc            # 核心（numpy / scipy）
pip install "sonoloc[torch]"   # 额外启用神经网络模型
```

## 快速上手

仿真一个已知方位的场景，再用 SRP-PHAT 找回方向：

```python
import numpy as np
from sonoloc.config import SonolocConfig
from sonoloc.data.scene import Scene, SoundEvent
from sonoloc.data.simulate import simulate_scene
from sonoloc.io.geometry import tetrahedral_array
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
sonoloc doa scene.wav --method music
```

## 文档

- [架构总览](docs/architecture.md)
- [使用指南](docs/usage.md)
- [设计说明](docs/design-notes.md)
- [API 参考](docs/api-reference.md)

## 参与贡献

欢迎提 issue 与 PR，详见 [CONTRIBUTING](CONTRIBUTING.md)。

## 许可

[MIT](LICENSE) © Lucy Liang
