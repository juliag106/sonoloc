# 使用指南

## 安装

```bash
pip install sonoloc            # 核心功能（numpy / scipy / soundfile / pyyaml）
pip install "sonoloc[torch]"   # 额外启用 CRNN 等神经网络模型
pip install "sonoloc[dev]"     # 开发依赖（pytest / ruff / mypy）
```

## 生成一段合成场景

```python
import numpy as np
from sonoloc.config import SonolocConfig
from sonoloc.data.scene import Scene, SoundEvent
from sonoloc.data.simulate import simulate_scene
from sonoloc.io.geometry import tetrahedral_array

config = SonolocConfig(sample_rate=24000)
array = tetrahedral_array()
events = [
    SoundEvent(class_index=3, azimuth=np.deg2rad(-60), elevation=0.0, onset=0.2, offset=0.9),
    SoundEvent(class_index=7, azimuth=np.deg2rad(80), elevation=np.deg2rad(15), onset=1.0, offset=1.8),
]
scene = Scene(duration=2.0, sample_rate=24000, n_classes=13, events=events)
signal, labels = simulate_scene(scene, array, config, snr_db=6.0)  # 6 dB 噪声，鲁棒性评测
```

`labels` 包含逐帧的 `activity` / `azimuth` / `elevation`。

## 声源定位

```python
from sonoloc.localization import srp_phat, music

az, el = srp_phat(signal, array, config)         # 可控响应功率
az2, el2 = music(signal, array, config, n_sources=1)
```

低频段（波长远大于阵列孔径）方位分辨率有限，建议关注接近空间奈奎斯特频率的频带。

## 特征提取

```python
from sonoloc.features.pipeline import FeaturePipeline

features = FeaturePipeline(config, array)(signal)  # (通道 + 麦克风对, n_mels, n_frames)
```

## 弱标注聚合

```python
from sonoloc.detection.mil import clip_labels, median_filter

frame_probs = ...              # 模型输出的帧级概率 (n_frames, n_classes)
frame_probs = median_filter(frame_probs, size=5)
clip = clip_labels(frame_probs, method="linear_softmax", threshold=0.5)
```

## 评测

```python
from sonoloc.metrics import segment_detection_scores, localization_scores, aggregate_seld

det = segment_detection_scores(ref_active, pred_active, frames_per_segment=10)
loc = localization_scores(ref_active, ref_az, ref_el, pred_active, pred_az, pred_el)
print(aggregate_seld(det, loc)["seld_score"])
```

## 命令行

```bash
sonoloc info
sonoloc simulate scene.wav --azimuth 45 --elevation 10 --snr 6
sonoloc doa scene.wav --method music
sonoloc features scene.wav --output feats.npy
```

所有子命令都支持 `--config config.yaml` 从文件读取参数（`SonolocConfig.save` 可导出）。
