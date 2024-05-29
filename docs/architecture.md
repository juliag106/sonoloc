# 架构总览

sonoloc 按“信号 → 特征 → 检测/定位 → 指标”的数据流组织，各子包职责单一、
彼此解耦，经典方法与神经方法共用同一套配置与标签约定。

## 目录结构

```
sonoloc/
├── config.py           # SonolocConfig：STFT / mel / 标签帧率 / 阵列等共享参数
├── io/
│   ├── geometry.py     # MicArray 几何、球/笛卡尔坐标转换、阵列预设
│   └── audio.py        # 多通道读写与重采样
├── features/
│   ├── stft.py         # 多通道 STFT
│   ├── logmel.py       # mel 滤波器组 + log-mel
│   ├── gcc.py          # GCC-PHAT 时延估计
│   ├── intensity.py    # FOA 有源声强向量
│   └── pipeline.py     # 特征拼接
├── localization/
│   ├── steering.py     # 导向向量与时延
│   ├── grid.py         # 球面搜索网格
│   ├── srp_phat.py     # SRP-PHAT
│   └── music.py        # MUSIC
├── detection/
│   ├── pooling.py      # MIL 池化
│   └── mil.py          # 弱标注聚合与后处理
├── labels/
│   ├── events.py       # 事件类别词表
│   └── accdoa.py       # ACCDOA / multi-ACCDOA 编解码
├── metrics/            # ER / F、LE / LR、综合 SELD 分数
├── data/               # 场景描述、自由场渲染、噪声混合
├── models/             # 可选 PyTorch：CRNN + ACCDOA 头
└── cli.py              # 命令行入口
```

## 坐标与约定

- 坐标系：x 向前、y 向左、z 向上；方位角水平面内逆时针为正，仰角向上为正。
- 信号布局：统一采用**通道优先** `(n_channels, n_samples)`。
- STFT 布局：`(n_channels, n_freq, n_frames)`，`n_freq = n_fft // 2 + 1`。
- 标签帧率与 STFT 帧率解耦：`label_rate`（默认 10 Hz）用于事件区间与评测。

## 数据流

1. `io.audio.load_audio` 读入多通道信号（可重采样到统一采样率）。
2. `features.pipeline.FeaturePipeline` 生成 log-mel + GCC-PHAT 特征。
3. 定位：`localization.srp_phat` / `music` 直接从信号估计 DOA；
   检测：`detection` 把（模型或规则给出的）帧级概率聚合为事件。
4. `metrics` 用段级 ER/F 与 LE/LR 汇总为 SELD 分数。

神经路线用 `models.SeldModel` 输出逐帧 ACCDOA，再由 `labels.decode_accdoa`
解码为活跃度与方向，接入同一套指标。
