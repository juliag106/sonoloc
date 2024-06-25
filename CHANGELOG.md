# 更新日志

本项目遵循 [Keep a Changelog](https://keepachangelog.com/zh-CN/1.0.0/)，
版本号采用 [语义化版本](https://semver.org/lang/zh-CN/)。

## [Unreleased]

### 计划中

- 接入房间冲激响应（RIR）以支持混响场景。
- 内置 ACCDOA 训练循环与置换不变损失。

## [0.1.0] - 2026-07-05

### 新增

- 麦克风阵列几何与四面体阵列预设，球/笛卡尔坐标转换。
- 多通道 STFT、log-mel、GCC-PHAT 与 FOA 有源声强特征，及 `FeaturePipeline`。
- 经典 DOA 定位：SRP-PHAT 与 MUSIC，含导向向量与球面搜索网格。
- 弱标注 MIL 池化、帧级后处理与事件分段。
- 单/多轨 ACCDOA 编解码与事件类别词表。
- 段级 ER / F、定位 LE / LR 及综合 SELD 分数。
- 自由场平面波渲染与按 SNR 混合的扩散噪声。
- 可选 PyTorch 模型：CRNN 骨干 + ACCDOA 头。
- 命令行 `sonoloc doa / features / simulate / info`。

[Unreleased]: https://github.com/juliag106/sonoloc/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/juliag106/sonoloc/releases/tag/v0.1.0
