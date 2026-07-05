# sonoloc

多通道声学事件检测与声源定位（SELD）工具箱。

- 特征：多通道 log-mel、GCC-PHAT、FOA 声强向量
- 定位：SRP-PHAT、MUSIC 等经典 DOA 方法
- 检测：面向弱标注的多示例学习（MIL）池化

特征流水线 `FeaturePipeline` 会把逐通道 log-mel 与麦克风对之间的
GCC-PHAT 拼接成统一张量，作为下游模型或经典定位方法的输入。

> 仍在早期开发阶段。
