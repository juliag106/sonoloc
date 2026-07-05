# 示例 02：弱标注事件检测

只有片段级标签时，如何从帧级概率既得到片段级（弱）标签，又恢复事件区间。
本例用合成的高斯包络模拟模型输出，展示 MIL 池化、中值滤波与事件分段。

```bash
python examples/02_weak_sed/run.py
```

要点：

- `pool(..., method="linear_softmax")` 把帧级概率自加权聚合为片段级预测；
- `median_filter` 抹掉孤立的误检尖峰；
- `frames_to_events` 把阈值后的活跃帧合并成 `(类别, 起, 止)` 区间。
