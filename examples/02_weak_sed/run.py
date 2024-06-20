"""示例：弱标注下的事件聚合与后处理。

给定一段帧级类别概率（这里用合成的高斯包络模拟模型输出），演示如何：
1. 用 MIL 池化得到片段级（弱）标签；
2. 中值滤波 + 阈值后，从帧级概率恢复事件区间。

运行：
    python examples/02_weak_sed/run.py
"""

import numpy as np

from sonoloc.config import SonolocConfig
from sonoloc.detection.mil import clip_labels, frames_to_events, median_filter
from sonoloc.detection.pooling import pool


def _synthetic_frame_probs(n_frames: int, n_classes: int) -> np.ndarray:
    """构造帧级概率：第 1 类在中段活跃，其余类接近静默。"""
    t = np.arange(n_frames)
    probs = np.full((n_frames, n_classes), 0.02)
    center, width = n_frames // 2, n_frames // 8
    probs[:, 1] = 0.95 * np.exp(-0.5 * ((t - center) / width) ** 2)
    probs[20, 1] = 0.9  # 一个孤立的误检尖峰，用来展示中值滤波
    return probs


def main() -> None:
    config = SonolocConfig(sample_rate=24000)
    n_frames = int(round(2.0 * config.label_rate * 5))  # 2 秒、约 100 帧
    probs = _synthetic_frame_probs(n_frames, n_classes=13)

    clip = pool(probs, method="linear_softmax")
    weak = clip_labels(probs, method="linear_softmax", threshold=0.3)
    print("片段级概率 top-1 类别:", int(np.argmax(clip)))
    print("弱标签命中的类别:", np.where(weak)[0].tolist())

    smoothed = median_filter(probs, size=5)
    active = smoothed >= 0.5
    events = frames_to_events(active, hop_seconds=1.0 / (config.label_rate * 5))
    print("检出事件 (类别, 起, 止):")
    for cls, onset, offset in events:
        print(f"  class={cls} {onset:.2f}s -> {offset:.2f}s")


if __name__ == "__main__":
    main()
