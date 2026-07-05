"""全局特征与信号处理配置。

``SonolocConfig`` 汇总了 STFT、mel、标签帧率以及麦克风阵列等参数，
供特征提取、定位和数据仿真等模块共享，避免散落的魔法数字。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

import yaml

# 常见的多通道 SELD 默认值，参考 DCASE 任务三的设置。
DEFAULT_SAMPLE_RATE = 24000
DEFAULT_N_FFT = 1024
DEFAULT_HOP = 480  # 20 ms @ 24 kHz -> 50 帧/秒
SPEED_OF_SOUND = 343.0  # m/s, 20°C 空气中


@dataclass
class SonolocConfig:
    """特征提取与信号处理的共享配置。"""

    sample_rate: int = DEFAULT_SAMPLE_RATE
    n_fft: int = DEFAULT_N_FFT
    hop_length: int = DEFAULT_HOP
    win_length: int | None = None
    window: str = "hann"
    n_mels: int = 64
    fmin: float = 50.0
    fmax: float = 12000.0
    label_rate: float = 10.0  # Hz, 每 100 ms 一个标签帧
    sound_speed: float = SPEED_OF_SOUND
    array: str = "tetra"
    extra: dict[str, float] = field(default_factory=dict)

    @property
    def effective_win_length(self) -> int:
        """未显式指定时，窗长等于 ``n_fft``。"""
        return self.win_length if self.win_length is not None else self.n_fft

    @property
    def frames_per_second(self) -> float:
        """STFT 输出的帧率（帧/秒）。"""
        return self.sample_rate / self.hop_length

    @property
    def label_hop(self) -> int:
        """相邻标签帧之间的样本数。"""
        return int(round(self.sample_rate / self.label_rate))

    def to_dict(self) -> dict[str, Any]:
        """转换为普通字典，便于序列化。"""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SonolocConfig:
        """从字典构造配置，忽略未知字段。"""
        known = {f.name for f in fields(cls)}
        filtered = {k: v for k, v in data.items() if k in known}
        return cls(**filtered)

    def save(self, path: str | Path) -> None:
        """把配置写为 YAML 文件。"""
        with open(path, "w", encoding="utf-8") as handle:
            yaml.safe_dump(self.to_dict(), handle, allow_unicode=True, sort_keys=True)

    @classmethod
    def load(cls, path: str | Path) -> SonolocConfig:
        """从 YAML 文件读取配置。"""
        with open(path, encoding="utf-8") as handle:
            data = yaml.safe_load(handle) or {}
        return cls.from_dict(data)
