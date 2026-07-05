"""全局特征与信号处理配置。

``SonolocConfig`` 汇总了 STFT、mel、标签帧率以及麦克风阵列等参数，
供特征提取、定位和数据仿真等模块共享，避免散落的魔法数字。
"""

from __future__ import annotations

from dataclasses import dataclass, field

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
