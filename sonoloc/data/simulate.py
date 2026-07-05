"""自由场（free-field）多通道渲染。

采用远场平面波近似：对每个麦克风施加与其位置相关的分数延迟，把单声道
源信号渲染成多通道阵列信号。适合快速生成可控的 SELD 测试与演示数据，
不涉及房间混响（后续可接入镜像法 / RIR）。
"""

from __future__ import annotations

import numpy as np

from sonoloc.config import SonolocConfig
from sonoloc.data.scene import Scene, SoundEvent
from sonoloc.io.arrays import MicArray, get_array, sph2cart


def spatialize(
    mono: np.ndarray,
    array: MicArray,
    azimuth: float,
    elevation: float,
    sample_rate: int,
    sound_speed: float = 343.0,
) -> np.ndarray:
    """把单声道信号按给定方向渲染为多通道信号。"""
    mono = np.asarray(mono, dtype=np.float64)
    n = mono.shape[-1]
    direction = sph2cart(np.array(azimuth), np.array(elevation), 1.0)
    delays = -(array.positions @ direction) / sound_speed
    freqs = np.fft.rfftfreq(n, 1.0 / sample_rate)
    spectrum = np.fft.rfft(mono)
    channels = [np.fft.irfft(spectrum * np.exp(-2j * np.pi * freqs * d), n=n) for d in delays]
    return np.stack(channels, axis=0)


def _default_source(n_samples: int, sample_rate: int, rng: np.random.Generator) -> np.ndarray:
    """生成一段带限噪声作为默认事件源，并加淡入淡出包络。"""
    noise = rng.standard_normal(n_samples)
    freqs = np.fft.rfftfreq(n_samples, 1.0 / sample_rate)
    spectrum = np.fft.rfft(noise) * ((freqs > 300.0) & (freqs < 6000.0))
    source = np.fft.irfft(spectrum, n=n_samples)
    fade = min(256, n_samples // 8)
    if fade > 0:
        ramp = np.linspace(0.0, 1.0, fade)
        source[:fade] *= ramp
        source[-fade:] *= ramp[::-1]
    return source


def simulate_scene(
    scene: Scene,
    array: MicArray | None = None,
    config: SonolocConfig | None = None,
    seed: int = 0,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """渲染场景为 ``(signal, labels)``。

    ``signal`` 形状 ``(n_mics, n_samples)``；``labels`` 是逐帧的
    activity / azimuth / elevation 字典。
    """
    config = config or SonolocConfig()
    array = array or get_array(config.array)
    rng = np.random.default_rng(seed)

    sr = scene.sample_rate
    n_samples = int(round(scene.duration * sr))
    signal = np.zeros((array.n_mics, n_samples), dtype=np.float64)

    for event in scene.events:
        start = max(int(round(event.onset * sr)), 0)
        end = min(int(round(event.offset * sr)), n_samples)
        if end <= start:
            continue
        length = end - start
        source = event.signal if event.signal is not None else _default_source(length, sr, rng)
        source = np.asarray(source, dtype=np.float64)[:length]
        buffer = np.zeros(n_samples, dtype=np.float64)
        buffer[start:end] = source
        # TODO: 引入房间冲激响应以支持混响场景
        signal += spatialize(buffer, array, event.azimuth, event.elevation, sr, config.sound_speed)

    labels = scene.label_frames(config.label_rate)
    return signal, labels


def random_scene(
    n_classes: int,
    duration: float = 2.0,
    sample_rate: int = 24000,
    n_events: int = 1,
    seed: int = 0,
) -> Scene:
    """随机生成一个非重叠事件的简单场景，便于演示与测试。"""
    rng = np.random.default_rng(seed)
    events: list[SoundEvent] = []
    slot = duration / n_events
    for k in range(n_events):
        onset = k * slot + rng.uniform(0.0, slot * 0.3)
        offset = min(onset + rng.uniform(slot * 0.4, slot * 0.8), duration)
        events.append(
            SoundEvent(
                class_index=int(rng.integers(0, n_classes)),
                azimuth=float(rng.uniform(-np.pi, np.pi)),
                elevation=float(rng.uniform(-np.pi / 6, np.pi / 6)),
                onset=onset,
                offset=offset,
            )
        )
    return Scene(duration=duration, sample_rate=sample_rate, n_classes=n_classes, events=events)
