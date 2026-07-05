"""``sonoloc`` 命令行入口。"""

from __future__ import annotations

import argparse

import numpy as np

from sonoloc.config import SonolocConfig
from sonoloc.data.scene import Scene, SoundEvent
from sonoloc.data.simulate import simulate_scene
from sonoloc.features.pipeline import FeaturePipeline
from sonoloc.io.arrays import get_array
from sonoloc.io.audio import load_audio, save_audio
from sonoloc.localization.music import music
from sonoloc.localization.srp_phat import srp_phat
from sonoloc.version import __version__


def _config_from_args(args: argparse.Namespace) -> SonolocConfig:
    """从 ``--config`` YAML 或命令行参数构造配置。"""
    config = SonolocConfig.load(args.config) if getattr(args, "config", None) else SonolocConfig()
    if getattr(args, "array", None):
        config.array = args.array
    if getattr(args, "sample_rate", None):
        config.sample_rate = args.sample_rate
    return config


def _add_config_args(parser: argparse.ArgumentParser, with_sample_rate: bool = True) -> None:
    parser.add_argument("--array", default="tetra", help="麦克风阵列预设")
    parser.add_argument("--config", default=None, help="可选：从 YAML 读取配置")
    if with_sample_rate:
        parser.add_argument("--sample-rate", type=int, default=24000)


def _cmd_doa(args: argparse.Namespace) -> int:
    config = _config_from_args(args)
    array = get_array(config.array)
    signal, _sr = load_audio(args.audio, sample_rate=config.sample_rate)
    if args.method == "music":
        azimuth, elevation = music(signal, array, config)
    else:
        azimuth, elevation = srp_phat(signal, array, config)
    print(f"method={args.method} azimuth={azimuth:.1f} elevation={elevation:.1f}")
    return 0


def _cmd_features(args: argparse.Namespace) -> int:
    config = _config_from_args(args)
    array = get_array(config.array)
    signal, _sr = load_audio(args.audio, sample_rate=config.sample_rate)
    features = FeaturePipeline(config, array)(signal)
    print(f"features shape={features.shape}")
    if args.output:
        np.save(args.output, features)
        print(f"saved -> {args.output}")
    return 0


def _cmd_info(args: argparse.Namespace) -> int:
    config = _config_from_args(args)
    array = get_array(config.array)
    print(f"sonoloc {__version__}")
    print(f"sample_rate={config.sample_rate} n_fft={config.n_fft} hop={config.hop_length}")
    print(f"frames_per_second={config.frames_per_second:.2f} label_rate={config.label_rate}")
    print(f"array={array.name} n_mics={array.n_mics}")
    return 0


def _cmd_simulate(args: argparse.Namespace) -> int:
    config = _config_from_args(args)
    array = get_array(config.array)
    event = SoundEvent(
        class_index=0,
        azimuth=np.deg2rad(args.azimuth),
        elevation=np.deg2rad(args.elevation),
        onset=0.0,
        offset=args.duration,
    )
    scene = Scene(
        duration=args.duration,
        sample_rate=config.sample_rate,
        n_classes=13,
        events=[event],
    )
    signal, _labels = simulate_scene(scene, array, config, snr_db=args.snr, seed=args.seed)
    save_audio(args.output, signal, config.sample_rate)
    print(f"saved {signal.shape[0]}-ch scene -> {args.output}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="sonoloc", description="多通道 SELD 工具箱")
    parser.add_argument("--version", action="version", version=f"sonoloc {__version__}")
    parser.set_defaults(func=None)
    sub = parser.add_subparsers(dest="command")

    doa = sub.add_parser("doa", help="估计单声源方位（DOA）")
    doa.add_argument("audio", help="多通道音频文件路径")
    doa.add_argument("--method", choices=["srp-phat", "music"], default="srp-phat")
    _add_config_args(doa)
    doa.set_defaults(func=_cmd_doa)

    feats = sub.add_parser("features", help="提取 log-mel + GCC-PHAT 特征")
    feats.add_argument("audio", help="多通道音频文件路径")
    feats.add_argument("--output", default=None, help="可选：保存特征的 .npy 路径")
    _add_config_args(feats)
    feats.set_defaults(func=_cmd_features)

    info = sub.add_parser("info", help="打印默认配置与阵列信息")
    _add_config_args(info, with_sample_rate=False)
    info.set_defaults(func=_cmd_info)

    sim = sub.add_parser("simulate", help="生成一个合成的多通道场景")
    sim.add_argument("output", help="输出 wav 路径")
    sim.add_argument("--azimuth", type=float, default=30.0, help="方位角（度）")
    sim.add_argument("--elevation", type=float, default=0.0, help="仰角（度）")
    sim.add_argument("--duration", type=float, default=2.0, help="时长（秒）")
    sim.add_argument("--snr", type=float, default=None, help="扩散噪声 SNR（dB）")
    sim.add_argument("--seed", type=int, default=0)
    _add_config_args(sim)
    sim.set_defaults(func=_cmd_simulate)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if getattr(args, "func", None) is None:
        parser.print_help()
        return 0
    return int(args.func(args))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
