"""``sonoloc`` 命令行入口。"""

from __future__ import annotations

import argparse

import numpy as np

from sonoloc.config import SonolocConfig
from sonoloc.features.pipeline import FeaturePipeline
from sonoloc.io.arrays import get_array
from sonoloc.io.audio import load_audio
from sonoloc.localization.music import music
from sonoloc.localization.srp_phat import srp_phat
from sonoloc.version import __version__


def _cmd_doa(args: argparse.Namespace) -> int:
    config = SonolocConfig(sample_rate=args.sample_rate, array=args.array)
    array = get_array(config.array)
    signal, _sr = load_audio(args.audio, sample_rate=config.sample_rate)
    if args.method == "music":
        azimuth, elevation = music(signal, array, config)
    else:
        azimuth, elevation = srp_phat(signal, array, config)
    print(f"method={args.method} azimuth={azimuth:.1f} elevation={elevation:.1f}")
    return 0


def _cmd_features(args: argparse.Namespace) -> int:
    config = SonolocConfig(sample_rate=args.sample_rate, array=args.array)
    array = get_array(config.array)
    signal, _sr = load_audio(args.audio, sample_rate=config.sample_rate)
    features = FeaturePipeline(config, array)(signal)
    print(f"features shape={features.shape}")
    if args.output:
        np.save(args.output, features)
        print(f"saved -> {args.output}")
    return 0


def _cmd_info(args: argparse.Namespace) -> int:
    config = SonolocConfig(array=args.array)
    array = get_array(config.array)
    print(f"sonoloc {__version__}")
    print(f"sample_rate={config.sample_rate} n_fft={config.n_fft} hop={config.hop_length}")
    print(f"frames_per_second={config.frames_per_second:.2f} label_rate={config.label_rate}")
    print(f"array={array.name} n_mics={array.n_mics}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="sonoloc", description="多通道 SELD 工具箱")
    parser.add_argument("--version", action="version", version=f"sonoloc {__version__}")
    parser.set_defaults(func=None)
    sub = parser.add_subparsers(dest="command")

    doa = sub.add_parser("doa", help="估计单声源方位（DOA）")
    doa.add_argument("audio", help="多通道音频文件路径")
    doa.add_argument("--method", choices=["srp-phat", "music"], default="srp-phat")
    doa.add_argument("--array", default="tetra", help="麦克风阵列预设")
    doa.add_argument("--sample-rate", type=int, default=24000)
    doa.set_defaults(func=_cmd_doa)

    feats = sub.add_parser("features", help="提取 log-mel + GCC-PHAT 特征")
    feats.add_argument("audio", help="多通道音频文件路径")
    feats.add_argument("--array", default="tetra")
    feats.add_argument("--sample-rate", type=int, default=24000)
    feats.add_argument("--output", default=None, help="可选：保存特征的 .npy 路径")
    feats.set_defaults(func=_cmd_features)

    info = sub.add_parser("info", help="打印默认配置与阵列信息")
    info.add_argument("--array", default="tetra")
    info.set_defaults(func=_cmd_info)

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
