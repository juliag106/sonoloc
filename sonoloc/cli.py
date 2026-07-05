"""``sonoloc`` 命令行入口。"""

from __future__ import annotations

import argparse

from sonoloc.config import SonolocConfig
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
