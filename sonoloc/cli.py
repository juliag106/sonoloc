"""``sonoloc`` 命令行入口。

子命令（doa / features / simulate / info）在后续提交中补全。
"""

from __future__ import annotations

import argparse

from sonoloc.version import __version__


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="sonoloc", description="多通道 SELD 工具箱")
    parser.add_argument("--version", action="version", version=f"sonoloc {__version__}")
    parser.set_defaults(func=None)
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
