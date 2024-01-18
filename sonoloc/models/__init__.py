"""可选的 PyTorch 模型。

本子包依赖 ``torch``（``pip install sonoloc[torch]``）。未安装时导入会抛出
带明确提示的 ``ImportError``，核心的特征 / 定位 / 指标功能不受影响。
"""

from __future__ import annotations

try:
    import torch  # noqa: F401

    HAS_TORCH = True
except ImportError:  # pragma: no cover - 取决于是否安装 torch
    HAS_TORCH = False


def require_torch() -> None:
    """在需要 torch 的入口处调用，给出清晰的安装提示。"""
    if not HAS_TORCH:  # pragma: no cover
        raise ImportError("该功能需要 PyTorch，请先安装：pip install 'sonoloc[torch]'")
