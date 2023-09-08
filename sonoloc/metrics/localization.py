"""定位指标：DOA 角误差（LE）与定位召回率（LR）。

角度均以弧度输入，误差以角度（度）返回。定位误差只在参考与系统
都判为活跃的位置上统计；定位召回率衡量参考事件被系统检出的比例。
"""

from __future__ import annotations

import numpy as np


def angular_distance(
    az1: np.ndarray, el1: np.ndarray, az2: np.ndarray, el2: np.ndarray
) -> np.ndarray:
    """两个方向之间的球面夹角（度）。"""
    cos_angle = np.sin(el1) * np.sin(el2) + np.cos(el1) * np.cos(el2) * np.cos(az1 - az2)
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    return np.rad2deg(np.arccos(cos_angle))


def localization_scores(
    ref_active: np.ndarray,
    ref_az: np.ndarray,
    ref_el: np.ndarray,
    pred_active: np.ndarray,
    pred_az: np.ndarray,
    pred_el: np.ndarray,
) -> dict[str, float]:
    """返回定位误差 ``localization_error`` 与定位召回 ``localization_recall``。"""
    ref_active = np.asarray(ref_active, dtype=bool)
    pred_active = np.asarray(pred_active, dtype=bool)
    both = ref_active & pred_active

    if both.any():
        errors = angular_distance(ref_az[both], ref_el[both], pred_az[both], pred_el[both])
        localization_error = float(errors.mean())
    else:
        localization_error = 180.0  # 没有可比对的检出，按最大误差计

    n_ref = int(ref_active.sum())
    localization_recall = float(both.sum()) / max(n_ref, 1)
    return {
        "localization_error": localization_error,
        "localization_recall": localization_recall,
    }
