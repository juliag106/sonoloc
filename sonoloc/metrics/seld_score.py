"""综合 SELD 分数。

把四个子指标聚合成单一分数，便于模型选择与早停：

``SELD = (ER + (1 - F) + LE / 180 + (1 - LR)) / 4``

数值越小越好，取值范围约为 ``[0, 1]``。
"""

from __future__ import annotations


def seld_score(
    error_rate: float,
    f_score: float,
    localization_error: float,
    localization_recall: float,
) -> float:
    """由四个子指标计算综合 SELD 分数（越小越好）。"""
    normalized_le = min(localization_error, 180.0) / 180.0
    return (error_rate + (1.0 - f_score) + normalized_le + (1.0 - localization_recall)) / 4.0


def aggregate_seld(detection: dict[str, float], localization: dict[str, float]) -> dict[str, float]:
    """合并检测与定位指标字典，并附加综合 SELD 分数。"""
    score = seld_score(
        detection["error_rate"],
        detection["f_score"],
        localization["localization_error"],
        localization["localization_recall"],
    )
    return {**detection, **localization, "seld_score": score}
