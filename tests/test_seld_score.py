"""综合 SELD 分数测试。"""

from sonoloc.metrics.seld_score import aggregate_seld, seld_score


def test_perfect_scores_zero() -> None:
    score = seld_score(error_rate=0.0, f_score=1.0, localization_error=0.0, localization_recall=1.0)
    assert score == 0.0


def test_worst_scores_one() -> None:
    score = seld_score(
        error_rate=1.0, f_score=0.0, localization_error=180.0, localization_recall=0.0
    )
    assert score == 1.0


def test_aggregate_merges_dicts() -> None:
    detection = {"error_rate": 0.2, "f_score": 0.8, "precision": 0.9, "recall": 0.7}
    localization = {"localization_error": 18.0, "localization_recall": 0.9}
    merged = aggregate_seld(detection, localization)
    assert "seld_score" in merged
    assert merged["f_score"] == 0.8
    assert 0.0 <= merged["seld_score"] <= 1.0
