"""特征流水线测试。"""

import numpy as np

from sonoloc.config import SonolocConfig
from sonoloc.features.pipeline import FeaturePipeline
from sonoloc.io.arrays import tetrahedral_array


def test_pipeline_stacked_shape() -> None:
    config = SonolocConfig(sample_rate=24000, n_mels=64)
    array = tetrahedral_array()
    signal = np.random.default_rng(0).standard_normal((4, 24000))
    features = FeaturePipeline(config, array)(signal)
    # 通道数 = 4 路 log-mel + C(4,2)=6 个麦克风对的 GCC
    assert features.shape[0] == 4 + 6
    assert features.shape[1] == 64


def test_pipeline_is_finite() -> None:
    config = SonolocConfig(sample_rate=16000, n_mels=40)
    array = tetrahedral_array()
    signal = np.random.default_rng(1).standard_normal((4, 16000))
    features = FeaturePipeline(config, array)(signal)
    assert np.all(np.isfinite(features))
