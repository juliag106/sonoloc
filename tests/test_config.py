"""配置序列化测试。"""

from sonoloc.config import SonolocConfig


def test_config_defaults() -> None:
    config = SonolocConfig()
    assert config.sample_rate == 24000
    assert config.frames_per_second == 24000 / 480
    assert config.effective_win_length == config.n_fft


def test_config_yaml_round_trip(tmp_path) -> None:
    config = SonolocConfig(sample_rate=16000, n_mels=40, array="tetra")
    path = tmp_path / "config.yaml"
    config.save(path)
    loaded = SonolocConfig.load(path)
    assert loaded == config


def test_from_dict_ignores_unknown_keys() -> None:
    config = SonolocConfig.from_dict({"sample_rate": 48000, "not_a_field": 1})
    assert config.sample_rate == 48000
