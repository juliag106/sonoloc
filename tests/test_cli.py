"""CLI 冒烟测试。"""

import numpy as np

from sonoloc.cli import main
from sonoloc.io.audio import save_audio


def _write_scene(path, sr=24000, n=24000, channels=4) -> None:
    rng = np.random.default_rng(0)
    save_audio(str(path), rng.standard_normal((channels, n)) * 0.01, sr)


def test_cli_info(capsys) -> None:
    assert main(["info"]) == 0
    out = capsys.readouterr().out
    assert "sonoloc" in out


def test_cli_no_command_prints_help(capsys) -> None:
    assert main([]) == 0
    assert "usage" in capsys.readouterr().out.lower()


def test_cli_doa_runs(tmp_path, capsys) -> None:
    wav = tmp_path / "scene.wav"
    _write_scene(wav)
    assert main(["doa", str(wav), "--method", "srp-phat"]) == 0
    assert "azimuth" in capsys.readouterr().out


def test_cli_simulate_and_features(tmp_path, capsys) -> None:
    wav = tmp_path / "sim.wav"
    assert main(["simulate", str(wav), "--azimuth", "20", "--duration", "1.0"]) == 0
    assert wav.exists()
    assert main(["features", str(wav)]) == 0
    assert "shape" in capsys.readouterr().out
