"""sonoloc: 多通道声学事件检测与声源定位（SELD）工具箱。"""

from sonoloc.config import SonolocConfig
from sonoloc.data.scene import Scene, SoundEvent
from sonoloc.data.simulate import random_scene, simulate_scene
from sonoloc.detection.pooling import pool
from sonoloc.features.pipeline import FeaturePipeline
from sonoloc.io.audio import load_audio, save_audio
from sonoloc.io.geometry import MicArray, get_array, tetrahedral_array
from sonoloc.labels.accdoa import decode_accdoa, encode_accdoa
from sonoloc.localization.music import music
from sonoloc.localization.srp_phat import srp_phat
from sonoloc.metrics.seld_score import seld_score
from sonoloc.version import __version__

__all__ = [
    "FeaturePipeline",
    "MicArray",
    "Scene",
    "SonolocConfig",
    "SoundEvent",
    "__version__",
    "decode_accdoa",
    "encode_accdoa",
    "get_array",
    "load_audio",
    "music",
    "pool",
    "random_scene",
    "save_audio",
    "seld_score",
    "simulate_scene",
    "srp_phat",
    "tetrahedral_array",
]
