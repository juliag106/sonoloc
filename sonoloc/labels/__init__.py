"""标签工具：事件类别词表与 ACCDOA 编解码。"""

from sonoloc.labels.accdoa import decode_accdoa, encode_accdoa
from sonoloc.labels.events import DEFAULT_CLASSES, EventVocabulary

__all__ = [
    "DEFAULT_CLASSES",
    "EventVocabulary",
    "decode_accdoa",
    "encode_accdoa",
]
