"""事件类别词表。

默认给出一组与 DCASE 任务三相近的 13 类日常声学事件，
用户也可以自定义类别列表。
"""

from __future__ import annotations

# 默认事件类别（顺序即类别索引）。
DEFAULT_CLASSES: tuple[str, ...] = (
    "female_speech",
    "male_speech",
    "clapping",
    "telephone",
    "laughter",
    "domestic_sounds",
    "footsteps",
    "door",
    "music",
    "musical_instrument",
    "water_tap",
    "bell",
    "knock",
)


class EventVocabulary:
    """事件类别名与索引之间的双向映射。"""

    def __init__(self, classes: tuple[str, ...] | None = None) -> None:
        self.classes = tuple(classes) if classes is not None else DEFAULT_CLASSES
        self._to_index = {name: i for i, name in enumerate(self.classes)}

    def __len__(self) -> int:
        return len(self.classes)

    def index(self, name: str) -> int:
        return self._to_index[name]

    def name(self, index: int) -> str:
        return self.classes[index]
