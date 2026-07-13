from __future__ import annotations

from typing import TYPE_CHECKING

from .config import (
    DEFAULT_CONFIG_NAME,
    DEFAULT_HF_REPO_ID,
    DEFAULT_MODEL_FILENAME,
    default_config_path,
)

if TYPE_CHECKING:
    from decoder.pretrained import Unicodec as Unicodec

__all__ = [
    "DEFAULT_CONFIG_NAME",
    "DEFAULT_HF_REPO_ID",
    "DEFAULT_MODEL_FILENAME",
    "Unicodec",
    "default_config_path",
]


def __getattr__(name: str):
    if name == "Unicodec":
        from decoder.pretrained import Unicodec

        return Unicodec
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
