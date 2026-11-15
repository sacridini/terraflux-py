# SPDX-License-Identifier: GPL-3.0-or-later
"""Reading what landschaft (terraflux until 1.58's name) wrote: its pickles, its module names."""

from __future__ import annotations

import io
import pickle
from typing import Any, BinaryIO

__all__ = ["load", "loads", "module_name"]


def module_name(name: str) -> str:
    """A module name with the old package name (``landschaft``, ``landschaft.fad``) as ``terraflux``'s."""
    if name == "landschaft" or name.startswith("landschaft."):
        return "terraflux" + name[len("landschaft"):]
    return name


class _Unpickler(pickle.Unpickler):
    def find_class(self, module: str, name: str) -> Any:
        return super().find_class(module_name(module), name)


def load(file: BinaryIO) -> Any:
    """``pickle.load`` that finds the classes of ``landschaft.*`` in ``terraflux.*``."""
    return _Unpickler(file).load()


def loads(data: bytes) -> Any:
    """``pickle.loads`` that finds the classes of ``landschaft.*`` in ``terraflux.*``."""
    return _Unpickler(io.BytesIO(data)).load()
