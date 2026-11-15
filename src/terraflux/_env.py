# SPDX-License-Identifier: GPL-3.0-or-later
"""Environment variables: ``TERRAFLUX_<NAME>``, else the name landschaft used (``LANDSCHAFT_<NAME>``),
and the move of the folders of that older name."""

from __future__ import annotations

import os
from pathlib import Path

__all__ = ["get", "migrate"]


def get(name: str, default: str | None = None) -> str | None:
    """``TERRAFLUX_<name>`` when set, else ``LANDSCHAFT_<name>`` (until 1.58 the only name), else ``default``."""
    value = os.environ.get("TERRAFLUX_" + name)
    if value is None:
        value = os.environ.get("LANDSCHAFT_" + name)
    return default if value is None else value


def migrate(old: Path, new: Path) -> Path:
    """The folder to use for ``new``, a folder that landschaft kept at ``old``.

    ``new`` when it exists, or when neither does. When only ``old`` exists it is moved to ``new`` (a rename:
    nothing is copied or deleted); if the move fails (another volume, permissions), ``old`` is used as it is."""
    try:
        if not old.exists() or old == new or new.exists():
            return new
        new.parent.mkdir(parents=True, exist_ok=True)
        old.rename(new)
        return new
    except OSError:
        return old if old.exists() and not new.exists() else new
