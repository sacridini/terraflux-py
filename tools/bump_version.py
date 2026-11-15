# SPDX-License-Identifier: GPL-3.0-or-later
"""python tools/bump_version.py 1.59.0: sets the version in pyproject.toml and in the compatibility
package compat/landschaft/pyproject.toml (its version and its terraflux-py== pin)."""

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def sub(path: Path, pattern: str, repl: str) -> None:
    text = path.read_text(encoding="utf-8")
    new, n = re.subn(pattern, repl, text, count=1, flags=re.M)
    if n != 1:
        sys.exit(f"{path}: {pattern!r} not found")
    path.write_text(new, encoding="utf-8", newline="")


def main() -> None:
    if len(sys.argv) != 2 or not re.fullmatch(r"\d+\.\d+\.\d+", sys.argv[1]):
        sys.exit("usage: bump_version.py X.Y.Z")
    v = sys.argv[1]
    sub(ROOT / "pyproject.toml", r'^version = "[^"]*"', f'version = "{v}"')
    compat = ROOT / "compat" / "landschaft" / "pyproject.toml"
    sub(compat, r'^version = "[^"]*"', f'version = "{v}"')
    sub(compat, r'"terraflux-py==[^"]*"', f'"terraflux-py=={v}"')
    print(f"version {v} set in pyproject.toml and compat/landschaft/pyproject.toml")


if __name__ == "__main__":
    main()
