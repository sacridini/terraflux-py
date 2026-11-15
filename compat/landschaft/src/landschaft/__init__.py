# SPDX-License-Identifier: GPL-3.0-or-later
"""landschaft is now terraflux. `landschaft` is the very same module object as `terraflux`, and every
`landschaft.<sub>` the same object as `terraflux.<sub>` (so no module is imported twice and pickles
that name `landschaft.<module>.<Class>` load)."""

import importlib
import importlib.abc
import importlib.util
import sys
import warnings

import terraflux

warnings.warn("landschaft is now terraflux: pip install terraflux, import terraflux as tf",
              DeprecationWarning, stacklevel=2)  # importlib's own frames are skipped: the user's import


class _Alias(importlib.abc.Loader):
    """Hands back the module terraflux already has instead of loading a copy."""

    def __init__(self, module):
        self.module = module

    def create_module(self, spec):
        return self.module

    def exec_module(self, module):
        pass  # already executed as terraflux.<sub>


class _Finder(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path=None, target=None):
        if not name.startswith("landschaft."):
            return None
        try:
            module = importlib.import_module("terraflux." + name[len("landschaft."):])
        except ModuleNotFoundError as e:
            if e.name == name.replace("landschaft", "terraflux", 1):
                return None  # no such submodule: the usual ModuleNotFoundError
            raise
        return importlib.util.spec_from_loader(name, _Alias(module))


if not any(isinstance(f, _Finder) for f in sys.meta_path):
    sys.meta_path.insert(0, _Finder())

# `import landschaft` returns sys.modules["landschaft"] once this file has run: terraflux itself
sys.modules[__name__] = terraflux
