# landschaft is now terraflux

The package `landschaft` was renamed to **terraflux**:

```
pip install terraflux-py
```

```python
import terraflux as tf
```

This package only exists so that old code keeps working: it installs `terraflux-py` (pinned to the same
version) and `import landschaft as ls` gives the very same module (`landschaft.mspa is terraflux.mspa`),
with a `DeprecationWarning`. The command `landschaft` still runs. It will be dropped in the next major version.

- Source: https://github.com/sacridini/terraflux-py
- Documentation: https://sacridini.github.io/terraflux/
