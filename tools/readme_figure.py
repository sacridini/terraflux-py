# SPDX-License-Identifier: GPL-3.0-or-later
"""Draw the README's figure: the bundled forest map and its MSPA, in GuidosToolbox's colours.

    python tools/readme_figure.py            # writes assets/readme/mspa-example.png

Needs matplotlib and rasterio. The README links the image from GitHub (raw.githubusercontent.com), so that it
also shows on PyPI.
"""

import os
import tempfile

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from matplotlib.patches import Patch

import terraflux as tf

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "assets", "readme", "mspa-example.png")

forest = tf.load_raster(tf.example_data("input"))
morph = tf.mspa(forest, edge_width=2)
with tempfile.TemporaryDirectory() as tmp:
    path = os.path.join(tmp, "mspa.tif")
    tf.write_raster(morph, path)  # the GeoTIFF carries GTB's colour table
    with rasterio.open(path) as ds:
        mspa = ds.read(1)
        table = ds.colormap(1)

lut = np.zeros((256, 3))
for value, rgba in table.items():
    lut[value] = np.array(rgba[:3]) / 255.0

fwd = np.asarray(forest)
inp = np.empty(fwd.shape + (3,))
inp[:] = (0.86, 0.86, 0.86)  # non-forest, as MSPA's background
inp[fwd == 2] = (0.0, 0.47, 0.0)  # forest
inp[fwd == 0] = (1.0, 1.0, 1.0)  # missing

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11})
fig, axes = plt.subplots(1, 2, figsize=(12, 6.4), dpi=150, gridspec_kw={"wspace": 0.04})
for ax, img, title in [
    (axes[0], inp, "Forest and non-forest (bundled sample map)"),
    (axes[1], lut[mspa], "Its MSPA classes, edge width 2"),
]:
    ax.imshow(img, interpolation="nearest")
    ax.set_title(title, fontsize=12.5, color="#222222", pad=8)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color("#bbbbbb")
        spine.set_linewidth(0.8)

classes = [("Core", 17), ("Islet", 9), ("Perforation", 105), ("Edge", 3), ("Loop", 65),
           ("Bridge", 33), ("Branch", 1), ("Background", 0), ("Opening", 220)]
handles = [Patch(facecolor=lut[v], edgecolor="#888888", linewidth=0.5, label=name) for name, v in classes]
fig.legend(handles=handles, loc="lower center", ncol=len(handles), frameon=False, fontsize=10.5,
           handlelength=1.3, handleheight=1.0, columnspacing=1.4, bbox_to_anchor=(0.5, 0.015))
fig.subplots_adjust(left=0.01, right=0.99, top=0.93, bottom=0.1)
os.makedirs(os.path.dirname(OUT), exist_ok=True)
fig.savefig(OUT, facecolor="white", dpi=150)
print(OUT)
