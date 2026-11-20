# SPDX-License-Identifier: GPL-3.0-or-later
"""The distance between utilisation distributions: the earth mover's distance (move2utils' ``emd``).

How much probability has to move, and how far, to turn one UD into another (Wasserstein-1; Rubner et al. 2000):
Sinkhorn's regularised version as R's move2utils (Kranstauber, Safi & Scharf 2026) computes it, or the exact one of
emdist's C code (Rubner's), which move2utils also offers.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from .. import _core
from .._threads import resolve

__all__ = ["ud_distance"]


def _layers(ud) -> list[tuple[str, np.ndarray, np.ndarray, np.ndarray, object]]:
    """(name, values, x, y, crs) of each UD layer: a (band, y, x) DataArray, a list or a dict of (y, x) ones."""
    import xarray as xr

    items: list[tuple[str, object]] = []
    if isinstance(ud, xr.DataArray):
        if ud.ndim == 2:
            items = [("ud_1", ud)]
        elif ud.ndim == 3:
            band = ud.dims[0]
            names = None
            if band in ud.coords and ud[band].dtype.kind in "OUS":
                names = [str(b) for b in ud[band].values]
            elif isinstance(ud.attrs.get("long_name"), (list, tuple)) and len(ud.attrs["long_name"]) == ud.sizes[band]:
                names = [str(b) for b in ud.attrs["long_name"]]
            items = [(names[i] if names else f"ud_{i + 1}", ud.isel({band: i})) for i in range(ud.sizes[band])]
        else:
            raise ValueError("a UD: a (y, x) raster, or (band, y, x) with a band per track")
    elif isinstance(ud, dict):
        items = [(str(k), v) for k, v in ud.items()]
    elif isinstance(ud, (list, tuple)):
        items = [(f"ud_{i + 1}", v) for i, v in enumerate(ud)]
    else:
        raise TypeError("ud: a DataArray with a band per UD, or a list or a dict of (y, x) DataArrays")
    out = []
    for name, da in items:
        if not isinstance(da, xr.DataArray):
            raise TypeError(f"{name}: a DataArray")
        if da.ndim == 3 and da.shape[0] == 1:
            da = da.isel({da.dims[0]: 0})
        if da.ndim != 2:
            raise ValueError(f"{name}: a (y, x) raster")
        v = np.array(da.values, dtype=np.float64).ravel()
        nodata = getattr(getattr(da, "rio", None), "nodata", None)
        if nodata is not None and not np.isnan(nodata):
            v[v == nodata] = np.nan
        crs = getattr(getattr(da, "rio", None), "crs", None)
        out.append((name, v, np.asarray(da["x"].values, dtype=np.float64), np.asarray(da["y"].values, dtype=np.float64),
                    crs))
    return out


def ud_distance(ud, ud_b=None, method: str = "sinkhorn", mask_quantile: float = 0.999, reg: float = 0.05,
                max_iter: int = 500, tol: float = 1e-9, gc: bool = False, threshold: float | None = None, *,
                exact_max_iter: int = 500, n_threads: int | None = None):
    """The earth mover's distance between utilisation distributions (move2utils' ``emd``).

    The cost of moving one UD's probability onto the other's: the sum, over the pairs of cells, of the probability moved
    times the distance between the cells' centres, least over all ways of moving it (Wasserstein-1). Both UDs
    must be on the same grid. Each UD is first cut to the cells inside its ``mask_quantile`` volume contour (the
    cells holding the most probability, together that much of it), and its weights renormalised; this keeps the cost
    matrix small and, on UDs with support, loses nothing.

    ``method="sinkhorn"`` (the default) is Cuturi's (2013) entropic regularisation: the distances scaled to [0, 1] by
    their largest, ``reg`` the regularisation, iterated until the scaling vector changes less than ``tol`` or ``max_iter``
    times. Close to the exact value (a fraction of a percent on typical UDs) and far faster. ``method="exact"`` is
    Rubner's transportation solver as the R package emdist has it (Euclidean cost, the weights and coordinates in
    single precision as there, at most 500 improving iterations), practical on coarse grids only.

    Parameters
    ----------
    ud
        A DataArray ``(band, y, x)`` with a band for each UD (the dBBMM UD of several tracks), a list or a dict of
        ``(y, x)`` DataArrays. With ``ud_b``, one UD (a ``(y, x)`` raster or a single band).
    ud_b
        The other UD, to give the distance between two UDs.
    method
        ``"sinkhorn"`` or ``"exact"``.
    mask_quantile
        In (0, 1]: the cells outside this volume contour are dropped; 1 keeps (almost) all.
    reg
        Sinkhorn's regularisation, applied to the distance scaled to [0, 1]: smaller is closer to the exact value
        and needs more iterations (below 0.01 it may underflow).
    max_iter, tol
        Sinkhorn's iterations and the change of the scaling vector (largest absolute difference) to stop at.
    gc
        Great-circle (haversine, on a sphere of 6 378 137 m) instead of Euclidean distances between the cells'
        centres: for UDs on longitude / latitude grids, the distances in metres. A longitude / latitude grid without
        ``gc`` is warned about.
    threshold
        Clip the cost at this distance (metres with ``gc``, else the grid's units): the EMD-hat of Pele and Werman
        (2009). Not with ``method="exact"``.
    exact_max_iter
        The iterations of the exact solver (500, as emd() leaves emdist's default).
    n_threads
        Threads (default: most of the cores) of Sinkhorn's matrix products; the same result with any number.

    Returns
    -------
    float or pandas.DataFrame
        The distance (in the grid's units, metres with ``gc``) between ``ud`` and ``ud_b``, or, for a
        stack of three or more UDs, a table of the pairs: ``a``, ``b`` (the names of the bands, ``ud_1``... without
        them) and ``distance``, each UD before the ones after it.

    References
    ----------
    Cuturi, M. (2013). Sinkhorn distances: lightspeed computation of optimal transport. Advances in Neural Information
    Processing Systems 26.

    Rubner, Y., Tomasi, C., Guibas, L. J. (2000). The earth mover's distance as a metric for image retrieval.
    International Journal of Computer Vision 40, 99-121.

    Kranstauber, B., Safi, K., Scharf, A. K. (2026). move2utils. bioRxiv doi:10.64898/2026.07.07.736908

    Examples
    --------
    >>> import terraflux as tf
    >>> import numpy as np, xarray as xr
    >>> x, y = np.arange(20) * 10.0 + 5, np.arange(20)[::-1] * 10.0 + 5
    >>> gx, gy = np.meshgrid(x, y)
    >>> def blob(cx, cy):
    ...     d = np.exp(-((gx - cx) ** 2 + (gy - cy) ** 2) / (2 * 20.0 ** 2))
    ...     return xr.DataArray(d / d.sum(), coords={"y": y, "x": x}, dims=("y", "x"))
    >>> round(tf.move.ud_distance(blob(60, 100), blob(140, 100), method="exact"))   # the blobs are 80 m apart
    80
    """
    if method not in ("sinkhorn", "exact"):
        raise ValueError('method: "sinkhorn" or "exact"')
    if not (isinstance(mask_quantile, (int, float, np.integer, np.floating)) and 0 < mask_quantile <= 1):
        raise ValueError("mask_quantile: a number in (0, 1]")
    if not (isinstance(reg, (int, float, np.integer, np.floating)) and reg > 0):
        raise ValueError("reg: a positive number")
    if not isinstance(gc, (bool, np.bool_)):
        raise TypeError("gc: True or False")
    if threshold is not None and not (np.isfinite(threshold) and threshold > 0):
        raise ValueError("threshold: a positive number or None")
    if int(max_iter) < 1 or int(exact_max_iter) < 1:
        raise ValueError("max_iter: at least 1")
    if method == "exact" and gc:
        raise ValueError('method="exact" uses the Euclidean distance and cannot honour gc=True: use method="sinkhorn"')
    if method == "exact" and threshold is not None:
        raise ValueError('method="exact" cannot honour threshold: use method="sinkhorn"')
    layers = _layers(ud)
    if ud_b is not None:
        other = _layers(ud_b)
        if len(layers) != 1 or len(other) != 1:
            raise ValueError("with ud_b, ud and ud_b are one UD each")
        layers = [layers[0], other[0]]
        pair = True
    else:
        pair = len(layers) == 2
    if len(layers) < 2:
        raise ValueError("need at least two UDs")
    ref = layers[0]
    for name, v, x, y, _ in layers[1:]:
        if v.size != ref[1].size or not (np.array_equal(x, ref[2]) and np.array_equal(y, ref[3])):
            raise ValueError("All UDs must share the same raster geometry.")
    crs = ref[4]
    if crs is not None and getattr(crs, "is_geographic", False) and not gc:
        warnings.warn("Input UDs appear to be on a longitude/latitude grid but `gc=False`. Euclidean distance on degrees "
                      "mixes north-south metres with cosine-shrunk east-west metres and is not meaningful at "
                      "non-equatorial latitudes. Either project the UDs to a planar CRS first, or pass `gc=True` "
                      "to use Haversine distances.", stacklevel=2)
    xs, ys = ref[2], ref[3]
    ncol = xs.size
    threads = resolve(n_threads)
    prepared = []
    for name, v, *_ in layers:
        cells, w = _core.move_ud_prepare(v, float(mask_quantile))
        prepared.append((np.ascontiguousarray(xs[cells % ncol]), np.ascontiguousarray(ys[cells // ncol]), w))
    thr = float("nan") if threshold is None else float(threshold)
    rows = []
    for i in range(len(layers) - 1):
        for j in range(i + 1, len(layers)):
            ax, ay, p = prepared[i]
            bx, by, q = prepared[j]
            if method == "sinkhorn":
                d, _ = _core.move_emd_sinkhorn(ax, ay, p, bx, by, q, bool(gc), thr, float(reg), int(max_iter), float(tol),
                                               threads)
            else:
                d, hit = _core.move_emd_exact(ax, ay, p, bx, by, q, int(exact_max_iter))
                if hit:
                    warnings.warn(f"emd: Maximum number of iterations has been reached ({int(exact_max_iter)})",
                                  stacklevel=2)
            rows.append((layers[i][0], layers[j][0], d))
    if pair:
        return float(rows[0][2])
    return pd.DataFrame(rows, columns=["a", "b", "distance"])
