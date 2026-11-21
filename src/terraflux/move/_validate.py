# SPDX-License-Identifier: GPL-3.0-or-later
"""Does the use of the animals follow the connectivity surface? (Zeller et al. 2018's validation, in numbers.)

Not in R: neither amt nor move2utils has it, so the definitions here are ours (see :func:`validate`).
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd

from ._corridor import quantile7
from ._tracks import _TRACK_NAMES, ATTRS, OLD_ATTRS

__all__ = ["Validation", "validate"]

_KINDS = ("current", "resistance", "binary")


@dataclass
class Validation:
    """What :func:`validate` gives: ``table`` (a row per track, ``pooled`` last when there are several), ``note`` (a
    short reading of the pooled row), ``threshold`` (the surface value from which a cell is a corridor, NaN for a
    binary surface), ``corridor`` (a 1 / 0 / NaN raster of the corridor cells on the surface's grid) and ``kind``."""
    table: pd.DataFrame
    note: str
    threshold: float
    corridor: object
    kind: str

    def __repr__(self) -> str:
        return f"Validation(kind={self.kind!r}, threshold={self.threshold:.6g}, tracks={len(self.table)})\n{self.note}"


# ------------------------------------------------------------------------------------------ the numbers

def _sum(a) -> float:
    """The sum in order (numpy's own is pairwise): the same bits as the operator's loop."""
    a = np.asarray(a, dtype=np.float64)
    return float(np.cumsum(a)[-1]) if a.size else 0.0


def _ranks(x: np.ndarray) -> np.ndarray:
    """Average ranks (ties share the mean of their ranks), 1-based: scipy's ``rankdata`` and R's ``rank``."""
    order = np.argsort(x, kind="stable")
    s = x[order]
    n = len(x)
    start = np.flatnonzero(np.r_[True, s[1:] != s[:-1]])
    end = np.r_[start[1:], n]
    avg = (start + end + 1) / 2.0
    out = np.empty(n)
    out[order] = np.repeat(avg, end - start)
    return out


def spearman(x, y) -> float:
    """Spearman's rank correlation: Pearson's correlation of the average ranks (as ``scipy.stats.spearmanr`` and R's
    ``cor(method = "spearman")``); NaN for fewer than three pairs or a variable that does not vary."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    n = len(x)
    if n < 3:
        return float("nan")
    rx, ry = _ranks(x), _ranks(y)
    mx, my = _sum(rx) / n, _sum(ry) / n
    dx, dy = rx - mx, ry - my
    sxx, syy = _sum(dx * dx), _sum(dy * dy)
    if sxx == 0.0 or syy == 0.0:
        return float("nan")
    return min(1.0, max(-1.0, _sum(dx * dy) / math.sqrt(sxx * syy)))


# ------------------------------------------------------------------------------------------ the surface

class _Grid:
    """A north-up raster's cells: the corner, the cell size and the values (row 0 is the northernmost)."""

    def __init__(self, da, what: str):
        import xarray as xr

        if not isinstance(da, xr.DataArray):
            raise TypeError(f"{what}: a DataArray (a raster)")
        if da.ndim == 3 and da.shape[0] == 1:
            da = da.isel({da.dims[0]: 0})
        if da.ndim != 2:
            raise ValueError(f"{what}: a (y, x) raster")
        x = np.asarray(da["x"].values, dtype=np.float64)
        y = np.asarray(da["y"].values, dtype=np.float64)
        v = np.array(da.values, dtype=np.float64)
        nodata = getattr(getattr(da, "rio", None), "nodata", None)
        if nodata is not None and not np.isnan(nodata):
            v[v == nodata] = np.nan
        v[~np.isfinite(v)] = np.nan
        if len(x) < 2 or len(y) < 2:
            raise ValueError(f"{what}: at least two rows and two columns")
        self.flipped = bool(y[0] < y[-1])
        if self.flipped:
            y, v = y[::-1], v[::-1]
        if x[0] > x[-1]:
            raise ValueError(f"{what}: x must increase along the columns")
        self.ny, self.nx = v.shape
        self.xres = (x[-1] - x[0]) / (self.nx - 1)
        self.yres = (y[0] - y[-1]) / (self.ny - 1)
        self.xmin = float(x[0] - self.xres / 2)
        self.ymax = float(y[0] + self.yres / 2)
        self.values = v
        self.da = da
        crs = getattr(getattr(da, "rio", None), "crs", None)
        self.crs = None
        if crs is not None:
            import pyproj

            self.crs = pyproj.CRS.from_user_input(crs)

    def same_as(self, other: _Grid) -> bool:
        tol = 1e-9
        return (self.values.shape == other.values.shape and abs(self.xmin - other.xmin) <= tol * self.xres
                and abs(self.ymax - other.ymax) <= tol * self.yres and abs(self.xres - other.xres) <= tol * self.xres
                and abs(self.yres - other.yres) <= tol * self.yres)

    def centres(self) -> tuple[np.ndarray, np.ndarray]:
        x = self.xmin + (np.arange(self.nx) + 0.5) * self.xres
        y = self.ymax - (np.arange(self.ny) + 0.5) * self.yres
        return x, y

    def cells(self, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(row, column, inside) of the cells holding the points (nearest, an edge belongs to the cell after it)."""
        with np.errstate(invalid="ignore"):
            c = np.floor((x - self.xmin) / self.xres)
            r = np.floor((self.ymax - y) / self.yres)
        inside = np.isfinite(c) & np.isfinite(r) & (c >= 0) & (c < self.nx) & (r >= 0) & (r < self.ny)
        c = np.where(inside, c, 0).astype(np.int64)
        r = np.where(inside, r, 0).astype(np.int64)
        return r, c, inside


def _project(x: np.ndarray, y: np.ndarray, src, dst) -> tuple[np.ndarray, np.ndarray]:
    """Coordinates from the CRS ``src`` to ``dst`` (unchanged where either is missing or they agree)."""
    if src is None or dst is None or src == dst:
        return x, y
    from pyproj import Transformer

    tx, ty = Transformer.from_crs(src, dst, always_xy=True).transform(x, y)
    return np.asarray(tx, dtype=np.float64), np.asarray(ty, dtype=np.float64)


# ------------------------------------------------------------------------------------------ the use

def _ud_layers(use) -> list[tuple[str, object]]:
    import xarray as xr

    if use.ndim == 2:
        return [("ud_1", use)]
    if use.ndim != 3:
        raise ValueError("a UD: a (y, x) raster, or (band, y, x) with a band per track")
    band = use.dims[0]
    names = None
    if band in use.coords and use[band].dtype.kind in "OUS":
        names = [str(b) for b in use[band].values]
    elif isinstance(use.attrs.get("long_name"), (list, tuple)) and len(use.attrs["long_name"]) == use.sizes[band]:
        names = [str(b) for b in use.attrs["long_name"]]
    assert isinstance(use, xr.DataArray)
    return [(names[i] if names else f"ud_{i + 1}", use.isel({band: i})) for i in range(use.sizes[band])]


def _track_column(df, track):
    if track is not None:
        if track not in df.columns:
            raise KeyError(f"track column {track!r} not found")
        return track
    info = df.attrs.get(ATTRS) or df.attrs.get(OLD_ATTRS) or {}
    if info.get("track") in df.columns:
        return info["track"]
    for name in _TRACK_NAMES:
        if name in df.columns:
            return name
    return None


def _samples(df, track, spacing, grid: _Grid):
    """The points the use stands for, by track: (names, [(x, y, weight)]) in the surface's CRS. A point weighs 1; a line
    is cut into pieces no longer than ``spacing`` and each piece's midpoint weighs its length."""
    geom = df.geometry
    if len(geom) == 0:
        raise ValueError("use: no locations")
    if geom.isna().any() or geom.is_empty.any():
        raise ValueError("use: some geometries are empty: drop them first")
    kinds = set(geom.geom_type)
    if kinds == {"Point"}:
        lines = False
    elif kinds == {"LineString"}:
        lines = True
    elif kinds == {"LineString", "Point"}:
        raise ValueError("use: points or lines, not both")
    else:
        raise ValueError("use: points (the tracks), or two-point lines (tf.move.corridor_lines); got " + ", ".join(sorted(kinds)))
    col = _track_column(df, track)
    labels = df[col].astype(str).to_numpy() if col is not None else np.full(len(df), "use", dtype=object)
    names = list(pd.unique(pd.Series(labels)))
    out = []
    if not lines:
        px, py = _project(geom.x.to_numpy(np.float64), geom.y.to_numpy(np.float64), df.crs, grid.crs)
        for nm in names:
            m = labels == nm
            out.append((px[m], py[m], np.ones(int(m.sum()))))
        return names, out
    step = float(spacing) if spacing is not None else min(grid.xres, grid.yres) / 2
    per = {nm: ([], [], []) for nm in names}
    for lab, g in zip(labels, geom):
        c = np.asarray(g.coords, dtype=np.float64)
        x, y = _project(c[:, 0], c[:, 1], df.crs, grid.crs)
        for i in range(len(x) - 1):
            dx, dy = x[i + 1] - x[i], y[i + 1] - y[i]
            length = math.sqrt(dx * dx + dy * dy)
            if not length > 0:
                continue
            n = max(1, math.ceil(length / step))
            t = (np.arange(n) + 0.5) / n
            per[lab][0].append(x[i] + t * dx)
            per[lab][1].append(y[i] + t * dy)
            per[lab][2].append(np.full(n, length / n))
    for nm in names:
        a = per[nm]
        out.append(tuple(np.concatenate(p) if p else np.empty(0) for p in a))
    return names, out


# ------------------------------------------------------------------------------------------ the measures

def _measures(u, s, corr, rho_cells, coverage, sign) -> dict:
    """``u`` the use of each analysis cell (summing to one for a UD), ``s`` the surface there, ``corr`` the corridor
    flags there, ``rho_cells`` the positions the Spearman runs over (in order)."""
    total = _sum(u)
    if not total > 0:
        return {"n_cells": 0, "rho": float("nan"), "use_fraction": float("nan"), "area_fraction": float("nan"),
                "use_ratio": float("nan"), "surface_used": float("nan"), "surface_available": float("nan"),
                "coverage": coverage}
    use_frac = _sum(np.where(corr, u, 0.0)) / total
    n = len(s)
    area_frac = float(np.count_nonzero(corr)) / n
    ratio = use_frac / area_frac if area_frac > 0 else float("nan")
    return {"n_cells": len(rho_cells), "rho": spearman(u[rho_cells], sign * s[rho_cells]),
            "use_fraction": use_frac, "area_fraction": area_frac, "use_ratio": ratio,
            "surface_used": _sum(u * s) / total, "surface_available": _sum(s) / n, "coverage": coverage}


def _support_cells(u: np.ndarray, support: float) -> np.ndarray:
    """The positions of the cells inside the ``support`` volume contour of the use, the most used first."""
    idx = np.flatnonzero(u > 0)
    if idx.size == 0:
        return idx
    order = idx[np.argsort(-u[idx], kind="stable")]
    if support >= 1:
        return order
    cum = np.cumsum(u[order])
    k = int(np.searchsorted(cum, support * cum[-1], side="left")) + 1
    return order[:min(k, len(order))]


def _fmt(x: float, sig: int = 3) -> str:
    return format(x, f".{sig}g")


def _note(label: str, r: dict, kind: str) -> str:
    use, area, ratio, rho = r["use_fraction"], r["area_fraction"], r["use_ratio"], r["rho"]
    if not math.isfinite(use):
        return f"{label}: no use fell on cells with data, so nothing can be compared."
    pct = lambda f: _fmt(100 * f, 3) + "%"
    if not math.isfinite(ratio):
        head = f"{label}: the corridors cover none of the compared area, so the use ratio is undefined."
    elif ratio >= 1.5:
        head = (f"{label}: {pct(use)} of the use lies in the corridors, {pct(area)} of the area: {_fmt(ratio)} times "
                "its share, so the animals favour the corridors.")
    elif ratio <= 1 / 1.5:
        head = (f"{label}: {pct(use)} of the use lies in the corridors, {pct(area)} of the area: {_fmt(ratio)} times "
                "its share, so the animals avoid the corridors.")
    else:
        head = (f"{label}: {pct(use)} of the use lies in the corridors, {pct(area)} of the area: {_fmt(ratio)} times "
                "its share, about what chance gives.")
    if math.isnan(rho):
        tail = " The rank correlation cannot be computed (too few cells or a constant variable)."
    else:
        a = abs(rho)
        word = "no" if a < 0.1 else "a weak" if a < 0.3 else "a moderate" if a < 0.5 else "a strong"
        if word == "no":
            tail = f" Spearman's rho is {_fmt(rho)}: no rank relation between the use and the surface."
        else:
            sense = "the use is higher where the surface is better" if rho > 0 else \
                "the use is higher where the surface is worse"
            tail = f" Spearman's rho is {_fmt(rho)}: {word} rank relation ({sense})."
    if r["coverage"] < 0.9:
        tail += f" Only {pct(r['coverage'])} of the use falls on cells with data."
    return head + tail


def validate(use, surface, *, kind: str = "current", threshold: float | None = None, quantile: float = 0.9,
             support: float = 0.99, track: str | None = None, spacing: float | None = None) -> Validation:
    """Compare where the animals went with a connectivity surface (Zeller et al. 2018's validation with movement data).

    Not in R (amt and move2utils have nothing like it): the definitions are ours, following what Zeller et al. (2018)
    measure, how well a connectivity surface *captures* the locations of the animals. Their paper (Diversity and
    Distributions 24:868-879, doi:10.1111/ddi.12742) validates resistance surfaces and corridors of cost distance and
    circuit theory with the GPS locations of dispersing pumas; its methods were not available to the author of this
    function, only the abstract, so the measures below are the usual ones of that kind and not a copy of theirs.

    **The surface** gives each cell a *score*, higher where the surface says movement is better: the surface itself
    for ``kind="current"`` (Circuitscape's or Omniscape's current, a cost-distance kernel, any "higher is better"
    map) and for ``"binary"`` (a least-cost corridor: non-zero cells are the corridor), minus the surface for
    ``"resistance"`` (lower is better). Cells without a value are left out of everything. **The corridors** are the
    cells with a score at or above a cut: ``threshold`` (a surface value: from it up for a current, up to it for a
    resistance) or else the ``quantile`` of the scores over the whole surface (R's type 7; 0.9 makes the best 10 % of
    the cells the corridors); a binary surface is its own corridor and takes neither.

    **The use** is one of

    - a UD (a ``(y, x)`` raster or ``(band, y, x)``, a band per track, from :func:`dbbmm_ud`, :func:`dbgb_ud` or
      :func:`hr_ud`). It is read on the surface's grid: the same grid cell by cell, otherwise the surface at each UD cell
      centre (nearest cell, the centres reprojected when the CRSs differ). Each band is cut to the cells where the surface
      has a value and renormalised to sum to 1 over them (``coverage``: the part of its mass that was on such cells).
      The compared area is those cells (the UD's grid, with the surface).
    - points (a GeoDataFrame of locations, ``track`` the column of the track, found like the other functions of
      ``tf.move`` do; none means one track). Each point counts 1 in the surface cell that holds it; a point off the
      surface or on a cell without a value is left out (``coverage``: the share kept). The compared area is the cells
      with a value inside the bounding box of the kept points' cells, track by track.
    - lines (two-point segments, :func:`corridor_lines`, or longer ones). Each segment is cut into
      ``ceil(length / spacing)`` equal pieces, ``spacing`` defaulting to half the surface's smaller cell side, and each
      piece counts its length at its midpoint, in the cell holding it: the use is the length of line in each cell.
      Otherwise as points, with coverage by length.

    **The measures** (a row per track, then ``pooled``: the mean of the tracks' normalised UDs, or all the points or all
    the lines together, with their bounding box):

    - ``rho``: Spearman's rank correlation (average ranks for ties, as ``scipy.stats.spearmanr``) between the use and
      the score over cells, so positive always means that the animals use better cells more, also for a resistance. For
      a UD the cells are those inside its ``support`` volume contour (the most probable cells holding that share of the
      probability; ``support=1`` takes all with a positive value) and the UD's probability is the use of each; it is
      the rank correlation of the surface with the UD, not weighted by it. For points and lines, all compared cells, the
      use being the number of points or the length of line they hold (most cells are 0: a tie). NaN with fewer than
      three cells or a constant variable. No p-value: cells next to each other are not independent.
    - ``use_fraction``: the share of the use (UD probability, points, length) inside the corridors, of all the use that
      fell on cells with data.
    - ``area_fraction``: the share of the compared cells that are corridors, what a use blind to the surface would give.
    - ``use_ratio``: ``use_fraction / area_fraction``, Zeller's "used against available"; above 1 the corridors catch more
      of the use than their area, 1 is no better than chance, NaN when no cell is a corridor.
    - ``surface_used``, ``surface_available``: the mean surface value (the surface itself, not the score) weighted by
      the use, and over the compared cells.
    - ``n_cells`` (the cells the correlation ran over) and ``coverage``.

    Sums run in order, so the result does not depend on the number of threads.

    Parameters
    ----------
    use
        A UD, points or lines, as above.
    surface
        A ``(y, x)`` DataArray, north up.
    kind
        ``"current"`` (default), ``"resistance"`` or ``"binary"``.
    threshold, quantile
        The corridor cut, as above (``quantile`` in (0, 1), used when ``threshold`` is None).
    support
        The volume contour, in (0, 1], of a UD's cells for the correlation (0.99).
    track
        The track column of points and lines.
    spacing
        The longest piece of a line, in the surface's units.

    Returns
    -------
    Validation
        ``table`` (columns ``track``, ``n_cells``, ``rho``, ``use_fraction``, ``area_fraction``, ``use_ratio``,
        ``surface_used``, ``surface_available``, ``coverage``), ``note`` (the pooled row, or the only one, in words),
        ``threshold`` (in the surface's units), ``corridor`` (a raster) and ``kind``.

    References
    ----------
    Zeller, K. A., Jennings, M. K., Vickers, T. W., Ernest, H. B., Cushman, S. A., Boyce, W. M. (2018). Are all data types
    and connectivity models created equal? Validating common connectivity approaches with dispersal data. Diversity and
    Distributions 24, 868-879. doi:10.1111/ddi.12742

    Examples
    --------
    >>> import terraflux as tf
    >>> import numpy as np, xarray as xr
    >>> x, y = np.arange(30) * 10.0 + 5, np.arange(30)[::-1] * 10.0 + 5
    >>> current = xr.DataArray(np.exp(-((x[None, :] - 150) / 40) ** 2) * np.ones((30, 1)), coords={"y": y, "x": x},
    ...                        dims=("y", "x"))
    >>> gx, gy = np.meshgrid(x, y)
    >>> ud = np.exp(-((gx - 160) ** 2 + (gy - 150) ** 2) / (2 * 30.0 ** 2))
    >>> ud = xr.DataArray(ud / ud.sum(), coords={"y": y, "x": x}, dims=("y", "x"))
    >>> v = tf.move.validate(ud, current, quantile=0.8)
    >>> bool(v.table["use_ratio"][0] > 1.5)
    True
    """
    if kind not in _KINDS:
        raise ValueError('kind: "current", "resistance" or "binary"')
    if kind == "binary" and threshold is not None:
        raise ValueError('kind="binary": the surface is its own corridor, a threshold has no meaning')
    if threshold is not None and not (isinstance(threshold, (int, float, np.integer, np.floating))
                                      and not isinstance(threshold, bool) and np.isfinite(threshold)):
        raise ValueError("threshold: a finite number or None")
    if kind != "binary" and threshold is None and not (isinstance(quantile, (int, float, np.integer, np.floating))
                                                       and 0 < quantile < 1):
        raise ValueError("quantile: a number in (0, 1)")
    if not (isinstance(support, (int, float, np.integer, np.floating)) and 0 < support <= 1):
        raise ValueError("support: a number in (0, 1]")
    if spacing is not None and not (np.isfinite(spacing) and spacing > 0):
        raise ValueError("spacing: a positive number or None")
    grid = _Grid(surface, "surface")
    s_all = grid.values
    finite = np.isfinite(s_all)
    if not finite.any():
        raise ValueError("surface: no cell has a value")
    sign = -1.0 if kind == "resistance" else 1.0
    score = s_all if kind != "resistance" else -s_all
    if kind == "binary":
        corr_all = finite & (s_all != 0)
        cut = float("nan")
        if not corr_all.any():
            raise ValueError("surface: a binary surface with no corridor cell")
    else:
        if threshold is not None:
            cut_score = float(threshold) if kind == "current" else -float(threshold)
        else:
            cut_score = quantile7(score[finite], float(quantile))
        corr_all = finite & (score >= cut_score)
        cut = cut_score if kind == "current" else -cut_score
    rows: list[tuple[str, dict]] = []

    import xarray as xr

    if isinstance(use, xr.DataArray):
        layers = _ud_layers(use)
        cells = []
        uds = []
        for name, da in layers:
            g = _Grid(da, name)
            v = g.values
            if np.nanmin(v, initial=0.0) < 0:
                raise ValueError(f"{name}: a UD has no negative values")
            if g.same_as(grid) and not (g.crs is not None and grid.crs is not None and g.crs != grid.crs):
                raw = v
                sidx = None
            else:
                gx, gy = g.centres()
                mx, my = np.meshgrid(gx, gy)
                px, py = _project(mx.ravel(), my.ravel(), g.crs, grid.crs)
                r, c, inside = grid.cells(px, py)
                raw = v.ravel()
                sidx = (r * grid.nx + c, inside)
            cells.append(sidx)
            uds.append((name, raw))
        # the analysis cells: those of the first UD's grid with a surface value (the same for every band)
        first = cells[0]
        for sidx in cells[1:]:
            if (sidx is None) != (first is None) or (sidx is not None and not np.array_equal(sidx[0], first[0])):
                raise ValueError("All UDs must share the same raster geometry.")
        if first is None:
            s = s_all.ravel()
            corr = corr_all.ravel()
            ok = finite.ravel()
        else:
            flat = first[0]
            inside = first[1]
            s = np.where(inside, s_all.ravel()[flat], np.nan)
            corr = inside & corr_all.ravel()[flat]
            ok = inside & finite.ravel()[flat]
        pos = np.flatnonzero(ok)
        s_a, corr_a = s[pos], corr[pos]
        norm = []
        for name, raw in uds:
            u = raw.ravel() if raw.ndim > 1 else raw
            u = np.where(np.isfinite(u), u, 0.0)
            total_all = _sum(u)
            u = u[pos]
            tot = _sum(u)
            if not tot > 0:
                raise ValueError(f"{name}: the UD has no mass on cells where the surface has a value")
            norm.append((name, u / tot, tot / total_all if total_all > 0 else float("nan")))
        for name, u, cov in norm:
            rows.append((name, _measures(u, s_a, corr_a, _support_cells(u, float(support)), cov, sign)))
        if len(norm) > 1:
            pooled = np.zeros(len(pos))
            for _, u, _ in norm:
                pooled = pooled + u
            pooled = pooled / len(norm)
            cov = _sum(np.array([c for _, _, c in norm])) / len(norm)
            rows.append(("pooled", _measures(pooled, s_a, corr_a, _support_cells(pooled, float(support)), cov, sign)))
    else:
        import geopandas as gpd

        if not isinstance(use, gpd.GeoDataFrame):
            raise TypeError("use: a UD (a DataArray), or a GeoDataFrame of points or lines")
        names, samples = _samples(use, track, spacing, grid)
        stacks = []
        for nm, (x, y, w) in zip(names, samples):
            r, c, inside = grid.cells(x, y)
            valid = inside & finite[r, c]
            stacks.append((r[valid], c[valid], w[valid], _sum(w[valid]), _sum(w)))
        for nm, (r, c, w, kept, whole) in zip(names, stacks):
            rows.append((nm, _on_cells(r, c, w, kept, whole, grid, s_all, finite, corr_all, sign)))
        if len(names) > 1:
            r = np.concatenate([a[0] for a in stacks])
            c = np.concatenate([a[1] for a in stacks])
            w = np.concatenate([a[2] for a in stacks])
            kept = _sum(np.array([a[3] for a in stacks]))
            whole = _sum(np.array([a[4] for a in stacks]))
            rows.append(("pooled", _on_cells(r, c, w, kept, whole, grid, s_all, finite, corr_all, sign)))

    cols = ["n_cells", "rho", "use_fraction", "area_fraction", "use_ratio", "surface_used", "surface_available",
            "coverage"]
    table = pd.DataFrame([{"track": nm, **m} for nm, m in rows], columns=["track", *cols])
    table["n_cells"] = table["n_cells"].astype("int64")
    last = rows[-1]
    note = _note("pooled" if last[0] == "pooled" else last[0], last[1], kind)
    flag = np.where(finite, corr_all.astype(np.float64), np.nan)
    cor = xr.DataArray(flag[::-1] if grid.flipped else flag, coords=grid.da.coords, dims=grid.da.dims, name="corridor")
    if grid.da.rio.crs is not None:
        cor = cor.rio.write_crs(grid.da.rio.crs)
    for nm, m in rows:
        if not m["coverage"] > 0 and nm != "pooled":
            warnings.warn(f"{nm}: none of the use falls on cells with data", stacklevel=2)
    return Validation(table, note, cut, cor, kind)


def _on_cells(r, c, w, kept, whole, grid: _Grid, s_all, finite, corr_all, sign) -> dict:
    """The measures of weighted points (their cells ``r``, ``c`` and weights ``w``) on the surface."""
    coverage = kept / whole if whole > 0 else float("nan")
    if len(r) == 0 or not kept > 0:
        return _measures(np.zeros(1), np.zeros(1), np.zeros(1, bool), np.empty(0, np.int64), coverage, sign)
    r0, r1, c0, c1 = int(r.min()), int(r.max()), int(c.min()), int(c.max())
    h, wd = r1 - r0 + 1, c1 - c0 + 1
    use = np.bincount((r - r0) * wd + (c - c0), weights=w, minlength=h * wd)
    sub_s = s_all[r0:r1 + 1, c0:c1 + 1].ravel()
    sub_ok = finite[r0:r1 + 1, c0:c1 + 1].ravel()
    sub_c = corr_all[r0:r1 + 1, c0:c1 + 1].ravel()
    pos = np.flatnonzero(sub_ok)
    u, s, cr = use[pos], sub_s[pos], sub_c[pos]
    return _measures(u, s, cr, np.arange(len(pos)), coverage, sign)
