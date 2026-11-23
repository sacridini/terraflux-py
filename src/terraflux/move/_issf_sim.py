# SPDX-License-Identifier: GPL-3.0-or-later
"""Simulation from a step selection model: amt's ``redistribution_kernel`` and ``simulate_path``, resistance and use maps.

Signer et al. (2024) simulate where an animal goes from a fitted integrated step selection function: candidate steps
are drawn from the (tentative) step length and turn angle distributions, each weighted by the selection function
``exp(b . x)``, and one is drawn; repeated, it is a path. This is amt 0.3.1's continuous-space simulation, computed in
C++ with R's generator and R's arithmetic, so the paths are R's, bit by bit, where amt's errors do not interfere.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .. import _core
from ._issf import ISSF, _columns_of, _MATH, _arith, _codes, _raw, _recipe, fit_issf
from ._ssf import Distribution, _layers, _seed, extract_covariates, random_steps
from ._steps import _seconds, filter_min_n_burst, resample, steps
from ._tracks import ATTRS, info_of

_ONE_HOUR = pd.Timedelta("1h")
_OP_FN = {"log": 12, "exp": 13, "sqrt": 14, "sin": 15, "cos": 16, "log1p": 17}
_OP_BIN = {"+": 20, "-": 21, "*": 22, "/": 23, "^": 24}
_MOVE = ("sl_", "ta_", "log_sl_", "cos_ta_")


# ------------------------------------------------------------------------------------------------ the start


@dataclass
class Start:
    """Where a simulation starts (amt's ``make_start``): a location, the direction of the step before it, the time
    and the time between steps.

    Attributes
    ----------
    x, y : float
    ta : float
        The direction the animal came from (radians, anticlockwise from east): the first turn angles are relative to it.
    t : pandas.Timestamp or float
    dt : pandas.Timedelta or float
    crs
    """
    x: float = 0.0
    y: float = 0.0
    ta: float = 0.0
    t: object = None
    dt: object = None
    crs: object = None


def make_start(xy=(0.0, 0.0), ta: float = 0.0, *, time=None, dt=None, crs=None) -> Start:
    """The start of a simulation (amt's ``make_start``).

    Parameters
    ----------
    xy
        The location ``(x, y)``.
    ta
        The direction of the step before (radians).
    time
        The time of the start (a timestamp or a number; default 1970-01-01 UTC, where amt takes the clock).
    dt
        The time between steps (a ``pandas.Timedelta``, a string such as ``"30min"``, or a number of seconds, or of
        units when ``time`` is a number; default 1 hour, as amt's).
    crs
        The CRS of the coordinates (kept for the output).

    Examples
    --------
    >>> import terraflux as tf
    >>> s = tf.move.make_start((500.0, 500.0), 0.5, time="2020-01-01", dt="30min")
    >>> s.x, s.ta, s.dt
    (500.0, 0.5, Timedelta('0 days 00:30:00'))
    """
    x, y = (float(v) for v in xy)
    if not np.isfinite([x, y, ta]).all():
        raise ValueError("xy and ta: finite numbers")
    if not (isinstance(time, (int, float, np.integer, np.floating)) and not isinstance(time, bool)):
        t = pd.Timestamp("1970-01-01", tz="UTC") if time is None else pd.Timestamp(time)
        if t.tzinfo is None:
            t = t.tz_localize("UTC")
        if dt is None:
            d = _ONE_HOUR
        elif isinstance(dt, (int, float, np.integer, np.floating)):
            d = pd.Timedelta(seconds=float(dt))
        else:
            d = pd.Timedelta(dt)
    else:
        t = float(time)
        d = 1.0 if dt is None else float(dt)
    return Start(x, y, float(ta), t, d, crs)


def _as_start(start) -> Start:
    if isinstance(start, Start):
        return start
    if isinstance(start, (tuple, list, np.ndarray)) and len(start) in (2, 3):
        return make_start(tuple(start[:2]), float(start[2]) if len(start) == 3 else 0.0)
    raise TypeError("start: tf.move.make_start(...) or a tuple (x, y[, direction])")


# ----------------------------------------------------------------------------------- the compiled formula


def _program(node, leaf, ops: list, args: list) -> None:
    """The reverse Polish program of an arithmetic expression of the formula (see ``_arith``)."""
    k = node[0]
    if k == "num":
        ops.append(0)
        args.append(float(node[1]))
    elif k in ("paren", "pos"):
        _program(node[1], leaf, ops, args)
    elif k == "neg":
        _program(node[1], leaf, ops, args)
        ops.append(10)
        args.append(0.0)
    elif k == "name":
        op, a = leaf(node[1])
        ops.append(op)
        args.append(float(a))
    elif k == "call":
        name, a = node[1], node[2]
        if name == "I":
            _program(a[0], leaf, ops, args)
        elif name == "abs":
            _program(a[0], leaf, ops, args)
            ops.append(11)
            args.append(0.0)
        elif name in _MATH:
            _program(a[0], leaf, ops, args)
            ops.append(_OP_FN[_MATH[name]])
            args.append(0.0)
        else:
            raise ValueError(f"formula: {name}() cannot be inside an expression")
    else:
        _, op, left, right = node
        if op not in _OP_BIN:
            raise ValueError(f"formula: {op!r} is not supported inside an expression")
        _program(left, leaf, ops, args)
        _program(right, leaf, ops, args)
        ops.append(_OP_BIN[op])
        args.append(0.0)


def _leaf_of(names: list[str]):
    """name -> (op, argument): the covariates (``x``, ``x_end`` at the end of the step, ``x_start`` at its start) and
    the movement terms."""
    index = {n: i for i, n in enumerate(names)}

    def leaf(name: str):
        if name in index:
            return 1, index[name]
        if name.endswith("_end") and name[:-4] in index:
            return 1, index[name[:-4]]
        if name.endswith("_start") and name[:-6] in index:
            return 2, index[name[:-6]]
        if name in _MOVE:
            return 3 + _MOVE.index(name), 0
        raise ValueError(f"the term {name!r} of the model is not a covariate ({', '.join(f'{n}_end, {n}_start' for n in names[:3])}"
                         f"{'...' if len(names) > 3 else ''}) nor one of {', '.join(_MOVE)}")
    return leaf


def _formula(model: ISSF, names: list[str]) -> dict:
    """The compact description of the model's formula for the C++ core."""
    terms = model._terms
    sp = terms.spec
    cols_names, cols = _recipe(terms)
    if cols_names != list(model.coef.index):
        raise ValueError("the columns of the model do not match its coefficients")
    leaf = _leaf_of(names)
    ops, args, var_ptr, var_factor, level_ptr, levels = [], [], [0], [], [0], []
    for i, v in enumerate(sp.variables):
        if i == sp.strata:
            ops.append(0)
            args.append(0.0)
            var_factor.append(0)
        elif v.kind == "num":
            _program(v.ast, leaf, ops, args)
            var_factor.append(0)
        else:
            _program(v.inner, leaf, ops, args)
            var_factor.append(1)
            try:
                levels.extend(float(x) for x in v.levels)
            except (TypeError, ValueError):
                raise ValueError(f"the levels of {v.label!r} are not numbers: a raster's values are") from None
        var_ptr.append(len(ops))
        level_ptr.append(len(levels))
    col_ptr, col_var, col_level = [0], [], []
    for c in cols:
        for vi, lv in c:
            col_var.append(vi)
            col_level.append(lv)
        col_ptr.append(len(col_var))
    coef = np.where(np.isnan(model.coef.to_numpy(np.float64)), 0.0, model.coef.to_numpy(np.float64))
    return {"var_ptr": var_ptr, "op": ops, "arg": args, "var_factor": var_factor, "level_ptr": level_ptr,
            "level": levels, "col_ptr": col_ptr, "col_var": col_var, "col_level": col_level, "coef": coef.tolist()}


def _grids(covariates):
    """(names, values, grids, bbox, crs, template) of the covariates' layers."""
    from .._zonal import _raster
    from ..io import _transform

    layers = _layers(covariates)
    if not layers:
        raise ValueError("covariates: at least one raster")
    names, values, grids = [], [], []
    ext = None
    for name, r, v in layers:
        tr = _transform(r)
        if tr.b != 0 or tr.d != 0:
            raise ValueError("rotated rasters are not supported")
        ny, nx = r.shape[-2:]
        xmin, ymax = tr.c, tr.f
        xmax, ymin = xmin + nx * tr.a, ymax + ny * tr.e
        grids.append([xmin, xmax, ymin, ymax, (xmax - xmin) / nx, (ymax - ymin) / ny])
        e = (xmin, xmax, ymin, ymax)
        ext = e if ext is None else (max(ext[0], e[0]), min(ext[1], e[1]), max(ext[2], e[2]), min(ext[3], e[3]))
        names.append(name)
        values.append(np.ascontiguousarray(v, dtype=np.float64))
    if len(set(names)) != len(names):
        raise ValueError(f"covariates: the layers need different names, got {names}")
    return names, values, np.asarray(grids, np.float64), ext, layers[0][1].rio.crs, layers[0][1]


# ------------------------------------------------------------------------------------------ the kernel


@dataclass
class RedistributionKernel:
    """One cycle of a simulation from an iSSF and where it left R's generator (amt's ``redistribution_kernel``).

    Attributes
    ----------
    sample : pandas.DataFrame or None
        The ``n_sample`` end points drawn (``x_``, ``y_``, ``t2_``); None when the cycle stopped (see ``stopped``).
    stopped : str or None
        Why the cycle stopped without drawing, if it did.
    model, start, n_control, n_sample, tolerance_outside, sl_distr, ta_distr
        The arguments of the call.
    state : numpy.ndarray
        The generator's state (R's Mersenne-Twister: 625 words): :func:`simulate_path` goes on from it, as R's
        generator is the same one after ``redistribution_kernel``; the call changes it.
    """
    model: ISSF
    start: Start
    sample: pd.DataFrame | None
    n_control: int
    n_sample: int
    tolerance_outside: float
    sl_distr: Distribution
    ta_distr: Distribution
    max_dist: float | None
    normalize: bool
    stopped: str | None
    state: np.ndarray = field(repr=False)
    _spec: dict = field(repr=False, default=None)


def _spec(model: ISSF, covariates, sl: Distribution, ta: Distribution, n_control: int, tol: float) -> dict:
    names, values, grids, ext, _, _ = _grids(covariates)
    return {"sl_name": sl.name, "ta_name": ta.name, "sl_par": sl._draw_params(), "ta_par": ta._draw_params(),
            "bbox": list(ext), "n_control": int(n_control), "tolerance_outside": float(tol), "layers": values,
            "grids": grids, "formula": _formula(model, names)}


def _distributions(model, sl_distr, ta_distr):
    sl = sl_distr if sl_distr is not None else model.sl_distr
    ta = ta_distr if ta_distr is not None else model.ta_distr
    if sl is None or ta is None:
        raise ValueError("the model has no step length and turn angle distributions (the tracks had several, or it is "
                         "not from random_steps): pass sl_distr and ta_distr, or use make_issf_model")
    for d, what in ((sl, "sl_distr"), (ta, "ta_distr")):
        if not isinstance(d, Distribution):
            raise TypeError(f"{what}: a Distribution (tf.move.make_distribution, tf.move.fit_distr)")
    if sl.name not in ("gamma", "exp", "lnorm", "hnorm", "unif"):
        raise ValueError(f"sl_distr: {sl.name!r} is not a distribution of step lengths")
    if ta.name not in ("vonmises", "unif"):
        raise ValueError(f"ta_distr: {ta.name!r} is not a distribution of turn angles")
    return sl, ta


def redistribution_kernel(model, start, covariates, *, sl_distr=None, ta_distr=None, max_dist=None,
                          n_control: int = 1_000_000, n_sample: int = 1, normalize: bool = True,
                          tolerance_outside: float = 0.0, landscape: str = "continuous",
                          seed: int | None = None) -> RedistributionKernel:
    """Where the animal goes next, from a fitted iSSF: amt's ``redistribution_kernel`` in continuous space.

    ``n_control`` candidate steps are drawn from the step length and turn angle distributions the model was fitted
    with (the tentative ones, as amt), from ``start`` and the direction of the step before; the covariates are read at
    their ends (and starts); each candidate has the weight ``exp(b . x)`` of the model matrix of the formula
    (movement terms included: it is importance sampling, as Signer et al. 2024); ``n_sample`` of them are drawn with
    those weights. The result is :func:`simulate_path`'s first cycle and leaves the generator where the next ones go on
    (R's: with ``seed``, ``set.seed(seed)``; the numbers are R's, bit by bit).

    Parameters
    ----------
    model
        An :class:`ISSF` (:func:`fit_issf`, :func:`issf`, :func:`make_issf_model`).
    start
        A :class:`Start` (:func:`make_start`) or ``(x, y)`` / ``(x, y, direction)``.
    covariates
        The rasters of the model's covariates (a DataArray, dict or list, as :func:`extract_covariates`), in the CRS of
        the start. The terms of the formula name them: ``name_end`` (or ``name``) at the end of the candidate step,
        ``name_start`` at its start; ``sl_``, ``ta_``, ``log_sl_`` and ``cos_ta_`` are the step's length and turn angle
        (amt wants you to make the last two in ``fun``, whose default does not and fails; they are made here).
    sl_distr, ta_distr
        The distributions to draw from (default: those of the model).
    max_dist
        Kept for amt's signature; amt uses it only in discrete space.
    n_control
        The candidates per cycle (amt's default, a million).
    n_sample
        The points drawn per cycle, without replacement.
    normalize
        Kept for amt's signature (it only normalises a raster kernel, not in continuous space).
    tolerance_outside
        The fraction of the candidates that could be drawn (positive weight) and end outside the study area (the
        extent of the covariates) above which the cycle stops without drawing. 0 by default as amt, which counts every
        candidate: one in a million outside, and amt stops, though a candidate with a covariate that is missing has no
        weight and is never drawn; here only the ones that could be drawn count.
    landscape
        ``"continuous"``. amt's ``"discrete"`` (candidates on the cells around the start, from GEOS's buffer polygon)
        is not implemented: it biases the step lengths to the cell size (Signer et al. 2024), and the cells at the
        edge of the circle depend on GEOS.
    seed
        The seed of R's generator.

    Returns
    -------
    RedistributionKernel
        With ``sample``, a table of ``x_``, ``y_`` and ``t2_`` (the time at the end of the step).

    Notes
    -----
    Differences from amt 0.3.1 (continuous space), whose errors are corrected: (1) the movement terms ``log_sl_`` and
    ``cos_ta_`` (``sl_`` and ``ta_`` too) need no ``fun``: amt's default ``fun`` does not make them and the weights fail
    ("object 'cos_ta_' not found"); (2) ``tolerance_outside`` counts the candidates that could be drawn, not all (above);
    (3) a step length distribution ``"hnorm"`` works: amt evaluates ``max.dist`` by default, which asks for ``qhnorm``, a
    function that does not exist; (4) ``max_dist`` is not computed; (5) a coefficient that is NaN (a collinear
    column) counts as zero; R's weights are all NA then; (6) the time and ``dt`` of the start are kept along the
    path (:func:`simulate_path`: amt makes each new start with the time of the step before and the default ``dt``);
    (7) the errors of amt's discrete mode (``hnorm`` and ``lnorm`` kernels, the terms split at a ``+``, factors) do not
    arise: discrete space is not here. The formula's terms are R's: columns, ``log``, ``exp``, ``sqrt``, ``sin``,
    ``cos``, ``log1p``, ``abs``, ``I()``, interactions, ``factor()`` of a covariate; a term of time (``t1_``, ``t2_``,
    which a ``fun`` of amt could turn into the hour...) is not available.

    References
    ----------
    Signer, J., Fieberg, J., Reineking, B., Schlaegel, U., Smith, B., Balkenhol, N. & Avgar, T. (2024). Simulating
    animal space use from fitted integrated Step-Selection Functions (iSSF). Methods in Ecology and Evolution 15,
    43-50. Avgar, T., Potts, J. R., Lewis, M. A. & Boyce, M. S. (2016). Integrated step selection analysis: bridging
    the gap between resource selection and animal movement. Methods in Ecology and Evolution 7, 619-630.

    Examples
    --------
    >>> import numpy as np, xarray as xr
    >>> import terraflux as tf
    >>> x = np.arange(0.5, 100.5)
    >>> cov = xr.DataArray(np.sin(x / 8)[None, :] + np.cos(x / 9)[:, None], dims=("y", "x"), coords={"y": x[::-1], "x": x},
    ...                    name="cov").rio.write_crs("EPSG:32633")
    >>> m = tf.move.make_issf_model({"cov_end": 1.0, "log_sl_": 0.2}, tf.move.make_distribution("gamma", shape=2, scale=3),
    ...                             tf.move.make_distribution("vonmises", kappa=1))
    >>> k = tf.move.redistribution_kernel(m, (50.0, 50.0), cov, n_control=1000, tolerance_outside=1, seed=1)
    >>> p = tf.move.simulate_path(k, 5)
    >>> p.shape
    (6, 5)
    """
    if landscape != "continuous":
        if landscape == "discrete":
            raise NotImplementedError("landscape='discrete' is not implemented (candidates on the cells around the "
                                      "start follow GEOS's circle and bias the step lengths to the cell size; use "
                                      "'continuous', as Signer et al. 2024 recommend)")
        raise ValueError("landscape: 'continuous'")
    if not isinstance(model, ISSF):
        raise TypeError("model: an ISSF (tf.move.fit_issf, tf.move.make_issf_model)")
    st = _as_start(start)
    if int(n_control) < 1 or int(n_sample) < 1:
        raise ValueError("n_control and n_sample: 1 or more")
    if not 0.0 <= float(tolerance_outside) <= 1.0:
        raise ValueError("tolerance_outside: a fraction, from 0 to 1")
    sl, ta = _distributions(model, sl_distr, ta_distr)
    spec = _spec(model, covariates, sl, ta, n_control, tolerance_outside)
    state = _core.move_sim_seed(_seed(seed))
    res = _core.move_sim_kernel(state, st.x, st.y, st.ta, int(n_sample), spec)
    stopped, sample = None, None
    if res["status"] == 0:
        t2 = st.t + st.dt
        sample = pd.DataFrame({"x_": res["x"], "y_": res["y"], "t2_": [t2] * len(res["x"])})
    elif res["status"] == 1:
        stopped = (f"more than {100 * tolerance_outside:.3g}% of the candidate steps that could be drawn end outside "
                   "the study area")
        warnings.warn(f"{stopped}. Terminating simulations here.", RuntimeWarning, stacklevel=2)
    else:
        raise ValueError("too few positive probabilities: no candidate step has a weight above 0 (the covariates are "
                         "missing at the end of all of them, or the weights overflow)")
    return RedistributionKernel(model, st, sample, int(n_control), int(n_sample), float(tolerance_outside), sl, ta,
                                max_dist, bool(normalize), stopped, res["state"], spec)


def simulate_path(kernel: RedistributionKernel, n_steps: int = 100, *, start=None, verbose: bool = False) -> pd.DataFrame:
    """A path simulated from a fitted iSSF (amt's ``simulate_path`` of a redistribution kernel).

    Each step is one cycle of :func:`redistribution_kernel` (``n_control`` candidates, one drawn) from the end of the
    step before, with its direction (``atan2``) as the direction of the step before. The generator is the kernel's:
    the path goes on from where :func:`redistribution_kernel` left it, as in R (with the same seed, the path is the one
    amt gives); a second call goes on from where the first stopped.

    Parameters
    ----------
    kernel
        The result of :func:`redistribution_kernel`.
    n_steps
        The steps.
    start
        Another start (default: the kernel's).
    verbose
        Not used (as in amt).

    Returns
    -------
    pandas.DataFrame
        ``n_steps + 1`` rows: ``x_``, ``y_``, ``t_`` (the start's time plus ``dt`` for each step), ``dt`` and ``step_``
        (0 for the start). If an animal leaves the study area (see ``tolerance_outside``) or no candidate can be
        drawn, the path stops with a warning and the rows after it are NaN in ``x_`` and ``y_``.

    Examples
    --------
    >>> import numpy as np, xarray as xr
    >>> import terraflux as tf
    >>> x = np.arange(0.5, 100.5)
    >>> cov = xr.DataArray(np.sin(x / 8)[None, :] + np.cos(x / 9)[:, None], dims=("y", "x"), coords={"y": x[::-1], "x": x},
    ...                    name="cov").rio.write_crs("EPSG:32633")
    >>> m = tf.move.make_issf_model({"cov_end": 1.0}, tf.move.make_distribution("gamma", shape=2, scale=3),
    ...                             tf.move.make_distribution("vonmises", kappa=1))
    >>> k = tf.move.redistribution_kernel(m, (50.0, 50.0), cov, n_control=500, tolerance_outside=1, seed=3)
    >>> tf.move.simulate_path(k, 3).columns.tolist()
    ['x_', 'y_', 't_', 'dt', 'step_']
    """
    if not isinstance(kernel, RedistributionKernel):
        raise TypeError("kernel: the result of tf.move.redistribution_kernel")
    n = int(n_steps)
    if n < 0:
        raise ValueError("n_steps: 0 or more")
    st = kernel.start if start is None else _as_start(start)
    res = _core.move_sim_path(kernel.state, st.x, st.y, st.ta, n, kernel._spec)
    kernel.state = res["state"]
    if isinstance(st.t, pd.Timestamp):
        t = pd.date_range(st.t, periods=n + 1, freq=st.dt) if n >= 0 else []
    else:
        t = st.t + st.dt * np.arange(n + 1)
    out = pd.DataFrame({"x_": res["x"], "y_": res["y"], "t_": t, "dt": [st.dt] * (n + 1), "step_": np.arange(n + 1)})
    out.attrs[ATTRS] = {"crs": st.crs}
    done = int(res["done"])
    if res["status"] == 1:
        warnings.warn(f"Simulation stopped after {done} time steps, because the animal stepped out of the landscape.",
                      RuntimeWarning, stacklevel=2)
    elif res["status"] == 2:
        warnings.warn(f"Simulation stopped after {done} time steps: no candidate step has a weight above 0.",
                      RuntimeWarning, stacklevel=2)
    return out


# --------------------------------------------------------------------------------------------- resistance


def _habitat(model: ISSF, names: list[str]):
    """The columns of the model that depend on habitat only: ([(name, factors)], dropped names, start-dropped names)."""
    terms = model._terms
    sp = terms.spec
    cols_names, cols = _recipe(terms)
    leaf = _leaf_of(names)
    kind = {}
    for i, v in enumerate(sp.variables):
        if i == sp.strata:
            continue
        ks = set()
        for nm in _columns_of(v.ast):
            op, _ = leaf(nm)
            ks.add({1: "end", 2: "start"}.get(op, "move"))
        kind[i] = ks
    keep, move, start = [], [], []
    for nm, c in zip(cols_names, cols):
        ks = set().union(*(kind[vi] for vi, _ in c))
        if "move" in ks:
            move.append(nm)
        elif "start" in ks:
            start.append(nm)
        else:
            keep.append((nm, c))
    return keep, move, start


def resistance(model: ISSF, covariates, *, transform: str = "negexp", c: float = 8.0, r_max: float = 100.0,
               scale: str = "minmax"):
    """A resistance surface from the habitat part of a step selection model.

    The relative selection ``w = exp(sum(b_k x_k))`` of each cell, from the covariates' terms alone: the terms of
    movement (``sl_``, ``log_sl_``, ``cos_ta_``, ``ta_``) and the interactions with them, and covariates at the start of
    the step, are left out (with a warning that says which), since a cell has no step. The selection becomes resistance
    (high where the animal avoids), ready for :func:`terraflux.cost.transition` or :func:`terraflux.circuit`.

    Parameters
    ----------
    model
        An :class:`ISSF` (:func:`fit_issf`, :func:`issf`, :func:`make_issf_model`).
    covariates
        The rasters of the covariates, on one grid (a DataArray, dict or list); the result has their grid.
    transform
        How the selection becomes resistance, with ``h`` the selection scaled to [0, 1] (``scale``):

        * ``"negexp"``: ``r_max - (r_max - 1) * (1 - exp(-c h)) / (1 - exp(-c))``, a negative exponential of the habitat
          suitability, the kind of curve Keeley et al. (2016) recommend for turning suitability into resistance; 1 for the
          best cell, ``r_max`` for the worst; ``c`` > 0 bends it (near 0 it is the linear one);
        * ``"linear"``: ``r_max - (r_max - 1) * h``;
        * ``"inverse"``: ``1 / w``, not scaled (the inverse of the relative selection, as Zeller et al. 2012 list among
          the ways of turning a selection into a resistance);
        * ``"log"``: ``1 + c * (log(max w) - log(w))``, linear in the linear predictor.
    c
        The shape of ``"negexp"``, the slope of ``"log"``.
    r_max
        The resistance of the least selected cell (``"negexp"`` and ``"linear"``); the best has 1.
    scale
        ``"minmax"``: ``h = (w - min w) / (max w - min w)``; ``"max"``: ``h = w / max w``.

    Returns
    -------
    xarray.DataArray
        The resistance on the grid of the covariates, named ``"resistance"``; NaN where a covariate of the kept terms is NaN.

    Notes
    -----
    The formulas of ``"negexp"``, ``"linear"``, ``"log"`` and ``"inverse"`` are this module's: the papers' shapes
    (Keeley et al. 2016 compare linear and nonlinear conversions and recommend a negative exponential; Zeller et al.
    2012 review how selection functions become resistance) were not checked against their full text.

    References
    ----------
    Zeller, K. A., McGarigal, K. & Whiteley, A. R. (2012). Estimating landscape resistance to movement: a review.
    Landscape Ecology 27, 777-797. Keeley, A. T. H., Beier, P. & Gagnon, J. W. (2016). Estimating landscape resistance
    from habitat suitability: effects of data source and nonlinearities. Landscape Ecology 31, 2151-2162.

    Examples
    --------
    >>> import numpy as np, xarray as xr
    >>> import terraflux as tf
    >>> cov = xr.DataArray(np.linspace(0, 1, 12).reshape(3, 4), dims=("y", "x"),
    ...                    coords={"y": [2.5, 1.5, 0.5], "x": [0.5, 1.5, 2.5, 3.5]}, name="forest").rio.write_crs("EPSG:32633")
    >>> m = tf.move.make_issf_model({"forest_end": 2.0})
    >>> r = tf.move.resistance(m, cov)
    >>> float(r.min()), float(r.max())
    (1.0, 100.0)
    """
    if not isinstance(model, ISSF):
        raise TypeError("model: an ISSF")
    if transform not in ("negexp", "linear", "inverse", "log"):
        raise ValueError("transform: 'negexp', 'linear', 'inverse' or 'log'")
    if scale not in ("minmax", "max"):
        raise ValueError("scale: 'minmax' or 'max'")
    if not (np.isfinite(c) and c > 0) or not (np.isfinite(r_max) and r_max > 1):
        raise ValueError("c: above 0; r_max: above 1")
    names, values, grids, _, crs, tmpl = _grids(covariates)
    shape = values[0].shape
    if any(v.shape != shape for v in values) or any(not np.allclose(g, grids[0]) for g in grids):
        raise ValueError("covariates: the rasters must share one grid")
    keep, move, start = _habitat(model, names)
    if move or start:
        out = []
        if move:
            out.append(f"the terms of movement {move}")
        if start:
            out.append(f"the terms at the start of the step {start}")
        warnings.warn("resistance: left out " + " and ".join(out) + ": a cell has no step", UserWarning, stacklevel=2)
    if not keep:
        raise ValueError("the model has no term of habitat left")
    n = int(np.prod(shape))
    df = pd.DataFrame({k: v.ravel() for nm, v in zip(names, values) for k in (nm, nm + "_end")})
    sp = model._terms.spec
    vals = {}
    coef = model.coef
    eta = np.zeros(n)
    with np.errstate(all="ignore"):
        for nm, cols in keep:
            b = coef[nm]
            if np.isnan(b):
                continue
            col = None
            for vi, lv in cols:
                v = sp.variables[vi]
                if vi not in vals:
                    vals[vi] = (np.broadcast_to(np.asarray(_arith(v.ast, df, n), np.float64), (n,)) if v.kind == "num"
                                else _codes(_raw(v.inner, df, n), v.levels, v.label))
                x = vals[vi] if lv < 0 else np.where(vals[vi] < 0, np.nan, (vals[vi] == lv).astype(np.float64))
                col = x if col is None else col * x
            eta = eta + col * b
        bad = np.isnan(eta)
        e = np.where(bad, np.nan, eta)
        emax = np.nanmax(e) if (~bad).any() else np.nan
        # w / max(w) = exp(eta - max eta): no overflow, the scale cancels in every transformation but "inverse"
        wr = np.exp(e - emax)
        if transform == "inverse":
            r = np.exp(-e)
        elif transform == "log":
            r = 1.0 + c * (emax - e)
        else:
            wmin = np.nanmin(wr)
            h = wr if scale == "max" else (wr - wmin) / (1.0 - wmin) if wmin < 1.0 else np.ones_like(wr)
            if transform == "linear":
                r = r_max - (r_max - 1.0) * h
            else:
                r = r_max - (r_max - 1.0) * (1.0 - np.exp(-c * h)) / (1.0 - np.exp(-c))
    r = np.where(bad, np.nan, r).reshape(shape)
    base = tmpl if tmpl.ndim == 2 else tmpl.isel({tmpl.dims[0]: 0}, drop=True)
    res = base.astype(np.float64).copy(data=r)
    res.name = "resistance"
    res.attrs = {}
    try:
        res = res.rio.write_nodata(np.nan)
    except Exception:  # noqa: BLE001 - a raster without a CRS keeps its values
        pass
    return res


# ---------------------------------------------------------------------------------------------- path density


def path_density(paths, like, *, by: str | None = None, include_start: bool = False):
    """The use of a landscape by simulated paths: the share of positions in each cell.

    The end points of the steps of one or many paths (what :func:`simulate_path` returns) counted in the cells of
    ``like`` (terra's ``cellFromXY`` rule) and divided by their total: the cells sum to 1. It is the utilisation
    distribution of steady state of Signer et al. (2024) (the step end points of long paths, from several starts if
    need be) without smoothing: where they smooth with a kernel density estimator, smooth this raster yourself
    (:func:`terraflux.move.hr_kde` on the positions, or a filter on the result).

    Parameters
    ----------
    paths
        A table with ``x_`` and ``y_`` (a path, or several stacked), or a list of tables (pooled).
    like
        A raster whose grid is the result's; cells that are NaN in it are NaN in the result and take no positions.
    by
        A column that names the paths (``"track"``...): the result has one band per value, each summing to 1.
    include_start
        Count the first position of each path (``step_`` 0, the start), which is not the end of a step (default no).

    Returns
    -------
    xarray.DataArray
        On the grid of ``like`` (``(y, x)``, or ``(by, y, x)``): the shares; 0 where no position fell, NaN where ``like`` is.

    Examples
    --------
    >>> import numpy as np, pandas as pd, xarray as xr
    >>> import terraflux as tf
    >>> like = xr.DataArray(np.zeros((2, 2)), dims=("y", "x"), coords={"y": [1.5, 0.5], "x": [0.5, 1.5]}).rio.write_crs("EPSG:32633")
    >>> p = pd.DataFrame({"x_": [0.2, 0.7, 1.7, 1.8], "y_": [0.3, 0.4, 1.6, 1.7], "step_": [0, 1, 2, 3]})
    >>> tf.move.path_density(p, like).values.round(2)
    array([[0.  , 0.67],
           [0.33, 0.  ]])
    """
    from ..sdm._points import grid_cells

    tables = list(paths) if isinstance(paths, (list, tuple)) else [paths]
    df = pd.concat([pd.DataFrame(t) for t in tables], ignore_index=True)
    for col in ("x_", "y_"):
        if col not in df.columns:
            raise ValueError(f"paths: a table with the columns x_ and y_ (missing {col!r})")
    if by is not None and by not in df.columns:
        raise ValueError(f"by: the column {by!r} is not in paths")
    if not include_start and "step_" in df.columns:
        df = df[df["step_"] != 0]
    _, _, _, _, _, tmpl = _grids(like)
    base = tmpl if tmpl.ndim == 2 else tmpl.isel({tmpl.dims[0]: 0}, drop=True)
    ny, nx = base.shape
    land = np.asarray(_layers(like)[0][2])
    ok = ~np.isnan(land.ravel())

    def share(sub):
        cells = grid_cells(sub["x_"].to_numpy(np.float64), sub["y_"].to_numpy(np.float64), base)
        cells = cells[cells >= 0]
        cells = cells[ok[cells]]
        counts = np.bincount(cells, minlength=ny * nx).astype(np.float64)
        total = counts.sum()
        if total == 0:
            raise ValueError("no position of the paths falls on a cell of like")
        return np.where(ok, counts / total, np.nan).reshape(ny, nx)

    if by is None:
        out = base.astype(np.float64).copy(data=share(df))
        out.name = "density"
        out.attrs = {}
        return out
    groups = [(k, share(g)) for k, g in df.groupby(by, sort=True)]
    import xarray as xr

    out = xr.concat([base.astype(np.float64).copy(data=a) for _, a in groups], dim=by)
    out = out.assign_coords({by: [k for k, _ in groups]})
    out.name = "density"
    out.attrs = {}
    return out


# ------------------------------------------------------------------------------------------ the shortcut


def issf(tracks, covariates, *, interval=None, tolerance=None, n_control: int = 10, formula: str | None = None,
         where: str = "end", seed: int | None = None) -> ISSF:
    """An integrated step selection function from tracks and covariates, in one call (not in amt).

    The sequence of amt's tutorial: :func:`resample` (with ``interval``), :func:`filter_min_n_burst`, :func:`steps`,
    :func:`random_steps`, :func:`extract_covariates` and :func:`fit_issf`.

    Parameters
    ----------
    tracks
        A GeoDataFrame of points in a projected CRS (:func:`terraflux.load_tracks`).
    covariates
        The rasters (a DataArray, dict or list), named as the terms of the formula will be.
    interval
        The regular time between locations (seconds, a string such as ``"30min"``, or a timedelta): the tracks are
        resampled to it, in bursts. None: the tracks are taken as they are.
    tolerance
        How far from ``interval`` a time lag may be (default a tenth of it).
    n_control
        Random steps per observed step.
    formula
        R's formula (see :func:`fit_issf`). Default ``case_ ~ <the covariates> + log(sl_) + cos(ta_) +
        strata(stratum_)`` (the covariates as they are extracted: ``name``, or ``name_start`` and ``name_end`` with
        ``where="both"``). Without ``strata(...)`` the stratum is added.
    where
        Where the covariates are read: ``"end"`` of the step (the default), ``"start"`` or ``"both"``.
    seed
        The seed of R's generator for the random steps.

    Returns
    -------
    ISSF
        With ``steps`` (the table the fit used), ``report`` (a short text: the tracks, the steps and the table of
        coefficients) and the distributions ``sl_distr`` / ``ta_distr`` for :func:`redistribution_kernel`.

    Notes
    -----
    The strata are made by ``stratum_``, a column of one number for each track and step: the ``step_id_`` of
    :func:`random_steps` restarts for each track, so using it alone would merge the steps of different tracks.

    Examples
    --------
    >>> import terraflux as tf
    >>> tracks = tf.load_tracks(tf.move.example_data("fishers"))
    """
    if where not in ("end", "start", "both"):
        raise ValueError('where: "end", "start" or "both"')
    t = tracks
    if interval is not None:
        tol = _seconds(tolerance, "tolerance") if tolerance is not None else 0.1 * _seconds(interval, "interval")
        t = filter_min_n_burst(resample(t, interval, tol), 3)
    rs = random_steps(steps(t), n_control, seed=seed)
    ex = extract_covariates(rs, covariates, where=where)
    layer_names = [nm for nm, _, _ in _layers(covariates)]
    terms = [nm + s for nm in layer_names for s in ({"end": [""], "start": [""], "both": ["_start", "_end"]}[where])]
    track = info_of(rs)["track"]
    ex["stratum_"] = ex.groupby([track, "step_id_"], sort=False).ngroup().to_numpy() + 1
    if formula is None:
        formula = "case_ ~ " + " + ".join(terms) + " + log(sl_) + cos(ta_) + strata(stratum_)"
    elif "strata(" not in formula:
        formula = formula + " + strata(stratum_)"
    fit = fit_issf(ex, formula)
    fit.steps = ex
    n_tracks = int(ex[track].nunique())
    fit.report = (f"iSSF of {n_tracks} track(s): {int(ex['case_'].sum())} steps, each with {int(n_control)} random "
                  f"steps ({len(ex)} rows; {fit.nevent} steps and {fit.n} rows used, {fit.n_omitted} left out for "
                  f"missing values).\n{fit.summary().round(4).to_string()}")
    return fit
