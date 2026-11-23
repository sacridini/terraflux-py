# SPDX-License-Identifier: GPL-3.0-or-later
"""Step selection functions: conditional logistic regression and the log-RSS (amt's ``fit_issf`` and ``log_rss``).

amt 0.3.1 fits a step selection function (``fit_issf``, ``fit_ssf`` and ``fit_clogit`` are the same function there)
with ``survival::clogit``, whose default ``method = "exact"`` is Cox's regression with one death time per stratum
(``coxph(Surv(rep(1, n), case) ~ x + strata(step_id_), method = "exact")``, Therneau's ``coxexact``). This is that
fit, computed as survival 3.7-0 does it in R 4.4.2 (the same iterations, the same sums in the same order, the same
``exp`` and ``log`` of R's Windows build), so the coefficients, their covariance, the log-likelihoods, the iteration
count, the tests and the log-RSS are R's bits. The formula is R's, in a string, for the terms step selection uses.
"""

from __future__ import annotations

import math
import re
import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .. import _core

_EPS = 1e-9
_TOLER_CHOL = 2.0 ** -39   # .Machine$double.eps^.75
_TOLER_INF = math.sqrt(_EPS)
_ITER_MAX = 20

# ------------------------------------------------------------------------------------------------ the formula

_TOKEN = re.compile(r"\s*(?:(?P<num>(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?)|(?P<name>[A-Za-z.][A-Za-z0-9._]*)"
                    r"|`(?P<bq>[^`]+)`|(?P<op>%in%|[~+\-*/^:(),]))")
# binding powers (left, right): R's + - below * / below : below unary minus below ^ (right to left)
_INFIX = {"+": (10, 11), "-": (10, 11), "*": (20, 21), "/": (20, 21), ":": (30, 31), "^": (50, 49)}
_UNARY = 40
_MATH = {"log": "log", "exp": "exp", "sqrt": "sqrt", "sin": "sin", "cos": "cos", "log1p": "log1p"}
_SUPPORTED = "log, exp, sqrt, sin, cos, log1p, abs, I, factor and strata"


def _tokens(text: str) -> list[tuple[str, str]]:
    out, pos = [], 0
    text = text.rstrip()
    while pos < len(text):
        m = _TOKEN.match(text, pos)
        if m is None:
            raise ValueError(f"formula: cannot read {text[pos:pos + 12]!r} (position {pos + 1})")
        pos = m.end()
        kind = m.lastgroup
        out.append(("name" if kind == "bq" else kind, m.group(kind)))
    return out


class _Parser:
    def __init__(self, text: str):
        self.t = _tokens(text)
        self.i = 0

    def peek(self):
        return self.t[self.i] if self.i < len(self.t) else (None, None)

    def take(self):
        tok = self.peek()
        self.i += 1
        return tok

    def expect(self, op: str):
        k, v = self.take()
        if k != "op" or v != op:
            raise ValueError(f"formula: expected {op!r}, found {v!r}")

    def expr(self, rbp: int = 0):
        left = self.prefix()
        while True:
            k, v = self.peek()
            if k != "op" or v not in _INFIX and v != "%in%":
                return left
            if v == "%in%":
                raise ValueError("formula: %in% is not supported")
            lbp, nbp = _INFIX[v]
            if lbp <= rbp:
                return left
            self.take()
            left = ("bin", v, left, self.expr(nbp))

    def prefix(self):
        k, v = self.take()
        if k == "num":
            return ("num", float(v))
        if k == "name":
            if self.peek() == ("op", "("):
                self.take()
                args = []
                if self.peek() != ("op", ")"):
                    while True:
                        args.append(self.expr(0))
                        if self.peek() == ("op", ","):
                            self.take()
                            continue
                        break
                self.expect(")")
                return ("call", v, args)
            return ("name", v)
        if k == "op" and v == "(":
            e = self.expr(0)
            self.expect(")")
            return ("paren", e)
        if k == "op" and v in "+-":
            return ("neg" if v == "-" else "pos", self.expr(_UNARY))
        raise ValueError("formula: the expression ends where a term is expected" if k is None
                         else f"formula: unexpected {v!r}")


def _num_text(v: float) -> str:
    return str(int(v)) if v == int(v) and abs(v) < 1e15 else format(v, ".15g")


def _deparse(a) -> str:
    """R's deparse of the call (the labels of terms): spaces around + - *, none around / ^ :."""
    k = a[0]
    if k == "num":
        return _num_text(a[1])
    if k == "name":
        return a[1]
    if k == "call":
        return f"{a[1]}({', '.join(_deparse(x) for x in a[2])})"
    if k == "paren":
        return f"({_deparse(a[1])})"
    if k == "neg":
        return f"-{_deparse(a[1])}"
    if k == "pos":
        return f"+{_deparse(a[1])}"
    _, op, left, right = a
    sp = " " if op in "+-*" else ""
    return f"{_deparse(left)}{sp}{op}{sp}{_deparse(right)}"


@dataclass
class _Var:
    label: str
    ast: tuple
    kind: str = "num"          # "num", "factor" (a factor(...) call or a column that is not a number)
    inner: tuple | None = None   # a factor(...)'s argument
    levels: list | None = None
    labels: list | None = None


@dataclass
class _Spec:
    """A parsed formula: the response, its variables (by first appearance) and the terms (tuples of variable indices)."""
    text: str
    response: str
    variables: list
    terms: list
    strata: int | None   # index of the strata(...) variable


def _parse_formula(text: str) -> _Spec:
    if not isinstance(text, str) or "~" not in text:
        raise ValueError("formula: a string such as 'case_ ~ elevation + log(sl_) + strata(step_id_)'")
    lhs, rhs = text.split("~", 1)
    resp = lhs.strip()
    if not re.fullmatch(r"[A-Za-z.][A-Za-z0-9._]*|`[^`]+`", resp):
        raise ValueError(f"formula: the left side must be the name of the column of cases, not {resp!r}")
    resp = resp.strip("`")
    p = _Parser(rhs)
    ast = p.expr(0)
    if p.peek()[0] is not None:
        raise ValueError(f"formula: unexpected {p.peek()[1]!r}")
    variables: list[_Var] = []
    index: dict[str, int] = {}

    def var(node) -> tuple[int, ...]:
        label = _deparse(node)
        if label not in index:
            index[label] = len(variables)
            variables.append(_Var(label, node))
        return (index[label],)

    def cross(a, b):
        return [tuple(sorted(set(x) | set(y))) for x in a for y in b]

    def expand(node) -> list[tuple]:
        k = node[0]
        if k == "num":
            if node[1] not in (0.0, 1.0):
                raise ValueError(f"formula: the number {_num_text(node[1])} is not a term")
            return []   # 0 and 1 are the intercept, which the fit always has (as coxph)
        if k == "paren":
            return expand(node[1])
        if k == "pos":
            return expand(node[1])
        if k == "neg":
            if node[1][0] == "num":
                return []
            raise ValueError("formula: a minus sign takes out a term or the intercept (- 1): '-x' is not a term")
        if k in ("name", "call"):
            if k == "call":
                _check_call(node)
            return [var(node)]
        _, op, left, right = node
        if op == "+":
            a, b = expand(left), expand(right)
            return a + [t for t in b if t not in a]
        if op == "-":
            a = expand(left)
            b = [] if right[0] == "num" else expand(right)
            return [t for t in a if t not in b]
        if op == ":":
            return cross(expand(left), expand(right))
        if op == "*":
            a, b = expand(left), expand(right)
            terms = a + [t for t in b if t not in a]
            return terms + [t for t in cross(a, b) if t not in terms]
        raise ValueError(f"formula: {op!r} is not supported between terms (write I(a {op} b))")

    terms = []
    for t in expand(ast):
        if t not in terms:
            terms.append(t)
    terms.sort(key=len)   # main effects first, then the interactions (R's terms, keep.order = FALSE)
    s = [i for i, v in enumerate(variables) if v.ast[0] == "call" and v.ast[1] == "strata"]
    if len(s) > 1:
        raise ValueError("formula: only one strata(...) is allowed")
    if s and any(s[0] in t and len(t) > 1 for t in terms):
        raise ValueError("formula: strata(...) cannot be in an interaction")
    return _Spec(text.strip(), resp, variables, terms, s[0] if s else None)


def _check_call(node):
    _, name, args = node
    if name in _MATH or name == "abs":
        if len(args) != 1:
            raise ValueError(f"formula: {name}() takes one argument ({name}(x, base) is not supported)")
    elif name in ("I", "factor", "strata"):
        if len(args) != 1:
            raise ValueError(f"formula: {name}() takes one argument"
                             + (" (combine the columns of the strata first)" if name == "strata" else ""))
    else:
        raise ValueError(f"formula: the function {name}() is not supported (only {_SUPPORTED}; "
                         "tan, poly, ns, scale, offset and cluster are left out)")


# --------------------------------------------------------------------------------------------- the values

def _column(df, name: str):
    if name not in df.columns:
        raise ValueError(f"the column {name!r} of the formula is not in the data")
    return df[name]


def _is_number(s) -> bool:
    return s.dtype.kind in "iuf" and not isinstance(s.dtype, pd.CategoricalDtype)


def _arith(node, df, n: int):
    """The values of an arithmetic expression: a float64 array (or a number)."""
    k = node[0]
    if k == "num":
        return node[1]
    if k == "paren" or k == "pos":
        return _arith(node[1], df, n)
    if k == "neg":
        return -_arith(node[1], df, n)
    if k == "name":
        s = _column(df, node[1])
        if s.dtype.kind in "iufb" and not isinstance(s.dtype, pd.CategoricalDtype):
            return s.to_numpy(np.float64, na_value=np.nan) if s.dtype.kind != "b" else s.to_numpy(np.float64)
        raise ValueError(f"the column {node[1]!r} is not a number: it cannot be used in an expression")
    if k == "call":
        name, args = node[1], node[2]
        if name == "I":
            return _arith(args[0], df, n)
        if name == "abs":
            return np.abs(_arith(args[0], df, n))
        if name in _MATH:
            x = np.ascontiguousarray(np.broadcast_to(np.asarray(_arith(args[0], df, n), np.float64), (n,)))
            return _core.move_issf_fn(_MATH[name], x)
        raise ValueError(f"formula: {name}() cannot be inside an expression")
    _, op, left, right = node
    a, b = _arith(left, df, n), _arith(right, df, n)
    with np.errstate(all="ignore"):
        if op == "+":
            return a + b
        if op == "-":
            return a - b
        if op == "*":
            return a * b
        if op == "/":
            return a / b
        if op == "^":
            x = np.ascontiguousarray(np.broadcast_to(np.asarray(a, np.float64), (n,)))
            y = np.ascontiguousarray(np.broadcast_to(np.asarray(b, np.float64), (n,)))
            return _core.move_issf_pow(x, y)
    raise ValueError(f"formula: {op!r} is not supported inside an expression")


def _levels_of(raw) -> list:
    """The levels of factor(x): a categorical's categories, False and True for a logical, else the sorted values."""
    s = pd.Series(raw) if not isinstance(raw, pd.Series) else raw
    if isinstance(s.dtype, pd.CategoricalDtype):
        return list(s.cat.categories)
    if s.dtype.kind == "b":
        return [False, True]
    vals = s.dropna().unique().tolist()
    try:
        return sorted(vals)
    except TypeError:
        raise ValueError(f"cannot order the values of {s.name!r} into levels (mixed types)") from None


def _label(v) -> str:
    if isinstance(v, (bool, np.bool_)):
        return "TRUE" if v else "FALSE"
    if isinstance(v, (float, np.floating)):
        return str(int(v)) if v == int(v) and abs(v) < 1e15 else format(float(v), ".15g")
    return str(v)


def _codes(raw, levels: list, label: str) -> np.ndarray:
    """The 0-based level of each value, -1 for a missing one; a value that is none of the levels is an error."""
    s = raw if isinstance(raw, pd.Series) else pd.Series(raw)
    if isinstance(s.dtype, pd.CategoricalDtype):
        s = s.astype(object)
    na = s.isna().to_numpy()
    codes = pd.Index(levels).get_indexer(s.to_numpy(object if s.dtype == object else None)).astype(np.int64)
    codes[na] = -1
    bad = (codes < 0) & ~na
    if bad.any():
        new = pd.unique(s[bad]).tolist()
        raise ValueError(f"{label}: new levels {new} (the fit has {[_label(v) for v in levels]})")
    return codes


def _raw(node, df, n):
    """The values a factor() is made of: a column as it is, or the values of an expression."""
    if node[0] == "name":
        return _column(df, node[1])
    if node[0] == "paren":
        return _raw(node[1], df, n)
    return pd.Series(np.broadcast_to(np.asarray(_arith(node, df, n), np.float64), (n,)))


def _set_kind(v: _Var, df, n: int):
    """Decides at the fit whether a variable is a number or a factor, and its levels."""
    a = v.ast
    if a[0] == "call" and a[1] == "factor":
        v.kind, v.inner = "factor", a[2][0]
    elif a[0] == "name":
        s = _column(df, a[1])
        v.kind = "num" if _is_number(s) else "factor"
        v.inner = a if v.kind == "factor" else None
    else:
        v.kind = "num"
    if v.kind == "factor":
        v.levels = _levels_of(_raw(v.inner, df, n))
        v.labels = [_label(x) for x in v.levels]


def _strata_codes(raw, label: str) -> np.ndarray:
    """strata(x): the codes of factor(x) (levels in order: numbers by value, text by sort), -1 for a missing one."""
    return _codes(raw, _levels_of(raw), label)


# --------------------------------------------------------------------------------------------- the terms

@dataclass
class _Terms:
    """What the formula needs to build the model matrix of a fit, or of new data."""
    spec: _Spec
    columns: list           # the names of the columns of the model matrix
    strata_arg: str | None = None

    def design(self, df, n: int | None = None) -> tuple[np.ndarray, list]:
        """The model matrix of ``df`` (n x p, float64; NaN where a value is missing) and its column names."""
        n = len(df) if n is None else n
        sp = self.spec
        vals = {}
        for i, v in enumerate(sp.variables):
            if i == sp.strata:
                continue
            if v.kind == "num":
                vals[i] = np.ascontiguousarray(np.broadcast_to(np.asarray(_arith(v.ast, df, n), np.float64), (n,)))
            else:
                vals[i] = _codes(_raw(v.inner, df, n), v.levels, v.label)
        in_model = set(sp.terms)
        cols: list[np.ndarray] = []
        names: list[str] = []
        for t in sp.terms:
            if sp.strata is not None and t == (sp.strata,):
                continue
            part: list[tuple[str, np.ndarray | None]] = [("", None)]
            for i in t:
                v = sp.variables[i]
                if v.kind == "num":
                    vcols = [(v.label, vals[i])]
                else:
                    margin = tuple(j for j in t if j != i)
                    first = 1 if (not margin or margin in in_model) else 0   # contrasts, or an indicator of each level
                    if len(v.levels) - first < 1:
                        raise ValueError(f"{v.label}: a factor with fewer than 2 levels has no contrasts")
                    c = vals[i]
                    vcols = [(v.label + v.labels[k],
                              np.where(c < 0, np.nan, (c == k).astype(np.float64))) for k in range(first, len(v.levels))]
                part = [((pn + ":" + vn) if pn else vn, vv if pv is None else pv * vv)
                        for vn, vv in vcols for pn, pv in part]
            for nm, arr in part:
                names.append(nm)
                cols.append(arr)
        if not cols:
            return np.empty((n, 0)), []
        return np.column_stack(cols), names


# ------------------------------------------------------------------------------------------------ the fit

@dataclass
class ISSF:
    """A fitted step selection function (the object of ``survival::clogit`` that amt's ``fit_issf`` returns).

    Attributes
    ----------
    coef : pandas.Series
        The coefficients by the names of the model matrix's columns (``log(sl_)``, ``landuse30``,
        ``log(sl_):elevation``...); NaN for a column found collinear with the others (R's NA).
    vcov : pandas.DataFrame
        Their covariance (the inverse of the information; a zero row and column for a NaN coefficient).
    loglik : tuple of float
        The log-likelihood at zero and at the fit.
    iter : int
        The Newton-Raphson iterations.
    n, nevent : int
        The rows and the cases used (after leaving out the rows with a missing value).
    wald, score : float
        The Wald test (``coxph.wtest`` of the coefficients) and the score test of the fit.
    means : pandas.Series
        The values the columns were centred on (0 for a column of only -1, 0 and 1), which the log-RSS needs.
    formula : str
    converged : bool
    """
    coef: pd.Series
    vcov: pd.DataFrame
    loglik: tuple
    iter: int
    n: int
    nevent: int
    wald: float
    score: float
    means: pd.Series
    formula: str
    converged: bool
    n_omitted: int = 0
    _terms: _Terms = field(default=None, repr=False)
    _data: pd.DataFrame | None = field(default=None, repr=False)
    _strata: np.ndarray | None = field(default=None, repr=False)

    @property
    def edf(self) -> int:
        """The coefficients that are not NaN (``extractAIC``'s degrees of freedom)."""
        return int(self.coef.notna().sum())

    @property
    def aic(self) -> float:
        """``extractAIC(fit)[2]``: -2 * the log-likelihood + 2 * edf."""
        return -2 * self.loglik[1] + 2 * self.edf

    def model_matrix(self, new_data) -> pd.DataFrame:
        """The model matrix of ``new_data`` (not centred), the columns of the fit; NaN where a value is missing.

        A factor takes the levels of the fit (a value that is none of them is an error).
        """
        x, names = self._terms.design(new_data)
        return pd.DataFrame(x, columns=names, index=getattr(new_data, "index", None))

    def summary(self) -> pd.DataFrame:
        """``summary(fit)$coefficients``: coef, exp(coef), se, z and the p-value ``pchisq(z^2, 1, lower.tail = FALSE)``."""
        beta = self.coef.to_numpy(np.float64)
        se = _sqrt_diag(self.vcov.to_numpy(np.float64))
        with np.errstate(all="ignore"):
            z = beta / se
            p = _core.move_issf_pchisq(np.ascontiguousarray(z * z), 1.0, False)
        return pd.DataFrame({"coef": beta, "exp_coef": _core.move_issf_fn("exp", beta), "se": se, "z": z, "p": p},
                            index=self.coef.index)

    def confint(self, level: float = 0.95) -> pd.DataFrame:
        """``confint(fit)``: the Wald intervals ``coef +- qnorm(1 - (1 - level) / 2) * se`` (``confint.default``)."""
        if not 0.0 < level < 1.0:
            raise ValueError("level: between 0 and 1")
        a = (1 - level) / 2
        fac = [_core.move_issf_qnorm(a), _core.move_issf_qnorm(1 - a)]
        ses = _sqrt_diag(self.vcov.to_numpy(np.float64))
        beta = self.coef.to_numpy(np.float64)
        return pd.DataFrame({f"{100 * q:.3g} %": beta + ses * f for q, f in zip((a, 1 - a), fac)}, index=self.coef.index)


def _sqrt_diag(v: np.ndarray) -> np.ndarray:
    """sqrt(diag(var)): R takes the square roots of a vector in extended precision, that of one number correctly rounded."""
    d = np.ascontiguousarray(np.diag(v))
    return _core.move_issf_fn("sqrt1" if len(d) == 1 else "sqrt", d)


def fit_issf(data, formula: str, *, strata=None, model: bool = True) -> ISSF:
    """A step selection function by conditional logistic regression (amt's ``fit_issf``, ``fit_ssf``, ``fit_clogit``).

    Fits ``survival::clogit(formula, data)`` as R's survival 3.7-0 does, exact partial likelihood (``method =
    "exact"``, ``coxexact``): the coefficients, covariance, log-likelihoods, number of iterations, tests, AIC and the
    log-RSS of :func:`log_rss` equal R's bit by bit. ``fit_ssf`` and ``fit_clogit`` are this function, as in amt.
    The concordance of ``coxph`` (``summary(fit)`` prints it) is not computed.

    Parameters
    ----------
    data
        A table of the steps and their random steps (:func:`random_steps`, with the covariates of
        :func:`extract_covariates`), one row per step.
    formula
        R's formula in a string, ``"case_ ~ x + log(sl_) + cos(ta_) + x:log(sl_) + I(x^2) + landuse +
        strata(step_id_)"``. The left side is the column of the cases (True / False or 1 / 0). The terms are columns,
        ``log``, ``exp``, ``sqrt``, ``sin``, ``cos``, ``log1p``, ``abs`` and ``I(...)`` of them (``+ - * / ^``),
        ``factor(x)``, and ``a:b``, ``a*b``; ``- 1`` and ``+ 0`` are ignored, as in R (the fit has no intercept
        either way). A column that is not a number is a factor with R's treatment contrasts (the first level, in the
        order of the categories, or sorted, is the reference); an interaction of a factor with a number needs the
        factor's main effect and the number's. ``strata(col)`` gives the strata, one column: the rows with the same
        value are one stratum, in the order of the sorted values (or categories). The ``step_id_`` of
        :func:`random_steps` restarts for each track, so the steps of several tracks share strata unless the ids are
        made unique first (amt fits the column it is given). The order of the terms and the names of the columns are
        R's: the interactions after the main effects, ``elevation:log(sl_)`` as ``log(sl_):elevation`` when
        ``log(sl_)`` came first.
    strata
        The column (name) or the values of the strata, instead of ``strata(...)`` in the formula.
    model
        Keep the rows used (needed by ``log_rss(..., ci="boot")``).

    Returns
    -------
    ISSF
        Its ``coef``, ``vcov``, ``loglik``, ``iter``, ``n``, ``nevent``, ``wald``, ``score``, ``means`` and ``aic``;
        :meth:`ISSF.summary`, :meth:`ISSF.confint` and :meth:`ISSF.model_matrix`.

    Notes
    -----
    Rows with a missing value in any column of the formula (or a transformation of it, such as ``log`` of a
    negative number) are left out, as R's ``na.omit`` does; ``log(0)`` is an error ("an infinite predictor").
    A column whose values are all equal and not 0 / 1 / -1 is an error (R fails in ``scale``); a column of ones, or
    one made of the others, gets a NaN coefficient. The columns are centred and scaled as the fit runs, as R does,
    so ``means`` are the centres. A stratum with several cases is fitted with the exact (Gail et al.) recursions.
    The strata are sorted as R sorts ``factor(x)``: numbers by value, text by code point (R by the locale's
    collation: the order of the sums can differ in the last bit for text with mixed case or accents).

    References
    ----------
    Therneau, T. M. survival: Survival Analysis (R package, 3.7-0); Gail, M. H., Lubin, J. H. & Rubinstein, L. V.
    (1981). Likelihood calculations for matched case-control studies and survival studies with tied death times.
    Biometrika 68, 703-707; Fortin et al. (2005) Ecology 86; Avgar et al. (2016) Methods Ecol. Evol. 7; Signer,
    Fieberg & Avgar (2019) Ecol. Evol. 9.

    Examples
    --------
    >>> import numpy as np, pandas as pd, terraflux as tf
    >>> rng = np.random.default_rng(1)
    >>> d = pd.DataFrame({"step_id_": np.repeat(np.arange(60), 5), "x": rng.normal(size=300)})
    >>> d["case_"] = np.tile([True, False, False, False, False], 60)
    >>> d.loc[d.case_, "x"] += 0.7
    >>> m = tf.move.fit_issf(d, "case_ ~ x + strata(step_id_)")
    >>> m.nevent, m.iter, m.coef.index.tolist()
    (60, 5, ['x'])
    """
    sp = _parse_formula(formula)
    if strata is not None and sp.strata is not None:
        raise ValueError("strata: given and in the formula (strata(...)) too")
    if strata is None and sp.strata is None:
        raise ValueError("No strata is provided, please make sure the formula includes a strata: strata(step_id_)")
    if not isinstance(data, pd.DataFrame):
        data = pd.DataFrame(data)
    return _fit(data, sp, strata, model)


def _fit(data: pd.DataFrame, sp: _Spec, strata, model: bool) -> ISSF:
    n = len(data)
    if n == 0:
        raise ValueError("data: no rows")
    sp = _Spec(sp.text, sp.response, [_Var(v.label, v.ast) for v in sp.variables], sp.terms, sp.strata)
    for i, v in enumerate(sp.variables):
        if i != sp.strata:
            _set_kind(v, data, n)
    terms = _Terms(sp, [])
    x, names = terms.design(data, n)
    if x.shape[1] == 0:
        raise ValueError("the formula has no covariates")
    # the cases: True / False or 0 / 1
    case = _column(data, sp.response)
    if case.dtype.kind == "b":
        status = case.to_numpy(np.float64)
    elif case.dtype.kind in "iuf":
        status = case.to_numpy(np.float64, na_value=np.nan)
        ok = status[~np.isnan(status)]
        if len(ok) and not np.isin(ok, (0.0, 1.0)).all():
            raise ValueError(f"{sp.response}: the cases must be True / False or 0 / 1")
    elif case.dtype == object or case.dtype.name == "boolean":
        status = np.where(case.isna().to_numpy(), np.nan, case.fillna(False).astype(bool).to_numpy().astype(np.float64))
    else:
        raise ValueError(f"{sp.response}: the cases must be True / False or 0 / 1")
    # the strata
    if strata is None:
        sv = sp.variables[sp.strata].ast[2][0]
        sraw, strata_name = _raw(sv, data, n), _deparse(sv)
    elif isinstance(strata, str):
        sraw, strata_name = _column(data, strata), strata
    else:
        sraw, strata_name = pd.Series(np.asarray(strata)), "strata"
        if len(sraw) != n:
            raise ValueError("strata: one value per row")
    scodes = _strata_codes(sraw, f"strata({strata_name})")
    keep = ~(np.isnan(x).any(axis=1) | np.isnan(status) | (scodes < 0))
    if not keep.any():
        raise ValueError("no row without a missing value")
    xk, sk, stk = x[keep], status[keep], scodes[keep]
    if not np.isfinite(xk).all():
        raise ValueError("data contains an infinite predictor (log(0), a division by zero...)")
    if sk.sum() == 0:
        raise ValueError("no case among the rows used")
    xk = np.asfortranarray(xk)
    try:
        r = _core.move_issf_fit(xk, sk.astype(np.uint8), stk.astype(np.int32), _ITER_MAX, _EPS, _TOLER_CHOL, True)
    except ValueError as e:
        const = [nm for j, nm in enumerate(names) if xk[:, j].min() == xk[:, j].max()
                 and not np.isin(xk[0, j], (-1.0, 0.0, 1.0))]
        raise ValueError(f"the column(s) {const or names} have a constant value other than 0, 1 and -1 "
                         f"(R: {e})") from None
    coef = r["coef"]
    p = len(names)
    flag, u, vs, cs = r["flag"], r["u"], r["var_scaled"], r["coef_scaled"]
    if flag == 1000:
        warnings.warn("Ran out of iterations and did not converge", RuntimeWarning, stacklevel=3)
    else:
        infs = np.zeros(p)
        for j in range(p):
            acc = 0.0
            for l in range(p):
                acc = acc + u[l] * vs[l, j]
            infs[j] = abs(acc)
        bad = (infs > _EPS) & (infs > _TOLER_INF * np.abs(cs))
        if bad.any():
            cols = ",".join(str(j + 1) for j in np.flatnonzero(bad))
            warnings.warn(f"Loglik converged before variable {cols}; beta may be infinite.", RuntimeWarning,
                          stacklevel=3)
    var = r["var"]
    nab = ~np.isnan(coef)
    wald = (float(_core.move_issf_wald(np.ascontiguousarray(var[np.ix_(nab, nab)].T), np.ascontiguousarray(coef[nab])))
            if nab.any() else float("nan"))
    fit = ISSF(coef=pd.Series(coef, index=names, name="coef"), vcov=pd.DataFrame(var, index=names, columns=names),
               loglik=(float(r["loglik"][0]), float(r["loglik"][1])), iter=int(r["iter"]), n=int(keep.sum()),
               nevent=int(sk.sum()), wald=wald, score=float(r["score"]), means=pd.Series(r["means"], index=names),
               formula=sp.text, converged=flag != 1000, n_omitted=int(n - keep.sum()))
    terms.columns = names
    fit._terms = terms
    if model:
        used = sorted({c for v in sp.variables for c in _columns_of(v.ast)} | {sp.response}, key=str)
        fit._data = data.loc[keep, [c for c in data.columns if c in used]].reset_index(drop=True)
        fit._strata = np.asarray(sraw)[keep]
    return fit


def _columns_of(node) -> set:
    k = node[0]
    if k == "name":
        return {node[1]}
    if k == "call":
        return set().union(*(_columns_of(a) for a in node[2])) if node[2] else set()
    if k in ("paren", "neg", "pos"):
        return _columns_of(node[1])
    if k == "bin":
        return _columns_of(node[2]) | _columns_of(node[3])
    return set()


fit_ssf = fit_clogit = fit_issf


# -------------------------------------------------------------------------------------------------- log-RSS

def _append_x1(s: str) -> str:
    if s.endswith("_x1"):
        return s
    return s + "x1" if s.endswith("_") else s + "_x1"


def _frame(x, what: str) -> pd.DataFrame:
    if not isinstance(x, pd.DataFrame):
        raise TypeError(f"{what}: a pandas DataFrame of the covariates")
    return x


def _check_levels(fit: ISSF, x1: pd.DataFrame, x2: pd.DataFrame):
    """amt's check_factors: the levels of a factor column in x1 and x2 are those of the fit."""
    for v in fit._terms.spec.variables:
        if v.kind != "factor" or v.ast[0] != "name":
            continue
        for what, x in (("x1", x1), ("x2", x2)):
            col = x[v.ast[1]] if v.ast[1] in x.columns else None
            if col is not None and isinstance(col.dtype, pd.CategoricalDtype) and list(col.cat.categories) != v.levels:
                raise ValueError(f"factor levels do not match.\nError with factor {v.label!r}:\n"
                                 f"  * levels from data do not match levels from {what}.")


def _lp(fit: ISSF, x: pd.DataFrame, what: str) -> np.ndarray:
    """predict(fit, x, type = "lp", reference = "sample") plus ``uncenter``: the centred row times the coefficients, in
    R's order, and then the sum of coef * means."""
    try:
        m, _ = fit._terms.design(x)
    except ValueError as e:
        raise ValueError(f"{what}: {e}") from None
    coef = fit.coef.to_numpy(np.float64)
    means = fit.means.to_numpy(np.float64)
    c0 = np.where(np.isnan(coef), 0.0, coef)
    acc = np.zeros(len(m))
    with np.errstate(all="ignore"):
        for j in range(m.shape[1]):
            acc = acc + (m[:, j] - means[j]) * c0[j]
        prod = coef * means
    prod = np.ascontiguousarray(prod[~np.isnan(prod)])
    unc = _core.move_issf_sum(prod)
    return (acc + 0.0) + unc


def _log_rss_point(fit: ISSF, x1: pd.DataFrame, x2: pd.DataFrame) -> np.ndarray:
    _check_levels(fit, x1, x2)
    return _lp(fit, x1, "x1") - _lp(fit, x2, "x2")[0]


def log_rss(fit: ISSF, x1, x2, *, ci: str | None = None, ci_level: float = 0.95, n_boot: int = 1000,
            seed: int | None = None) -> pd.DataFrame:
    """The log relative selection strength of the covariates ``x1`` against ``x2`` (amt's ``log_rss``).

    ``log_rss = log(w(x1) / w(x2))``: how much more (or less) a step ending where the covariates are ``x1`` is selected
    than one ending where they are ``x2`` (Avgar et al. 2017), from a fit of :func:`fit_issf`. As amt, it is
    ``predict(fit, x1, type = "lp", reference = "sample")`` less that of ``x2``, each plus ``sum(coef * means)``, so the
    values are R's bits (not ``(x1 - x2) . coef``, which differs in the last digits). A NaN coefficient counts as zero.

    Parameters
    ----------
    fit
        An :class:`ISSF`.
    x1
        A DataFrame with the covariates of the formula, one row per point to compare (columns the formula does not
        use may be there; ``case_`` and the strata are not needed).
    x2
        A DataFrame of one row: the reference.
    ci
        ``None`` (no interval), ``"se"`` (the Wald interval from the covariance of the coefficients: the standard
        error of ``(x1 - x2) b``) or ``"boot"`` (a bootstrap, below).
    ci_level
        The confidence level.
    n_boot
        The bootstrap samples.
    seed
        The seed of R's generator (``set.seed(seed)``) for the bootstrap: the draws are R's.

    Returns
    -------
    pandas.DataFrame
        The columns of ``x1`` named as amt names them (``elevation_x1``, ``sl_x1``), ``log_rss`` and, with ``ci``,
        ``lwr`` and ``upr``.

    Notes
    -----
    The bootstrap is amt's: each sample draws, with replacement, as many strata as the data has cases (R's
    ``sample``), refits the model and takes its log-RSS; the interval is ``mle + (q - mean(boot))`` with ``q`` the
    ``(1 - level) / 2`` and ``(1 + level) / 2`` quantiles (type 7) of the samples. A sample whose refit fails (a level of
    a factor missing...) is drawn again (up to 49 times). It differs from amt (0.3.1), which in R: (1) merges a stratum
    drawn twice into one stratum with two cases, so the refit uses the exact recursions of tied deaths and not a
    bootstrap of the strata; here each draw is a stratum of its own (the ids 1, 2, ...), (2) stops with "object 'sl_' not
    found" when the formula has ``log(sl_)`` or any transformation (it refits on the transformed columns of the model
    frame), (3) fails with one row in ``x1``. All three work here; the numbers are those of R with the same
    corrections (``tools/amt_fixed.R``). It needs ``model=True`` in the fit.

    References
    ----------
    Avgar, T., Lele, S. R., Keim, J. L. & Boyce, M. S. (2017). Relative selection strength: quantifying effect size
    in habitat- and step-selection inference. Ecology and Evolution 7, 5322-5330.
    Signer, J., Fieberg, J. & Avgar, T. (2019). Animal movement tools (amt). Ecology and Evolution 9, 880-890.
    Fieberg, J., Signer, J., Smith, B. & Avgar, T. (2021). A 'How to' guide for interpreting parameters in
    habitat-selection analyses. Journal of Animal Ecology 90, 1027-1043.

    Examples
    --------
    >>> import numpy as np, pandas as pd, terraflux as tf
    >>> rng = np.random.default_rng(1)
    >>> d = pd.DataFrame({"step_id_": np.repeat(np.arange(60), 5), "x": rng.normal(size=300)})
    >>> d["case_"] = np.tile([True, False, False, False, False], 60)
    >>> d.loc[d.case_, "x"] += 0.7
    >>> m = tf.move.fit_issf(d, "case_ ~ x + strata(step_id_)")
    >>> r = tf.move.log_rss(m, pd.DataFrame({"x": [0.0, 1.0]}), pd.DataFrame({"x": [0.0]}), ci="se")
    >>> r.columns.tolist()
    ['x_x1', 'log_rss', 'lwr', 'upr']
    """
    if not isinstance(fit, ISSF):
        raise TypeError("fit: an ISSF (tf.move.fit_issf)")
    x1, x2 = _frame(x1, "x1"), _frame(x2, "x2")
    if len(x2) != 1:
        raise ValueError(f"The data.frame provided as 'x2' should have exactly 1 row (it has {len(x2)}).")
    if ci not in (None, "se", "boot"):
        raise ValueError("ci: None, 'se' or 'boot'")
    if ci is not None:
        if not (isinstance(ci_level, (int, float)) and 0.0 <= ci_level <= 1.0):
            raise ValueError("ci_level: a number between 0 and 1")
        if ci == "boot" and (int(n_boot) != n_boot or n_boot < 1):
            raise ValueError("n_boot: a whole number, 1 or more")
        if ci == "boot" and fit._data is None:
            raise ValueError("Please refit your step selection model with model=True: the bootstrap needs the data.")
    lrss = _log_rss_point(fit, x1, x2)
    out = x1.copy()
    out.columns = [_append_x1(str(c)) for c in x1.columns]
    out["log_rss"] = lrss
    if ci == "se":
        m1, _ = fit._terms.design(x1)
        m2, _ = fit._terms.design(x2)
        delta = m1 - m2   # the rows of x1 less the one of x2
        v = fit.vcov.to_numpy(np.float64)
        p = delta.shape[1]
        with np.errstate(all="ignore"):
            dv = np.zeros_like(delta)
            for l in range(p):
                dv = dv + delta[:, [l]] * v[l, :]
            var = np.zeros(len(delta))
            for j in range(p):
                var = var + dv[:, j] * delta[:, j]
        se = _core.move_issf_fn("sqrt1" if len(var) == 1 else "sqrt", np.ascontiguousarray(var))
        zstar = _core.move_issf_qnorm(1 - ((1 - ci_level) / 2))
        out["lwr"] = lrss - zstar * se
        out["upr"] = lrss + zstar * se
    elif ci == "boot":
        lwr, upr = _boot(fit, x1, x2, lrss, ci_level, int(n_boot), seed)
        out["lwr"], out["upr"] = lwr, upr
    return out


def _boot(fit: ISSF, x1, x2, mle: np.ndarray, ci_level: float, n_boot: int, seed):
    from ._ssf import _seed

    data, strata = fit._data, fit._strata
    ids, first = pd.factorize(pd.Series(strata), sort=False)
    n_ids = len(first)
    rows = [np.flatnonzero(ids == k) for k in range(n_ids)]
    case = data[fit._terms.spec.response].to_numpy()
    n_obs = int(np.asarray(case, np.float64).sum())
    seed = _seed(seed)
    chunk = n_boot + 64
    draws = _core.move_issf_sample(seed, n_ids, n_obs, chunk)
    used = 0
    sp = fit._terms.spec
    arr = np.empty((len(x1), n_boot))
    for b in range(n_boot):
        attempts = 0
        while True:
            if used == chunk:
                chunk *= 2
                draws = _core.move_issf_sample(seed, n_ids, n_obs, chunk)
            d = draws[used] - 1
            used += 1
            attempts += 1
            idx = np.concatenate([rows[k] for k in d])
            new_ids = np.repeat(np.arange(1, len(d) + 1), [len(rows[k]) for k in d])
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    refit = _fit(data.iloc[idx].reset_index(drop=True), sp, new_ids, False)
                arr[:, b] = _log_rss_point(refit, x1, x2)
                break
            except ValueError:
                if attempts + 1 == 25:
                    warnings.warn("Having trouble resampling a valid dataset: the bootstrap refits keep failing",
                                  RuntimeWarning, stacklevel=3)
                if attempts + 1 == 50:
                    raise RuntimeError("Bootstrapping failed. The resampled datasets cannot be fitted.") from None
    p_lwr = (1 - ci_level) / 2
    probs = np.array([p_lwr, 1 - p_lwr])
    lwr, upr = np.empty(len(x1)), np.empty(len(x1))
    for i in range(len(x1)):
        q = _core.move_hr_quantile7(np.ascontiguousarray(arr[i]), probs)
        mean = _core.move_hr_mean(np.ascontiguousarray(arr[i]))
        lwr[i] = mle[i] + (q[0] - mean)
        upr[i] = mle[i] + (q[1] - mean)
    return lwr, upr
