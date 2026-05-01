r"""
Paper-grade LaTeX table render, Faz 19 Chunk C.5.

Render a :class:`~jacopy.frame_calc.component_tensor.ComponentTensor`
as a LaTeX ``align*`` block (or full document) listing only the
non-zero components. Designed for direct copy-paste into a paper.

.. code-block:: python

    from jacopy.frame_calc import levi_civita
    from jacopy.frame_calc.library import schwarzschild
    from jacopy.frame_calc.latex_table import to_latex_table

    F, g = schwarzschild()
    LC = levi_civita(g)
    print(to_latex_table(LC))
    # \begin{align*}
    # \Gamma^{t}{}_{tr} &= \frac{M}{r(r - 2M)} \\
    # \Gamma^{r}{}_{tt} &= \frac{M(r - 2M)}{r^{3}} \\
    # ...
    # \end{align*}

For Riemann ``(1, 3)`` and Ricci/metric ``(0, 2)`` tensors the row label
adopts the canonical symbol; rank-3 connection components are written
``\Gamma^a{}_{bc}``.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, Optional, Tuple

try:
    import sympy as sp
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "jacopy.frame_calc requires SymPy"
    ) from exc

from jacopy.frame_calc.component_tensor import (
    ComponentConnection,
    ComponentMetric,
    ComponentMetricInverse,
    ComponentTensor,
)
from jacopy.frame_calc.curvature import CurvatureTensor
from jacopy.frame_calc.einstein import EinsteinTensor
from jacopy.frame_calc.levi_civita import LeviCivitaConnection
from jacopy.frame_calc.ricci import RicciTensor


# --------------------------------------------------------------------- #
# Symbol selection                                                       #
# --------------------------------------------------------------------- #


def _default_symbol_for(tensor: ComponentTensor) -> str:
    r"""Return the standard LaTeX letter for a known tensor type."""
    if isinstance(tensor, LeviCivitaConnection):
        return r"\Gamma"
    if isinstance(tensor, ComponentConnection):
        return r"\Gamma"
    if isinstance(tensor, CurvatureTensor):
        return "R"
    if isinstance(tensor, RicciTensor):
        return r"\operatorname{Ric}"
    if isinstance(tensor, EinsteinTensor):
        return "G"
    if isinstance(tensor, ComponentMetric):
        return "g"
    if isinstance(tensor, ComponentMetricInverse):
        return "g"
    return "T"


def _index_label(name: str) -> str:
    r"""Convert a frame index name to its LaTeX form.

    Greek letters get backslashed (``theta`` → ``\theta``); other
    names pass through unchanged. Multi-character names are wrapped
    in braces to render correctly under super/subscript.
    """
    greek = {
        "alpha", "beta", "gamma", "delta", "epsilon", "zeta", "eta",
        "theta", "iota", "kappa", "lambda", "mu", "nu", "xi",
        "omicron", "pi", "rho", "sigma", "tau", "upsilon", "phi",
        "chi", "psi", "omega",
    }
    if name in greek:
        return "\\" + name
    return name


# --------------------------------------------------------------------- #
# Index formatting                                                       #
# --------------------------------------------------------------------- #


def _format_indices(
    upper: Tuple[str, ...], lower: Tuple[str, ...]
) -> str:
    r"""Format a single tensor entry's indices.

    Convention: upper indices first, then lower, with ``{}`` separator
    so they don't visually merge. ``a^{b}{}_{cd}`` rather than
    ``a^b_{cd}`` (which has an ambiguous baseline).

    Greek names are emitted as LaTeX commands (``\theta``); raw
    single-letter names pass through.
    """
    upper_str = "".join(_index_label(u) for u in upper)
    lower_str = "".join(_index_label(l) for l in lower)
    if upper_str and lower_str:
        return f"^{{{upper_str}}}{{}}_{{{lower_str}}}"
    if upper_str:
        return f"^{{{upper_str}}}"
    if lower_str:
        return f"_{{{lower_str}}}"
    return ""


# --------------------------------------------------------------------- #
# Public API                                                             #
# --------------------------------------------------------------------- #


def to_latex_table(
    tensor: ComponentTensor,
    *,
    symbol: Optional[str] = None,
    skip_redundant_symmetric: bool = True,
    full_document: bool = False,
    title: Optional[str] = None,
) -> str:
    r"""Render a :class:`ComponentTensor`'s non-zero components as a
    LaTeX ``align*`` block.

    Parameters
    ----------
    tensor
        The :class:`ComponentTensor` to render.
    symbol
        LaTeX symbol for the tensor. If ``None``, picks a default by
        type (Christoffel → ``\Gamma``, Riemann → ``R``, Ricci →
        ``\operatorname{Ric}``, etc.).
    skip_redundant_symmetric
        When ``True`` (default), for symmetric tensors (``g``, ``Ric``,
        ``G``) and for the Christoffel pair ``(b, c)``, redundant
        permutations are skipped (only ``(a, b)`` with ``a ≤ b``
        listed).
    full_document
        When ``True``, wrap the output in a full standalone LaTeX
        document. Useful for piping directly into a TeX renderer.
    title
        Optional title prepended as a ``\section*{…}`` (only meaningful
        when ``full_document=True``).

    Returns
    -------
    str
        LaTeX source for the table.

    Examples
    --------
    Schwarzschild Christoffels::

        from jacopy.frame_calc import levi_civita
        from jacopy.frame_calc.library import schwarzschild
        from jacopy.frame_calc.latex_table import to_latex_table

        F, g = schwarzschild()
        LC = levi_civita(g)
        print(to_latex_table(LC))
    """
    if not isinstance(tensor, ComponentTensor):
        raise TypeError(
            "to_latex_table: tensor must be a ComponentTensor"
        )

    sym = symbol if symbol is not None else _default_symbol_for(tensor)
    q, r = tensor.signature
    names = tensor.frame.index_names()

    nz = tensor.nonzero_components()
    rows: list[str] = []

    seen: set[Tuple[int, ...]] = set()
    for idx in sorted(nz.keys()):
        # Symmetry pruning
        if skip_redundant_symmetric:
            canonical = _canonical_index(idx, tensor)
            if canonical in seen:
                continue
            seen.add(canonical)

        upper = tuple(names[idx[i]] for i in range(q))
        lower = tuple(names[idx[i]] for i in range(q, q + r))
        idx_str = _format_indices(upper, lower)
        rhs = sp.latex(nz[idx])
        rows.append(f"{sym}{idx_str} &= {rhs}")

    if not rows:
        body = (
            r"\begin{align*}" + "\n"
            + r"\text{(all components zero)}" + "\n"
            + r"\end{align*}"
        )
    else:
        body = (
            r"\begin{align*}" + "\n"
            + " \\\\\n".join(rows) + "\n"
            + r"\end{align*}"
        )

    if not full_document:
        return body

    parts = [
        r"\documentclass{article}",
        r"\usepackage{amsmath, amssymb}",
        r"\begin{document}",
    ]
    if title:
        parts.append(rf"\section*{{{title}}}")
    parts.append(body)
    parts.append(r"\end{document}")
    return "\n".join(parts)


def _canonical_index(
    idx: Tuple[int, ...], tensor: ComponentTensor
) -> Tuple[int, ...]:
    r"""Map an index tuple to its canonical representative under known
    symmetries. Used by :func:`to_latex_table` to avoid listing both
    ``Ric_{ab}`` and ``Ric_{ba}`` (or both ``Γ^a_{bc}`` and
    ``Γ^a_{cb}``).
    """
    # Symmetric (0, 2) tensors: g, Ric, G, sort the two lower indices.
    if isinstance(
        tensor, (ComponentMetric, ComponentMetricInverse, RicciTensor, EinsteinTensor)
    ):
        a, b = idx
        return (min(a, b), max(a, b))
    # ComponentConnection / LeviCivita: torsion-free → symmetric in (b, c).
    if isinstance(tensor, ComponentConnection):
        a, b, c = idx
        return (a, min(b, c), max(b, c))
    return idx
