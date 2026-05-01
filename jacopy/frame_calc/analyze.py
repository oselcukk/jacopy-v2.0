r"""
One-shot ``analyze_metric`` helper, Faz 19 Chunk C.4.

Composes the 6-line standard workflow

.. code-block:: python

    F = CoordinateFrame(coords)
    g = ComponentMetric(F, matrix)
    LC = levi_civita(g)
    R = curvature(LC)
    Ric = ricci(LC)
    G = einstein_tensor(LC, g)

into a single :func:`analyze_metric` call returning a dict of all the
useful results plus :func:`~jacopy.frame_calc.kretschmann` for
singularity characterization. Drop a metric matrix in, get every
standard scalar / tensor out, useful for one-off testing where the
full workflow is overkill to retype.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence

try:
    import sympy as sp
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "jacopy.frame_calc requires SymPy"
    ) from exc

from jacopy.frame_calc.component_tensor import ComponentMetric
from jacopy.frame_calc.curvature import curvature
from jacopy.frame_calc.einstein import einstein_tensor
from jacopy.frame_calc.frame import CoordinateFrame
from jacopy.frame_calc.invariants import kretschmann, ricci_squared
from jacopy.frame_calc.levi_civita import levi_civita
from jacopy.frame_calc.ricci import ricci, ricci_scalar


def analyze_metric(
    matrix: Any,
    coords: Sequence[sp.Symbol],
    *,
    optimized: bool = False,
    name: Optional[str] = None,
) -> Dict[str, Any]:
    r"""Build a coordinate frame + metric, run the full pipeline, return
    every standard derived quantity.

    Parameters
    ----------
    matrix
        SymPy ``Matrix`` (or anything :class:`ComponentMetric` accepts).
    coords
        Sequence of SymPy ``Symbol`` s for the coordinates. Length must
        match the matrix's dimension.
    optimized
        Pass-through to :func:`levi_civita`,
        :func:`curvature`, :func:`ricci`,
        :func:`einstein_tensor`. Set ``True`` for Kerr-class metrics
        where mid-formula simplify dominates runtime.
    name
        Optional name for the constructed :class:`CoordinateFrame`.

    Returns
    -------
    dict
        Keys: ``frame``, ``metric``, ``inverse``, ``christoffel``,
        ``riemann``, ``ricci``, ``ricci_scalar``, ``einstein``,
        ``kretschmann``, ``ricci_squared``, ``is_vacuum``.

    Examples
    --------
    Schwarzschild on the fly::

        t, r, th, ph = sp.symbols("t r theta phi")
        M = sp.Symbol("M", positive=True)
        f = 1 - 2*M/r
        out = analyze_metric(
            sp.diag(-f, 1/f, r**2, r**2 * sp.sin(th)**2),
            [t, r, th, ph],
        )
        print(out['kretschmann'])    # 48 M^2 / r^6
        print(out['is_vacuum'])      # True
    """
    F = CoordinateFrame(list(coords), name=name) if name else CoordinateFrame(list(coords))
    g = ComponentMetric(F, matrix)

    LC = levi_civita(g, optimized=optimized)
    R = curvature(LC, optimized=optimized)
    Ric = ricci(LC, optimized=optimized)
    R_scalar = ricci_scalar(LC, g, optimized=optimized)
    G = einstein_tensor(LC, g, optimized=optimized)

    K = kretschmann(R, g, simplify=not optimized)
    Ric_sq = ricci_squared(Ric, g, simplify=not optimized)

    return {
        "frame": F,
        "metric": g,
        "inverse": g.inverse(),
        "christoffel": LC,
        "riemann": R,
        "ricci": Ric,
        "ricci_scalar": R_scalar,
        "einstein": G,
        "kretschmann": K,
        "ricci_squared": Ric_sq,
        "is_vacuum": G.is_vacuum(),
    }
