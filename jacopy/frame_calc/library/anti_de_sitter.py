r"""
Anti–de Sitter metric, maximally symmetric spacetime with negative
cosmological constant.

In static coordinates ``(t, r, θ, φ)``:

.. math::

    ds^2 = -f(r)\,dt^2 + f(r)^{-1}\,dr^2
         + r^2\,d\theta^2 + r^2 \sin^2\theta\,d\varphi^2,
    \qquad f(r) = 1 + \frac{|\Lambda|}{3} r^2.

Solution of vacuum Einstein equations with negative cosmological
constant: ``R_{ab} = -|\Lambda| g_{ab}``. Famous for AdS/CFT
correspondence, boundary CFT lives at conformal infinity.
"""

from __future__ import annotations

from typing import Optional, Tuple

import sympy as sp

from jacopy.frame_calc.component_tensor import ComponentMetric
from jacopy.frame_calc.frame import CoordinateFrame


def anti_de_sitter(
    Lambda_sym: Optional[sp.Symbol] = None,
) -> Tuple[CoordinateFrame, ComponentMetric]:
    r"""Anti–de Sitter metric in static coordinates.

    Parameters
    ----------
    Lambda_sym
        Cosmological constant magnitude. If ``None``, a fresh
        ``Symbol("Lambda", positive=True)`` is created, the metric
        applies the negative sign internally.

    Returns
    -------
    tuple[CoordinateFrame, ComponentMetric]
        Coordinates ordered ``(t, r, θ, φ)``.

    Notes
    -----
    No horizon, ``f(r) > 0`` everywhere. The conformal boundary at
    ``r → ∞`` is where the dual CFT lives in AdS/CFT.
    """
    if Lambda_sym is None:
        Lambda_sym = sp.Symbol("Lambda", positive=True)
    elif not isinstance(Lambda_sym, sp.Symbol):
        raise TypeError(
            "anti_de_sitter Lambda_sym must be a SymPy Symbol or None"
        )
    t = sp.Symbol("t", real=True)
    r = sp.Symbol("r", positive=True)
    theta = sp.Symbol("theta", positive=True)
    phi = sp.Symbol("phi", real=True)
    F = CoordinateFrame([t, r, theta, phi], name="anti_de_sitter")
    f = 1 + Lambda_sym * r ** 2 / 3
    g = ComponentMetric(
        F,
        sp.Matrix([
            [-f,              0,           0,                          0],
            [0,               1 / f,       0,                          0],
            [0,               0,           r ** 2,                     0],
            [0,               0,           0,    r ** 2 * sp.sin(theta) ** 2],
        ]),
    )
    return F, g
