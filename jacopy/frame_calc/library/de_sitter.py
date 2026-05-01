r"""
de Sitter metric, maximally symmetric spacetime with positive cosmological
constant.

In static coordinates ``(t, r, θ, φ)``:

.. math::

    ds^2 = -f(r)\,dt^2 + f(r)^{-1}\,dr^2
         + r^2\,d\theta^2 + r^2 \sin^2\theta\,d\varphi^2,
    \qquad f(r) = 1 - \frac{\Lambda}{3} r^2.

Solution of vacuum Einstein equations with cosmological constant:
``R_{ab} = \Lambda g_{ab}``, ``R = 4\Lambda``,
``G_{ab} + \Lambda g_{ab} = 0``.
"""

from __future__ import annotations

from typing import Optional, Tuple

import sympy as sp

from jacopy.frame_calc.component_tensor import ComponentMetric
from jacopy.frame_calc.frame import CoordinateFrame


def de_sitter(
    Lambda_sym: Optional[sp.Symbol] = None,
) -> Tuple[CoordinateFrame, ComponentMetric]:
    r"""de Sitter metric in static patch coordinates.

    Parameters
    ----------
    Lambda_sym
        Cosmological constant. If ``None``, a fresh
        ``Symbol("Lambda", positive=True)`` is created.

    Returns
    -------
    tuple[CoordinateFrame, ComponentMetric]
        Coordinates ordered ``(t, r, θ, φ)``.

    Notes
    -----
    Cosmological horizon at ``r = √(3/Λ)``. Maximally symmetric:
    constant Ricci tensor ``R_{ab} = Λ g_{ab}``. Used as the late-time
    asymptote of accelerating universes.
    """
    if Lambda_sym is None:
        Lambda_sym = sp.Symbol("Lambda", positive=True)
    elif not isinstance(Lambda_sym, sp.Symbol):
        raise TypeError(
            "de_sitter Lambda_sym must be a SymPy Symbol or None"
        )
    t = sp.Symbol("t", real=True)
    r = sp.Symbol("r", positive=True)
    theta = sp.Symbol("theta", positive=True)
    phi = sp.Symbol("phi", real=True)
    F = CoordinateFrame([t, r, theta, phi], name="de_sitter")
    f = 1 - Lambda_sym * r ** 2 / 3
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
