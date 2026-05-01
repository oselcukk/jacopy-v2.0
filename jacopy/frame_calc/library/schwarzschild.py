r"""
Schwarzschild metric, static, spherically symmetric vacuum solution.

In Schwarzschild coordinates ``(t, r, θ, φ)``:

.. math::

    ds^2 = -\left(1 - \tfrac{2M}{r}\right) dt^2
         + \left(1 - \tfrac{2M}{r}\right)^{-1} dr^2
         + r^2\,d\theta^2
         + r^2 \sin^2\theta\,d\varphi^2.

Vacuum solution of Einstein's field equations:
``G(∇, g)_{ab} ≡ 0``. The :func:`schwarzschild`-built metric
satisfies this identically through
:func:`jacopy.frame_calc.einstein_tensor`'s
:meth:`~jacopy.frame_calc.einstein.EinsteinTensor.is_vacuum` check.
"""

from __future__ import annotations

from typing import Optional, Tuple

import sympy as sp

from jacopy.frame_calc.component_tensor import ComponentMetric
from jacopy.frame_calc.frame import CoordinateFrame


def schwarzschild(
    M_sym: Optional[sp.Symbol] = None,
) -> Tuple[CoordinateFrame, ComponentMetric]:
    r"""Schwarzschild metric in Schwarzschild coordinates.

    Parameters
    ----------
    M_sym
        Mass parameter. If ``None``, a fresh
        ``Symbol("M", positive=True)`` is created. Pass an explicit
        symbol when you need to substitute a numeric value or
        compose with other expressions.

    Returns
    -------
    tuple[CoordinateFrame, ComponentMetric]
        Coordinates ordered ``(t, r, θ, φ)``; the frame's
        :meth:`~jacopy.frame_calc.frame.Frame.index_names` exposes
        these symbol names for display.

    Notes
    -----
    The metric has a coordinate singularity at ``r = 2M`` (the event
    horizon) and a curvature singularity at ``r = 0``. The
    :func:`schwarzschild`-built metric uses the symbols ``r > 0`` and
    ``M > 0`` declared positive so SymPy simplifies expressions in
    the exterior region.
    """
    if M_sym is None:
        M_sym = sp.Symbol("M", positive=True)
    elif not isinstance(M_sym, sp.Symbol):
        raise TypeError(
            "schwarzschild M_sym must be a SymPy Symbol or None"
        )
    t = sp.Symbol("t", real=True)
    r = sp.Symbol("r", positive=True)
    theta = sp.Symbol("theta", positive=True)
    phi = sp.Symbol("phi", real=True)
    F = CoordinateFrame([t, r, theta, phi], name="schwarzschild")
    factor = 1 - 2 * M_sym / r
    g = ComponentMetric(
        F,
        sp.Matrix([
            [-factor,         0,           0,                          0],
            [0,               1 / factor,  0,                          0],
            [0,               0,           r ** 2,                     0],
            [0,               0,           0,    r ** 2 * sp.sin(theta) ** 2],
        ]),
    )
    return F, g
