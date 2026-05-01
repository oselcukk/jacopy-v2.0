r"""
Gödel metric, rotating dust universe with closed timelike curves.

In Gödel's original coordinates ``(t, x, y, z)``:

.. math::

    ds^2 = -dt^2 - 2 e^{\omega x}\,dt\,dy
         + dx^2 - \tfrac{1}{2} e^{2 \omega x}\,dy^2 + dz^2.

Solution of Einstein's equations with rotating dust + cosmological
constant. Famous for admitting **closed timelike curves** through every
point, used as a benchmark for causality-violation arguments.

Discovered by Kurt Gödel (1949). The rotation parameter ``ω`` sets the
angular velocity scale.
"""

from __future__ import annotations

from typing import Optional, Tuple

import sympy as sp

from jacopy.frame_calc.component_tensor import ComponentMetric
from jacopy.frame_calc.frame import CoordinateFrame


def godel(
    omega_sym: Optional[sp.Symbol] = None,
) -> Tuple[CoordinateFrame, ComponentMetric]:
    r"""Gödel metric in original coordinates.

    Parameters
    ----------
    omega_sym
        Rotation parameter. If ``None``, a fresh
        ``Symbol("omega", positive=True)`` is created.

    Returns
    -------
    tuple[CoordinateFrame, ComponentMetric]
        Coordinates ordered ``(t, x, y, z)``.

    Notes
    -----
    The off-diagonal ``g_{ty} = -e^{\omega x}`` is what allows the
    closed timelike curves: a circle at large ``y`` becomes timelike
    once ``g_{yy} < 0``, which happens at finite ``y``-distance.

    Solves Einstein's equations with stress-energy of pressure-less
    rotating dust plus a cosmological constant
    ``Λ = -ω²/2`` and dust energy density ``ρ = ω² / 8πG``.
    """
    if omega_sym is None:
        omega_sym = sp.Symbol("omega", positive=True)
    elif not isinstance(omega_sym, sp.Symbol):
        raise TypeError("godel omega_sym must be a SymPy Symbol or None")
    t = sp.Symbol("t", real=True)
    x = sp.Symbol("x", real=True)
    y = sp.Symbol("y", real=True)
    z = sp.Symbol("z", real=True)
    F = CoordinateFrame([t, x, y, z], name="godel")
    e_wx = sp.exp(omega_sym * x)
    e_2wx = sp.exp(2 * omega_sym * x)
    g = ComponentMetric(
        F,
        sp.Matrix([
            [-1,            0,    -e_wx,                  0],
            [0,             1,    0,                      0],
            [-e_wx,         0,    -sp.Rational(1, 2) * e_2wx,    0],
            [0,             0,    0,                      1],
        ]),
    )
    return F, g
