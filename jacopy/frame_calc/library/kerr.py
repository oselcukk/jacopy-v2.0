r"""
Kerr metric, rotating vacuum solution.

In Boyer–Lindquist coordinates ``(t, r, θ, φ)``:

.. math::

    ds^2 = -\Big(1 - \frac{2Mr}{\Sigma}\Big) dt^2
         + \frac{\Sigma}{\Delta} dr^2
         + \Sigma\,d\theta^2
         + \sin^2\theta\Big(r^2 + a^2 + \frac{2Mra^2\sin^2\theta}{\Sigma}\Big) d\varphi^2
         - \frac{4Mra\sin^2\theta}{\Sigma}\,dt\,d\varphi,

where

.. math::

    \Sigma := r^2 + a^2\cos^2\theta, \qquad
    \Delta := r^2 - 2Mr + a^2.

Vacuum solution: ``G(∇, g) ≡ 0``. The full pipeline through
:func:`~jacopy.frame_calc.einstein_tensor` is computationally heavy
(60+ seconds typical), running the vacuum check on Kerr is a
deliberate exercise, not a default unit-test target.
"""

from __future__ import annotations

from typing import Optional, Tuple

import sympy as sp

from jacopy.frame_calc.component_tensor import ComponentMetric
from jacopy.frame_calc.frame import CoordinateFrame


def kerr(
    M_sym: Optional[sp.Symbol] = None,
    a_sym: Optional[sp.Symbol] = None,
) -> Tuple[CoordinateFrame, ComponentMetric]:
    r"""Kerr metric in Boyer–Lindquist coordinates.

    Parameters
    ----------
    M_sym
        Mass. If ``None``, fresh ``Symbol("M", positive=True)``.
    a_sym
        Spin parameter. If ``None``, fresh
        ``Symbol("a", real=True)``. Note: Kerr requires ``|a| ≤ M``;
        no constraint is imposed here.

    Returns
    -------
    tuple[CoordinateFrame, ComponentMetric]
        Coordinates ``(t, r, θ, φ)``; the metric has a non-zero
        ``dt dφ`` cross-term encoding frame dragging.

    Notes
    -----
    Setting ``a = 0`` recovers Schwarzschild; that limit can be checked
    via SymPy's ``subs`` on the resulting metric components, e.g.
    ``g[0, 0].subs(a_sym, 0) == -(1 - 2*M_sym/r)``.

    Full vacuum verification (``einstein_tensor.is_vacuum() == True``)
    is symbolically valid but computationally expensive, see
    :func:`~jacopy.frame_calc.library.schwarzschild` for a faster
    test target.
    """
    if M_sym is None:
        M_sym = sp.Symbol("M", positive=True)
    elif not isinstance(M_sym, sp.Symbol):
        raise TypeError("kerr M_sym must be a SymPy Symbol or None")
    if a_sym is None:
        a_sym = sp.Symbol("a", real=True)
    elif not isinstance(a_sym, sp.Symbol):
        raise TypeError("kerr a_sym must be a SymPy Symbol or None")

    t = sp.Symbol("t", real=True)
    r = sp.Symbol("r", positive=True)
    theta = sp.Symbol("theta", positive=True)
    phi = sp.Symbol("phi", real=True)

    Sigma = r ** 2 + a_sym ** 2 * sp.cos(theta) ** 2
    Delta = r ** 2 - 2 * M_sym * r + a_sym ** 2
    sin2 = sp.sin(theta) ** 2

    F = CoordinateFrame([t, r, theta, phi], name="kerr")
    g_tt = -(1 - 2 * M_sym * r / Sigma)
    g_tphi = -2 * M_sym * r * a_sym * sin2 / Sigma
    g_rr = Sigma / Delta
    g_thth = Sigma
    g_phph = sin2 * (
        r ** 2
        + a_sym ** 2
        + 2 * M_sym * r * a_sym ** 2 * sin2 / Sigma
    )
    g = ComponentMetric(
        F,
        sp.Matrix([
            [g_tt,    0,    0,    g_tphi],
            [0,       g_rr, 0,    0],
            [0,       0,    g_thth, 0],
            [g_tphi,  0,    0,    g_phph],
        ]),
    )
    return F, g
