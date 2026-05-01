r"""
Reissner–Nordström metric, static, spherically symmetric charged black hole.

In Schwarzschild-like coordinates ``(t, r, θ, φ)`` with mass ``M`` and
electric charge ``Q``:

.. math::

    ds^2 = -f(r)\,dt^2 + f(r)^{-1}\,dr^2
         + r^2\,d\theta^2 + r^2 \sin^2\theta\,d\varphi^2,
    \qquad f(r) = 1 - \frac{2M}{r} + \frac{Q^2}{r^2}.

**Not vacuum**, the Einstein tensor matches the stress-energy of the
electromagnetic field of a point charge:
``G_{ab} = 8\pi T^{(EM)}_{ab}``. Reduces to Schwarzschild when ``Q → 0``.
"""

from __future__ import annotations

from typing import Optional, Tuple

import sympy as sp

from jacopy.frame_calc.component_tensor import ComponentMetric
from jacopy.frame_calc.frame import CoordinateFrame


def reissner_nordstrom(
    M_sym: Optional[sp.Symbol] = None,
    Q_sym: Optional[sp.Symbol] = None,
) -> Tuple[CoordinateFrame, ComponentMetric]:
    r"""Reissner–Nordström metric in static spherical coordinates.

    Parameters
    ----------
    M_sym
        Mass parameter. If ``None``, a fresh
        ``Symbol("M", positive=True)`` is created.
    Q_sym
        Electric charge parameter. If ``None``, a fresh
        ``Symbol("Q", real=True)`` is created (charge can be either
        sign; squared in the metric).

    Returns
    -------
    tuple[CoordinateFrame, ComponentMetric]
        Coordinates ordered ``(t, r, θ, φ)``.

    Notes
    -----
    The horizons are at ``r_± = M ± √(M² − Q²)``. Extremal at
    ``|Q| = M``; naked singularity for ``|Q| > M``. Reduces to
    Schwarzschild when ``Q = 0``.
    """
    if M_sym is None:
        M_sym = sp.Symbol("M", positive=True)
    elif not isinstance(M_sym, sp.Symbol):
        raise TypeError(
            "reissner_nordstrom M_sym must be a SymPy Symbol or None"
        )
    if Q_sym is None:
        Q_sym = sp.Symbol("Q", real=True)
    elif not isinstance(Q_sym, sp.Symbol):
        raise TypeError(
            "reissner_nordstrom Q_sym must be a SymPy Symbol or None"
        )
    t = sp.Symbol("t", real=True)
    r = sp.Symbol("r", positive=True)
    theta = sp.Symbol("theta", positive=True)
    phi = sp.Symbol("phi", real=True)
    F = CoordinateFrame([t, r, theta, phi], name="reissner_nordstrom")
    f = 1 - 2 * M_sym / r + Q_sym ** 2 / r ** 2
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
