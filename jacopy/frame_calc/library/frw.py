r"""
Friedmann–Robertson–Walker (FRW) metric, homogeneous isotropic cosmology.

In comoving coordinates ``(t, r, θ, φ)``:

.. math::

    ds^2 = -dt^2 + a(t)^2\!\left[\frac{dr^2}{1 - k r^2}
           + r^2\,d\theta^2 + r^2\sin^2\theta\,d\varphi^2\right]

with curvature parameter ``k ∈ {-1, 0, +1}`` (open / flat / closed)
and scale factor ``a(t)``. The metric is **not** vacuum, its
Einstein tensor encodes the Friedmann equations once a stress-energy
``T`` is supplied.
"""

from __future__ import annotations

from typing import Optional, Tuple

import sympy as sp

from jacopy.frame_calc.component_tensor import ComponentMetric
from jacopy.frame_calc.frame import CoordinateFrame


def frw(
    a_func: Optional[sp.Function] = None,
    *,
    k: int = 0,
    t_sym: Optional[sp.Symbol] = None,
) -> Tuple[CoordinateFrame, ComponentMetric]:
    r"""Friedmann–Robertson–Walker metric in comoving coordinates.

    Parameters
    ----------
    a_func
        Scale factor as a SymPy applied function ``a(t)``. If ``None``,
        a fresh ``Function("a")(t_sym)`` is constructed. Pass an
        explicit applied function when you want to specialise to a
        specific evolution (e.g. ``a(t) = t**(2/3)`` for matter-dominated).
    k
        Curvature parameter, must be ``-1``, ``0``, or ``+1``.
        Default ``0`` (spatially flat).
    t_sym
        Time symbol. If ``None``, a fresh ``Symbol("t", real=True)`` is
        created. Pass explicitly to share with other expressions.

    Returns
    -------
    tuple[CoordinateFrame, ComponentMetric]
        Coordinates ordered ``(t, r, θ, φ)``. The metric carries
        explicit ``a(t)`` dependence so :class:`~jacopy.frame_calc.CoordinateFrame`'s
        :meth:`~jacopy.frame_calc.frame.CoordinateFrame.derivative` will
        produce ``a'(t)`` (i.e. ``Derivative(a(t), t)``) when
        Christoffels are computed.

    Notes
    -----
    For ``k = 0``: ``ds² = -dt² + a(t)²(dr² + r² dΩ²)``, flat space
    expanded by ``a(t)``.

    For ``k = ±1``: the radial term takes the curved form
    ``dr² / (1 − k r²)``.
    """
    if k not in (-1, 0, 1):
        raise ValueError(f"frw k must be -1, 0, or +1; got {k!r}")
    if t_sym is None:
        t_sym = sp.Symbol("t", real=True)
    elif not isinstance(t_sym, sp.Symbol):
        raise TypeError("frw t_sym must be a SymPy Symbol or None")
    if a_func is None:
        a = sp.Function("a")(t_sym)
    else:
        a = a_func
    r = sp.Symbol("r", positive=True)
    theta = sp.Symbol("theta", positive=True)
    phi = sp.Symbol("phi", real=True)
    F = CoordinateFrame([t_sym, r, theta, phi], name=f"frw(k={k})")
    g_rr = a ** 2 / (1 - k * r ** 2)
    g_thth = a ** 2 * r ** 2
    g_phph = a ** 2 * r ** 2 * sp.sin(theta) ** 2
    g = ComponentMetric(
        F,
        sp.Matrix([
            [-1,    0,       0,            0],
            [0,    g_rr,     0,            0],
            [0,    0,       g_thth,        0],
            [0,    0,       0,            g_phph],
        ]),
    )
    return F, g
