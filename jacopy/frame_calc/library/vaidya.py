r"""
Vaidya metric, radiating / accreting black hole with time-dependent mass.

In ingoing Eddington–Finkelstein null coordinates ``(v, r, θ, φ)`` with
mass function ``M(v)``:

.. math::

    ds^2 = -\!\left(1 - \tfrac{2 M(v)}{r}\right) dv^2
         + 2\,dv\,dr
         + r^2\,d\theta^2 + r^2 \sin^2\theta\,d\varphi^2.

The off-diagonal ``g_{vr} = 1`` is the hallmark of null coordinates.
Models a black hole emitting (or absorbing) null radiation; reduces to
ingoing-EF Schwarzschild when ``M(v) = const``. The Einstein tensor's
single non-zero component encodes the inflowing null energy:
``G_{vv} = 2 \dot{M}(v) / r^2``.
"""

from __future__ import annotations

from typing import Optional, Tuple

import sympy as sp

from jacopy.frame_calc.component_tensor import ComponentMetric
from jacopy.frame_calc.frame import CoordinateFrame


def vaidya(
    M_func: Optional[sp.Function] = None,
    *,
    v_sym: Optional[sp.Symbol] = None,
) -> Tuple[CoordinateFrame, ComponentMetric]:
    r"""Vaidya metric in ingoing Eddington–Finkelstein coordinates.

    Parameters
    ----------
    M_func
        Mass as a SymPy applied function ``M(v)``. If ``None``, a
        fresh ``Function("M")(v_sym)`` is constructed. Pass an
        explicit applied function to specialise (e.g. ``M(v) = v``
        for linear accretion).
    v_sym
        Advanced null coordinate symbol. If ``None``, a fresh
        ``Symbol("v", real=True)`` is created.

    Returns
    -------
    tuple[CoordinateFrame, ComponentMetric]
        Coordinates ordered ``(v, r, θ, φ)``. Note: ``v`` is the
        advanced null coordinate, not the usual time ``t``.

    Notes
    -----
    Reduces to ingoing-EF Schwarzschild when ``M(v) = M`` constant.
    The metric is **non-vacuum** for non-constant ``M(v)``, the
    Einstein tensor encodes the incoming null radiation flux.
    """
    if v_sym is None:
        v_sym = sp.Symbol("v", real=True)
    elif not isinstance(v_sym, sp.Symbol):
        raise TypeError("vaidya v_sym must be a SymPy Symbol or None")
    if M_func is None:
        M_func = sp.Function("M")(v_sym)
    r = sp.Symbol("r", positive=True)
    theta = sp.Symbol("theta", positive=True)
    phi = sp.Symbol("phi", real=True)
    F = CoordinateFrame([v_sym, r, theta, phi], name="vaidya")
    factor = 1 - 2 * M_func / r
    g = ComponentMetric(
        F,
        sp.Matrix([
            [-factor,         1,           0,                          0],
            [1,               0,           0,                          0],
            [0,               0,           r ** 2,                     0],
            [0,               0,           0,    r ** 2 * sp.sin(theta) ** 2],
        ]),
    )
    return F, g
