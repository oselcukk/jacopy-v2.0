r"""
Bianchi cosmologies, homogeneous but anisotropic models.

The Bianchi classification organises spatially homogeneous 3-geometries
by the Lie algebra structure of their isometry group. Three of the
most common types in cosmology:

* **Type I**, flat anisotropic; three independent expansion rates
  ``a(t), b(t), c(t)``. Reduces to flat FRW when ``a = b = c``.
* **Type V**, open anisotropic; like type I but with hyperbolic
  spatial slices.
* **Type IX**, closed anisotropic ("Mixmaster"); spatial slices
  are 3-spheres. Famous for chaotic dynamics near the Big Bang.

All three live on coordinates ``(t, x, y, z)``; the spatial part
encodes the Bianchi structure.
"""

from __future__ import annotations

from typing import Optional, Tuple

import sympy as sp

from jacopy.frame_calc.component_tensor import ComponentMetric
from jacopy.frame_calc.frame import CoordinateFrame


def _make_anisotropic_factors(
    a_func: Optional[sp.Function],
    b_func: Optional[sp.Function],
    c_func: Optional[sp.Function],
    t_sym: sp.Symbol,
) -> Tuple[sp.Expr, sp.Expr, sp.Expr]:
    """Resolve three scale-factor functions, defaulting to ``a(t), b(t), c(t)``."""
    a = a_func if a_func is not None else sp.Function("a")(t_sym)
    b = b_func if b_func is not None else sp.Function("b")(t_sym)
    c = c_func if c_func is not None else sp.Function("c")(t_sym)
    return a, b, c


def bianchi_I(
    a_func: Optional[sp.Function] = None,
    b_func: Optional[sp.Function] = None,
    c_func: Optional[sp.Function] = None,
    *,
    t_sym: Optional[sp.Symbol] = None,
) -> Tuple[CoordinateFrame, ComponentMetric]:
    r"""Bianchi type I, flat anisotropic cosmology.

    Metric: ``ds² = -dt² + a(t)² dx² + b(t)² dy² + c(t)² dz²``.

    Three independent scale factors. Reduces to spatially flat FRW
    when ``a = b = c``. Used as the simplest anisotropic cosmology
    deviation from FRW.
    """
    if t_sym is None:
        t_sym = sp.Symbol("t", real=True)
    elif not isinstance(t_sym, sp.Symbol):
        raise TypeError("bianchi_I t_sym must be a SymPy Symbol or None")
    a, b, c = _make_anisotropic_factors(a_func, b_func, c_func, t_sym)
    x = sp.Symbol("x", real=True)
    y = sp.Symbol("y", real=True)
    z = sp.Symbol("z", real=True)
    F = CoordinateFrame([t_sym, x, y, z], name="bianchi_I")
    g = ComponentMetric(
        F,
        sp.Matrix([
            [-1,    0,         0,         0],
            [0,    a ** 2,     0,         0],
            [0,    0,          b ** 2,    0],
            [0,    0,          0,         c ** 2],
        ]),
    )
    return F, g


def bianchi_V(
    a_func: Optional[sp.Function] = None,
    b_func: Optional[sp.Function] = None,
    c_func: Optional[sp.Function] = None,
    *,
    t_sym: Optional[sp.Symbol] = None,
) -> Tuple[CoordinateFrame, ComponentMetric]:
    r"""Bianchi type V, open anisotropic cosmology.

    Metric: ``ds² = -dt² + a(t)² dx² + e^{2x} (b(t)² dy² + c(t)² dz²)``.

    Spatial slices have negative curvature. Reduces to open FRW
    (``k = -1``) when ``a = b = c`` and the scale factor matches the
    Friedmann equation.
    """
    if t_sym is None:
        t_sym = sp.Symbol("t", real=True)
    elif not isinstance(t_sym, sp.Symbol):
        raise TypeError("bianchi_V t_sym must be a SymPy Symbol or None")
    a, b, c = _make_anisotropic_factors(a_func, b_func, c_func, t_sym)
    x = sp.Symbol("x", real=True)
    y = sp.Symbol("y", real=True)
    z = sp.Symbol("z", real=True)
    F = CoordinateFrame([t_sym, x, y, z], name="bianchi_V")
    e2x = sp.exp(2 * x)
    g = ComponentMetric(
        F,
        sp.Matrix([
            [-1,    0,             0,                 0],
            [0,    a ** 2,         0,                 0],
            [0,    0,              e2x * b ** 2,      0],
            [0,    0,              0,                 e2x * c ** 2],
        ]),
    )
    return F, g


def bianchi_IX(
    a_func: Optional[sp.Function] = None,
    b_func: Optional[sp.Function] = None,
    c_func: Optional[sp.Function] = None,
    *,
    t_sym: Optional[sp.Symbol] = None,
) -> Tuple[CoordinateFrame, ComponentMetric]:
    r"""Bianchi type IX, closed anisotropic cosmology (Mixmaster).

    In Euler-angle coordinates ``(ψ, θ, φ)`` on the 3-sphere, the
    invariant 1-forms are

    .. math::
        \sigma^1 &= -\sin\psi\,d\theta + \cos\psi \sin\theta\,d\varphi, \\
        \sigma^2 &= \cos\psi\,d\theta + \sin\psi \sin\theta\,d\varphi, \\
        \sigma^3 &= d\psi + \cos\theta\,d\varphi.

    Metric: ``ds² = -dt² + a(t)²(σ¹)² + b(t)²(σ²)² + c(t)²(σ³)²``.

    The Mixmaster universe, chaotic oscillations between
    ``a, b, c`` near the singularity (Belinski–Khalatnikov–Lifshitz).
    """
    if t_sym is None:
        t_sym = sp.Symbol("t", real=True)
    elif not isinstance(t_sym, sp.Symbol):
        raise TypeError("bianchi_IX t_sym must be a SymPy Symbol or None")
    a, b, c = _make_anisotropic_factors(a_func, b_func, c_func, t_sym)
    psi = sp.Symbol("psi", real=True)
    theta = sp.Symbol("theta", positive=True)
    phi = sp.Symbol("phi", real=True)
    F = CoordinateFrame([t_sym, psi, theta, phi], name="bianchi_IX")

    # Build the spatial metric from the three invariant 1-forms.
    # σ¹ = -sin(ψ) dθ + cos(ψ) sin(θ) dφ
    # σ² =  cos(ψ) dθ + sin(ψ) sin(θ) dφ
    # σ³ = dψ + cos(θ) dφ
    sin_psi, cos_psi = sp.sin(psi), sp.cos(psi)
    sin_th, cos_th = sp.sin(theta), sp.cos(theta)

    # Components against (dψ, dθ, dφ), each σ^I as a row.
    # σ¹: (0, -sinψ, cosψ sinθ)
    # σ²: (0,  cosψ, sinψ sinθ)
    # σ³: (1, 0,    cosθ)
    sigma_components = [
        (0, -sin_psi, cos_psi * sin_th),
        (0, cos_psi, sin_psi * sin_th),
        (1, 0, cos_th),
    ]
    # Spatial metric h_{ij} = a²σ¹_i σ¹_j + b²σ²_i σ²_j + c²σ³_i σ³_j
    weights = [a ** 2, b ** 2, c ** 2]
    h = [[sp.S.Zero for _ in range(3)] for _ in range(3)]
    for w, sig in zip(weights, sigma_components):
        for i in range(3):
            for j in range(3):
                h[i][j] += w * sig[i] * sig[j]

    g = ComponentMetric(
        F,
        sp.Matrix([
            [-1,        0,                          0,                          0],
            [0,         h[0][0],                    h[0][1],                    h[0][2]],
            [0,         h[1][0],                    h[1][1],                    h[1][2]],
            [0,         h[2][0],                    h[2][1],                    h[2][2]],
        ]).applyfunc(sp.simplify),
    )
    return F, g
