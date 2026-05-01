r"""
Curvature scalar invariants, Faz 19 Chunk A.

Coordinate-independent scalars built from Riemann/Ricci by raising
indices via the metric inverse and contracting all of them away. These
are the standard physics measures for distinguishing coordinate
singularities from real ones (Kretschmann at Schwarzschild horizon),
characterising conformal anomalies (Weyl²), and verifying conformal
flatness (Weyl in 4D, Cotton in 3D).

Functions
---------
* :func:`kretschmann`, ``K = R_{abcd} R^{abcd}``
* :func:`ricci_squared`, ``R_{ab} R^{ab}``
* :func:`cotton`, 3D Cotton tensor (vanishes iff conformally flat)

All functions accept a :class:`~jacopy.frame_calc.component_tensor.ComponentTensor`
representing Riemann ``(1, 3)`` or Ricci ``(0, 2)`` plus a
:class:`~jacopy.frame_calc.component_tensor.ComponentMetric`. Riemann's
natural index layout is ``R[a, b, c, d]`` with ``a`` upper and
``(b, c, d)`` lower, matching the output of
:func:`~jacopy.frame_calc.curvature.curvature`. This package's Riemann
has antisymmetry in the **first two lower indices** ``(b, c)``,
non-standard ordering vs. Wald's (last two lower indices). Kretschmann
and Ricci² are robust under any consistent convention because they
fully contract; Weyl/Weyl² are deferred pending convention
reconciliation.

**Sign convention note.** This package's Ricci tensor is sign-opposite
to Wald/Carroll; see ``faz18_progress.md``. The scalar invariants
below are sign-agnostic for *vacuum* spacetimes (Kretschmann), or
quadratic in Ricci so sign-magnitude is preserved. For non-vacuum,
trust the shape but expect a global sign opposite to many textbooks.
"""

from __future__ import annotations

from typing import Any

try:
    import sympy as sp
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "jacopy.frame_calc requires SymPy"
    ) from exc

from jacopy.frame_calc.component_tensor import (
    ComponentMetric,
    ComponentTensor,
)
from jacopy.frame_calc.frame import Frame


# --------------------------------------------------------------------- #
# Helpers                                                                #
# --------------------------------------------------------------------- #


def _lower_first_index(R: ComponentTensor, g: ComponentMetric) -> Any:
    r"""Compute ``R_{abcd} = g_{ae} R^e{}_{bcd}`` as an ``n×n×n×n`` array.

    Returns a SymPy ``MutableDenseNDimArray``.
    """
    n = g.frame.dim
    out = sp.MutableDenseNDimArray.zeros(n, n, n, n)
    for a in range(n):
        for b in range(n):
            for c in range(n):
                for d in range(n):
                    s = sp.S.Zero
                    for e in range(n):
                        s += g[a, e] * R[e, b, c, d]
                    out[a, b, c, d] = s
    return out


def _raise_lower_indices(
    R: ComponentTensor, g: ComponentMetric
) -> Any:
    r"""Raise the three lower indices of ``R^a{}_{bcd}`` via ``g^{-1}``.

    Returns ``R^{abcd} = g^{bf} g^{cg} g^{dh} R^a{}_{fgh}`` as an
    ``n×n×n×n`` array.
    """
    n = g.frame.dim
    g_inv = g.inverse()
    out = sp.MutableDenseNDimArray.zeros(n, n, n, n)
    for a in range(n):
        for b in range(n):
            for c in range(n):
                for d in range(n):
                    s = sp.S.Zero
                    for f in range(n):
                        for h in range(n):
                            for k in range(n):
                                s += (
                                    g_inv[b, f]
                                    * g_inv[c, h]
                                    * g_inv[d, k]
                                    * R[a, f, h, k]
                                )
                    out[a, b, c, d] = s
    return out


def _raise_both_ricci(
    Ric: ComponentTensor, g: ComponentMetric
) -> Any:
    r"""Compute ``Ric^{ab} = g^{ac} g^{bd} Ric_{cd}`` as an ``n×n`` array."""
    n = g.frame.dim
    g_inv = g.inverse()
    out = sp.MutableDenseNDimArray.zeros(n, n)
    for a in range(n):
        for b in range(n):
            s = sp.S.Zero
            for c in range(n):
                for d in range(n):
                    s += g_inv[a, c] * g_inv[b, d] * Ric[c, d]
            out[a, b] = s
    return out


# --------------------------------------------------------------------- #
# Kretschmann scalar                                                     #
# --------------------------------------------------------------------- #


def kretschmann(
    riemann: ComponentTensor,
    g: ComponentMetric,
    *,
    simplify: bool = True,
) -> sp.Expr:
    r"""Kretschmann scalar ``K = R_{abcd} R^{abcd}``.

    Parameters
    ----------
    riemann
        Riemann curvature tensor of signature ``(1, 3)``,
        ``riemann[a, b, c, d] = R^a{}_{bcd}``. Output of
        :func:`~jacopy.frame_calc.curvature.curvature`.
    g
        The :class:`ComponentMetric`.
    simplify
        Whether to apply :func:`sympy.simplify` to the result. Default
        ``True``. For Kerr-class metrics where simplify is the bottleneck,
        pass ``False`` and post-simplify yourself.

    Returns
    -------
    sp.Expr
        The Kretschmann scalar.

    Examples
    --------
    Schwarzschild horizon regularity::

        F, g = schwarzschild()
        R = curvature(levi_civita(g))
        K = kretschmann(R, g)
        assert sp.simplify(K - 48 * M**2 / r**6) == 0
        # K is finite at r=2M (horizon) but → ∞ as r → 0 (real singularity)
    """
    if not isinstance(riemann, ComponentTensor):
        raise TypeError("kretschmann: riemann must be a ComponentTensor")
    if not isinstance(g, ComponentMetric):
        raise TypeError("kretschmann: g must be a ComponentMetric")
    if riemann.signature != (1, 3):
        raise ValueError(
            "kretschmann: riemann signature must be (1, 3); "
            f"got {riemann.signature}"
        )
    if riemann.frame is not g.frame:
        raise ValueError(
            "kretschmann: riemann and g must share the same Frame"
        )

    n = g.frame.dim
    R_lower = _lower_first_index(riemann, g)
    R_upper = _raise_lower_indices(riemann, g)

    K = sp.S.Zero
    for a in range(n):
        for b in range(n):
            for c in range(n):
                for d in range(n):
                    K += R_lower[a, b, c, d] * R_upper[a, b, c, d]

    if simplify:
        K = sp.simplify(K)
    return K


# --------------------------------------------------------------------- #
# Ricci-squared                                                          #
# --------------------------------------------------------------------- #


def ricci_squared(
    ricci_tensor: ComponentTensor,
    g: ComponentMetric,
    *,
    simplify: bool = True,
) -> sp.Expr:
    r"""``Ric_{ab} Ric^{ab}``.

    Vanishes identically for vacuum spacetimes (Schwarzschild, Kerr).
    Used to characterise stress-energy in non-vacuum.
    """
    if not isinstance(ricci_tensor, ComponentTensor):
        raise TypeError(
            "ricci_squared: ricci_tensor must be a ComponentTensor"
        )
    if ricci_tensor.signature != (0, 2):
        raise ValueError(
            "ricci_squared: ricci_tensor signature must be (0, 2)"
        )
    if ricci_tensor.frame is not g.frame:
        raise ValueError(
            "ricci_squared: tensor and g must share the same Frame"
        )

    n = g.frame.dim
    Ric_upper = _raise_both_ricci(ricci_tensor, g)

    s = sp.S.Zero
    for a in range(n):
        for b in range(n):
            s += ricci_tensor[a, b] * Ric_upper[a, b]

    if simplify:
        s = sp.simplify(s)
    return s


# --------------------------------------------------------------------- #
# Cotton tensor (3D)                                                     #
# --------------------------------------------------------------------- #


def cotton(
    connection: Any,
    g: ComponentMetric,
    *,
    simplify: bool = True,
) -> ComponentTensor:
    r"""Cotton tensor ``C_{abc}``, 3D conformal-flatness measure.

    .. math::

        C_{abc} = \nabla_a S_{bc} - \nabla_b S_{ac},
        \qquad
        S_{ab} = R_{ab} - \tfrac{1}{4} R\, g_{ab}.

    Vanishes iff the metric is conformally flat. The 3D analog of the
    Weyl tensor (which vanishes identically in 3D).

    Parameters
    ----------
    connection
        A :class:`~jacopy.frame_calc.component_tensor.ComponentConnection`
        (typically Levi-Civita).
    g
        The :class:`ComponentMetric`. Must satisfy ``g.frame.dim == 3``.

    Returns
    -------
    ComponentTensor
        Rank-3 ``(0, 3)`` tensor antisymmetric in the first two indices.

    Raises
    ------
    ValueError
        If ``g.frame.dim != 3``. The Cotton tensor is only conformally
        invariant (and only useful) in 3D, in higher dimensions the
        Weyl tensor handles conformal flatness.
    """
    from jacopy.frame_calc.ricci import ricci, ricci_scalar

    if g.frame.dim != 3:
        raise ValueError(
            f"cotton: requires dim 3 (got {g.frame.dim}). "
            "In 4D+, use the Weyl tensor for conformal flatness checks."
        )
    if connection.frame is not g.frame:
        raise ValueError("cotton: connection and g must share Frame")

    n = 3
    Ric = ricci(connection)
    R = ricci_scalar(connection, g)
    frame = g.frame

    # Schouten tensor S_{bc} = R_{bc} - (1/4) R g_{bc}  (in 3D)
    S = sp.MutableDenseNDimArray.zeros(n, n)
    for b in range(n):
        for c in range(n):
            S[b, c] = Ric[b, c] - sp.Rational(1, 4) * R * g[b, c]

    # Covariant derivative ∇_a S_{bc}:
    #   ∇_a S_{bc} = e_a(S_{bc}) - Γ^d_{ab} S_{dc} - Γ^d_{ac} S_{bd}
    def _nabla_S(a: int, b: int, c: int) -> Any:
        v = frame.derivative(S[b, c], a)
        for d in range(n):
            v -= connection[d, a, b] * S[d, c]
            v -= connection[d, a, c] * S[b, d]
        return v

    out = sp.MutableDenseNDimArray.zeros(n, n, n)
    for a in range(n):
        for b in range(n):
            for c in range(n):
                val = _nabla_S(a, b, c) - _nabla_S(b, a, c)
                if simplify:
                    try:
                        val = sp.simplify(val)
                    except (TypeError, AttributeError):
                        pass
                out[a, b, c] = val
    return ComponentTensor(frame, out, signature=(0, 3))
