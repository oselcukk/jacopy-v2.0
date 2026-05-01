r"""
Custom-connection helpers, Faz 19 Chunk D.

Tutorial 25 demonstrates these patterns by hand; this module collapses
them into one-line factories. Each takes a metric ``g`` and a deformation
parameter (torsion tensor, 1-form for Weyl, etc.) and returns the full
:class:`~jacopy.frame_calc.component_tensor.ComponentConnection` ready
to feed into :func:`~jacopy.frame_calc.curvature.curvature` /
:func:`~jacopy.frame_calc.einstein.einstein_tensor`.

Functions
---------
* :func:`connection_with_torsion`, Levi-Civita + contorsion(T)
  (Einstein-Cartan)
* :func:`weyl_connection`, Levi-Civita + Weyl non-metricity from a
  1-form ``W``
* :func:`projective_connection`, Levi-Civita + projective deformation
  from a 1-form ``X`` (geodesic-preserving)

All three return a :class:`ComponentConnection` with the same shape as
:func:`~jacopy.frame_calc.levi_civita.levi_civita`'s output, so they
plug directly into the rest of the pipeline.
"""

from __future__ import annotations

from typing import Any, Optional, Sequence

try:
    import sympy as sp
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "jacopy.frame_calc requires SymPy"
    ) from exc

from jacopy.frame_calc.component_tensor import (
    ComponentConnection,
    ComponentMetric,
    ComponentTensor,
)
from jacopy.frame_calc.levi_civita import levi_civita


# --------------------------------------------------------------------- #
# Einstein-Cartan: connection with prescribed torsion                    #
# --------------------------------------------------------------------- #


def connection_with_torsion(
    g: ComponentMetric,
    T: ComponentTensor,
    *,
    optimized: bool = False,
) -> ComponentConnection:
    r"""Levi-Civita ``+`` contorsion of a prescribed torsion tensor.

    Given a metric ``g`` and a torsion tensor ``T^a_{bc}`` (antisymmetric
    in ``(b, c)``), the metric-compatible connection with that torsion is

    .. math::

        \tilde\Gamma^a{}_{bc} = \Gamma^a{}_{bc}(g) + K^a{}_{bc}(T),

    where ``Γ`` is Levi-Civita and the **contorsion**

    .. math::

        K^a{}_{bc} = \tfrac{1}{2} g^{ad}(T_{dbc} - T_{bcd} + T_{cdb}),
        \qquad T_{abc} = g_{ae} T^e{}_{bc}.

    This is determined uniquely by demanding the deformed connection
    is metric-compatible (``∇̃g = 0``) and has torsion equal to ``T``.

    Parameters
    ----------
    g
        Metric.
    T
        Torsion ``(1, 2)`` tensor; antisymmetry in ``(b, c)`` is *not*
        verified here, pass a tensor that already satisfies it.
    optimized
        Forwarded to the Levi-Civita base.

    Returns
    -------
    ComponentConnection
        The full Einstein-Cartan connection. Verifying
        :func:`~jacopy.frame_calc.torsion.torsion` on the result reproduces
        ``T``.
    """
    if not isinstance(g, ComponentMetric):
        raise TypeError("connection_with_torsion: g must be a ComponentMetric")
    if not isinstance(T, ComponentTensor):
        raise TypeError("connection_with_torsion: T must be a ComponentTensor")
    if T.signature != (1, 2):
        raise ValueError(
            f"connection_with_torsion: T signature must be (1, 2); got {T.signature}"
        )
    if T.frame is not g.frame:
        raise ValueError("connection_with_torsion: T and g must share Frame")

    n = g.frame.dim
    LC = levi_civita(g, optimized=optimized)
    g_inv = g.inverse()

    # Lower the upper index of T: T_{αβγ} where α is the lowered position.
    # T_lowered(α, β, γ) = sum_e g[α, e] * T^e_{βγ}.
    def T_lowered(alpha: int, beta: int, gamma: int) -> Any:
        s = sp.S.Zero
        for e in range(n):
            s += g[alpha, e] * T[e, beta, gamma]
        return s

    christoffel = sp.MutableDenseNDimArray.zeros(n, n, n)
    for a in range(n):
        for b in range(n):
            for c in range(n):
                # K^a_{bc} = (1/2) g^{ad} (T_{dbc} - T_{bcd} + T_{cdb})
                K = sp.S.Zero
                for d in range(n):
                    K += g_inv[a, d] * (
                        T_lowered(d, b, c)
                        - T_lowered(b, c, d)
                        + T_lowered(c, d, b)
                    )
                K = sp.Rational(1, 2) * K
                christoffel[a, b, c] = LC[a, b, c] + K

    return ComponentConnection(g.frame, christoffel)


# --------------------------------------------------------------------- #
# Weyl non-metricity                                                    #
# --------------------------------------------------------------------- #


def weyl_connection(
    g: ComponentMetric,
    W: Sequence[Any],
    *,
    optimized: bool = False,
) -> ComponentConnection:
    r"""Levi-Civita ``+`` Weyl deformation from a 1-form ``W_a``.

    A Weyl connection has non-metricity ``∇_a g_{bc} = -2 W_a g_{bc}``
    (the "Weyl gauge"). Its components are

    .. math::

        \tilde\Gamma^a{}_{bc} = \Gamma^a{}_{bc}(g)
                              + \delta^a_b W_c + \delta^a_c W_b
                              - g_{bc} W^a,
        \qquad W^a = g^{ad} W_d.

    Parameters
    ----------
    g
        Metric.
    W
        1-form components ``W_a`` as a sequence of SymPy expressions
        of length ``g.frame.dim``.
    optimized
        Forwarded to the Levi-Civita base.

    Returns
    -------
    ComponentConnection
        The Weyl-deformed connection.
    """
    if not isinstance(g, ComponentMetric):
        raise TypeError("weyl_connection: g must be a ComponentMetric")
    n = g.frame.dim
    W_components = list(W)
    if len(W_components) != n:
        raise ValueError(
            f"weyl_connection: W must have length {n}; got {len(W_components)}"
        )

    LC = levi_civita(g, optimized=optimized)
    g_inv = g.inverse()

    # W^a = g^{ad} W_d
    def W_upper(a: int) -> Any:
        s = sp.S.Zero
        for d in range(n):
            s += g_inv[a, d] * W_components[d]
        return s

    christoffel = sp.MutableDenseNDimArray.zeros(n, n, n)
    for a in range(n):
        Wa = W_upper(a)
        for b in range(n):
            for c in range(n):
                deformation = sp.S.Zero
                if a == b:
                    deformation += W_components[c]
                if a == c:
                    deformation += W_components[b]
                deformation -= g[b, c] * Wa
                christoffel[a, b, c] = LC[a, b, c] + deformation

    return ComponentConnection(g.frame, christoffel)


# --------------------------------------------------------------------- #
# Projective deformation                                                 #
# --------------------------------------------------------------------- #


def projective_connection(
    g: ComponentMetric,
    X: Sequence[Any],
    *,
    optimized: bool = False,
) -> ComponentConnection:
    r"""Levi-Civita ``+`` projective deformation from a 1-form ``X_a``.

    Projective deformations preserve **unparametrized geodesics**: any
    path geodesic for ``Γ`` remains a path geodesic for the deformed
    ``\tilde\Gamma`` (only the affine parameter rescales). The
    component-level form:

    .. math::

        \tilde\Gamma^a{}_{bc} = \Gamma^a{}_{bc} + \delta^a_b X_c
                              + \delta^a_c X_b.

    Parameters
    ----------
    g
        Metric (used for the Levi-Civita base; the deformation itself
        only uses the frame).
    X
        1-form components ``X_a`` of length ``g.frame.dim``.
    optimized
        Forwarded to the Levi-Civita base.

    Returns
    -------
    ComponentConnection
        The projective-deformed connection. Geodesics of this connection
        coincide as paths with those of ``g``'s Levi-Civita; the affine
        parameter shifts.
    """
    if not isinstance(g, ComponentMetric):
        raise TypeError("projective_connection: g must be a ComponentMetric")
    n = g.frame.dim
    X_components = list(X)
    if len(X_components) != n:
        raise ValueError(
            f"projective_connection: X must have length {n}; got {len(X_components)}"
        )

    LC = levi_civita(g, optimized=optimized)

    christoffel = sp.MutableDenseNDimArray.zeros(n, n, n)
    for a in range(n):
        for b in range(n):
            for c in range(n):
                deformation = sp.S.Zero
                if a == b:
                    deformation += X_components[c]
                if a == c:
                    deformation += X_components[b]
                christoffel[a, b, c] = LC[a, b, c] + deformation

    return ComponentConnection(g.frame, christoffel)
