r"""
Einstein tensor, Stage F.

The Einstein tensor of a connection ``∇`` and a metric ``g`` is the
``(0, 2)`` tensor

.. math::

    G(\nabla, g)_{ab}
    = \operatorname{Ric}(\nabla)_{ab}
      - \tfrac{1}{2}\,\mathcal{R}(\nabla, g)\,g_{ab}.

The Einstein field equations of general relativity (in geometric
units) read ``G = T``, where ``T`` is the stress-energy-momentum
tensor. A vacuum solution is one where ``G_{ab} ≡ 0`` for every
``(a, b)``, Schwarzschild is the canonical example, and the
:func:`einstein_tensor`-built tensor on Schwarzschild satisfies
:meth:`~jacopy.frame_calc.component_tensor.ComponentTensor.is_zero`
identically (after :func:`sympy.simplify` / :func:`sympy.trigsimp`).
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
    ComponentConnection,
    ComponentMetric,
    ComponentTensor,
)
from jacopy.frame_calc.ricci import (
    RicciTensor,
    ricci,
    ricci_scalar_from_ricci,
)


# --------------------------------------------------------------------- #
# EinsteinTensor                                                        #
# --------------------------------------------------------------------- #


class EinsteinTensor(ComponentTensor):
    r"""``(0, 2)`` symmetric Einstein tensor ``G_{ab}``."""

    __slots__ = ()

    def __init__(self, frame: Any, components: Any) -> None:
        super().__init__(frame, components, signature=(0, 2))

    def matrix(self) -> sp.Matrix:
        """Return the Einstein tensor as a SymPy ``Matrix``."""
        return sp.Matrix(self._components.tolist())

    def is_vacuum(self, *, simplify: bool = True) -> bool:
        """Convenience alias for :meth:`is_zero`.

        Vanishing of the Einstein tensor is the vacuum-solution
        condition of general relativity (equivalent to ``G = 0`` in
        the field equations). Reads more naturally at the call site
        than ``einstein_tensor(LC, g).is_zero()``.
        """
        return self.is_zero(simplify=simplify)

    def _rebuild_from_array(
        self, new_arr: sp.MutableDenseNDimArray
    ) -> "EinsteinTensor":
        out = object.__new__(EinsteinTensor)
        ComponentTensor.__init__(
            out, self._frame, new_arr, signature=(0, 2)
        )
        return out


# --------------------------------------------------------------------- #
# Factories                                                              #
# --------------------------------------------------------------------- #


def einstein_tensor(
    connection: ComponentConnection,
    g: ComponentMetric,
    *,
    optimized: bool = False,
) -> EinsteinTensor:
    r"""Compute the Einstein tensor from a connection and a metric.

    Runs the full pipeline ``connection → curvature → Ricci →
    Ricci scalar → G_{ab}`` end-to-end. For repeated computations
    where the Ricci tensor is already available, prefer
    :func:`einstein_from_ricci`. When ``optimized=True``, every
    intermediate stage skips per-entry simplify; the final ``G``
    components are stored in raw form and the user can apply
    :func:`sympy.simplify` (or
    :meth:`~jacopy.frame_calc.component_tensor.ComponentTensor.simplify`)
    at access time.
    """
    if not isinstance(connection, ComponentConnection):
        raise TypeError(
            "einstein_tensor expects a ComponentConnection, got "
            f"{type(connection).__name__}"
        )
    if not isinstance(g, ComponentMetric):
        raise TypeError(
            "einstein_tensor expects a ComponentMetric, got "
            f"{type(g).__name__}"
        )
    Ric = ricci(connection, optimized=optimized)
    return einstein_from_ricci(Ric, g, optimized=optimized)


def einstein_from_ricci(
    Ric: RicciTensor,
    g: ComponentMetric,
    *,
    optimized: bool = False,
) -> EinsteinTensor:
    r"""``G_{ab} = Ric_{ab} - ½ R g_{ab}`` from existing Ricci and metric."""
    if not isinstance(Ric, RicciTensor):
        raise TypeError(
            "einstein_from_ricci expects a RicciTensor, got "
            f"{type(Ric).__name__}"
        )
    if not isinstance(g, ComponentMetric):
        raise TypeError(
            "einstein_from_ricci expects a ComponentMetric, got "
            f"{type(g).__name__}"
        )
    if Ric.frame != g.frame:
        raise ValueError(
            "einstein_from_ricci: Ric and g must share a frame"
        )

    R_scalar = ricci_scalar_from_ricci(Ric, g, optimized=optimized)
    n = Ric.frame.dim
    components = sp.MutableDenseNDimArray.zeros(n, n)
    for a in range(n):
        for b in range(n):
            value = (
                Ric[a, b] - sp.Rational(1, 2) * R_scalar * g[a, b]
            )
            if not optimized:
                try:
                    value = sp.simplify(value)
                except (TypeError, AttributeError):
                    pass
            components[a, b] = value
    return EinsteinTensor(Ric.frame, components)
