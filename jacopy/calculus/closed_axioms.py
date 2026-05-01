r"""
Closed-form axiom, Faz 12.B #7.

Engine rewrite that consumes the :class:`~jacopy.core.properties.Closed`
registry property: any ``Act(d, ω)`` where the registry declares
``ω`` closed collapses to ``Integer(0)``. Mirrors the 2a/2b notebook
pattern of inline ``DOmegaClosed`` :class:`Definition` classes,
elevating it to a reusable engine rule.

The rule is registry-aware (mirrors
:class:`~jacopy.proof.expansion.IotaOnZeroFormDefinition` and friends):
the registry is supplied at construction time, the rule queries it on
``matches``. Passing ``registry=None`` makes the rule a permanent
no-op, useful for engines that don't have a registry context.
"""

from __future__ import annotations

from typing import Optional

from jacopy.algebra.derivation import Act
from jacopy.calculus.exterior_d import ExteriorDerivative
from jacopy.core.expr import Expr, Integer
from jacopy.core.properties import Closed
from jacopy.core.registry import PropertyRegistry
from jacopy.proof.expansion import Definition


class ClosedFormDefinition(Definition):
    r"""``d(ω) → 0`` whenever the registry declares ``ω`` :class:`Closed`.

    Fires on ``Act(d, ω)`` (the :class:`ExteriorDerivative` outermost
    head) when :meth:`PropertyRegistry.has` reports a :class:`Closed`
    property on ``ω``. The rewrite is unconditional: a closed form's
    exterior derivative is zero by definition of closedness.

    Construction takes the registry by keyword:

    .. code-block:: python

        engine = ExpansionEngine([
            ClosedFormDefinition(registry=reg),
            …
        ])

    With ``registry=None`` (the default), :meth:`matches` always
    returns ``False``, the rule is dormant. Callers who want this
    rule active must explicitly pass a registry.
    """

    name = "Closed: d(ω) = 0 when ω is declared closed"

    def __init__(self, *, registry: Optional[PropertyRegistry] = None) -> None:
        self._registry = registry

    def matches(self, expr: Expr) -> bool:
        if self._registry is None:
            return False
        if not isinstance(expr, Act):
            return False
        if not isinstance(expr.op, ExteriorDerivative):
            return False
        return self._registry.has(expr.arg, Closed)

    def rewrite(self, expr: Expr) -> Expr:
        return Integer(0)
