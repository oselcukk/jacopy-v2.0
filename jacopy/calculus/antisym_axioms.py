r"""
Registry-driven antisymmetry, Faz 12.B #11.

Engine rewrite that consumes the
:class:`~jacopy.core.properties.Antisymmetric` registry property.
When the registry declares a binary head antisymmetric (typical case:
a Schouten–Nijenhuis bivector ``π``), the rule canonicalises a pairing
``MultiEval(π, α, β)`` toward repr-sorted args and wraps the swap in
:class:`~jacopy.core.expr.Neg`:

.. math::

    \pi(\alpha, \beta) \;\longrightarrow\; -\pi(\beta, \alpha)
        \qquad \text{when } \mathrm{repr}(\alpha) > \mathrm{repr}(\beta).

Restricted to arity 2: :class:`Antisymmetric` is a binary-operator
property. Higher-arity alternating multilinear forms should be handled
through the ``alternating=True`` flag on :class:`MultiEval` and the
existing :class:`MultiEvalAlternatingNormalDefinition`.

This rule complements the flag-based alternating canonicaliser: it
fires when the bivector was *declared* antisymmetric in the registry
but the surrounding :class:`MultiEval` was constructed with
``alternating=False`` (or the default). After the rewrite the args are
sorted and the cumulative sign sits as a single :class:`Neg` wrapper
that downstream :func:`~jacopy.algorithms.simplify.simplify` collapses.
"""

from __future__ import annotations

from typing import Optional

from jacopy.core.expr import Expr, Neg
from jacopy.core.multi_eval import MultiEval
from jacopy.core.properties import Antisymmetric
from jacopy.core.registry import PropertyRegistry
from jacopy.proof.expansion import Definition


class RegistryAntiSymCanonicalDefinition(Definition):
    r"""Canonical-order swap on ``MultiEval`` whose head is registry-antisymmetric.

    Fires on ``MultiEval(head, α, β)`` (arity 2) when:

    * the registry has :class:`Antisymmetric` declared on ``head``, and
    * ``repr(α) > repr(β)``, i.e. the args are out of canonical order.

    Rewrites to ``Neg(MultiEval(head, β, α, …))`` preserving the
    ``alternating`` and ``slot_kind`` flags. Termination: each fire
    sorts the only out-of-order pair, so the rule applies at most once
    per node.

    Construction takes the registry as a keyword:

    .. code-block:: python

        engine = ExpansionEngine([
            RegistryAntiSymCanonicalDefinition(registry=reg),
            …
        ])

    With ``registry=None`` (the default), :meth:`matches` always
    returns ``False``, dormant.
    """

    name = "Antisymmetric (registry): π(α, β) = -π(β, α)"

    def __init__(self, *, registry: Optional[PropertyRegistry] = None) -> None:
        self._registry = registry

    def matches(self, expr: Expr) -> bool:
        if self._registry is None:
            return False
        if not isinstance(expr, MultiEval):
            return False
        if len(expr.args) != 2:
            return False
        if not self._registry.has(expr.head, Antisymmetric):
            return False
        return repr(expr.args[0]) > repr(expr.args[1])

    def rewrite(self, expr: Expr) -> Expr:
        a, b = expr.args
        return Neg(
            MultiEval(
                expr.head,
                b,
                a,
                alternating=expr.alternating,
                slot_kind=expr.slot_kind,
            )
        )
