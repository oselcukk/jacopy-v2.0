r"""
Non-degeneracy axiom — Faz 12.B #9.

Engine rewrite that consumes the
:class:`~jacopy.core.properties.NonDegenerate` registry property: a
``Sum`` shaped like ``ι_Y ω − ι_Z ω`` collapses to ``Y − Z`` whenever
``ω`` carries the ``NonDegenerate`` flag. The rewrite is the
contrapositive of injectivity:

* mathematically, ``ω`` non-degenerate ⇔ the bundle map
  ``X ↦ ι_X ω: TM → T*M`` is injective;
* injectivity of a linear map equates kernel-only to ``0``, so the
  difference ``ι_Y ω − ι_Z ω = 0`` forces ``Y − Z = 0``;
* the rule peels off the common ``ι_(·) ω`` shell, leaving the
  vector-field difference for the rest of the engine to canonicalize.

The rule is registry-aware (mirrors
:class:`~jacopy.calculus.closed_axioms.ClosedFormDefinition`): the
registry is supplied at construction time, the rule queries it on
``matches``. Passing ``registry=None`` makes the rule a permanent
no-op — useful for engines that don't have a registry context.
"""

from __future__ import annotations

from typing import Optional, Tuple

from jacopy.algebra.derivation import Act
from jacopy.calculus.interior import InteriorProduct
from jacopy.core.expr import Expr, Neg, Sum
from jacopy.core.properties import NonDegenerate
from jacopy.core.registry import PropertyRegistry
from jacopy.proof.expansion import Definition


def _ip_target(expr: Expr) -> Optional[Tuple[Expr, Expr, bool]]:
    """Decompose ``expr`` as ``(Y, ω, negated)`` for an interior-product term.

    Returns ``None`` when ``expr`` is not of shape ``ι_Y ω`` or
    ``Neg(ι_Y ω)`` (or further nested negations of either).
    """
    if isinstance(expr, Neg):
        sub = _ip_target(expr.arg)
        if sub is None:
            return None
        Y, omega, negated = sub
        return (Y, omega, not negated)
    if not isinstance(expr, Act):
        return None
    if not isinstance(expr.op, InteriorProduct):
        return None
    return (expr.op.vector_field, expr.arg, False)


class NonDegenerateInteriorEqualityDefinition(Definition):
    r"""``ι_Y ω − ι_Z ω → Y − Z`` whenever ``ω`` is :class:`NonDegenerate`.

    Fires on a two-term :class:`Sum` whose children, modulo a
    :class:`Neg` wrapper, are interior products of the *same* form
    ``ω`` against vector fields with opposite signs (i.e. a positive
    ``ι_Y ω`` and a negated ``Neg(ι_Z ω)``). The rewrite peels off the
    ``ι_(·) ω`` layer:

    .. code-block:: text

        Sum(Act(ι_Y, ω), Neg(Act(ι_Z, ω)))   →   Sum(Y, Neg(Z))

    With ``registry=None`` (the default) :meth:`matches` always
    returns ``False``. Callers who want the rule active must pass a
    registry that has at least one form declared
    :class:`NonDegenerate`.

    The rule scopes through the registry rather than per-instance:
    every form flagged ``NonDegenerate`` participates. Symplectic
    problems auto-declare the flag on their ``ω``, so a single
    registration on the engine suffices for every interior-product
    equality involving a non-degenerate form in the same proof.
    """

    name = "ι_Y ω − ι_Z ω = 0 ⇒ Y − Z = 0 [NonDegenerate(ω)]"

    def __init__(self, *, registry: Optional[PropertyRegistry] = None) -> None:
        self._registry = registry

    def _decompose(
        self, expr: Expr
    ) -> Optional[Tuple[Expr, Expr, Expr]]:
        """Return ``(Y, Z, ω)`` if ``expr`` matches the difference shape."""
        if self._registry is None:
            return None
        if not isinstance(expr, Sum):
            return None
        if len(expr.children) != 2:
            return None
        a, b = expr.children
        ta = _ip_target(a)
        tb = _ip_target(b)
        if ta is None or tb is None:
            return None
        Ya, omega_a, neg_a = ta
        Yb, omega_b, neg_b = tb
        if omega_a != omega_b:
            return None
        if not self._registry.has(omega_a, NonDegenerate):
            return None
        # Need exactly one positive and one negated child.
        if neg_a == neg_b:
            return None
        if neg_a:
            # b is the positive ι_Y ω, a is the negated ι_Z ω.
            return (Yb, Ya, omega_a)
        return (Ya, Yb, omega_a)

    def matches(self, expr: Expr) -> bool:
        return self._decompose(expr) is not None

    def rewrite(self, expr: Expr) -> Expr:
        decomposed = self._decompose(expr)
        if decomposed is None:
            return expr
        Y, Z, _omega = decomposed
        return Sum(Y, Neg(Z))
