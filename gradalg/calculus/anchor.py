"""
Anchor ``ρ: E → TM``.

For a Lie algebroid ``E`` over a manifold ``M`` the anchor is a
bundle map to the tangent bundle that is *linear* and *compatible
with brackets*:

    ρ([X, Y]_E) = [ρ(X), ρ(Y)]_{TM}.

At the symbolic level ``ρ`` is just a named linear operator that
applies to an operand via :class:`Act` and carries no graded-Leibniz
rule: its degree is ``0`` so the Koszul signs in :mod:`product_rule`
vanish anyway, but more importantly it is not a derivation on its
own algebra — it is a *morphism* between two different algebras.
That is why it lives here rather than as a subclass of a bracket
operator.

The compatibility axiom is not silently assumed. Instead we expose
:func:`bracket_compatibility_obstruction`, which returns the :class:`Expr`

    ρ([X, Y]_E) − [ρ(X), ρ(Y)]_{TM}

that ``simplify`` should reduce to ``0`` when the user has declared
the anchor compatible with the two given brackets. This keeps
compatibility as an explicit theorem-or-axiom choice rather than
hiding it in the type.
"""

from __future__ import annotations

from typing import Optional

from gradalg.algebra.derivation import Act, Derivation
from gradalg.brackets.base import GradedBracket
from gradalg.core.expr import Expr, Neg, Sum
from gradalg.core.registry import PropertyRegistry


class Anchor(Derivation):
    """Anchor morphism ``ρ: E → TM`` — a named degree-0 linear operator.

    Structurally a :class:`Derivation` of degree 0, which buys us the
    existing ``Act`` application machinery and degree-aware Leibniz
    (trivially zero-sign because of the degree). Semantically an
    anchor is *not* a derivation on a single algebra; the Leibniz
    behaviour that :mod:`product_rule` would apply is vacuous on the
    inputs it receives (single vector-field symbols), so no harm
    comes from the shared superclass.

    The constructor accepts an optional ``name`` (defaults to
    ``"ρ"``) and does not require any declaration of the source /
    target algebras at this layer — those live in higher-level
    algebroid objects and only matter when compatibility is being
    proved.
    """

    def __init__(self, name: str = "ρ") -> None:
        super().__init__(name, degree=0)


def bracket_compatibility_obstruction(
    anchor: Anchor,
    bracket_E: GradedBracket,
    bracket_TM: GradedBracket,
    X: Expr,
    Y: Expr,
    registry: Optional[PropertyRegistry] = None,
) -> Expr:
    """Return ``ρ([X, Y]_E) − [ρ(X), ρ(Y)]_{TM}`` as an :class:`Expr`.

    When ``simplify`` reduces the result to ``Integer(0)``, the anchor
    is compatible with the two brackets on the pair ``(X, Y)``. The
    obstruction form mirrors the axiom-obstruction helpers in
    :mod:`gradalg.brackets.base`: expose the *thing that should be
    zero* and let the caller choose how much work to do on it.

    Both brackets are expanded via :meth:`GradedBracket.expand`, so
    any grading declared on ``X`` / ``Y`` that the target bracket
    needs will be pulled from ``registry``.
    """
    lhs_inner = bracket_E.expand(X, Y, registry)
    lhs = Act(anchor, lhs_inner)
    rho_X = Act(anchor, X)
    rho_Y = Act(anchor, Y)
    rhs = bracket_TM.expand(rho_X, rho_Y, registry)
    return Sum(lhs, Neg(rhs))
