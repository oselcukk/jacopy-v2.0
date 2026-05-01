"""
Operator R-linearity axioms, supplementary expansion rules.

Two engine-level rewrite rules that distribute :class:`LieDerivative`
and :class:`ExteriorDerivative` over :class:`Sum` (and :class:`Neg`)
in their *argument* slot:

* :class:`LieDerivativeArgLinearityDefinition`,
  ``L_X(a + b + …) → L_X(a) + L_X(b) + …``
* :class:`ExteriorDerivativeLinearityDefinition`,
  ``d(a + b + …) → d(a) + d(b) + …``

Both operators are :math:`\\mathbb{R}`-linear by definition; the rules
make that linearity *visible* to the engine's bottom-up walk so a
fully-expanded LHS surfaces all of its terms as Sum-children rather
than burying them inside nested ``Act(L_X, Sum(…))`` /
``Act(d, Sum(…))`` blocks. Without these, a Koszul-bracket cyclic
Jacobi LHS that should expand to 27 syntactic terms collapses to 15
top-level children with 12 terms still hidden inside three
``L_X(Sum)`` and three ``d(Sum)`` blocks. With them, the same LHS
emerges as a flat 27-term Sum that the downstream folding axioms
(:class:`OpCommutatorVfDefinition`,
:class:`SnBivectorFormulaDefinition`) can pattern-match against.

Both rules are deliberately complementary to the existing
*field-slot* :math:`L_X` linearity (built ad-hoc in the 2f-deep
notebook): together they let the engine fully traverse compound
derivative expressions without requiring callers to hand-fold inner
Sum's first.
"""

from __future__ import annotations

from jacopy.algebra.derivation import Act
from jacopy.calculus.exterior_d import ExteriorDerivative
from jacopy.calculus.lie_derivative import LieDerivative
from jacopy.core.expr import Expr, Neg, Sum
from jacopy.proof.expansion import Definition


def _distribute_over_sum_or_neg(op: Expr, arg: Expr) -> Expr:
    """Build ``Sum(Act(op, c) for c in arg)`` (or ``Neg(Act(op, x))``).

    Centralised so the two rules below stay structurally identical,
    the only thing that varies is which operator class they target.
    """
    if isinstance(arg, Neg):
        return Neg(Act(op, arg.arg))
    terms = []
    for c in arg.children:
        if isinstance(c, Neg):
            terms.append(Neg(Act(op, c.arg)))
        else:
            terms.append(Act(op, c))
    return Sum.make(*terms)


# --------------------------------------------------------------------- #
# L_X(Sum) → Sum(L_X)                                                    #
# --------------------------------------------------------------------- #


class LieDerivativeArgLinearityDefinition(Definition):
    """``L_X(a + b + …) → L_X(a) + L_X(b) + …``, Lie derivative is
    R-linear in its argument.

    Fires on ``Act(LieDerivative, Sum)`` and ``Act(LieDerivative,
    Neg)``. Works for both ``"cartan"`` and ``"flow"`` mode Lie
    derivatives, the linearity holds independently of the defining
    formula, since :math:`L_X` is a derivation in either presentation.

    Scope is intentionally broad: any :class:`LieDerivative` whose
    argument is a :class:`Sum` (or :class:`Neg`) gets distributed.
    Pair this with field-slot linearity (e.g. an :math:`L_{X+Y} =
    L_X + L_Y` rule) for a fully-traversable Lie-derivative tree.
    """

    name = "L_X R-linearity in arg"

    def matches(self, expr: Expr) -> bool:
        return (
            isinstance(expr, Act)
            and isinstance(expr.op, LieDerivative)
            and isinstance(expr.arg, (Sum, Neg))
        )

    def rewrite(self, expr: Expr) -> Expr:
        return _distribute_over_sum_or_neg(expr.op, expr.arg)


# --------------------------------------------------------------------- #
# d(Sum) → Sum(d)                                                        #
# --------------------------------------------------------------------- #


class ExteriorDerivativeLinearityDefinition(Definition):
    """``d(a + b + …) → d(a) + d(b) + …``, exterior derivative is
    R-linear.

    Fires on ``Act(ExteriorDerivative, Sum)`` and
    ``Act(ExteriorDerivative, Neg)``. Like Lie-derivative linearity,
    this is part of the operator's *definition* (a graded derivation
    of degree :math:`+1`) and its absence in the engine merely keeps
    a fully-distributed expansion from surfacing as flat Sum-children.
    """

    name = "d R-linearity"

    def matches(self, expr: Expr) -> bool:
        return (
            isinstance(expr, Act)
            and isinstance(expr.op, ExteriorDerivative)
            and isinstance(expr.arg, (Sum, Neg))
        )

    def rewrite(self, expr: Expr) -> Expr:
        return _distribute_over_sum_or_neg(expr.op, expr.arg)
