"""
Pattern recognizers, pure shape inspection, no rewriting.

A :class:`Recognizer` answers a single question: "is this expression
an instance of the pattern I care about, and if so, what are its
pieces?". Recognizers don't rewrite, don't consult definitions, and
don't build proof steps, they only decompose an expression into the
structural fields a strategy or verifier-API entry point will hand
off to an actual proof tactic.

Splitting recognition from rewriting keeps the proof-layer
vocabulary composable. ``prove_antisymmetry`` wants to look at an
expression and answer "which A and B go into the commutator?"
*before* deciding which strategy to invoke. :class:`PatternGuided`
dispatches on pattern kind without the recognizers themselves
knowing what strategy will handle the hit. And a human caller can
ask a recognizer directly when they just want to confirm that an
expression has the shape they think it does.

Every recognizer exposes :meth:`recognize`, returning either a
pattern-specific :class:`Match` dataclass or ``None``. No exceptions
, a non-match is not an error condition.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

from jacopy.algebra.commutator import Commutator
from jacopy.algebra.derivation import Act
from jacopy.core.expr import Expr, Product


# --------------------------------------------------------------------- #
# Commutator                                                             #
# --------------------------------------------------------------------- #


@dataclass(frozen=True)
class CommutatorMatch:
    """Successful :class:`CommutatorRecognizer` result."""

    a: Expr
    b: Expr


class CommutatorRecognizer:
    """Match a bare :class:`Commutator` node and extract ``(a, b)``."""

    name = "commutator"

    def recognize(self, expr: Expr) -> Optional[CommutatorMatch]:
        if isinstance(expr, Commutator):
            return CommutatorMatch(a=expr.a, b=expr.b)
        return None


# --------------------------------------------------------------------- #
# Leibniz LHS                                                            #
# --------------------------------------------------------------------- #


@dataclass(frozen=True)
class LeibnizMatch:
    """Successful :class:`LeibnizRecognizer` result.

    ``op`` is the derivation being applied; ``factors`` are the
    operands of the inner :class:`Product` in order. ``factors`` is
    always length ≥ 2, a single-factor product would not exercise
    graded Leibniz.
    """

    op: Expr
    factors: Tuple[Expr, ...]


class LeibnizRecognizer:
    """Match ``Act(D, Product(a, b, …))``, the LHS of a graded Leibniz rule.

    The recognizer does *not* consult :func:`degree_of` or the
    registry; whether ``D`` actually behaves as a graded derivation is
    a semantic question for the strategy that consumes the match.
    This keeps recognition cheap and registry-independent.
    """

    name = "leibniz"

    def recognize(self, expr: Expr) -> Optional[LeibnizMatch]:
        if not isinstance(expr, Act):
            return None
        arg = expr.arg
        if not isinstance(arg, Product):
            return None
        factors = arg.children
        if len(factors) < 2:
            return None
        return LeibnizMatch(op=expr.op, factors=factors)


# --------------------------------------------------------------------- #
# Antisymmetry                                                           #
# --------------------------------------------------------------------- #


@dataclass(frozen=True)
class AntisymmetryMatch:
    """Successful :class:`AntisymmetryRecognizer` result.

    ``a`` and ``b`` are the commutator operands in the order they
    appear. An antisymmetry claim is a statement *about* the
    commutator, asking whether it swaps to a sign-correct form, so
    the recognizer returns the operands and leaves the sign
    computation to :mod:`jacopy.algebra.commutator`.
    """

    a: Expr
    b: Expr


class AntisymmetryRecognizer:
    """Match a :class:`Commutator` node posed as an antisymmetry claim.

    Structurally identical to :class:`CommutatorRecognizer` today,
    both entry points return on the same syntactic shape. The two
    recognizers are separate classes because the *intent* differs:
    :class:`CommutatorRecognizer` feeds decomposition tactics that
    expand a commutator to its signed :class:`Sum`;
    :class:`AntisymmetryRecognizer` feeds the verifier's
    antisymmetry entry point, which will later also accept
    :class:`Sum`-of-commutator forms (``[A, B] + sign·[B, A] == 0``).
    Keeping the class split now avoids a disruptive rename when that
    broader pattern lands.
    """

    name = "antisymmetry"

    def recognize(self, expr: Expr) -> Optional[AntisymmetryMatch]:
        if isinstance(expr, Commutator):
            return AntisymmetryMatch(a=expr.a, b=expr.b)
        return None
