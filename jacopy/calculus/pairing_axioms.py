"""
Pairing axioms — Faz 13.B.

Two engine-level rewrite rules that promote :class:`Pairing` from a
structurally inert two-arg node to a participating shape in
derived-bracket proofs:

* :class:`PairingLinearityDefinition` — bilinear distribution over
  :class:`Sum` in either slot. Distributes the first sum encountered;
  the engine fix-point loop reapplies the rule until both slots are
  fully expanded.
* :class:`PairingLieLeibnizDefinition` —
  ``L_X⟨α, Y⟩ → ⟨L_X α, Y⟩ + ⟨α, L_X Y⟩``. Pairing is a bilinear scalar
  (degree 0), so any degree-0 Lie derivative satisfies this Leibniz.
  Restricting the rule to :class:`LieDerivative` keeps the semantics
  unambiguous — Sharp/Flat acting on a Pairing has no geometric
  meaning, and a generic match would fire on those shapes.

Both rules are needed by the 2f-deep notebook: the cyclic Koszul
Jacobi sum produces nested Pairing-on-Sum and ``L_{ρa}⟨ρb, c⟩`` shapes
that no current engine rule rewrites.
"""

from __future__ import annotations

from jacopy.algebra.derivation import Act
from jacopy.calculus.lie_derivative import LieDerivative
from jacopy.calculus.pairing import Pairing
from jacopy.core.expr import Expr, Neg, Sum
from jacopy.proof.expansion import Definition


# --------------------------------------------------------------------- #
# Axiom 3 — Pairing R-linearity in either slot                           #
# --------------------------------------------------------------------- #


class PairingLinearityDefinition(Definition):
    """``⟨A+B, X⟩ → ⟨A,X⟩+⟨B,X⟩``, ``⟨α, X+Y⟩ → ⟨α,X⟩+⟨α,Y⟩``,
    ``⟨−A, X⟩ → −⟨A, X⟩``, ``⟨α, −X⟩ → −⟨α, X⟩``.

    Single Definition covering both slots and both shapes (Sum and
    Neg) — the engine's bottom-up walk visits the same Pairing
    repeatedly until neither slot holds a Sum/Neg, so one rule
    handles nested distributions and sign extractions without needing
    multiple separate rules.

    Distribution order: alpha-slot Sum first → X-slot Sum →
    alpha-slot Neg → X-slot Neg. Sums expand before Negs because
    Sum-distribution surfaces fresh Pairings that the engine's next
    pass then unwraps.
    """

    name = "Pairing R-linearity"

    def matches(self, expr: Expr) -> bool:
        return isinstance(expr, Pairing) and (
            isinstance(expr.alpha, Sum)
            or isinstance(expr.X, Sum)
            or isinstance(expr.alpha, Neg)
            or isinstance(expr.X, Neg)
        )

    def rewrite(self, expr: Expr) -> Expr:
        # Sum branch — alpha first, engine reapplies on X next pass.
        if isinstance(expr.alpha, Sum):
            return Sum.make(
                *(Pairing(c, expr.X) for c in expr.alpha.children)
            )
        if isinstance(expr.X, Sum):
            return Sum.make(
                *(Pairing(expr.alpha, c) for c in expr.X.children)
            )
        # Neg branch — extract the sign so collect_terms can cancel
        # ``⟨α, A⟩ + ⟨α, −A⟩`` once both pairings reach matching shape.
        if isinstance(expr.alpha, Neg):
            return Neg(Pairing(expr.alpha.arg, expr.X))
        return Neg(Pairing(expr.alpha, expr.X.arg))


# --------------------------------------------------------------------- #
# Axiom 4 — Pairing-Lie Leibniz                                          #
# --------------------------------------------------------------------- #


class PairingLieLeibnizDefinition(Definition):
    """``L_X⟨α, Y⟩ → ⟨L_X α, Y⟩ + ⟨α, L_X Y⟩`` — Lie Leibniz on Pairing.

    Restricted to :class:`LieDerivative` operators because the bilinear
    Leibniz is meaningful exactly for degree-0 vector field actions on
    the scalar produced by the pairing. A generic match on any
    :class:`~jacopy.algebra.derivation.Derivation` would fire on
    Sharp, Flat, and other tensorial maps where the rewrite makes no
    geometric sense.
    """

    name = "Pairing-Lie Leibniz"

    def matches(self, expr: Expr) -> bool:
        return (
            isinstance(expr, Act)
            and isinstance(expr.op, LieDerivative)
            and isinstance(expr.arg, Pairing)
        )

    def rewrite(self, expr: Expr) -> Expr:
        L = expr.op
        p = expr.arg
        return Sum(
            Pairing(Act(L, p.alpha), p.X),
            Pairing(p.alpha, Act(L, p.X)),
        )
