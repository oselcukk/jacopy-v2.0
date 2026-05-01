"""Tests for Faz 13.B, Pairing R-linearity and Pairing-Lie Leibniz."""

from jacopy.algebra.derivation import Act
from jacopy.algorithms.simplify import simplify
from jacopy.calculus.exterior_d import d
from jacopy.calculus.lie_derivative import lie_derivative
from jacopy.calculus.musical import sharp
from jacopy.calculus.pairing import Pairing, pairing
from jacopy.calculus.pairing_axioms import (
    PairingLieLeibnizDefinition,
    PairingLinearityDefinition,
)
from jacopy.core.expr import Sum, Symbol
from jacopy.proof.expansion import ExpansionEngine


# --------------------------------------------------------------------- #
# PairingLinearityDefinition (axiom 3)                                   #
# --------------------------------------------------------------------- #


class TestPairingLinearityMatches:
    def test_matches_sum_in_alpha_slot(self):
        a, b, X = Symbol("a"), Symbol("b"), Symbol("X")
        rule = PairingLinearityDefinition()
        assert rule.matches(Pairing(Sum(a, b), X))

    def test_matches_sum_in_X_slot(self):
        alpha, X, Y = Symbol("α"), Symbol("X"), Symbol("Y")
        rule = PairingLinearityDefinition()
        assert rule.matches(Pairing(alpha, Sum(X, Y)))

    def test_matches_sum_in_both_slots(self):
        a, b, X, Y = Symbol("a"), Symbol("b"), Symbol("X"), Symbol("Y")
        rule = PairingLinearityDefinition()
        assert rule.matches(Pairing(Sum(a, b), Sum(X, Y)))

    def test_no_match_on_atomic_pairing(self):
        rule = PairingLinearityDefinition()
        assert not rule.matches(Pairing(Symbol("α"), Symbol("X")))

    def test_no_match_on_non_pairing(self):
        rule = PairingLinearityDefinition()
        assert not rule.matches(Sum(Symbol("a"), Symbol("b")))


class TestPairingLinearityRewrite:
    def test_distributes_alpha_slot(self):
        a, b, X = Symbol("a"), Symbol("b"), Symbol("X")
        rule = PairingLinearityDefinition()
        out = rule.rewrite(Pairing(Sum(a, b), X))
        assert out == Sum(Pairing(a, X), Pairing(b, X))

    def test_distributes_X_slot(self):
        alpha, X, Y = Symbol("α"), Symbol("X"), Symbol("Y")
        rule = PairingLinearityDefinition()
        out = rule.rewrite(Pairing(alpha, Sum(X, Y)))
        assert out == Sum(Pairing(alpha, X), Pairing(alpha, Y))

    def test_alpha_first_when_both_summed(self):
        # alpha-slot wins on the first pass; X-slot expands on the
        # engine's second iteration.
        a, b, X, Y = Symbol("a"), Symbol("b"), Symbol("X"), Symbol("Y")
        rule = PairingLinearityDefinition()
        out = rule.rewrite(Pairing(Sum(a, b), Sum(X, Y)))
        # First pass: alpha slot only
        assert out == Sum(
            Pairing(a, Sum(X, Y)),
            Pairing(b, Sum(X, Y)),
        )

    def test_engine_fully_expands_both_slots(self):
        a, b, X, Y = Symbol("a"), Symbol("b"), Symbol("X"), Symbol("Y")
        engine = ExpansionEngine([PairingLinearityDefinition()])
        result, _ = engine.expand(Pairing(Sum(a, b), Sum(X, Y)))
        # Engine produces a structurally nested Sum during repeated
        # rewrites; simplify flattens it into the canonical 4-term form.
        flat = simplify(result)
        assert flat == Sum(
            Pairing(a, X), Pairing(a, Y),
            Pairing(b, X), Pairing(b, Y),
        )

    def test_engine_distributes_three_term_sum(self):
        a, b, c = Symbol("a"), Symbol("b"), Symbol("c")
        X = Symbol("X")
        engine = ExpansionEngine([PairingLinearityDefinition()])
        result, _ = engine.expand(Pairing(Sum(a, b, c), X))
        flat = simplify(result)
        assert flat == Sum(Pairing(a, X), Pairing(b, X), Pairing(c, X))


# --------------------------------------------------------------------- #
# PairingLieLeibnizDefinition (axiom 4)                                  #
# --------------------------------------------------------------------- #


class TestPairingLieLeibnizMatches:
    def test_matches_lie_on_pairing(self):
        X, alpha, Y = Symbol("X"), Symbol("α"), Symbol("Y")
        L = lie_derivative(X)
        rule = PairingLieLeibnizDefinition()
        assert rule.matches(Act(L, Pairing(alpha, Y)))

    def test_no_match_on_non_lie_op(self):
        # Sharp acting on a Pairing has no geometric meaning; the rule
        # restricts to LieDerivative on purpose.
        pi = Symbol("π")
        sh = sharp(pi)
        alpha, Y = Symbol("α"), Symbol("Y")
        rule = PairingLieLeibnizDefinition()
        assert not rule.matches(Act(sh, Pairing(alpha, Y)))

    def test_no_match_on_d_on_pairing(self):
        # d on a pairing is a different identity (Cartan), not the
        # bilinear Leibniz this axiom encodes.
        alpha, Y = Symbol("α"), Symbol("Y")
        rule = PairingLieLeibnizDefinition()
        assert not rule.matches(Act(d, Pairing(alpha, Y)))

    def test_no_match_on_lie_of_non_pairing(self):
        X = Symbol("X")
        L = lie_derivative(X)
        rule = PairingLieLeibnizDefinition()
        assert not rule.matches(Act(L, Symbol("β")))


class TestPairingLieLeibnizRewrite:
    def test_unfolds_to_two_term_sum(self):
        X, alpha, Y = Symbol("X"), Symbol("α"), Symbol("Y")
        L = lie_derivative(X)
        rule = PairingLieLeibnizDefinition()
        out = rule.rewrite(Act(L, Pairing(alpha, Y)))
        assert out == Sum(
            Pairing(Act(L, alpha), Y),
            Pairing(alpha, Act(L, Y)),
        )

    def test_engine_fires(self):
        X, alpha, Y = Symbol("X"), Symbol("α"), Symbol("Y")
        L = lie_derivative(X)
        engine = ExpansionEngine([PairingLieLeibnizDefinition()])
        result, steps = engine.expand(Act(L, Pairing(alpha, Y)))
        assert result == Sum(
            Pairing(Act(L, alpha), Y),
            Pairing(alpha, Act(L, Y)),
        )
        assert any("Pairing-Lie Leibniz" in s.rule for s in steps)


class TestPairingAxiomsCombined:
    def test_lie_through_pairing_atomic_then_leibniz(self):
        # When the Pairing's slots are atomic, Lie-Leibniz fires
        # directly without a Linearity prelude.
        X = Symbol("X")
        alpha, Y = Symbol("α"), Symbol("Y")
        L = lie_derivative(X)
        engine = ExpansionEngine(
            [
                PairingLieLeibnizDefinition(),
                PairingLinearityDefinition(),
            ]
        )
        result, _ = engine.expand(Act(L, Pairing(alpha, Y)))
        flat = simplify(result)
        assert flat == Sum(
            Pairing(Act(L, alpha), Y),
            Pairing(alpha, Act(L, Y)),
        )

    def test_inner_linearity_before_outer_lie(self):
        # When the inner Pairing has a summed slot, the engine's
        # bottom-up walk fires PairingLinearity first, peeling the
        # Sum out before any outer Lie-Leibniz step can fire. After
        # that the outer Act(L, Sum(...)) needs product_rule (not the
        # engine) to push L through; this test pins the engine-only
        # behaviour as a regression against accidental rule reordering.
        X = Symbol("X")
        a, b, Y = Symbol("a"), Symbol("b"), Symbol("Y")
        L = lie_derivative(X)
        engine = ExpansionEngine(
            [
                PairingLieLeibnizDefinition(),
                PairingLinearityDefinition(),
            ]
        )
        result, _ = engine.expand(Act(L, Pairing(Sum(a, b), Y)))
        # Engine-only: Linearity peels Sum-in-alpha; the outer Act(L,
        # Sum(...)) is left for the strategy's product_rule pass.
        assert result == Act(L, Sum(Pairing(a, Y), Pairing(b, Y)))
