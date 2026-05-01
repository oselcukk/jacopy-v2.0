"""Tests for operator R-linearity axioms (L_X arg, d arg)."""

from jacopy.algebra.derivation import Act
from jacopy.calculus.exterior_d import d as default_d
from jacopy.calculus.lie_derivative import lie_derivative
from jacopy.calculus.linearity_axioms import (
    ExteriorDerivativeLinearityDefinition,
    LieDerivativeArgLinearityDefinition,
)
from jacopy.core.expr import Neg, Sum, Symbol
from jacopy.proof.expansion import ExpansionEngine


# --------------------------------------------------------------------- #
# LieDerivativeArgLinearityDefinition                                    #
# --------------------------------------------------------------------- #


class TestLieDerivativeArgLinearity:
    def test_matches_sum_argument(self):
        X = Symbol("X")
        a, b = Symbol("a"), Symbol("b")
        rule = LieDerivativeArgLinearityDefinition()
        assert rule.matches(Act(lie_derivative(X), Sum(a, b)))

    def test_matches_neg_argument(self):
        X = Symbol("X")
        a = Symbol("a")
        rule = LieDerivativeArgLinearityDefinition()
        assert rule.matches(Act(lie_derivative(X), Neg(a)))

    def test_no_match_atomic_argument(self):
        X = Symbol("X")
        rule = LieDerivativeArgLinearityDefinition()
        assert not rule.matches(Act(lie_derivative(X), Symbol("a")))

    def test_no_match_non_lie(self):
        a, b = Symbol("a"), Symbol("b")
        rule = LieDerivativeArgLinearityDefinition()
        # Act with non-LieDerivative operator, shouldn't fire.
        assert not rule.matches(Act(default_d, Sum(a, b)))

    def test_distributes_over_sum(self):
        X = Symbol("X")
        a, b, c = Symbol("a"), Symbol("b"), Symbol("c")
        L = lie_derivative(X)
        rule = LieDerivativeArgLinearityDefinition()
        out = rule.rewrite(Act(L, Sum(a, b, c)))
        assert out == Sum(Act(L, a), Act(L, b), Act(L, c))

    def test_distributes_over_neg(self):
        X = Symbol("X")
        a = Symbol("a")
        L = lie_derivative(X)
        rule = LieDerivativeArgLinearityDefinition()
        assert rule.rewrite(Act(L, Neg(a))) == Neg(Act(L, a))

    def test_handles_neg_inside_sum(self):
        # Sum children may individually be Neg-wrapped.
        X = Symbol("X")
        a, b = Symbol("a"), Symbol("b")
        L = lie_derivative(X)
        rule = LieDerivativeArgLinearityDefinition()
        out = rule.rewrite(Act(L, Sum(a, Neg(b))))
        assert out == Sum(Act(L, a), Neg(Act(L, b)))


# --------------------------------------------------------------------- #
# ExteriorDerivativeLinearityDefinition                                  #
# --------------------------------------------------------------------- #


class TestExteriorDerivativeLinearity:
    def test_matches_sum_argument(self):
        a, b = Symbol("a"), Symbol("b")
        rule = ExteriorDerivativeLinearityDefinition()
        assert rule.matches(Act(default_d, Sum(a, b)))

    def test_matches_neg_argument(self):
        a = Symbol("a")
        rule = ExteriorDerivativeLinearityDefinition()
        assert rule.matches(Act(default_d, Neg(a)))

    def test_no_match_atomic_argument(self):
        rule = ExteriorDerivativeLinearityDefinition()
        assert not rule.matches(Act(default_d, Symbol("a")))

    def test_no_match_non_d(self):
        X = Symbol("X")
        a, b = Symbol("a"), Symbol("b")
        rule = ExteriorDerivativeLinearityDefinition()
        # Lie derivative of a Sum, different operator, shouldn't fire.
        assert not rule.matches(Act(lie_derivative(X), Sum(a, b)))

    def test_distributes_over_sum(self):
        a, b, c = Symbol("a"), Symbol("b"), Symbol("c")
        rule = ExteriorDerivativeLinearityDefinition()
        out = rule.rewrite(Act(default_d, Sum(a, b, c)))
        assert out == Sum(Act(default_d, a), Act(default_d, b), Act(default_d, c))

    def test_distributes_over_neg(self):
        a = Symbol("a")
        rule = ExteriorDerivativeLinearityDefinition()
        assert rule.rewrite(Act(default_d, Neg(a))) == Neg(Act(default_d, a))

    def test_handles_neg_inside_sum(self):
        a, b = Symbol("a"), Symbol("b")
        rule = ExteriorDerivativeLinearityDefinition()
        out = rule.rewrite(Act(default_d, Sum(a, Neg(b))))
        assert out == Sum(Act(default_d, a), Neg(Act(default_d, b)))


# --------------------------------------------------------------------- #
# Engine integration                                                     #
# --------------------------------------------------------------------- #


class TestLinearityAxiomsEngine:
    def test_engine_distributes_lie_arg(self):
        X = Symbol("X")
        a, b, c = Symbol("a"), Symbol("b"), Symbol("c")
        L = lie_derivative(X)
        engine = ExpansionEngine([LieDerivativeArgLinearityDefinition()])
        result, _ = engine.expand(Act(L, Sum(a, b, c)))
        assert result == Sum(Act(L, a), Act(L, b), Act(L, c))

    def test_engine_distributes_d(self):
        a, b, c = Symbol("a"), Symbol("b"), Symbol("c")
        engine = ExpansionEngine([ExteriorDerivativeLinearityDefinition()])
        result, _ = engine.expand(Act(default_d, Sum(a, b, c)))
        assert result == Sum(
            Act(default_d, a), Act(default_d, b), Act(default_d, c)
        )

    def test_engine_combined_distributes_through_compound(self):
        # d(L_X(a + b)), d's arg is a Lie act, but the inner Sum is
        # what triggers distribution. After the Lie-arg rule fires, d
        # ends up over a Sum and its own rule fires.
        X = Symbol("X")
        a, b = Symbol("a"), Symbol("b")
        L = lie_derivative(X)
        engine = ExpansionEngine(
            [
                LieDerivativeArgLinearityDefinition(),
                ExteriorDerivativeLinearityDefinition(),
            ]
        )
        result, _ = engine.expand(Act(default_d, Act(L, Sum(a, b))))
        assert result == Sum(
            Act(default_d, Act(L, a)),
            Act(default_d, Act(L, b)),
        )
