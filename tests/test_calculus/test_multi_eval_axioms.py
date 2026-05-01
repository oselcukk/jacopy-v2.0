"""Tests for MultiEval engine rules (Faz 12.A.0)."""

import pytest

from jacopy.algorithms.simplify import simplify
from jacopy.calculus.multi_eval_axioms import (
    MultiEvalAlternatingNormalDefinition,
    MultiEvalArgLinearityDefinition,
    MultiEvalHeadLinearityDefinition,
    MultiEvalRepeatArgZeroDefinition,
)
from jacopy.core.expr import Neg, Sum, Symbol, Zero
from jacopy.core.multi_eval import multi_eval
from jacopy.core.registry import PropertyRegistry
from jacopy.proof.expansion import ExpansionEngine


def _expand_and_simplify(engine: ExpansionEngine, expr):
    """Engine + simplify cascade, flatten nested Sums, drop zeros."""
    out, _ = engine.expand(expr)
    return simplify(out, PropertyRegistry())


# --------------------------------------------------------------------- #
# Repeat-arg → 0                                                         #
# --------------------------------------------------------------------- #


class TestRepeatArgZero:
    def test_matches_repeat_alternating(self):
        omega = Symbol("ω")
        X = Symbol("X")
        rule = MultiEvalRepeatArgZeroDefinition()
        assert rule.matches(multi_eval(omega, X, X))

    def test_no_match_when_distinct(self):
        omega = Symbol("ω")
        X, Y = Symbol("X"), Symbol("Y")
        rule = MultiEvalRepeatArgZeroDefinition()
        assert not rule.matches(multi_eval(omega, X, Y))

    def test_no_match_when_non_alternating(self):
        omega = Symbol("ω")
        X = Symbol("X")
        rule = MultiEvalRepeatArgZeroDefinition()
        assert not rule.matches(
            multi_eval(omega, X, X, alternating=False)
        )

    def test_rewrite_returns_zero(self):
        omega = Symbol("ω")
        X = Symbol("X")
        rule = MultiEvalRepeatArgZeroDefinition()
        assert rule.rewrite(multi_eval(omega, X, X)) is Zero

    def test_engine_zeros_repeat(self):
        omega = Symbol("ω")
        X = Symbol("X")
        engine = ExpansionEngine([MultiEvalRepeatArgZeroDefinition()])
        out, _ = engine.expand(multi_eval(omega, X, X))
        assert out is Zero

    def test_engine_zeros_repeat_in_three_arg_form(self):
        omega = Symbol("ω")
        X, Y = Symbol("X"), Symbol("Y")
        engine = ExpansionEngine([MultiEvalRepeatArgZeroDefinition()])
        out, _ = engine.expand(multi_eval(omega, X, Y, X))
        assert out is Zero


# --------------------------------------------------------------------- #
# Arg-slot linearity                                                    #
# --------------------------------------------------------------------- #


class TestArgLinearity:
    def test_matches_sum_in_arg_slot(self):
        omega = Symbol("ω")
        X, Y = Symbol("X"), Symbol("Y")
        rule = MultiEvalArgLinearityDefinition()
        assert rule.matches(multi_eval(omega, Sum(X, Y), X))

    def test_matches_neg_in_arg_slot(self):
        omega = Symbol("ω")
        X, Y = Symbol("X"), Symbol("Y")
        rule = MultiEvalArgLinearityDefinition()
        assert rule.matches(multi_eval(omega, X, Neg(Y)))

    def test_no_match_atomic_args(self):
        omega = Symbol("ω")
        X, Y = Symbol("X"), Symbol("Y")
        rule = MultiEvalArgLinearityDefinition()
        assert not rule.matches(multi_eval(omega, X, Y))

    def test_distributes_sum_in_first_slot(self):
        omega = Symbol("ω")
        X, Y, Z = Symbol("X"), Symbol("Y"), Symbol("Z")
        rule = MultiEvalArgLinearityDefinition()
        out = rule.rewrite(multi_eval(omega, Sum(X, Y), Z))
        assert out == Sum(multi_eval(omega, X, Z), multi_eval(omega, Y, Z))

    def test_distributes_sum_in_second_slot(self):
        omega = Symbol("ω")
        X, Y, Z = Symbol("X"), Symbol("Y"), Symbol("Z")
        rule = MultiEvalArgLinearityDefinition()
        out = rule.rewrite(multi_eval(omega, X, Sum(Y, Z)))
        assert out == Sum(multi_eval(omega, X, Y), multi_eval(omega, X, Z))

    def test_distributes_neg(self):
        omega = Symbol("ω")
        X, Y = Symbol("X"), Symbol("Y")
        rule = MultiEvalArgLinearityDefinition()
        out = rule.rewrite(multi_eval(omega, X, Neg(Y)))
        assert out == Neg(multi_eval(omega, X, Y))

    def test_engine_unfolds_compound(self):
        omega = Symbol("ω")
        X, Y, Z, W = (Symbol(s) for s in ("X", "Y", "Z", "W"))
        engine = ExpansionEngine([MultiEvalArgLinearityDefinition()])
        out = _expand_and_simplify(
            engine, multi_eval(omega, Sum(X, Y), Sum(Z, W))
        )
        # After simplify the four-term distribution surfaces as a flat Sum.
        terms = {
            multi_eval(omega, X, Z),
            multi_eval(omega, X, W),
            multi_eval(omega, Y, Z),
            multi_eval(omega, Y, W),
        }
        assert isinstance(out, Sum)
        assert set(out.children) == terms


# --------------------------------------------------------------------- #
# Head linearity                                                         #
# --------------------------------------------------------------------- #


class TestHeadLinearity:
    def test_matches_sum_head(self):
        alpha, beta = Symbol("α"), Symbol("β")
        X = Symbol("X")
        rule = MultiEvalHeadLinearityDefinition()
        assert rule.matches(multi_eval(Sum(alpha, beta), X))

    def test_matches_neg_head(self):
        alpha = Symbol("α")
        X = Symbol("X")
        rule = MultiEvalHeadLinearityDefinition()
        assert rule.matches(multi_eval(Neg(alpha), X))

    def test_no_match_atomic_head(self):
        omega = Symbol("ω")
        X = Symbol("X")
        rule = MultiEvalHeadLinearityDefinition()
        assert not rule.matches(multi_eval(omega, X))

    def test_distributes_sum_head(self):
        alpha, beta = Symbol("α"), Symbol("β")
        X, Y = Symbol("X"), Symbol("Y")
        rule = MultiEvalHeadLinearityDefinition()
        out = rule.rewrite(multi_eval(Sum(alpha, beta), X, Y))
        assert out == Sum(
            multi_eval(alpha, X, Y), multi_eval(beta, X, Y)
        )

    def test_distributes_neg_head(self):
        alpha = Symbol("α")
        X = Symbol("X")
        rule = MultiEvalHeadLinearityDefinition()
        out = rule.rewrite(multi_eval(Neg(alpha), X))
        assert out == Neg(multi_eval(alpha, X))


# --------------------------------------------------------------------- #
# Combined engine integration                                            #
# --------------------------------------------------------------------- #


class TestCombinedEngine:
    def _engine(self) -> ExpansionEngine:
        return ExpansionEngine(
            [
                MultiEvalArgLinearityDefinition(),
                MultiEvalHeadLinearityDefinition(),
                MultiEvalRepeatArgZeroDefinition(),
            ]
        )

    def test_head_and_arg_distribute_together(self):
        alpha, beta = Symbol("α"), Symbol("β")
        X, Y, Z = Symbol("X"), Symbol("Y"), Symbol("Z")
        engine = self._engine()
        out = _expand_and_simplify(
            engine, multi_eval(Sum(alpha, beta), X, Sum(Y, Z))
        )
        terms = {
            multi_eval(alpha, X, Y),
            multi_eval(alpha, X, Z),
            multi_eval(beta, X, Y),
            multi_eval(beta, X, Z),
        }
        assert isinstance(out, Sum)
        assert set(out.children) == terms

    def test_repeat_kills_one_branch(self):
        # ω(X + Y, X), distributes to ω(X, X) + ω(Y, X);
        # the first vanishes by alternation, leaving ω(Y, X).
        omega = Symbol("ω")
        X, Y = Symbol("X"), Symbol("Y")
        engine = self._engine()
        out = _expand_and_simplify(engine, multi_eval(omega, Sum(X, Y), X))
        assert out == multi_eval(omega, Y, X)


# --------------------------------------------------------------------- #
# Alternating canonicalize (bubble-swap toward repr-sorted args)         #
# --------------------------------------------------------------------- #


class TestAlternatingCanonicalize:
    def test_matches_inverted_pair(self):
        omega = Symbol("ω")
        X, Y = Symbol("X"), Symbol("Y")
        rule = MultiEvalAlternatingNormalDefinition()
        # repr("Y") > repr("X"), so ω(Y, X) is out of order.
        assert rule.matches(multi_eval(omega, Y, X))

    def test_no_match_when_sorted(self):
        omega = Symbol("ω")
        X, Y = Symbol("X"), Symbol("Y")
        rule = MultiEvalAlternatingNormalDefinition()
        assert not rule.matches(multi_eval(omega, X, Y))

    def test_no_match_when_non_alternating(self):
        omega = Symbol("ω")
        X, Y = Symbol("X"), Symbol("Y")
        rule = MultiEvalAlternatingNormalDefinition()
        assert not rule.matches(
            multi_eval(omega, Y, X, alternating=False)
        )

    def test_no_match_on_equal_adjacent_args(self):
        # repeat-arg case is owned by RepeatArgZero rule, not this one.
        omega = Symbol("ω")
        X = Symbol("X")
        rule = MultiEvalAlternatingNormalDefinition()
        assert not rule.matches(multi_eval(omega, X, X))

    def test_rewrite_swaps_and_negates(self):
        omega = Symbol("ω")
        X, Y = Symbol("X"), Symbol("Y")
        rule = MultiEvalAlternatingNormalDefinition()
        out = rule.rewrite(multi_eval(omega, Y, X))
        assert out == Neg(multi_eval(omega, X, Y))

    def test_engine_fixpoints_two_args(self):
        omega = Symbol("ω")
        X, Y = Symbol("X"), Symbol("Y")
        engine = ExpansionEngine([MultiEvalAlternatingNormalDefinition()])
        out = _expand_and_simplify(engine, multi_eval(omega, Y, X))
        assert out == Neg(multi_eval(omega, X, Y))

    def test_engine_three_args_reverse_to_sorted(self):
        # Reverse-order ω(Z, Y, X) → 3 inversions → parity = +Neg → sign = +1.
        # Bubble sort: (Z,Y,X) → -(Y,Z,X) → --(Y,X,Z) → ---(X,Y,Z) → 3× Neg.
        # simplify collapses cascade: 3 Negs = -1 sign.
        omega = Symbol("ω")
        X, Y, Z = Symbol("X"), Symbol("Y"), Symbol("Z")
        engine = ExpansionEngine([MultiEvalAlternatingNormalDefinition()])
        out = _expand_and_simplify(engine, multi_eval(omega, Z, Y, X))
        assert out == Neg(multi_eval(omega, X, Y, Z))

    def test_engine_swap_pair_in_three_args(self):
        # ω(X, Z, Y), single inversion (Z, Y) → +Neg → -ω(X, Y, Z).
        omega = Symbol("ω")
        X, Y, Z = Symbol("X"), Symbol("Y"), Symbol("Z")
        engine = ExpansionEngine([MultiEvalAlternatingNormalDefinition()])
        out = _expand_and_simplify(engine, multi_eval(omega, X, Z, Y))
        assert out == Neg(multi_eval(omega, X, Y, Z))

    def test_engine_already_sorted_unchanged(self):
        omega = Symbol("ω")
        X, Y, Z = Symbol("X"), Symbol("Y"), Symbol("Z")
        engine = ExpansionEngine([MultiEvalAlternatingNormalDefinition()])
        sorted_expr = multi_eval(omega, X, Y, Z)
        out, steps = engine.expand(sorted_expr)
        assert out is sorted_expr
        assert list(steps) == []

    def test_first_inversion_helper(self):
        X, Y, Z = Symbol("X"), Symbol("Y"), Symbol("Z")
        find = MultiEvalAlternatingNormalDefinition._first_inversion
        assert find((X, Y, Z)) == -1
        assert find((Y, X, Z)) == 0
        assert find((X, Z, Y)) == 1
        assert find((Z, Y, X)) == 0  # leftmost inversion
