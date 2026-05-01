"""Tests for the BracketApply-side closure axioms, Q9 Stage 9.C."""

from __future__ import annotations

import pytest

from jacopy.algebra.derivation import Act, Derivation
from jacopy.brackets.base import BracketApply, GradedBracket
from jacopy.brackets.koszul import KoszulBracket
from jacopy.calculus.anchor import Anchor
from jacopy.calculus.bracket_apply_axioms import (
    BracketApplyAntiSymmetryDefinition,
    BracketApplyArgAntisymmetryDefinition,
    BracketApplyJacobiDefinition,
    BracketApplyNegLinearityDefinition,
    BracketApplySumLinearityDefinition,
)
from jacopy.brackets.custom import CustomBracket
from jacopy.core.expr import Expr, Integer, Neg, Sum, Symbol


# --------------------------------------------------------------------- #
# Fixtures                                                               #
# --------------------------------------------------------------------- #


def koszul() -> KoszulBracket:
    return KoszulBracket(Anchor(name="ρ"))


def syms3() -> tuple[Symbol, Symbol, Symbol]:
    return Symbol("α"), Symbol("β"), Symbol("γ")


# --------------------------------------------------------------------- #
# Sum / Neg linearity                                                    #
# --------------------------------------------------------------------- #


class TestBracketApplySumLinearity:
    def test_distributes_left_slot(self):
        k = koszul()
        a, b, c = syms3()
        rule = BracketApplySumLinearityDefinition(k)
        expr = BracketApply(k, Sum(a, b), c)
        assert rule.matches(expr)
        out = rule.rewrite(expr)
        assert out == Sum(BracketApply(k, a, c), BracketApply(k, b, c))

    def test_distributes_right_slot(self):
        k = koszul()
        a, b, c = syms3()
        rule = BracketApplySumLinearityDefinition(k)
        expr = BracketApply(k, a, Sum(b, c))
        out = rule.rewrite(expr)
        assert out == Sum(BracketApply(k, a, b), BracketApply(k, a, c))

    def test_no_match_when_no_sum(self):
        k = koszul()
        a, b, _ = syms3()
        rule = BracketApplySumLinearityDefinition(k)
        assert not rule.matches(BracketApply(k, a, b))

    def test_scope_skips_other_brackets(self):
        k1 = koszul()
        k2 = KoszulBracket(Anchor(name="ρ2"))
        rule = BracketApplySumLinearityDefinition(k1)
        a, b, c = syms3()
        # k2 is a different KoszulBracket instance (different anchor), rule
        # must not fire on it.
        assert not rule.matches(BracketApply(k2, Sum(a, b), c))

    def test_rejects_non_bracket(self):
        with pytest.raises(TypeError):
            BracketApplySumLinearityDefinition("nope")  # type: ignore[arg-type]


class TestBracketApplyNegLinearity:
    def test_pulls_neg_from_left(self):
        k = koszul()
        a, b, _ = syms3()
        rule = BracketApplyNegLinearityDefinition(k)
        expr = BracketApply(k, Neg(a), b)
        assert rule.matches(expr)
        out = rule.rewrite(expr)
        assert out == Neg(BracketApply(k, a, b))

    def test_pulls_neg_from_right(self):
        k = koszul()
        a, b, _ = syms3()
        rule = BracketApplyNegLinearityDefinition(k)
        expr = BracketApply(k, a, Neg(b))
        out = rule.rewrite(expr)
        assert out == Neg(BracketApply(k, a, b))

    def test_double_neg_cancels(self):
        k = koszul()
        a, b, _ = syms3()
        rule = BracketApplyNegLinearityDefinition(k)
        expr = BracketApply(k, Neg(a), Neg(b))
        out = rule.rewrite(expr)
        assert out == BracketApply(k, a, b)

    def test_scope_skips_other_brackets(self):
        k1 = koszul()
        k2 = KoszulBracket(Anchor(name="ρ2"))
        rule = BracketApplyNegLinearityDefinition(k1)
        a, b, _ = syms3()
        assert not rule.matches(BracketApply(k2, Neg(a), b))


# --------------------------------------------------------------------- #
# Arg antisymmetry, atom level                                          #
# --------------------------------------------------------------------- #


class TestBracketApplyArgAntisymmetry:
    def test_swaps_when_repr_decreases(self):
        k = koszul()
        a, _, c = syms3()  # repr('γ') > repr('α')
        rule = BracketApplyArgAntisymmetryDefinition(k)
        expr = BracketApply(k, c, a)
        assert rule.matches(expr)
        out = rule.rewrite(expr)
        assert out == Neg(BracketApply(k, a, c))

    def test_no_match_when_already_canonical(self):
        k = koszul()
        a, _, c = syms3()
        rule = BracketApplyArgAntisymmetryDefinition(k)
        # repr('α') < repr('γ'), already canonical
        assert not rule.matches(BracketApply(k, a, c))

    def test_no_match_when_args_equal(self):
        k = koszul()
        a, _, _ = syms3()
        rule = BracketApplyArgAntisymmetryDefinition(k)
        assert not rule.matches(BracketApply(k, a, a))

    def test_accepts_higher_degree_bracket(self):
        # Higher-degree antisym brackets use the default literal-antisym
        # convention from ``GradedBracket.pair_swap_sign`` (returns -1).
        # Constructing the rule succeeds; the rewrite emits ``Neg(...)``
        # exactly like the degree-0 case. Subclasses override
        # ``pair_swap_sign`` to encode full graded behavior.
        deg2 = CustomBracket("[·,·]_2", lambda a, b, r: a, degree=2)
        rule = BracketApplyArgAntisymmetryDefinition(deg2)
        a, _, c = syms3()
        expr = BracketApply(deg2, c, a)
        assert rule.matches(expr)
        out = rule.rewrite(expr)
        assert out == Neg(BracketApply(deg2, a, c))

    def test_rejects_non_antisymmetric_bracket(self):
        bad = CustomBracket(
            "[·,·]_sym",
            lambda a, b, r: a,
            degree=0,
            is_graded_antisymmetric=False,
        )
        with pytest.raises(ValueError):
            BracketApplyArgAntisymmetryDefinition(bad)

    def test_graded_symmetric_swap_drops_neg(self):
        # Subclass override: ``pair_swap_sign`` returns +1 (the pair
        # behaves graded-symmetrically). Then ``[c, a] → +[a, c]``,
        # rewrite emits the canonical bracket without an outer Neg.
        class GradedSymPairBracket(CustomBracket):
            def pair_swap_sign(self, a, b, registry=None):
                return +1

        bracket = GradedSymPairBracket(
            "[·,·]_gsym", lambda a, b, r: a, degree=0,
        )
        rule = BracketApplyArgAntisymmetryDefinition(bracket)
        a, _, c = syms3()
        expr = BracketApply(bracket, c, a)
        assert rule.matches(expr)
        out = rule.rewrite(expr)
        assert out == BracketApply(bracket, a, c)


# --------------------------------------------------------------------- #
# Sum-level antisymmetry                                                 #
# --------------------------------------------------------------------- #


class TestBracketApplyAntiSymmetry:
    def test_cancels_bare_pair(self):
        k = koszul()
        a, b, _ = syms3()
        rule = BracketApplyAntiSymmetryDefinition(k)
        expr = Sum(BracketApply(k, a, b), BracketApply(k, b, a))
        assert rule.matches(expr)
        out = rule.rewrite(expr)
        assert out == Integer(0)

    def test_cancels_inside_act_wrapper(self):
        k = koszul()
        a, b, _ = syms3()
        f = Symbol("f")
        rule = BracketApplyAntiSymmetryDefinition(k)
        expr = Sum(
            Act(BracketApply(k, a, b), f),
            Act(BracketApply(k, b, a), f),
        )
        assert rule.matches(expr)
        out = rule.rewrite(expr)
        # Both children consumed (no sibling left); zero.
        assert out == Integer(0)

    def test_skips_when_wrappers_differ(self):
        # Same bracket pair under different wrappers must not cancel,
        # ``[a,b](f) + [b,a](g)`` is not zero.
        k = koszul()
        a, b, _ = syms3()
        f, g = Symbol("f"), Symbol("g")
        rule = BracketApplyAntiSymmetryDefinition(k)
        expr = Sum(
            Act(BracketApply(k, a, b), f),
            Act(BracketApply(k, b, a), g),
        )
        assert not rule.matches(expr)

    def test_keeps_extra_summands(self):
        k = koszul()
        a, b, c = syms3()
        rule = BracketApplyAntiSymmetryDefinition(k)
        extra = Symbol("X")
        expr = Sum(
            BracketApply(k, a, b),
            BracketApply(k, b, a),
            extra,
        )
        out = rule.rewrite(expr)
        assert out == extra


# --------------------------------------------------------------------- #
# Cyclic Jacobi                                                          #
# --------------------------------------------------------------------- #


class TestBracketApplyJacobi:
    def test_bare_cyclic_triple_collapses(self):
        k = koszul()
        a, b, c = syms3()
        rule = BracketApplyJacobiDefinition(k)
        expr = Sum(
            BracketApply(k, a, BracketApply(k, b, c)),
            BracketApply(k, b, BracketApply(k, c, a)),
            BracketApply(k, c, BracketApply(k, a, b)),
        )
        assert rule.matches(expr)
        out = rule.rewrite(expr)
        assert out == Integer(0)

    def test_inner_anti_sym_variant_collapses(self):
        # ``[A,[C,B]] + [B,[A,C]] + [C,[B,A]]`` is the cyclic with
        # inner anti-symmetry applied to each, same triple, all signs
        # net to ``-1``, still cancels.
        k = koszul()
        a, b, c = syms3()
        rule = BracketApplyJacobiDefinition(k)
        expr = Sum(
            BracketApply(k, a, BracketApply(k, c, b)),
            BracketApply(k, b, BracketApply(k, a, c)),
            BracketApply(k, c, BracketApply(k, b, a)),
        )
        assert rule.matches(expr)

    def test_scope_skips_other_brackets(self):
        k1 = koszul()
        k2 = KoszulBracket(Anchor(name="ρ2"))
        a, b, c = syms3()
        rule = BracketApplyJacobiDefinition(k1)
        expr = Sum(
            BracketApply(k2, a, BracketApply(k2, b, c)),
            BracketApply(k2, b, BracketApply(k2, c, a)),
            BracketApply(k2, c, BracketApply(k2, a, b)),
        )
        # Wrong bracket, no triple found.
        assert not rule.matches(expr)

    def test_accepts_higher_degree_bracket(self):
        # Higher-degree antisym brackets fire the Jacobi rule via the
        # default literal Jacobi convention (jacobi_term_sign = +1,
        # pair_swap_sign = -1). Subclasses override the hooks for full
        # graded behavior.
        deg2 = CustomBracket("[·,·]_2", lambda a, b, r: a, degree=2)
        rule = BracketApplyJacobiDefinition(deg2)
        a, b, c = syms3()
        expr = Sum(
            BracketApply(deg2, a, BracketApply(deg2, b, c)),
            BracketApply(deg2, b, BracketApply(deg2, c, a)),
            BracketApply(deg2, c, BracketApply(deg2, a, b)),
        )
        assert rule.matches(expr)
        assert rule.rewrite(expr) == Integer(0)
