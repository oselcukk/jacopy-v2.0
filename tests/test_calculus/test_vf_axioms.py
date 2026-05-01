"""Tests for Faz 13.C, LieBracketVF + Op commutator + Lie-Jacobi VF."""

import pytest

from jacopy.algebra.derivation import Act, Derivation
from jacopy.algebra.lie_bracket_vf import LieBracketVF, lie_bracket_vf
from jacopy.calculus.lie_derivative import lie_derivative
from jacopy.calculus.vf_axioms import (
    LieVfJacobiDefinition,
    OpCommutatorVfDefinition,
)
from jacopy.core.expr import Integer, Neg, Sum, Symbol
from jacopy.core.symbolic_degree import Degree
from jacopy.proof.expansion import ExpansionEngine


# --------------------------------------------------------------------- #
# LieBracketVF atom                                                      #
# --------------------------------------------------------------------- #


class TestLieBracketVF:
    def test_is_degree_zero_derivation(self):
        X, Y = Symbol("X"), Symbol("Y")
        b = lie_bracket_vf(X, Y)
        assert isinstance(b, Derivation)
        assert b.degree == Degree.const(0)

    def test_default_name(self):
        X, Y = Symbol("X"), Symbol("Y")
        assert lie_bracket_vf(X, Y).name == "[X,Y]_VF"

    def test_custom_name(self):
        X, Y = Symbol("X"), Symbol("Y")
        assert lie_bracket_vf(X, Y, name="W").name == "W"

    def test_carries_arguments(self):
        X, Y = Symbol("X"), Symbol("Y")
        b = lie_bracket_vf(X, Y)
        assert b.X is X and b.Y is Y

    def test_equality_on_arguments(self):
        X, Y = Symbol("X"), Symbol("Y")
        assert lie_bracket_vf(X, Y) == lie_bracket_vf(X, Y)

    def test_distinct_arguments_give_distinct_atoms(self):
        X, Y, Z = Symbol("X"), Symbol("Y"), Symbol("Z")
        assert lie_bracket_vf(X, Y) != lie_bracket_vf(X, Z)
        # Order matters at the atomic level, antisymmetry is an
        # axiom that lives at the rewrite layer, not in identity.
        assert lie_bracket_vf(X, Y) != lie_bracket_vf(Y, X)

    def test_requires_expr(self):
        with pytest.raises(TypeError):
            LieBracketVF("X", Symbol("Y"))  # type: ignore[arg-type]
        with pytest.raises(TypeError):
            LieBracketVF(Symbol("X"), "Y")  # type: ignore[arg-type]


# --------------------------------------------------------------------- #
# OpCommutatorVfDefinition (axiom 5)                                     #
# --------------------------------------------------------------------- #


class TestOpCommutatorMatches:
    def test_matches_canonical_pair(self):
        X, Y, w = Symbol("X"), Symbol("Y"), Symbol("ω")
        L_X, L_Y = lie_derivative(X), lie_derivative(Y)
        rule = OpCommutatorVfDefinition()
        s = Sum(
            Act(L_X, Act(L_Y, w)),
            Neg(Act(L_Y, Act(L_X, w))),
        )
        assert rule.matches(s)

    def test_matches_reversed_order(self):
        X, Y, w = Symbol("X"), Symbol("Y"), Symbol("ω")
        L_X, L_Y = lie_derivative(X), lie_derivative(Y)
        rule = OpCommutatorVfDefinition()
        # Negative term first, positive term second, both should match.
        s = Sum(
            Neg(Act(L_Y, Act(L_X, w))),
            Act(L_X, Act(L_Y, w)),
        )
        assert rule.matches(s)

    def test_no_match_on_same_lie_derivative(self):
        # L_X(L_X(w)) - L_X(L_X(w)) is just 0, not a vf bracket pair.
        X, w = Symbol("X"), Symbol("ω")
        L_X = lie_derivative(X)
        rule = OpCommutatorVfDefinition()
        s = Sum(
            Act(L_X, Act(L_X, w)),
            Neg(Act(L_X, Act(L_X, w))),
        )
        assert not rule.matches(s)

    def test_no_match_on_different_operands(self):
        X, Y, w1, w2 = (
            Symbol("X"), Symbol("Y"), Symbol("ω1"), Symbol("ω2")
        )
        L_X, L_Y = lie_derivative(X), lie_derivative(Y)
        rule = OpCommutatorVfDefinition()
        s = Sum(
            Act(L_X, Act(L_Y, w1)),
            Neg(Act(L_Y, Act(L_X, w2))),
        )
        assert not rule.matches(s)

    def test_no_match_on_atomic_term(self):
        rule = OpCommutatorVfDefinition()
        assert not rule.matches(Symbol("ω"))


class TestOpCommutatorRewrite:
    def test_collapses_pair_to_lie_bracket_vf(self):
        X, Y, w = Symbol("X"), Symbol("Y"), Symbol("ω")
        L_X, L_Y = lie_derivative(X), lie_derivative(Y)
        rule = OpCommutatorVfDefinition()
        s = Sum(
            Act(L_X, Act(L_Y, w)),
            Neg(Act(L_Y, Act(L_X, w))),
        )
        out = rule.rewrite(s)
        L_XY = lie_derivative(LieBracketVF(X, Y))
        assert out == Act(L_XY, w)

    def test_keeps_non_matching_terms(self):
        X, Y, w = Symbol("X"), Symbol("Y"), Symbol("ω")
        extra = Symbol("γ")
        L_X, L_Y = lie_derivative(X), lie_derivative(Y)
        rule = OpCommutatorVfDefinition()
        s = Sum(
            Act(L_X, Act(L_Y, w)),
            Neg(Act(L_Y, Act(L_X, w))),
            extra,
        )
        out = rule.rewrite(s)
        L_XY = lie_derivative(LieBracketVF(X, Y))
        assert out == Sum(Act(L_XY, w), extra)

    def test_engine_fires(self):
        X, Y, w = Symbol("X"), Symbol("Y"), Symbol("ω")
        L_X, L_Y = lie_derivative(X), lie_derivative(Y)
        engine = ExpansionEngine([OpCommutatorVfDefinition()])
        s = Sum(
            Act(L_X, Act(L_Y, w)),
            Neg(Act(L_Y, Act(L_X, w))),
        )
        result, steps = engine.expand(s)
        L_XY = lie_derivative(LieBracketVF(X, Y))
        assert result == Act(L_XY, w)
        assert any("L_{[X,Y]_VF}" in step.rule for step in steps)


# --------------------------------------------------------------------- #
# LieVfJacobiDefinition                                                  #
# --------------------------------------------------------------------- #


class TestLieVfJacobiMatches:
    def test_matches_canonical_cyclic_triple(self):
        A, B, C, w = Symbol("A"), Symbol("B"), Symbol("C"), Symbol("ω")
        rule = LieVfJacobiDefinition()
        # [A, [B, C]_VF]_VF + [B, [C, A]_VF]_VF + [C, [A, B]_VF]_VF
        t1 = Act(lie_derivative(LieBracketVF(A, LieBracketVF(B, C))), w)
        t2 = Act(lie_derivative(LieBracketVF(B, LieBracketVF(C, A))), w)
        t3 = Act(lie_derivative(LieBracketVF(C, LieBracketVF(A, B))), w)
        s = Sum(t1, t2, t3)
        assert rule.matches(s)

    def test_no_match_on_two_term_subset(self):
        A, B, C, w = Symbol("A"), Symbol("B"), Symbol("C"), Symbol("ω")
        rule = LieVfJacobiDefinition()
        t1 = Act(lie_derivative(LieBracketVF(A, LieBracketVF(B, C))), w)
        t2 = Act(lie_derivative(LieBracketVF(B, LieBracketVF(C, A))), w)
        assert not rule.matches(Sum(t1, t2))

    def test_no_match_on_different_operands(self):
        A, B, C = Symbol("A"), Symbol("B"), Symbol("C")
        rule = LieVfJacobiDefinition()
        t1 = Act(
            lie_derivative(LieBracketVF(A, LieBracketVF(B, C))), Symbol("w1"),
        )
        t2 = Act(
            lie_derivative(LieBracketVF(B, LieBracketVF(C, A))), Symbol("w2"),
        )
        t3 = Act(
            lie_derivative(LieBracketVF(C, LieBracketVF(A, B))), Symbol("w3"),
        )
        assert not rule.matches(Sum(t1, t2, t3))

    def test_no_match_when_inner_not_lie_bracket_vf(self):
        A, w = Symbol("A"), Symbol("ω")
        # Outer is a LieBracketVF, but inner is a bare Symbol, not the
        # nested pattern this axiom captures.
        rule = LieVfJacobiDefinition()
        t = Act(lie_derivative(LieBracketVF(A, Symbol("B"))), w)
        assert not rule.matches(Sum(t, t, t))


class TestLieVfJacobiRewrite:
    def test_collapses_to_zero_when_only_triple(self):
        A, B, C, w = Symbol("A"), Symbol("B"), Symbol("C"), Symbol("ω")
        rule = LieVfJacobiDefinition()
        t1 = Act(lie_derivative(LieBracketVF(A, LieBracketVF(B, C))), w)
        t2 = Act(lie_derivative(LieBracketVF(B, LieBracketVF(C, A))), w)
        t3 = Act(lie_derivative(LieBracketVF(C, LieBracketVF(A, B))), w)
        s = Sum(t1, t2, t3)
        assert rule.rewrite(s) == Integer(0)

    def test_preserves_other_terms(self):
        A, B, C, w = Symbol("A"), Symbol("B"), Symbol("C"), Symbol("ω")
        extra = Symbol("residue")
        rule = LieVfJacobiDefinition()
        t1 = Act(lie_derivative(LieBracketVF(A, LieBracketVF(B, C))), w)
        t2 = Act(lie_derivative(LieBracketVF(B, LieBracketVF(C, A))), w)
        t3 = Act(lie_derivative(LieBracketVF(C, LieBracketVF(A, B))), w)
        s = Sum(t1, t2, t3, extra)
        out = rule.rewrite(s)
        assert out == extra

    def test_engine_fires(self):
        A, B, C, w = Symbol("A"), Symbol("B"), Symbol("C"), Symbol("ω")
        engine = ExpansionEngine([LieVfJacobiDefinition()])
        t1 = Act(lie_derivative(LieBracketVF(A, LieBracketVF(B, C))), w)
        t2 = Act(lie_derivative(LieBracketVF(B, LieBracketVF(C, A))), w)
        t3 = Act(lie_derivative(LieBracketVF(C, LieBracketVF(A, B))), w)
        result, steps = engine.expand(Sum(t1, t2, t3))
        assert result == Integer(0)
        assert any("Jacobi" in step.rule for step in steps)
