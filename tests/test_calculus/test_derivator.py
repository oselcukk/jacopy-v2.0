"""Tests for the derivator helper (Faz 15.A).

The helper builds the inert Sum
``φ[u,v]_E − [φu, v]_E − [u, φv]_E``. Stage A only verifies the shape;
engine-driven evaluation lives in Faz 15.B.
"""

import pytest

from jacopy.algebra.derivation import Act, Derivation
from jacopy.brackets.base import BracketApply
from jacopy.brackets.koszul import KoszulBracket
from jacopy.brackets.lie import LieBracket
from jacopy.calculus.derivator import derivator
from jacopy.calculus.lie_derivative import lie_derivative
from jacopy.core.expr import Neg, Sum, Symbol


@pytest.fixture
def koszul():
    rho = Derivation("ρ", 0)
    return KoszulBracket(rho)


@pytest.fixture
def L_U():
    U = Derivation("U", 0)
    return lie_derivative(U)


# --------------------------------------------------------------------- #
# Shape                                                                 #
# --------------------------------------------------------------------- #


class TestDerivatorShape:
    def test_returns_three_term_sum(self, koszul, L_U):
        eta, mu = Symbol("η"), Symbol("μ")
        result = derivator(L_U, koszul, eta, mu)
        assert isinstance(result, Sum)
        assert len(result.children) == 3

    def test_first_term_is_phi_on_bracket(self, koszul, L_U):
        eta, mu = Symbol("η"), Symbol("μ")
        result = derivator(L_U, koszul, eta, mu)
        first = result.children[0]
        assert isinstance(first, Act)
        assert first.op == L_U
        assert isinstance(first.arg, BracketApply)
        assert first.arg.bracket is koszul
        assert first.arg.a is eta
        assert first.arg.b is mu

    def test_second_term_is_neg_bracket_with_phi_left(self, koszul, L_U):
        eta, mu = Symbol("η"), Symbol("μ")
        result = derivator(L_U, koszul, eta, mu)
        second = result.children[1]
        assert isinstance(second, Neg)
        inner = second.arg
        assert isinstance(inner, BracketApply)
        assert inner.bracket is koszul
        assert inner.a == Act(L_U, eta)
        assert inner.b is mu

    def test_third_term_is_neg_bracket_with_phi_right(self, koszul, L_U):
        eta, mu = Symbol("η"), Symbol("μ")
        result = derivator(L_U, koszul, eta, mu)
        third = result.children[2]
        assert isinstance(third, Neg)
        inner = third.arg
        assert isinstance(inner, BracketApply)
        assert inner.bracket is koszul
        assert inner.a is eta
        assert inner.b == Act(L_U, mu)


# --------------------------------------------------------------------- #
# Bracket-flexibility                                                   #
# --------------------------------------------------------------------- #


class TestDerivatorBracketAgnostic:
    def test_works_with_lie_bracket(self, L_U):
        u, v = Symbol("u"), Symbol("v")
        result = derivator(L_U, LieBracket(), u, v)
        assert isinstance(result, Sum)
        assert all(
            isinstance(child, (Act, Neg)) for child in result.children
        )


# --------------------------------------------------------------------- #
# Type guards                                                           #
# --------------------------------------------------------------------- #


class TestDerivatorTypeGuards:
    def test_rejects_non_expr_phi(self, koszul):
        with pytest.raises(TypeError):
            derivator("not-an-expr", koszul, Symbol("u"), Symbol("v"))  # type: ignore[arg-type]

    def test_rejects_non_bracket(self, L_U):
        with pytest.raises(TypeError):
            derivator(L_U, "not-a-bracket", Symbol("u"), Symbol("v"))  # type: ignore[arg-type]

    def test_rejects_non_expr_u(self, koszul, L_U):
        with pytest.raises(TypeError):
            derivator(L_U, koszul, "not-an-expr", Symbol("v"))  # type: ignore[arg-type]

    def test_rejects_non_expr_v(self, koszul, L_U):
        with pytest.raises(TypeError):
            derivator(L_U, koszul, Symbol("u"), "not-an-expr")  # type: ignore[arg-type]

    def test_accepts_optional_registry_kwarg(self, koszul, L_U):
        from jacopy.core.registry import PropertyRegistry
        reg = PropertyRegistry()
        result = derivator(L_U, koszul, Symbol("u"), Symbol("v"), registry=reg)
        assert isinstance(result, Sum)
