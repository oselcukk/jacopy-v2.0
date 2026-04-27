"""Tests for jacopy.calculus.anchor."""

import pytest

from jacopy.algebra.derivation import Act, Derivation, degree_of
from jacopy.algorithms.simplify import simplify
from jacopy.brackets.lie import LieBracket, lie
from jacopy.calculus.anchor import Anchor, bracket_compatibility_obstruction
from jacopy.core.expr import Integer, Neg, Product, Sum, Symbol
from jacopy.core.properties import Graded
from jacopy.core.registry import PropertyRegistry
from jacopy.core.symbolic_degree import Degree


class TestConstruction:
    def test_default_name_and_degree(self):
        rho = Anchor()
        assert rho.name == "ρ"
        assert rho.degree == Degree.const(0)

    def test_custom_name(self):
        rho = Anchor("ρ_E")
        assert rho.name == "ρ_E"

    def test_is_a_derivation(self):
        assert isinstance(Anchor(), Derivation)

    def test_operator_degree_is_zero(self):
        assert degree_of(Anchor()) == Degree.const(0)


class TestBracketCompatibility:
    def test_obstruction_shape(self):
        """ρ([X,Y]_E) − [ρ(X), ρ(Y)]_{TM} with both brackets = Lie."""
        reg = PropertyRegistry()
        X, Y = Symbol("X"), Symbol("Y")
        reg.declare(X, Graded(degree=0))
        reg.declare(Y, Graded(degree=0))
        rho = Anchor()
        bracket_E = LieBracket(name="[·,·]_E")
        bracket_TM = lie  # standard Lie bracket
        obs = bracket_compatibility_obstruction(
            rho, bracket_E, bracket_TM, X, Y, reg
        )
        expected = Sum(
            Act(rho, Sum(Product(X, Y), Neg(Product(Y, X)))),
            Neg(Sum(
                Product(Act(rho, X), Act(rho, Y)),
                Neg(Product(Act(rho, Y), Act(rho, X))),
            )),
        )
        assert obs == expected
