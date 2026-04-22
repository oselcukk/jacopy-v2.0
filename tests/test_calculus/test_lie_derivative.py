"""Tests for gradalg.calculus.lie_derivative."""

import pytest

from gradalg.algebra.derivation import Act, Derivation, compose, degree_of
from gradalg.calculus.exterior_d import ExteriorDerivative, d
from gradalg.calculus.interior import InteriorProduct, interior
from gradalg.calculus.lie_derivative import (
    DEFINITIONS,
    LieDerivative,
    cartan_expansion,
    cartan_obstruction,
    lie_derivative,
)
from gradalg.core.expr import Neg, Product, Sum, Symbol
from gradalg.core.symbolic_degree import Degree


class TestConstruction:
    def test_default_name_from_vector_field(self):
        X = Symbol("X")
        L = lie_derivative(X)
        assert L.name == "L_X"
        assert L.degree == Degree.const(0)
        assert L.vector_field is X
        assert L.definition == "cartan"

    def test_flow_mode(self):
        X = Symbol("X")
        L = lie_derivative(X, definition="flow")
        assert L.definition == "flow"

    def test_rejects_bad_definition(self):
        X = Symbol("X")
        with pytest.raises(ValueError):
            lie_derivative(X, definition="bogus")

    def test_rejects_non_expr(self):
        with pytest.raises(TypeError):
            LieDerivative("X")  # type: ignore[arg-type]

    def test_is_a_derivation(self):
        X = Symbol("X")
        assert isinstance(lie_derivative(X), Derivation)

    def test_operator_degree_is_zero(self):
        X = Symbol("X")
        assert degree_of(lie_derivative(X)) == Degree.const(0)

    def test_definitions_constant_exposed(self):
        assert "flow" in DEFINITIONS
        assert "cartan" in DEFINITIONS


class TestCartanExpansion:
    def test_shape_is_sum_of_two_compositions(self):
        """d ∘ ι_X + ι_X ∘ d."""
        X = Symbol("X")
        iota_X = interior(X)
        expanded = cartan_expansion(X, iota=iota_X)
        assert expanded == Sum(
            compose(d, iota_X),
            compose(iota_X, d),
        )

    def test_default_d_is_singleton(self):
        X = Symbol("X")
        iota_X = interior(X)
        expanded = cartan_expansion(X, iota=iota_X)
        # First composition starts with d, not a fresh instance.
        first = expanded.children[0]
        assert isinstance(first, Product)
        assert first.children[0] is d

    def test_custom_d_respected(self):
        X = Symbol("X")
        d_E = ExteriorDerivative("d_E")
        iota_X = interior(X)
        expanded = cartan_expansion(X, d=d_E, iota=iota_X)
        first = expanded.children[0]
        assert first.children[0] is d_E


class TestCartanObstruction:
    def test_shape_is_lhs_minus_rhs(self):
        """L_X(ω) − (d ∘ ι_X + ι_X ∘ d)(ω)."""
        X = Symbol("X")
        omega = Symbol("omega")
        L = lie_derivative(X)
        iota_X = interior(X)
        obs = cartan_obstruction(L, omega, iota=iota_X)
        expected = Sum(
            Act(L, omega),
            Neg(Act(cartan_expansion(X, iota=iota_X), omega)),
        )
        assert obs == expected
