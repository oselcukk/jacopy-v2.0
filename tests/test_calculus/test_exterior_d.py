"""Tests for jacopy.calculus.exterior_d."""

import pytest

from jacopy.algebra.derivation import Act, Derivation, compose, degree_of
from jacopy.algorithms.product_rule import product_rule
from jacopy.algorithms.simplify import simplify
from jacopy.calculus.exterior_d import (
    ExteriorDerivative,
    apply_d_squared_zero,
    d,
)
from jacopy.core.expr import Integer, Neg, Product, Sum, Symbol
from jacopy.core.properties import Graded
from jacopy.core.registry import PropertyRegistry
from jacopy.core.symbolic_degree import Degree


class TestConstruction:
    def test_default_name_and_degree(self):
        e = ExteriorDerivative()
        assert e.name == "d"
        assert e.degree == Degree.const(1)

    def test_custom_name(self):
        e = ExteriorDerivative("d_E")
        assert e.name == "d_E"
        assert e.degree == Degree.const(1)

    def test_singleton_is_exterior_derivative(self):
        assert isinstance(d, ExteriorDerivative)
        assert d.name == "d"

    def test_equality_on_name_and_degree(self):
        # Two ExteriorDerivative("d") compare equal (Derivation is
        # structural on (name, degree)).
        assert ExteriorDerivative() == ExteriorDerivative()
        assert ExteriorDerivative("d_E") != ExteriorDerivative("d")

    def test_is_a_derivation(self):
        assert isinstance(d, Derivation)


class TestLeibnizViaProductRule:
    def test_d_of_product_graded_leibniz(self):
        """d(α ∧ β) = dα ∧ β + (−1)^{|α|} α ∧ dβ.

        Here α is degree 1, β is degree 2, so the sign on the second
        term is −1.
        """
        reg = PropertyRegistry()
        alpha = Symbol("alpha")
        beta = Symbol("beta")
        reg.declare(alpha, Graded(degree=1))
        reg.declare(beta, Graded(degree=2))

        expanded = product_rule(d(Product(alpha, beta)), reg)
        expected = Sum(
            Product(Act(d, alpha), beta),
            Neg(Product(alpha, Act(d, beta))),
        )
        assert expanded == expected

    def test_d_of_function_stays_inert(self):
        """d on a degree-0 operand is kept inert, it's the atomic 1-form df."""
        reg = PropertyRegistry()
        f = Symbol("f")
        reg.declare(f, Graded(degree=0))
        # Single-factor operand: no Leibniz to apply, node stays as Act.
        assert product_rule(d(f), reg) == Act(d, f)

    def test_d_is_linear(self):
        reg = PropertyRegistry()
        a, b = Symbol("a"), Symbol("b")
        reg.declare(a, Graded(degree=1))
        reg.declare(b, Graded(degree=1))
        assert product_rule(d(Sum(a, b)), reg) == Sum(
            Act(d, a), Act(d, b)
        )


class TestDSquaredZero:
    def test_nested_act_collapses(self):
        """d(d(x)) → 0 at the element level."""
        x = Symbol("x")
        assert apply_d_squared_zero(d(d(x))) == Integer(0)

    def test_composition_collapses(self):
        """(d ∘ d)(x) → 0 at the operator level."""
        x = Symbol("x")
        expr = Act(compose(d, d), x)
        assert apply_d_squared_zero(expr) == Integer(0)

    def test_composition_with_trailing_operators_collapses(self):
        """(d ∘ d ∘ E)(x) → 0, leading d² vanishes the whole stack."""
        x = Symbol("x")
        E = Derivation("E", degree=1)
        expr = Act(compose(d, d, E), x)
        assert apply_d_squared_zero(expr) == Integer(0)

    def test_zero_falls_out_of_enclosing_sum(self):
        """d(d(a)) + b → b after simplify composes with apply_d_squared_zero."""
        reg = PropertyRegistry()
        a, b = Symbol("a"), Symbol("b")
        reg.declare(a, Graded(degree=1))
        reg.declare(b, Graded(degree=1))
        before = Sum(d(d(a)), b)
        zapped = apply_d_squared_zero(before)
        assert simplify(zapped) == b

    def test_single_d_untouched(self):
        """A lone d is preserved."""
        x = Symbol("x")
        assert apply_d_squared_zero(d(x)) == d(x)

    def test_only_target_d_is_zeroed(self):
        """A Lie-algebroid d_E does not cancel against the standard d."""
        x = Symbol("x")
        d_E = ExteriorDerivative("d_E")
        # d(d_E(x)) is *not* d², because the outer and inner are
        # distinct operators.
        expr = d(d_E(x))
        assert apply_d_squared_zero(expr) == expr  # unchanged
        # But applying the pass targeted at d_E leaves the same
        # expression alone too, the leftmost d is not d_E.
        assert apply_d_squared_zero(expr, target=d_E) == expr

    def test_operator_degree_is_plus_one(self):
        assert degree_of(d) == Degree.const(1)
