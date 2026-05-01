"""Tests for jacopy.calculus.interior."""

import pytest

from jacopy.algebra.derivation import Act, Derivation, compose, degree_of
from jacopy.algorithms.product_rule import product_rule
from jacopy.algorithms.simplify import simplify
from jacopy.calculus.exterior_d import ExteriorDerivative, d
from jacopy.calculus.interior import (
    InteriorProduct,
    apply_iota_axioms,
    apply_iota_squared_zero,
    interior,
)
from jacopy.core.expr import Integer, Neg, Product, Sum, Symbol
from jacopy.core.properties import Graded
from jacopy.core.registry import PropertyRegistry
from jacopy.core.symbolic_degree import Degree


class TestConstruction:
    def test_default_name_from_vector_field(self):
        X = Symbol("X")
        iota_X = interior(X)
        assert iota_X.name == "ι_X"
        assert iota_X.degree == Degree.const(-1)
        assert iota_X.vector_field is X

    def test_custom_name(self):
        X = Symbol("X")
        iota_X = interior(X, name="ι_X^E")
        assert iota_X.name == "ι_X^E"

    def test_rejects_non_expr(self):
        with pytest.raises(TypeError):
            InteriorProduct("X")  # type: ignore[arg-type]

    def test_equality_on_name_and_degree(self):
        X = Symbol("X")
        Y = Symbol("Y")
        assert interior(X) == interior(X)
        assert interior(X) != interior(Y)

    def test_is_a_derivation(self):
        X = Symbol("X")
        assert isinstance(interior(X), Derivation)

    def test_operator_degree_is_minus_one(self):
        X = Symbol("X")
        assert degree_of(interior(X)) == Degree.const(-1)


class TestLeibnizViaProductRule:
    def test_iota_of_product_graded_leibniz(self):
        """ι_X(α ∧ β) = ι_X(α) ∧ β + (−1)^{|α|} α ∧ ι_X(β).

        α degree 1, β degree 2 → second term sign flips.
        """
        reg = PropertyRegistry()
        X = Symbol("X")
        reg.declare(X, Graded(degree=0))
        alpha, beta = Symbol("alpha"), Symbol("beta")
        reg.declare(alpha, Graded(degree=1))
        reg.declare(beta, Graded(degree=2))

        iota_X = interior(X)
        expanded = product_rule(iota_X(Product(alpha, beta)), reg)
        expected = Sum(
            Product(Act(iota_X, alpha), beta),
            Neg(Product(alpha, Act(iota_X, beta))),
        )
        assert expanded == expected


class TestIotaSquaredZero:
    def test_nested_act_collapses(self):
        X = Symbol("X")
        iota_X = interior(X)
        assert apply_iota_squared_zero(iota_X(iota_X(Symbol("a"))), iota_X) == Integer(0)

    def test_composition_collapses(self):
        X = Symbol("X")
        iota_X = interior(X)
        expr = Act(compose(iota_X, iota_X), Symbol("a"))
        assert apply_iota_squared_zero(expr, iota_X) == Integer(0)

    def test_only_same_field_collapses(self):
        """ι_X ∘ ι_Y does NOT collapse, they're different operators."""
        X, Y = Symbol("X"), Symbol("Y")
        iota_X, iota_Y = interior(X), interior(Y)
        expr = iota_X(iota_Y(Symbol("a")))
        # Target = iota_X: outer is iota_X but inner is iota_Y, no match.
        assert apply_iota_squared_zero(expr, iota_X) == expr

    def test_zero_falls_out_of_enclosing_sum(self):
        reg = PropertyRegistry()
        a, b = Symbol("a"), Symbol("b")
        reg.declare(a, Graded(degree=2))
        reg.declare(b, Graded(degree=1))
        X = Symbol("X")
        iota_X = interior(X)
        before = Sum(iota_X(iota_X(a)), b)
        zapped = apply_iota_squared_zero(before, iota_X)
        assert simplify(zapped) == b


class TestIotaOnFunctions:
    def test_vanishes_on_degree_zero_symbol(self):
        """ι_X(f) = 0 when f is a declared 0-form."""
        reg = PropertyRegistry()
        f = Symbol("f")
        reg.declare(f, Graded(degree=0))
        X = Symbol("X")
        iota_X = interior(X)
        assert apply_iota_axioms(iota_X(f), iota_X, reg) == Integer(0)

    def test_does_not_touch_unregistered_operand(self):
        """Unknown-degree operand stays put, don't silently assume 0-form."""
        f = Symbol("f")  # not declared
        X = Symbol("X")
        iota_X = interior(X)
        expr = iota_X(f)
        assert apply_iota_axioms(expr, iota_X) == expr

    def test_does_not_touch_one_form(self):
        """ι_X on a 1-form stays inert, needs further reasoning."""
        reg = PropertyRegistry()
        alpha = Symbol("alpha")
        reg.declare(alpha, Graded(degree=1))
        X = Symbol("X")
        iota_X = interior(X)
        expr = iota_X(alpha)
        assert apply_iota_axioms(expr, iota_X, reg) == expr


class TestIotaOnExactOneForms:
    def test_pairing_with_derivation(self):
        """ι_X(df) = X(f) when X is supplied as a derivation on functions."""
        reg = PropertyRegistry()
        f = Symbol("f")
        reg.declare(f, Graded(degree=0))
        X_deriv = Derivation("X", degree=0)
        iota_X = interior(X_deriv)
        # ι_X(d(f)) → X(f).
        assert apply_iota_axioms(
            iota_X(d(f)), iota_X, reg, X=X_deriv
        ) == Act(X_deriv, f)

    def test_pairing_requires_explicit_X(self):
        """Without an X Derivation, ι_X(df) isn't paired, function clause vanishes
        df to X-dependent form only when X is given; absent X, df is degree 1
        so the 0-form vanish clause doesn't fire either, and the expression
        stays inert."""
        reg = PropertyRegistry()
        f = Symbol("f")
        reg.declare(f, Graded(degree=0))
        iota_X = interior(Symbol("X"))
        expr = iota_X(d(f))
        assert apply_iota_axioms(expr, iota_X, reg) == expr

    def test_pairing_restricted_to_matching_d(self):
        """A Lie-algebroid d_E inside ι_X(d_E(f)) doesn't pair against the standard d."""
        reg = PropertyRegistry()
        f = Symbol("f")
        reg.declare(f, Graded(degree=0))
        d_E = ExteriorDerivative("d_E")
        X_deriv = Derivation("X", degree=0)
        iota_X = interior(X_deriv)
        expr = iota_X(d_E(f))
        # Default d is the singleton, not d_E, so no pairing fires.
        result = apply_iota_axioms(expr, iota_X, reg, X=X_deriv)
        assert result == expr
        # Pass d=d_E explicitly and it pairs.
        assert apply_iota_axioms(
            expr, iota_X, reg, d=d_E, X=X_deriv
        ) == Act(X_deriv, f)

    def test_pairing_doesnt_fire_on_one_form_symbol(self):
        """ι_X(α) where α is a declared 1-form (not of the form d(f)) stays inert."""
        reg = PropertyRegistry()
        alpha = Symbol("alpha")
        reg.declare(alpha, Graded(degree=1))
        X_deriv = Derivation("X", degree=0)
        iota_X = interior(X_deriv)
        expr = iota_X(alpha)
        assert apply_iota_axioms(expr, iota_X, reg, X=X_deriv) == expr
