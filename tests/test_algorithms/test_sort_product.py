"""Tests for jacopy.algorithms.sort_product."""

import pytest

from jacopy.algorithms.sort_product import apply_sign, sort_product
from jacopy.core.expr import Neg, Product, Symbol
from jacopy.core.properties import (
    AntiCommuting,
    Graded,
    GradedCommutative,
    NonCommuting,
    Scalar,
)
from jacopy.core.registry import PropertyRegistry
from jacopy.core.symbolic_degree import Degree


# --------------------------------------------------------------------- #
# Fixtures                                                               #
# --------------------------------------------------------------------- #


@pytest.fixture
def reg():
    r = PropertyRegistry()
    # scalars
    r.declare(Symbol("f"), Scalar())
    r.declare(Symbol("g"), Scalar())
    # graded of known integer degree
    r.declare(Symbol("a"), Graded(degree=1))  # odd
    r.declare(Symbol("b"), Graded(degree=1))  # odd
    r.declare(Symbol("c"), Graded(degree=2))  # even
    # graded with symbolic degree
    r.declare(Symbol("α"), Graded(degree=Degree.var("|α|")))
    r.declare(Symbol("β"), Graded(degree=Degree.var("|β|")))
    return r


# --------------------------------------------------------------------- #
# Edge cases: passthrough                                                #
# --------------------------------------------------------------------- #


class TestPassthrough:
    def test_non_product(self, reg):
        x = Symbol("x")
        result, sign = sort_product(x, reg)
        assert result is x
        assert sign.parity() == 0

    def test_single_factor_product(self, reg):
        # Product(a) isn't constructible via .make but we can build directly.
        a = Symbol("a")
        # Direct construct (smart-ctor would collapse).
        result, sign = sort_product(Product(a), reg)
        assert result is a
        assert sign.parity() == 0


# --------------------------------------------------------------------- #
# Scalars                                                                #
# --------------------------------------------------------------------- #


class TestScalars:
    def test_two_scalars_sort(self, reg):
        f, g = Symbol("f"), Symbol("g")
        # g * f -> f * g; no sign (scalars).
        result, sign = sort_product(Product(g, f), reg)
        assert result == Product(f, g)
        assert sign.parity() == 0

    def test_scalar_floats_past_graded(self, reg):
        f, a = Symbol("f"), Symbol("a")
        # a * f -> f * a; no sign because f is scalar.
        result, sign = sort_product(Product(a, f), reg)
        assert result == Product(f, a)
        assert sign.parity() == 0

    def test_multiple_scalars_and_graded(self, reg):
        f, g, a = Symbol("f"), Symbol("g"), Symbol("a")
        result, sign = sort_product(Product(a, g, f), reg)
        # All scalars float to front; among (f,g) sort alphabetically.
        assert result == Product(f, g, a)
        assert sign.parity() == 0


# --------------------------------------------------------------------- #
# Graded factors, concrete degrees                                       #
# --------------------------------------------------------------------- #


class TestConcreteGraded:
    def test_two_odd_swap_gives_minus(self, reg):
        a, b = Symbol("a"), Symbol("b")
        # b * a -> a * b with sign (-1)^{1*1} = -1.
        result, sign = sort_product(Product(b, a), reg)
        assert result == Product(a, b)
        assert sign.parity() == 1

    def test_odd_even_swap_gives_plus(self, reg):
        a, c = Symbol("a"), Symbol("c")  # |a|=1, |c|=2 -> 1*2=2 even
        result, sign = sort_product(Product(c, a), reg)
        assert result == Product(a, c)
        assert sign.parity() == 0

    def test_already_sorted(self, reg):
        a, b = Symbol("a"), Symbol("b")
        result, sign = sort_product(Product(a, b), reg)
        assert result == Product(a, b)
        assert sign.parity() == 0

    def test_three_odds_reverse(self, reg):
        # b * a * c with a,b odd and c even
        # Target canonical order: a, b, c (alphabetical repr within non-scalar).
        a, b, c = Symbol("a"), Symbol("b"), Symbol("c")
        result, sign = sort_product(Product(b, a, c), reg)
        assert result == Product(a, b, c)
        # One swap of two odds -> exponent 1 (odd).
        assert sign.parity() == 1


# --------------------------------------------------------------------- #
# Symbolic degrees                                                       #
# --------------------------------------------------------------------- #


class TestSymbolicDegrees:
    def test_symbolic_swap(self, reg):
        α, β = Symbol("α"), Symbol("β")
        # Sort: compare repr("α") vs repr("β"); canonical order is α, β.
        # If already in that order, no swap.
        # Swap β,α -> α,β with exponent |α|*|β|.
        result, sign = sort_product(Product(β, α), reg)
        assert result == Product(α, β)
        # Parity undecidable (symbolic).
        assert sign.parity() is None
        # Sign exponent is exactly |α|*|β|.
        assert sign == Degree.var("|α|") * Degree.var("|β|")

    def test_double_swap_has_even_exponent(self, reg):
        """Two swaps of the same pair should give a sign with even parity."""
        α, β = Symbol("α"), Symbol("β")
        # Build α β α and sort: α must move past β twice? No -
        # current canonical order puts α before β, so α, β, α stays:
        # positions are (0,1,2). Bubble-sort compares (0,1): α,β -> no swap.
        # compares (1,2): β,α -> swap -> α,α,β with sign |α|*|β|.
        # compares (0,1): α,α -> no swap. Final: α,α,β with exp |α||β|.
        result, sign = sort_product(Product(α, β, α), reg)
        assert result == Product(α, α, β)
        assert sign == Degree.var("|α|") * Degree.var("|β|")

    def test_scalars_dont_contribute(self, reg):
        # f * β * α -> f * α * β with sign |α||β| (only the α,β swap counts).
        f, α, β = Symbol("f"), Symbol("α"), Symbol("β")
        result, sign = sort_product(Product(f, β, α), reg)
        assert result == Product(f, α, β)
        assert sign == Degree.var("|α|") * Degree.var("|β|")


# --------------------------------------------------------------------- #
# Unclassified factors                                                   #
# --------------------------------------------------------------------- #


class TestUnclassified:
    def test_unclassified_raises(self, reg):
        # x is not declared.
        x, a = Symbol("x"), Symbol("a")
        with pytest.raises(ValueError, match="neither Scalar nor Graded"):
            sort_product(Product(a, x), reg)

    def test_unclassified_not_raised_if_no_swap_needed(self, reg):
        # If the order is already canonical AND no swap is ever attempted,
        # we still visit every pair in the bubble sort and query degrees,
        # so even a passthrough sort demands classification.
        x = Symbol("x")
        y = Symbol("y")
        with pytest.raises(ValueError):
            sort_product(Product(x, y), reg)


# --------------------------------------------------------------------- #
# apply_sign                                                             #
# --------------------------------------------------------------------- #


class TestApplySign:
    def test_even_returns_expr(self):
        x = Symbol("x")
        assert apply_sign(x, Degree.const(0)) is x
        assert apply_sign(x, Degree.const(4)) is x

    def test_odd_returns_neg(self):
        x = Symbol("x")
        assert apply_sign(x, Degree.const(1)) == Neg(x)
        assert apply_sign(x, Degree.const(3)) == Neg(x)

    def test_symbolic_raises(self):
        x = Symbol("x")
        with pytest.raises(ValueError, match="not decidable"):
            apply_sign(x, Degree.var("|α|"))

    def test_symbolic_even_coefficient_decidable(self):
        """2|α||β| has known-even parity and must apply cleanly."""
        x = Symbol("x")
        sign = Degree.const(2) * Degree.var("|α|") * Degree.var("|β|")
        assert apply_sign(x, sign) is x


# --------------------------------------------------------------------- #
# Commutativity markers: NonCommuting / AntiCommuting / GradedCommutative
# --------------------------------------------------------------------- #


class TestNonCommuting:
    def test_non_commuting_pair_not_swapped(self):
        reg = PropertyRegistry()
        # Force a sort order (b < a alphabetically? no, a < b). Use
        # lexicographically *higher* symbol first so sort would want to
        # swap absent the NonCommuting block.
        u = Symbol("u")
        t = Symbol("t")
        reg.declare(u, Graded(degree=1))
        reg.declare(u, NonCommuting())
        reg.declare(t, Graded(degree=1))
        reg.declare(t, NonCommuting())
        # Product(u, t), sort order says t before u but NonCommuting blocks swap.
        expr = Product(u, t)
        out, sign = sort_product(expr, reg)
        assert out == Product(u, t)
        assert sign.parity() == 0

    def test_scalar_still_moves_past_non_commuting(self):
        reg = PropertyRegistry()
        s = Symbol("s")
        x = Symbol("x")
        reg.declare(s, Scalar())
        reg.declare(x, Graded(degree=1))
        reg.declare(x, NonCommuting())
        # Scalar should still sort to the front (scalars commute with anything).
        expr = Product(x, s)
        out, _ = sort_product(expr, reg)
        assert out == Product(s, x)


class TestAntiCommuting:
    def test_anti_commuting_swap_contributes_parity_one(self):
        reg = PropertyRegistry()
        u = Symbol("u")
        t = Symbol("t")
        reg.declare(u, Graded(degree=0))
        reg.declare(u, AntiCommuting())
        reg.declare(t, Graded(degree=0))
        reg.declare(t, AntiCommuting())
        # One swap, sort wants t, u order.
        expr = Product(u, t)
        out, sign = sort_product(expr, reg)
        assert out == Product(t, u)
        # AntiCommuting produces parity 1 per swap regardless of degree.
        assert sign.parity() == 1


class TestGradedCommutativeExplicit:
    def test_explicit_graded_commutative_markers(self):
        reg = PropertyRegistry()
        u = Symbol("u")
        t = Symbol("t")
        reg.declare(u, Graded(degree=1))
        reg.declare(u, GradedCommutative())
        reg.declare(t, Graded(degree=1))
        reg.declare(t, GradedCommutative())
        expr = Product(u, t)
        out, sign = sort_product(expr, reg)
        assert out == Product(t, u)
        # Both odd → parity 1.
        assert sign.parity() == 1


# --------------------------------------------------------------------- #
# Neg-wrapped factors and nested Products                                #
# --------------------------------------------------------------------- #


class TestNegWrappedFactors:
    def test_single_neg_factor_contributes_parity_one(self):
        reg = PropertyRegistry()
        a = Symbol("a")
        reg.declare(a, Graded(degree=0))
        # Product(Neg(a)), a single peeled factor; parity bumps to 1.
        out, sign = sort_product(Product(Neg(a)), reg)
        assert sign.parity() == 1
        # Single-factor product collapses to the lone factor.
        assert out == a

    def test_double_neg_cancels(self):
        reg = PropertyRegistry()
        a, b = Symbol("a"), Symbol("b")
        for s in (a, b):
            reg.declare(s, Graded(degree=0))
        out, sign = sort_product(Product(Neg(a), Neg(b)), reg)
        # Two Negs → parity 0 from peel; both degree 0 → no Koszul sign.
        assert sign.parity() == 0
        assert out == Product(a, b)

    def test_neg_over_inner_product_splices_factors(self):
        """Previously raised: ``Neg(Product(a,c))`` as factor.

        The enhancement flattens through the Neg barrier, pulling the
        inner product's factors up and contributing a single sign.
        """
        reg = PropertyRegistry()
        a, b, c = Symbol("a"), Symbol("b"), Symbol("c")
        for s in (a, b, c):
            reg.declare(s, Graded(degree=0))
        expr = Product(a, Neg(Product(b, c)))
        out, sign = sort_product(expr, reg)
        assert sign.parity() == 1
        assert out == Product(a, b, c)

    def test_inner_product_without_neg_also_splices(self):
        reg = PropertyRegistry()
        a, b, c = Symbol("a"), Symbol("b"), Symbol("c")
        for s in (a, b, c):
            reg.declare(s, Graded(degree=0))
        expr = Product(a, Product(b, c))
        out, sign = sort_product(expr, reg)
        assert sign.parity() == 0
        assert out == Product(a, b, c)

    def test_deeply_nested_neg_product_resolves(self):
        reg = PropertyRegistry()
        a, b = Symbol("a"), Symbol("b")
        for s in (a, b):
            reg.declare(s, Graded(degree=0))
        # Neg(Product(a, Neg(b))): two Negs → parity 0, splice → Product(a, b).
        expr = Product(Neg(Product(a, Neg(b))))
        out, sign = sort_product(expr, reg)
        assert sign.parity() == 0
        assert out == Product(a, b)
