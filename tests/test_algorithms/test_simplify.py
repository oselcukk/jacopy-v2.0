"""Tests for jacopy.algorithms.simplify, the Faz 2 pipeline."""

import pytest

from jacopy.algorithms.simplify import simplify
from jacopy.core.expr import (
    Integer,
    Neg,
    Product,
    Sum,
    Symbol,
    Zero,
)
from jacopy.core.properties import Graded, Scalar
from jacopy.core.registry import PropertyRegistry


class TestWithoutRegistry:
    def test_atom_passthrough(self):
        x = Symbol("x")
        assert simplify(x) is x

    def test_flatten_and_collect(self):
        x = Symbol("x")
        expr = Sum(Sum(x, x), Sum(x, x))
        out = simplify(expr)
        assert out == Product(Integer(4), x)

    def test_distribute_then_collect(self):
        a, b = Symbol("a"), Symbol("b")
        # a*(b+b) = a*b + a*b = 2*a*b
        expr = Product(a, Sum(b, b))
        out = simplify(expr)
        assert out == Product(Integer(2), a, b)

    def test_cancellation(self):
        x = Symbol("x")
        out = simplify(Sum(x, Neg(x)))
        assert out is Zero

    def test_empty_sum_collapses_via_flatten(self):
        out = simplify(Sum(Sum()))
        assert out is Zero


class TestWithRegistry:
    @pytest.fixture
    def reg(self):
        r = PropertyRegistry()
        return r

    def test_scalar_factor_moves_front(self, reg):
        a, b = Symbol("a"), Symbol("b")
        reg.declare(a, Scalar())
        reg.declare(b, Graded(degree=1))
        # (b * a), b has higher sort key; a should move to front.
        expr = Product(b, a)
        out = simplify(expr, reg)
        # Scalars bucket before graded; a first.
        assert out == Product(a, b)

    def test_koszul_sign_folded(self, reg):
        a, b = Symbol("a"), Symbol("b")
        reg.declare(a, Graded(degree=1))
        reg.declare(b, Graded(degree=1))
        # Both odd: swap flips sign.
        # repr("a") < repr("b") so Product(b, a) should sort to -Product(a, b).
        expr = Product(b, a)
        out = simplify(expr, reg)
        assert out == Neg(Product(a, b))

    def test_symbolic_sign_preserves_order(self, reg):
        """If sign parity is symbolic, the Product stays in pre-sort order."""
        from jacopy.core.symbolic_degree import Degree
        a, b = Symbol("a"), Symbol("b")
        reg.declare(a, Graded(degree=Degree.var("|α|")))
        reg.declare(b, Graded(degree=Degree.var("|β|")))
        expr = Product(b, a)
        out = simplify(expr, reg)
        # Unchanged at the product level (sign is symbolic).
        assert out == Product(b, a)

    def test_pipeline_full(self, reg):
        """distribute → sort → collect all working together."""
        a, b, c = Symbol("a"), Symbol("b"), Symbol("c")
        for s in (a, b, c):
            reg.declare(s, Scalar())
        # a*(b+c) + a*b = 2ab + ac (after distribute + collect).
        expr = Sum(Product(a, Sum(b, c)), Product(a, b))
        out = simplify(expr, reg)
        # Expected: 2*a*b + a*c (scalars commute, but order preserved
        # within a term; scalar-only so no sign).
        expected_terms = {Product(Integer(2), a, b), Product(a, c)}
        assert isinstance(out, Sum)
        assert set(out.children) == expected_terms


class TestFixPoint:
    def test_converges(self):
        x = Symbol("x")
        # A single pass already finds the fix-point on this input.
        expr = Sum(Sum(x, x), Sum(x, x, x))
        out = simplify(expr)
        assert out == Product(Integer(5), x)

    def test_max_iterations_argument_accepted(self):
        x = Symbol("x")
        out = simplify(Sum(x, x), max_iterations=1)
        assert out == Product(Integer(2), x)
