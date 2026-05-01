"""Tests for jacopy.algorithms.collect_terms."""

from jacopy.algorithms.collect_terms import CollectTerms, collect_terms
from jacopy.core.expr import (
    Integer,
    Neg,
    Product,
    Sum,
    Symbol,
    Zero,
)


class TestBasic:
    def test_like_terms_merge(self):
        x = Symbol("x")
        out = collect_terms(Sum(x, x))
        assert out == Product(Integer(2), x)

    def test_coefficients_add(self):
        x = Symbol("x")
        out = collect_terms(Sum(Product(Integer(2), x), Product(Integer(3), x)))
        assert out == Product(Integer(5), x)

    def test_opposite_terms_cancel(self):
        x = Symbol("x")
        out = collect_terms(Sum(x, Neg(x)))
        assert out is Zero

    def test_distinct_terms_stay(self):
        x, y = Symbol("x"), Symbol("y")
        out = collect_terms(Sum(x, y))
        # Order preservation is a bonus: CollectTerms doesn't reorder.
        assert out == Sum(x, y)

    def test_constant_fold(self):
        out = collect_terms(Sum(Integer(2), Integer(3)))
        assert out == Integer(5)


class TestNested:
    def test_flat_nested_sum(self):
        x = Symbol("x")
        # Sum(Sum(x, x), x) collapses to 3*x via the flatten step.
        out = collect_terms(Sum(Sum(x, x), x))
        assert out == Product(Integer(3), x)

    def test_inside_product(self):
        x, y = Symbol("x"), Symbol("y")
        # Product is a barrier, Sum-of-Sum only merges within a Sum.
        inner = Sum(x, x)
        expr = Product(inner, y)
        out = collect_terms(expr)
        # (x+x) collapsed to 2*x, Product rebuilt.
        assert out == Product(Product(Integer(2), x), y)

    def test_atom_passthrough(self):
        x = Symbol("x")
        assert collect_terms(x) is x


class TestIdempotent:
    def test_already_collected(self):
        x, y = Symbol("x"), Symbol("y")
        expr = Sum(Product(Integer(2), x), y)
        assert collect_terms(expr) == expr

    def test_double_call(self):
        x, y = Symbol("x"), Symbol("y")
        expr = Sum(x, x, y, y, y)
        once = collect_terms(expr)
        twice = collect_terms(once)
        assert once == twice


class TestAlgorithmWrapper:
    def test_can_apply_multi_child_sum(self):
        x = Symbol("x")
        assert CollectTerms().can_apply(Sum(x, x))

    def test_can_apply_false_on_atom(self):
        assert not CollectTerms().can_apply(Symbol("x"))

    def test_run_returns_stepresult(self):
        x = Symbol("x")
        r = CollectTerms().run(Sum(x, x))
        assert r.changed
        assert r.after == Product(Integer(2), x)
