"""Tests for jacopy.algorithms.flatten."""

from jacopy.algorithms.flatten import Flatten, flatten
from jacopy.core.expr import (
    Integer,
    Neg,
    Power,
    Product,
    Sum,
    Symbol,
)


class TestFlattenSum:
    def test_nested_sum(self):
        x, y, z = Symbol("x"), Symbol("y"), Symbol("z")
        expr = Sum(Sum(x, y), z)
        assert flatten(expr) == Sum(x, y, z)

    def test_deeply_nested(self):
        x, y, z, w = Symbol("x"), Symbol("y"), Symbol("z"), Symbol("w")
        expr = Sum(Sum(Sum(x, y), z), w)
        assert flatten(expr) == Sum(x, y, z, w)

    def test_single_child_collapses(self):
        x = Symbol("x")
        expr = Sum(x)
        assert flatten(expr) is x


class TestFlattenProduct:
    def test_nested_product(self):
        x, y, z = Symbol("x"), Symbol("y"), Symbol("z")
        expr = Product(Product(x, y), z)
        assert flatten(expr) == Product(x, y, z)

    def test_nested_does_not_reorder(self):
        x, y, z = Symbol("x"), Symbol("y"), Symbol("z")
        expr = Product(x, Product(y, z))
        assert flatten(expr).children == (x, y, z)

    def test_single_child_collapses(self):
        x = Symbol("x")
        assert flatten(Product(x)) is x


class TestFlattenPreservesOtherNodes:
    def test_atom_passthrough(self):
        x = Symbol("x")
        assert flatten(x) is x
        assert flatten(Integer(5)) == Integer(5)

    def test_neg_descends(self):
        x, y = Symbol("x"), Symbol("y")
        expr = Neg(Sum(Sum(x, y)))
        # Sum inside Neg must flatten even though Neg is not associative.
        out = flatten(expr)
        assert out == Neg(Sum(x, y))

    def test_power_children_flatten(self):
        x, y = Symbol("x"), Symbol("y")
        expr = Power(Sum(Sum(x, y)), Integer(2))
        assert flatten(expr) == Power(Sum(x, y), Integer(2))


class TestIdempotent:
    def test_already_flat(self):
        x, y, z = Symbol("x"), Symbol("y"), Symbol("z")
        expr = Sum(x, y, z)
        assert flatten(expr) == expr

    def test_double_call(self):
        x, y, z = Symbol("x"), Symbol("y"), Symbol("z")
        expr = Sum(Sum(x, y), Product(Product(y, z)))
        once = flatten(expr)
        twice = flatten(once)
        assert once == twice


class TestAlgorithmWrapper:
    def test_can_apply_true_on_nested(self):
        x, y = Symbol("x"), Symbol("y")
        assert Flatten().can_apply(Sum(Sum(x, y)))

    def test_can_apply_false_on_flat(self):
        x, y = Symbol("x"), Symbol("y")
        assert not Flatten().can_apply(Sum(x, y))

    def test_can_apply_false_on_atom(self):
        assert not Flatten().can_apply(Symbol("x"))

    def test_run_returns_stepresult(self):
        x, y, z = Symbol("x"), Symbol("y"), Symbol("z")
        r = Flatten().run(Sum(Sum(x, y), z))
        assert r.changed
        assert r.after == Sum(x, y, z)
