"""Tests for jacopy.algorithms.distribute."""

from jacopy.algorithms.distribute import Distribute, distribute
from jacopy.core.expr import (
    Integer,
    Neg,
    Product,
    Sum,
    Symbol,
)


class TestBasicDistribute:
    def test_left_distribute(self):
        a, b, c = Symbol("a"), Symbol("b"), Symbol("c")
        expr = Product(a, Sum(b, c))
        out = distribute(expr)
        assert out == Sum(Product(a, b), Product(a, c))

    def test_right_distribute(self):
        a, b, c = Symbol("a"), Symbol("b"), Symbol("c")
        expr = Product(Sum(a, b), c)
        out = distribute(expr)
        assert out == Sum(Product(a, c), Product(b, c))

    def test_atom_passthrough(self):
        x = Symbol("x")
        assert distribute(x) is x

    def test_sum_without_product_passthrough(self):
        x, y = Symbol("x"), Symbol("y")
        expr = Sum(x, y)
        assert distribute(expr) == expr


class TestPreservesFactorOrder:
    def test_non_commutative_order(self):
        # X * (Y + Z) * W must yield X*Y*W + X*Z*W (order preserved).
        X, Y, Z, W = Symbol("X"), Symbol("Y"), Symbol("Z"), Symbol("W")
        expr = Product(X, Sum(Y, Z), W)
        out = distribute(expr)
        assert out == Sum(Product(X, Y, W), Product(X, Z, W))

    def test_factors_never_reordered(self):
        a, b, c, d = Symbol("a"), Symbol("b"), Symbol("c"), Symbol("d")
        expr = Product(a, b, Sum(c, d))
        out = distribute(expr)
        # a*b*c + a*b*d, NOT c*a*b + d*a*b.
        assert out == Sum(Product(a, b, c), Product(a, b, d))


class TestMultipleSums:
    def test_two_sums_multiply(self):
        # (a + b) * (c + d) = a*c + a*d + b*c + b*d (order preserved).
        a, b, c, d = Symbol("a"), Symbol("b"), Symbol("c"), Symbol("d")
        expr = Product(Sum(a, b), Sum(c, d))
        out = distribute(expr)
        assert out == Sum(
            Product(a, c),
            Product(a, d),
            Product(b, c),
            Product(b, d),
        )

    def test_three_sums(self):
        a, b = Symbol("a"), Symbol("b")
        expr = Product(Sum(a, b), Sum(a, b), Sum(a, b))
        out = distribute(expr)
        # 2^3 = 8 terms.
        assert isinstance(out, Sum)
        assert len(out.children) == 8


class TestNestedDistribute:
    def test_recurses_into_children(self):
        a, b, c, d = Symbol("a"), Symbol("b"), Symbol("c"), Symbol("d")
        # Sum containing a distributable product must still distribute.
        inner = Product(a, Sum(b, c))
        expr = Sum(inner, d)
        out = distribute(expr)
        assert out == Sum(Sum(Product(a, b), Product(a, c)), d)

    def test_distributes_under_neg(self):
        # Neg doesn't stop distribution from touching its child.
        a, b, c = Symbol("a"), Symbol("b"), Symbol("c")
        expr = Neg(Product(a, Sum(b, c)))
        out = distribute(expr)
        assert out == Neg(Sum(Product(a, b), Product(a, c)))


class TestScalarFactors:
    def test_integer_factor(self):
        a, b = Symbol("a"), Symbol("b")
        expr = Product(Integer(2), Sum(a, b))
        out = distribute(expr)
        assert out == Sum(Product(Integer(2), a), Product(Integer(2), b))


class TestIdempotent:
    def test_already_distributed(self):
        a, b, c = Symbol("a"), Symbol("b"), Symbol("c")
        expr = Sum(Product(a, b), Product(a, c))
        assert distribute(expr) == expr

    def test_double_call(self):
        a, b, c = Symbol("a"), Symbol("b"), Symbol("c")
        expr = Product(a, Sum(b, c))
        once = distribute(expr)
        twice = distribute(once)
        assert once == twice


class TestAlgorithmWrapper:
    def test_can_apply_true(self):
        a, b, c = Symbol("a"), Symbol("b"), Symbol("c")
        assert Distribute().can_apply(Product(a, Sum(b, c)))

    def test_can_apply_false_on_pure_sum(self):
        a, b = Symbol("a"), Symbol("b")
        assert not Distribute().can_apply(Sum(a, b))

    def test_run_returns_stepresult(self):
        a, b, c = Symbol("a"), Symbol("b"), Symbol("c")
        r = Distribute().run(Product(a, Sum(b, c)))
        assert r.changed
