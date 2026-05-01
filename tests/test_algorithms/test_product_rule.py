"""Tests for jacopy.algorithms.product_rule."""

import pytest

from jacopy.algebra.derivation import Act, Derivation, compose
from jacopy.algorithms.product_rule import ProductRule, product_rule
from jacopy.core.expr import Integer, Neg, Product, Sum, Symbol
from jacopy.core.properties import Graded, Scalar
from jacopy.core.registry import PropertyRegistry


# --------------------------------------------------------------------- #
# Fixtures                                                               #
# --------------------------------------------------------------------- #


@pytest.fixture
def reg():
    r = PropertyRegistry()
    # Declare a zoo of symbols with known degrees.
    r.declare(Symbol("f"), Scalar())
    r.declare(Symbol("g"), Scalar())
    r.declare(Symbol("a"), Graded(degree=1))  # odd
    r.declare(Symbol("b"), Graded(degree=1))  # odd
    r.declare(Symbol("c"), Graded(degree=2))  # even
    return r


# --------------------------------------------------------------------- #
# Linearity                                                              #
# --------------------------------------------------------------------- #


class TestLinearity:
    def test_act_on_sum_distributes(self, reg):
        d = Derivation("d", degree=1)
        x, y = Symbol("x"), Symbol("y")
        reg.declare(x, Scalar())
        reg.declare(y, Scalar())
        out = product_rule(d(Sum(x, y)), reg)
        assert out == Sum(Act(d, x), Act(d, y))

    def test_act_on_neg(self, reg):
        d = Derivation("d", degree=1)
        x = Symbol("x")
        reg.declare(x, Scalar())
        out = product_rule(d(Neg(x)), reg)
        assert out == Neg(Act(d, x))

    def test_act_on_atom_stays_inert(self):
        d = Derivation("d", degree=1)
        x = Symbol("x")
        # No rewrite to perform, no registry needed.
        assert product_rule(d(x)) == Act(d, x)


# --------------------------------------------------------------------- #
# Graded Leibniz                                                         #
# --------------------------------------------------------------------- #


class TestGradedLeibniz:
    def test_even_derivation_two_factors(self, reg):
        """|d|=0 → no sign: d(x*y) = d(x)*y + x*d(y)."""
        d = Derivation("d", degree=0)
        x, y = Symbol("x"), Symbol("y")
        reg.declare(x, Scalar())
        reg.declare(y, Scalar())
        out = product_rule(d(Product(x, y)), reg)
        assert out == Sum(
            Product(Act(d, x), y),
            Product(x, Act(d, y)),
        )

    def test_odd_derivation_two_odd_factors(self, reg):
        """|d|=1, |a|=1 → sign on second term is (-1)^1 = -1.

        d(a*b) = d(a)*b − a*d(b)
        """
        d = Derivation("d", degree=1)
        a, b = Symbol("a"), Symbol("b")
        out = product_rule(d(Product(a, b)), reg)
        assert out == Sum(
            Product(Act(d, a), b),
            Neg(Product(a, Act(d, b))),
        )

    def test_three_factors(self, reg):
        """d(a*b*c) with |d|=1, |a|=|b|=1, |c|=2.

        Running degrees: 0, 1, 2. Signs: +, −, +.
        """
        d = Derivation("d", degree=1)
        a, b, c = Symbol("a"), Symbol("b"), Symbol("c")
        out = product_rule(d(Product(a, b, c)), reg)
        assert out == Sum(
            Product(Act(d, a), b, c),
            Neg(Product(a, Act(d, b), c)),
            Product(a, b, Act(d, c)),
        )

    def test_scalars_carry_no_sign(self, reg):
        """Scalar factors contribute degree 0, running parity stays."""
        d = Derivation("d", degree=1)
        f, a, b = Symbol("f"), Symbol("a"), Symbol("b")
        # f scalar, a,b odd. Running degrees: 0, 0, 1.
        # Signs: +, +, −.
        out = product_rule(d(Product(f, a, b)), reg)
        assert out == Sum(
            Product(Act(d, f), a, b),
            Product(f, Act(d, a), b),
            Neg(Product(f, a, Act(d, b))),
        )


# --------------------------------------------------------------------- #
# Bottom-up / recursion                                                  #
# --------------------------------------------------------------------- #


class TestRecursion:
    def test_nested_act_expands_inner_first(self, reg):
        """d(e(a*b)), inner e expanded first, then outer d sees a Sum."""
        d = Derivation("d", degree=0)
        e = Derivation("e", degree=0)
        a, b = Symbol("a"), Symbol("b")
        out = product_rule(d(e(Product(a, b))), reg)
        # Inner: e(a*b) = e(a)*b + a*e(b).
        # Outer applies to a Sum → d distributes:
        # d(e(a)*b) + d(a*e(b))
        #   = d(e(a))*b + e(a)*d(b) + d(a)*e(b) + a*d(e(b))  (|d|=0, no signs)
        # Sum.make flattens nested Sum results into a single flat 4-term Sum.
        assert out == Sum(
            Product(Act(d, Act(e, a)), b),
            Product(Act(e, a), Act(d, b)),
            Product(Act(d, a), Act(e, b)),
            Product(a, Act(d, Act(e, b))),
        )

    def test_expansion_inside_larger_tree(self, reg):
        """product_rule descends into non-Act parents too."""
        d = Derivation("d", degree=0)
        x, y, z = Symbol("x"), Symbol("y"), Symbol("z")
        for s in (x, y, z):
            reg.declare(s, Scalar())
        expr = Sum(z, d(Product(x, y)))
        out = product_rule(expr, reg)
        assert out == Sum(
            z,
            Sum(Product(Act(d, x), y), Product(x, Act(d, y))),
        )

    def test_atom_untouched(self):
        x = Symbol("x")
        assert product_rule(x) is x


# --------------------------------------------------------------------- #
# Degree requirements                                                    #
# --------------------------------------------------------------------- #


class TestDegreeErrors:
    def test_unregistered_factor_raises(self, reg):
        """An unknown factor that sits before a split aborts expansion."""
        d = Derivation("d", degree=1)
        a = Symbol("a")  # declared in fixture as Graded(1)
        # q is not declared anywhere.
        q = Symbol("q")
        # First factor's degree is needed to compute the running parity
        # before the second split point.
        with pytest.raises(ValueError):
            product_rule(d(Product(q, a)), reg)

    def test_symbolic_parity_mid_product_raises(self, reg):
        """A symbolic-degree factor blocking a split should raise."""
        from jacopy.core.symbolic_degree import Degree

        d = Derivation("d", degree=1)
        alpha = Symbol("α")
        reg.declare(alpha, Graded(degree=Degree.var("|α|")))
        # a is already Graded(1) in the fixture.
        a = Symbol("a")
        # Factors α, a. Running parity before a's split: |α|, symbolic.
        with pytest.raises(ValueError, match="symbolic"):
            product_rule(d(Product(alpha, a)), reg)


# --------------------------------------------------------------------- #
# Algorithm wrapper                                                      #
# --------------------------------------------------------------------- #


class TestComposedOperator:
    """Act(compose(D1, ..., Dn), x) unfolds right-to-left into nested
    Acts and Leibniz-expands each layer. This is the only place
    composition actually acts on an operand."""

    def test_composed_op_on_atom_unfolds(self, reg):
        """Atom operand → no Leibniz to apply; just nested Acts."""
        D1 = Derivation("D1", degree=0)
        D2 = Derivation("D2", degree=0)
        x = Symbol("x")
        reg.declare(x, Scalar())
        out = product_rule(Act(compose(D1, D2), x), reg)
        assert out == Act(D1, Act(D2, x))

    def test_composed_op_on_product_expands_each_layer(self, reg):
        """(D1 ∘ D2)(a*b) with |D1|=|D2|=0 (no signs):

        Inner D2(a*b) → D2(a)*b + a*D2(b).
        Outer D1 distributes over the Sum, then Leibniz on each product:
          D1(D2(a)*b) → D1(D2(a))*b + D2(a)*D1(b)
          D1(a*D2(b)) → D1(a)*D2(b) + a*D1(D2(b))
        Sum.make flattens to a single 4-term Sum.
        """
        D1 = Derivation("D1", degree=0)
        D2 = Derivation("D2", degree=0)
        a, b = Symbol("a"), Symbol("b")
        out = product_rule(Act(compose(D1, D2), Product(a, b)), reg)
        assert out == Sum(
            Product(Act(D1, Act(D2, a)), b),
            Product(Act(D2, a), Act(D1, b)),
            Product(Act(D1, a), Act(D2, b)),
            Product(a, Act(D1, Act(D2, b))),
        )

    def test_single_op_compose_equivalent_to_bare_op(self, reg):
        """compose(D) is D itself; Act(compose(D), ...) behaves like Act(D, ...)."""
        D = Derivation("D", degree=0)
        a, b = Symbol("a"), Symbol("b")
        direct = product_rule(Act(D, Product(a, b)), reg)
        via_compose = product_rule(Act(compose(D), Product(a, b)), reg)
        assert direct == via_compose


class TestProductRuleAlgorithm:
    def test_can_apply_true(self):
        d = Derivation("d", degree=1)
        x = Symbol("x")
        assert ProductRule().can_apply(d(x))

    def test_can_apply_false_no_act(self):
        x = Symbol("x")
        assert not ProductRule().can_apply(Sum(x, Symbol("y")))

    def test_run_returns_stepresult(self, reg):
        d = Derivation("d", degree=1)
        a, b = Symbol("a"), Symbol("b")
        alg = ProductRule(reg)
        r = alg.run(d(Product(a, b)))
        assert r.changed
        assert r.after == Sum(
            Product(Act(d, a), b),
            Neg(Product(a, Act(d, b))),
        )

    def test_run_unchanged_on_plain_expr(self):
        x = Symbol("x")
        r = ProductRule().run(x)
        assert not r.changed
        assert r.after is x
