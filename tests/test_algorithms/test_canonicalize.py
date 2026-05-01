"""Tests for jacopy.algorithms.canonicalize."""

import pytest

from jacopy.algorithms.canonicalize import (
    canonical_hash,
    canonicalize,
    semantically_equal,
)
from jacopy.core.expr import (
    Integer,
    Neg,
    One,
    Power,
    Product,
    Rational,
    Sum,
    Symbol,
    Zero,
)


# --------------------------------------------------------------------- #
# Atoms                                                                 #
# --------------------------------------------------------------------- #


class TestAtoms:
    def test_symbol_passthrough(self):
        x = Symbol("x")
        assert canonicalize(x) is x

    def test_integer_passthrough(self):
        assert canonicalize(Integer(7)) == Integer(7)

    def test_rational_passthrough(self):
        assert canonicalize(Rational(2, 3)) == Rational(2, 3)


# --------------------------------------------------------------------- #
# Neg                                                                   #
# --------------------------------------------------------------------- #


class TestNeg:
    def test_double_neg(self):
        x = Symbol("x")
        assert canonicalize(Neg(Neg(x))) == x

    def test_triple_neg(self):
        x = Symbol("x")
        assert canonicalize(Neg(Neg(Neg(x)))) == Neg(x)

    def test_neg_integer(self):
        assert canonicalize(Neg(Integer(5))) == Integer(-5)

    def test_neg_rational(self):
        assert canonicalize(Neg(Rational(2, 3))) == Rational(-2, 3)

    def test_neg_zero(self):
        assert canonicalize(Neg(Integer(0))) == Zero


# --------------------------------------------------------------------- #
# Sum                                                                   #
# --------------------------------------------------------------------- #


class TestSumConstantFolding:
    def test_fold_two_integers(self):
        assert canonicalize(Sum(Integer(2), Integer(3))) == Integer(5)

    def test_fold_mixed_numerics(self):
        # 2 + 1/3 = 7/3
        assert canonicalize(Sum(Integer(2), Rational(1, 3))) == Rational(7, 3)

    def test_fold_to_zero(self):
        assert canonicalize(Sum(Integer(5), Integer(-5))) == Zero

    def test_fold_rationals(self):
        assert (
            canonicalize(Sum(Rational(1, 2), Rational(1, 3)))
            == Rational(5, 6)
        )


class TestSumLikeTerms:
    def test_x_plus_x(self):
        x = Symbol("x")
        assert canonicalize(x + x) == Integer(2) * x

    def test_two_x_plus_three_x(self):
        x = Symbol("x")
        assert canonicalize(Integer(2) * x + Integer(3) * x) == Integer(5) * x

    def test_cancellation(self):
        x = Symbol("x")
        assert canonicalize(x + Neg(x)) == Zero

    def test_coefficient_of_one_dropped(self):
        x, y = Symbol("x"), Symbol("y")
        # After merging: 1*x + 1*y stays as x + y
        r = canonicalize(x + y)
        assert r == Sum(x, y) or r == Sum(y, x)  # either order depending on sort

    def test_merge_with_negated_factor(self):
        x = Symbol("x")
        # 3x + (-x) = 2x
        assert canonicalize(Integer(3) * x + Neg(x)) == Integer(2) * x


class TestSumOrdering:
    def test_deterministic(self):
        x, y = Symbol("x"), Symbol("y")
        # Same set of terms in different order should canonicalize identically.
        assert canonicalize(x + y) == canonicalize(y + x)

    def test_constant_at_end(self):
        x = Symbol("x")
        r = canonicalize(Integer(5) + x)
        # The sum has a non-constant term and a constant; constant trails.
        assert isinstance(r, Sum)
        assert r.children[-1] == Integer(5)


# --------------------------------------------------------------------- #
# Product                                                               #
# --------------------------------------------------------------------- #


class TestProductNumericFolding:
    def test_two_integers(self):
        assert canonicalize(Product(Integer(2), Integer(3))) == Integer(6)

    def test_interleaved(self):
        x = Symbol("x")
        # 2 * x * 3 -> 6 * x
        assert canonicalize(Product(Integer(2), x, Integer(3))) == Integer(6) * x

    def test_fold_to_zero(self):
        x = Symbol("x")
        assert canonicalize(Product(Integer(2), Zero, x)) == Zero

    def test_fold_rational_factor(self):
        x = Symbol("x")
        assert (
            canonicalize(Product(Rational(1, 2), Integer(3), x))
            == Rational(3, 2) * x
        )

    def test_coefficient_minus_one_becomes_neg(self):
        x = Symbol("x")
        # (-1) * x should canonicalize to Neg(x).
        assert canonicalize(Product(Integer(-1), x)) == Neg(x)

    def test_coefficient_one_disappears(self):
        x = Symbol("x")
        assert canonicalize(Product(Integer(1), x)) == x


class TestProductOrderPreserved:
    def test_non_numeric_order(self):
        x, y = Symbol("x"), Symbol("y")
        # x * y should NOT become y * x (non-commutative by default).
        r = canonicalize(x * y)
        assert r == Product(x, y)

    def test_sign_propagation(self):
        x, y = Symbol("x"), Symbol("y")
        # (-x) * y = -(x*y) after canonicalization
        r = canonicalize(Neg(x) * y)
        assert r == Neg(Product(x, y))


# --------------------------------------------------------------------- #
# Power                                                                 #
# --------------------------------------------------------------------- #


class TestPower:
    def test_exp_zero(self):
        x = Symbol("x")
        assert canonicalize(Power(x, Integer(0))) == One

    def test_exp_one(self):
        x = Symbol("x")
        assert canonicalize(Power(x, Integer(1))) == x

    def test_base_one(self):
        assert canonicalize(Power(One, Symbol("n"))) == One

    def test_zero_to_positive(self):
        assert canonicalize(Power(Zero, Integer(3))) == Zero

    def test_fold_numeric_power(self):
        assert canonicalize(Power(Integer(2), Integer(3))) == Integer(8)

    def test_fold_rational_power(self):
        assert canonicalize(Power(Rational(1, 2), Integer(3))) == Rational(1, 8)

    def test_symbolic_exp_preserved(self):
        x, n = Symbol("x"), Symbol("n")
        r = canonicalize(Power(x, n))
        assert r == Power(x, n)


# --------------------------------------------------------------------- #
# Idempotency                                                           #
# --------------------------------------------------------------------- #


class TestIdempotent:
    def test_sum(self):
        x, y = Symbol("x"), Symbol("y")
        once = canonicalize(x + x + Integer(2) + Neg(y))
        twice = canonicalize(once)
        assert once == twice

    def test_product(self):
        x = Symbol("x")
        once = canonicalize(Product(Integer(2), Integer(3), x))
        twice = canonicalize(once)
        assert once == twice

    def test_nested(self):
        x = Symbol("x")
        expr = Neg(Neg(Neg(Sum(x, x, Integer(0)))))
        once = canonicalize(expr)
        twice = canonicalize(once)
        assert once == twice


# --------------------------------------------------------------------- #
# Nested shapes                                                         #
# --------------------------------------------------------------------- #


class TestNested:
    def test_neg_of_sum_distributes(self):
        x, y = Symbol("x"), Symbol("y")
        # -(x + y) → -x + -y. Distributing Neg through Sum lets
        # downstream collect_terms cancel across signs; the prior
        # no-distribute behavior left cancellations trapped inside a
        # Neg envelope. Output order is stable-sorted by repr.
        r = canonicalize(Neg(x + y))
        assert r == Sum(Neg(x), Neg(y)) or r == Sum(Neg(y), Neg(x))

    def test_neg_of_sum_cancels_across_signs(self):
        """X − (X − Y) → Y, the whole point of distributing Neg
        through Sum."""
        x, y = Symbol("x"), Symbol("y")
        r = canonicalize(Sum(x, Neg(Sum(x, Neg(y)))))
        assert r == y

    def test_nested_sum_in_product(self):
        x, y = Symbol("x"), Symbol("y")
        # (x + x) * y = 2x * y at the Sum level; Product stays.
        r = canonicalize(Sum(x, x) * y)
        assert r == Product(Integer(2), x, y) or r == Product(Integer(2), x) * y


# --------------------------------------------------------------------- #
# semantically_equal / canonical_hash                                   #
# --------------------------------------------------------------------- #


class TestSemanticEquality:
    def test_like_terms_merge(self):
        x = Symbol("x")
        assert semantically_equal(x + x, Integer(2) * x)

    def test_commuted_sum_equal(self):
        x, y = Symbol("x"), Symbol("y")
        # Sum reordering is part of canonical form.
        assert semantically_equal(x + y, y + x)

    def test_double_negation(self):
        x = Symbol("x")
        assert semantically_equal(-(-x), x)

    def test_constant_folding(self):
        assert semantically_equal(Integer(2) + Integer(3), Integer(5))

    def test_distinct_not_equal(self):
        x, y = Symbol("x"), Symbol("y")
        assert not semantically_equal(x + y, x * y)

    def test_non_commutative_product_not_equal(self):
        x, y = Symbol("x"), Symbol("y")
        # Core stays non-commutative; graded sort is a separate pass.
        assert not semantically_equal(x * y, y * x)

    def test_reflexive_and_symmetric(self):
        x = Symbol("x")
        a = x + x
        b = Integer(2) * x
        assert semantically_equal(a, a)
        assert semantically_equal(a, b) == semantically_equal(b, a)


class TestCanonicalHash:
    def test_hash_agrees_with_semantic_equality(self):
        x = Symbol("x")
        assert canonical_hash(x + x) == canonical_hash(Integer(2) * x)

    def test_hash_is_int(self):
        assert isinstance(canonical_hash(Symbol("x")), int)

    def test_different_shapes_likely_distinct(self):
        x, y = Symbol("x"), Symbol("y")
        # Not a hard requirement, but these commonly differ.
        assert canonical_hash(x + y) != canonical_hash(x * y)

    def test_hash_stable_across_calls(self):
        x = Symbol("x")
        h1 = canonical_hash(x + x)
        h2 = canonical_hash(x + x)
        assert h1 == h2
