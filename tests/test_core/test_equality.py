"""Tests for jacopy.core.equality."""

import pytest

from jacopy.core.expr import Integer, Product, Sum, Symbol
from jacopy.core.equality import alpha_equal, structural_equal, sum_bag_equal
from jacopy.core.properties import Graded, Scalar
from jacopy.core.wildcards import SeqWildcard, Wildcard


# --------------------------------------------------------------------- #
# structural_equal                                                       #
# --------------------------------------------------------------------- #


class TestStructuralEqual:
    def test_matches(self):
        x = Symbol("x")
        assert structural_equal(x, x)
        assert structural_equal(x + Symbol("y"), Symbol("x") + Symbol("y"))

    def test_order_matters(self):
        x, y = Symbol("x"), Symbol("y")
        assert not structural_equal(x + y, y + x)


# --------------------------------------------------------------------- #
# alpha_equal                                                           #
# --------------------------------------------------------------------- #


class TestAlphaEqual:
    def test_same_name_same_pattern(self):
        p = Wildcard("A") + Symbol("x")
        assert alpha_equal(p, p)

    def test_simple_rename(self):
        lhs = Wildcard("A") + Symbol("x")
        rhs = Wildcard("B") + Symbol("x")
        assert alpha_equal(lhs, rhs)

    def test_non_wildcard_parts_must_match(self):
        lhs = Wildcard("A") + Symbol("x")
        rhs = Wildcard("B") + Symbol("y")
        assert not alpha_equal(lhs, rhs)

    def test_consistent_renaming_across_occurrences(self):
        lhs = Wildcard("A") + Wildcard("A")
        rhs = Wildcard("B") + Wildcard("B")
        assert alpha_equal(lhs, rhs)

    def test_inconsistent_renaming_fails(self):
        lhs = Wildcard("A") + Wildcard("A")
        rhs = Wildcard("B") + Wildcard("C")
        assert not alpha_equal(lhs, rhs)

    def test_bijection_required(self):
        """Two distinct wildcards can't collapse into one."""
        lhs = Wildcard("A") + Wildcard("B")
        rhs = Wildcard("C") + Wildcard("C")
        assert not alpha_equal(lhs, rhs)

    def test_type_filter_must_match(self):
        lhs = Wildcard("A", type_filter=Scalar)
        rhs = Wildcard("B", type_filter=Scalar)
        assert alpha_equal(lhs, rhs)

    def test_different_type_filters_fail(self):
        lhs = Wildcard("A", type_filter=Scalar)
        rhs = Wildcard("B", type_filter=Graded)
        assert not alpha_equal(lhs, rhs)

    def test_filter_vs_no_filter_fails(self):
        lhs = Wildcard("A", type_filter=Scalar)
        rhs = Wildcard("B")
        assert not alpha_equal(lhs, rhs)

    def test_wildcard_vs_symbol_fails(self):
        lhs = Wildcard("A")
        rhs = Symbol("x")
        assert not alpha_equal(lhs, rhs)

    def test_seq_wildcards_rename(self):
        lhs = Sum(Wildcard("A"), SeqWildcard("rest"))
        rhs = Sum(Wildcard("B"), SeqWildcard("tail"))
        assert alpha_equal(lhs, rhs)

    def test_seq_vs_single_wildcard_fails(self):
        lhs = SeqWildcard("rest")
        rhs = Wildcard("B")
        assert not alpha_equal(lhs, rhs)

    def test_nested(self):
        lhs = Wildcard("A") + Wildcard("B") * Wildcard("A")
        rhs = Wildcard("X") + Wildcard("Y") * Wildcard("X")
        assert alpha_equal(lhs, rhs)

    def test_atoms_without_wildcards(self):
        assert alpha_equal(Symbol("x"), Symbol("x"))
        assert not alpha_equal(Symbol("x"), Symbol("y"))
        assert alpha_equal(Integer(5), Integer(5))


# --------------------------------------------------------------------- #
# sum_bag_equal                                                         #
# --------------------------------------------------------------------- #


class TestSumBagEqual:
    def test_permuted_sum_equal(self):
        x, y, z = Symbol("x"), Symbol("y"), Symbol("z")
        assert sum_bag_equal(Sum(x, y, z), Sum(z, x, y))

    def test_same_sum_equal(self):
        x, y = Symbol("x"), Symbol("y")
        assert sum_bag_equal(x + y, x + y)

    def test_different_multiplicities_fail(self):
        x, y = Symbol("x"), Symbol("y")
        assert not sum_bag_equal(Sum(x, x, y), Sum(x, y, y))

    def test_different_lengths_fail(self):
        x, y = Symbol("x"), Symbol("y")
        assert not sum_bag_equal(Sum(x, y), Sum(x, y, Symbol("z")))

    def test_non_sum_falls_back_to_structural(self):
        x = Symbol("x")
        assert sum_bag_equal(x, x)
        assert not sum_bag_equal(x, Symbol("y"))

    def test_different_types_fail(self):
        x, y = Symbol("x"), Symbol("y")
        assert not sum_bag_equal(Sum(x, y), Product(x, y))

    def test_nested_sums_not_recursive(self):
        """Documented behavior: inner Sums are compared structurally."""
        x, y, z = Symbol("x"), Symbol("y"), Symbol("z")
        lhs = Sum(Sum(x, y), z)
        rhs = Sum(Sum(y, x), z)
        # Top-level children: (Sum(x,y), z) vs (Sum(y,x), z)
        # Sum(x,y) != Sum(y,x) structurally, so Counter differs.
        assert not sum_bag_equal(lhs, rhs)

    def test_product_not_permutable(self):
        """Only Sum is commutative; Product order matters."""
        x, y = Symbol("x"), Symbol("y")
        assert not sum_bag_equal(x * y, y * x)
