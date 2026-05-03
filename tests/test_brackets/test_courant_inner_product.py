"""Tests for the CourantInnerProduct Expr node and its unfold rule."""

import pytest

from jacopy.brackets.courant_inner_product import (
    CourantInnerProduct,
    CourantInnerProductDefinition,
    courant_inner_product,
)
from jacopy.brackets.dorfman import SectionPair
from jacopy.calculus.pairing import Pairing
from jacopy.core.expr import Product, Rational, Sum, Symbol


# --------------------------------------------------------------------- #
# Construction                                                           #
# --------------------------------------------------------------------- #


class TestConstruction:
    def test_smoke(self):
        X, alpha = Symbol("X"), Symbol("α")
        Y, beta = Symbol("Y"), Symbol("β")
        ip = courant_inner_product(SectionPair(X, alpha), SectionPair(Y, beta))
        assert isinstance(ip, CourantInnerProduct)

    def test_repr(self):
        X, alpha = Symbol("X"), Symbol("α")
        Y, beta = Symbol("Y"), Symbol("β")
        ip = courant_inner_product(SectionPair(X, alpha), SectionPair(Y, beta))
        assert ip._repr_inner() == "⟨(X, α), (Y, β)⟩"

    def test_left_right_accessors(self):
        a, b = Symbol("a"), Symbol("b")
        ip = CourantInnerProduct(a, b)
        assert ip.left is a
        assert ip.right is b

    def test_children(self):
        a, b = Symbol("a"), Symbol("b")
        ip = CourantInnerProduct(a, b)
        assert ip.children == (a, b)

    def test_rejects_non_expr_left(self):
        with pytest.raises(TypeError):
            CourantInnerProduct("not_expr", Symbol("b"))

    def test_rejects_non_expr_right(self):
        with pytest.raises(TypeError):
            CourantInnerProduct(Symbol("a"), 42)


# --------------------------------------------------------------------- #
# Equality / hash                                                        #
# --------------------------------------------------------------------- #


class TestEquality:
    def test_structural_equality(self):
        X, alpha = Symbol("X"), Symbol("α")
        Y, beta = Symbol("Y"), Symbol("β")
        ip1 = CourantInnerProduct(SectionPair(X, alpha), SectionPair(Y, beta))
        ip2 = CourantInnerProduct(SectionPair(X, alpha), SectionPair(Y, beta))
        assert ip1 == ip2
        assert hash(ip1) == hash(ip2)

    def test_swap_changes_identity(self):
        a, b = Symbol("a"), Symbol("b")
        assert CourantInnerProduct(a, b) != CourantInnerProduct(b, a)


# --------------------------------------------------------------------- #
# Unfold definition                                                      #
# --------------------------------------------------------------------- #


class TestUnfold:
    def test_matches_section_pair_pair(self):
        X, alpha = Symbol("X"), Symbol("α")
        Y, beta = Symbol("Y"), Symbol("β")
        ip = CourantInnerProduct(SectionPair(X, alpha), SectionPair(Y, beta))
        assert CourantInnerProductDefinition().matches(ip)

    def test_does_not_match_opaque_left(self):
        a = Symbol("a")
        Y, beta = Symbol("Y"), Symbol("β")
        ip = CourantInnerProduct(a, SectionPair(Y, beta))
        assert not CourantInnerProductDefinition().matches(ip)

    def test_does_not_match_opaque_right(self):
        X, alpha = Symbol("X"), Symbol("α")
        b = Symbol("b")
        ip = CourantInnerProduct(SectionPair(X, alpha), b)
        assert not CourantInnerProductDefinition().matches(ip)

    def test_does_not_match_non_inner_product(self):
        assert not CourantInnerProductDefinition().matches(Symbol("foo"))

    def test_rewrite_yields_pairing_sum(self):
        X, alpha = Symbol("X"), Symbol("α")
        Y, beta = Symbol("Y"), Symbol("β")
        ip = CourantInnerProduct(SectionPair(X, alpha), SectionPair(Y, beta))
        result = CourantInnerProductDefinition().rewrite(ip)
        expected = Product(
            Rational(1, 2),
            Sum(Pairing(alpha, Y), Pairing(beta, X)),
        )
        assert result == expected
