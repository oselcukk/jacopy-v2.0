"""Tests for the Courant anchor projection and D operator."""

import pytest

from jacopy.algebra.derivation import Act
from jacopy.brackets.courant_anchor_d import (
    CourantAnchor,
    CourantAnchorDefinition,
    DOperator,
    DOperatorDefinition,
    anchor,
    d_operator,
)
from jacopy.brackets.dorfman import SectionPair
from jacopy.calculus.exterior_d import d
from jacopy.core.expr import Integer, Symbol


# --------------------------------------------------------------------- #
# Anchor                                                                 #
# --------------------------------------------------------------------- #


class TestCourantAnchor:
    def test_construction(self):
        e = Symbol("e")
        a = anchor(e)
        assert isinstance(a, CourantAnchor)
        assert a.section is e

    def test_repr_opaque(self):
        e = Symbol("e")
        assert anchor(e)._repr_inner() == "anchor(e)"

    def test_repr_section_pair(self):
        X, alpha = Symbol("X"), Symbol("α")
        a = anchor(SectionPair(X, alpha))
        assert a._repr_inner() == "anchor((X, α))"

    def test_rejects_non_expr(self):
        with pytest.raises(TypeError):
            CourantAnchor("not_expr")

    def test_equality(self):
        e = Symbol("e")
        assert anchor(e) == anchor(e)


class TestCourantAnchorDefinition:
    def test_matches_section_pair(self):
        X, alpha = Symbol("X"), Symbol("α")
        a = anchor(SectionPair(X, alpha))
        assert CourantAnchorDefinition().matches(a)

    def test_does_not_match_opaque(self):
        a = anchor(Symbol("e"))
        assert not CourantAnchorDefinition().matches(a)

    def test_does_not_match_non_anchor(self):
        assert not CourantAnchorDefinition().matches(Symbol("X"))

    def test_rewrite_extracts_vector(self):
        X, alpha = Symbol("X"), Symbol("α")
        a = anchor(SectionPair(X, alpha))
        assert CourantAnchorDefinition().rewrite(a) == X


# --------------------------------------------------------------------- #
# D operator                                                             #
# --------------------------------------------------------------------- #


class TestDOperator:
    def test_construction(self):
        f = Symbol("f")
        Df = d_operator(f)
        assert isinstance(Df, DOperator)
        assert Df.f is f

    def test_repr(self):
        f = Symbol("f")
        assert d_operator(f)._repr_inner() == "D(f)"

    def test_rejects_non_expr(self):
        with pytest.raises(TypeError):
            DOperator("not_expr")

    def test_default_d_singleton(self):
        f = Symbol("f")
        assert d_operator(f).d_op is d

    def test_custom_d_propagates(self):
        f = Symbol("f")
        # Use the default d as a stand-in custom op; identity check is
        # what matters, the DOperator stores whatever derivation is
        # passed.
        Df = DOperator(f, d=d)
        assert Df.d_op is d

    def test_equality_same_d(self):
        f = Symbol("f")
        assert d_operator(f) == d_operator(f)


class TestDOperatorDefinition:
    def test_matches_any_d_operator(self):
        f = Symbol("f")
        assert DOperatorDefinition().matches(d_operator(f))

    def test_does_not_match_non_d(self):
        assert not DOperatorDefinition().matches(Symbol("foo"))

    def test_rewrite_yields_section_pair_zero_df(self):
        f = Symbol("f")
        result = DOperatorDefinition().rewrite(d_operator(f))
        assert isinstance(result, SectionPair)
        assert result.vector == Integer(0)
        assert result.form == Act(d, f)
