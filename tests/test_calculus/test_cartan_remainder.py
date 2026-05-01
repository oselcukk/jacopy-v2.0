"""Tests for Cartan-remainder operator atoms (Faz 15.A).

Covers both standard ``K_V`` (forms-side) and tilde ``K̃_η``
(multivectors-side). Stage A is the inert-atom layer only, defining
rewrites land in Faz 15.B.
"""

import pytest

from jacopy.algebra.derivation import Act, Derivation
from jacopy.calculus.cartan_remainder import CartanRemainder, K
from jacopy.calculus.tilde import (
    K_tilde,
    TildeCartanRemainder,
)
from jacopy.core.expr import Symbol
from jacopy.core.symbolic_degree import Degree


# --------------------------------------------------------------------- #
# Standard Cartan remainder K_V                                          #
# --------------------------------------------------------------------- #


class TestCartanRemainder:
    def test_is_degree_zero_derivation(self):
        V = Symbol("V")
        op = K(V)
        assert isinstance(op, CartanRemainder)
        assert isinstance(op, Derivation)
        assert op.degree == Degree.const(0)

    def test_carries_vector_field(self):
        V = Symbol("V")
        assert K(V).vector_field is V

    def test_default_name_uses_subscript(self):
        V = Symbol("V")
        assert K(V).name == "K_V"

    def test_custom_name_override(self):
        V = Symbol("V")
        assert K(V, name="K_custom").name == "K_custom"

    def test_equality_structural_over_vector_field(self):
        V = Symbol("V")
        assert K(V) == K(V)
        assert hash(K(V)) == hash(K(V))

    def test_inequality_for_distinct_vector_fields(self):
        assert K(Symbol("V")) != K(Symbol("W"))

    def test_inequality_for_distinct_names(self):
        V = Symbol("V")
        assert K(V) != K(V, name="K_other")

    def test_act_application_builds_inert_node(self):
        V = Symbol("V")
        omega = Symbol("ω")
        node = Act(K(V), omega)
        assert isinstance(node, Act)
        assert node.op == K(V)
        assert node.arg is omega

    def test_rejects_non_expr_vector_field(self):
        with pytest.raises(TypeError):
            K("not-an-expr")  # type: ignore[arg-type]


# --------------------------------------------------------------------- #
# Tilde Cartan remainder K̃_η                                            #
# --------------------------------------------------------------------- #


class TestTildeCartanRemainder:
    def test_is_degree_zero_derivation(self):
        eta = Symbol("η")
        pi = Symbol("π")
        op = K_tilde(eta, pi)
        assert isinstance(op, TildeCartanRemainder)
        assert isinstance(op, Derivation)
        assert op.degree == Degree.const(0)

    def test_carries_form_and_bivector(self):
        eta = Symbol("η")
        pi = Symbol("π")
        op = K_tilde(eta, pi)
        assert op.form is eta
        assert op.bivector is pi

    def test_default_name_uses_form_subscript(self):
        eta = Symbol("η")
        pi = Symbol("π")
        assert K_tilde(eta, pi).name == "K̃_η"

    def test_custom_name_override(self):
        eta = Symbol("η")
        pi = Symbol("π")
        assert K_tilde(eta, pi, name="K̃_custom").name == "K̃_custom"

    def test_equality_structural_over_form_and_bivector(self):
        eta = Symbol("η")
        pi = Symbol("π")
        assert K_tilde(eta, pi) == K_tilde(eta, pi)
        assert hash(K_tilde(eta, pi)) == hash(K_tilde(eta, pi))

    def test_inequality_on_distinct_forms(self):
        pi = Symbol("π")
        assert K_tilde(Symbol("η"), pi) != K_tilde(Symbol("μ"), pi)

    def test_inequality_on_distinct_bivectors(self):
        eta = Symbol("η")
        assert K_tilde(eta, Symbol("π")) != K_tilde(eta, Symbol("π'"))

    def test_inequality_on_distinct_names(self):
        eta = Symbol("η")
        pi = Symbol("π")
        assert K_tilde(eta, pi) != K_tilde(eta, pi, name="K̃_other")

    def test_act_application_builds_inert_node(self):
        eta = Symbol("η")
        pi = Symbol("π")
        V = Symbol("V")
        node = Act(K_tilde(eta, pi), V)
        assert isinstance(node, Act)
        assert node.op == K_tilde(eta, pi)
        assert node.arg is V

    def test_rejects_non_expr_form(self):
        with pytest.raises(TypeError):
            K_tilde("not-an-expr", Symbol("π"))  # type: ignore[arg-type]

    def test_rejects_non_expr_bivector(self):
        with pytest.raises(TypeError):
            K_tilde(Symbol("η"), "not-an-expr")  # type: ignore[arg-type]


# --------------------------------------------------------------------- #
# Cross-family disjointness                                              #
# --------------------------------------------------------------------- #


class TestStandardVsTildeDisjoint:
    """Standard K_V and tilde K̃_η are different operator families even
    when their underlying parameters share a Symbol, the indexing
    semantics differ (vector field vs. form), so they must not alias."""

    def test_standard_and_tilde_are_distinct_types(self):
        x = Symbol("x")
        pi = Symbol("π")
        assert not isinstance(K(x), TildeCartanRemainder)
        assert not isinstance(K_tilde(x, pi), CartanRemainder)
