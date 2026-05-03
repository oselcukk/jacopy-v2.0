"""Tests for the LWX Courant bracket on a triangular Lie bialgebroid."""

import pytest

from jacopy.algebra.derivation import Act
from jacopy.brackets.courant_lwx import LWXCourantBracket
from jacopy.brackets.dorfman import SectionPair
from jacopy.core.expr import Symbol
from jacopy.core.properties import Graded
from jacopy.core.registry import PropertyRegistry
from jacopy.library.triangular_lie_bialgebroid import TriangularLieBialgebroid


# --------------------------------------------------------------------- #
# Fixtures                                                               #
# --------------------------------------------------------------------- #


@pytest.fixture
def pi():
    return Symbol("π")


@pytest.fixture
def tlb(pi):
    return TriangularLieBialgebroid(pi)


@pytest.fixture
def bracket(tlb):
    return LWXCourantBracket(tlb)


@pytest.fixture
def operands():
    U, V = Symbol("U"), Symbol("V")
    omega, eta = Symbol("ω"), Symbol("η")
    return SectionPair(U, omega), SectionPair(V, eta)


@pytest.fixture
def registry():
    reg = PropertyRegistry()
    for sym, deg in [
        (Symbol("U"), 0),
        (Symbol("V"), 0),
        (Symbol("ω"), 1),
        (Symbol("η"), 1),
    ]:
        reg.declare(sym, Graded(degree=deg))
    return reg


# --------------------------------------------------------------------- #
# Construction                                                           #
# --------------------------------------------------------------------- #


class TestConstruction:
    def test_smoke(self, bracket):
        assert isinstance(bracket, LWXCourantBracket)

    def test_default_name(self, bracket):
        assert bracket.name == "[·,·]_LWX"

    def test_twisted_name(self, tlb):
        H = Symbol("H")
        b = LWXCourantBracket(tlb, background_H=H)
        assert "H" in b.name
        assert "LWX" in b.name

    def test_custom_name(self, tlb):
        b = LWXCourantBracket(tlb, name="MyLWX")
        assert b.name == "MyLWX"

    def test_axiom_flags(self, bracket):
        # Dorfman-style: not antisymmetric, satisfies Leibniz, conditional Jacobi.
        assert bracket.is_graded_antisymmetric is False
        assert bracket.satisfies_leibniz is True
        assert bracket.satisfies_graded_jacobi is None

    def test_carries_bialgebroid(self, bracket, tlb):
        assert bracket.bialgebroid is tlb

    def test_untwisted_by_default(self, bracket):
        assert bracket.is_twisted is False
        assert bracket.background_H is None

    def test_twisted(self, tlb):
        H = Symbol("H")
        b = LWXCourantBracket(tlb, background_H=H)
        assert b.is_twisted is True
        assert b.background_H == H

    def test_rejects_non_expr_H(self, tlb):
        with pytest.raises(TypeError, match="Expr"):
            LWXCourantBracket(tlb, background_H="H")

    def test_rejects_invalid_bialgebroid(self):
        # Plain Symbol is missing the required attributes.
        with pytest.raises(TypeError, match="missing attribute"):
            LWXCourantBracket(Symbol("not_tlb"))


# --------------------------------------------------------------------- #
# Expansion                                                              #
# --------------------------------------------------------------------- #


class TestExpansion:
    def test_expand_returns_section_pair(self, bracket, operands, registry):
        a, b = operands
        result = bracket.expand(a, b, registry)
        assert isinstance(result, SectionPair)

    def test_rejects_non_section_pair_left(self, bracket, registry):
        with pytest.raises(TypeError, match="SectionPair"):
            bracket.expand(Symbol("not_pair"), SectionPair(Symbol("V"), Symbol("η")), registry)

    def test_rejects_non_section_pair_right(self, bracket, registry):
        with pytest.raises(TypeError, match="SectionPair"):
            bracket.expand(SectionPair(Symbol("U"), Symbol("ω")), Symbol("not_pair"), registry)

    def test_vector_half_carries_lie_bracket(self, bracket, operands, registry):
        a, b = operands
        result = bracket.expand(a, b, registry)
        # The TM-side Lie bracket [U,V] = U·V - V·U appears.
        repr_v = result.vector._repr_inner()
        assert "U * V" in repr_v
        assert "V * U" in repr_v

    def test_vector_half_carries_tilde_lie(self, bracket, operands, registry):
        """Vector half includes L̃_ω V and -L̃_η U terms."""
        a, b = operands
        result = bracket.expand(a, b, registry)
        repr_v = result.vector._repr_inner()
        assert "L̃_ω(V)" in repr_v
        assert "L̃_η(U)" in repr_v

    def test_vector_half_carries_d_tilde_iota_tilde(self, bracket, operands, registry):
        """Vector half includes -d̃ι̃_η U term."""
        a, b = operands
        result = bracket.expand(a, b, registry)
        repr_v = result.vector._repr_inner()
        assert "d̃(ι̃_η(U))" in repr_v

    def test_form_half_carries_koszul(self, bracket, operands, registry):
        """Form half includes the Koszul bracket [ω, η]_{T*M}."""
        a, b = operands
        result = bracket.expand(a, b, registry)
        repr_f = result.form._repr_inner()
        # Koszul expand emits L_π♯(ω) and pairing-of-π♯ terms.
        assert "π♯" in repr_f

    def test_form_half_carries_lie_terms(self, bracket, operands, registry):
        """Form half includes L_U η and -L_V ω."""
        a, b = operands
        result = bracket.expand(a, b, registry)
        repr_f = result.form._repr_inner()
        assert "L_U(η)" in repr_f
        assert "L_V(ω)" in repr_f

    def test_form_half_carries_d_iota_v_omega(self, bracket, operands, registry):
        """Form half includes +d ι_V ω (no ½ — LWX convention)."""
        a, b = operands
        result = bracket.expand(a, b, registry)
        repr_f = result.form._repr_inner()
        assert "d(ι_V(ω))" in repr_f


# --------------------------------------------------------------------- #
# H-twist                                                                #
# --------------------------------------------------------------------- #


class TestHTwist:
    def test_twisted_form_half_includes_iota_iota_H(self, tlb, operands, registry):
        H = Symbol("H")
        b = LWXCourantBracket(tlb, background_H=H)
        a_, b_ = operands
        result = b.expand(a_, b_, registry)
        assert "ι_V(ι_U(H))" in result.form._repr_inner()

    def test_untwisted_form_half_excludes_H(self, bracket, operands, registry):
        a, b = operands
        result = bracket.expand(a, b, registry)
        assert "H" not in result.form._repr_inner()

    def test_twisted_vector_half_unchanged(self, tlb, operands, registry):
        H = Symbol("H")
        b_t = LWXCourantBracket(tlb, background_H=H)
        b_u = LWXCourantBracket(tlb)
        a, b = operands
        # Vector half identical between twisted and untwisted (twist
        # only contributes to form half).
        assert (
            b_t.expand(a, b, registry).vector
            == b_u.expand(a, b, registry).vector
        )
