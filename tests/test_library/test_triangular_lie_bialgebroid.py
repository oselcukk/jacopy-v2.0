"""Tests for ``jacopy.library.triangular_lie_bialgebroid``."""

from __future__ import annotations

import pytest

from jacopy.algebra.derivation import Act
from jacopy.brackets.koszul import KoszulBracket
from jacopy.brackets.lie import LieBracket
from jacopy.calculus.exterior_d import ExteriorDerivative, d as default_d
from jacopy.calculus.interior import (
    InteriorProduct,
    interior as default_interior,
)
from jacopy.calculus.lie_derivative import (
    LieDerivative,
    lie_derivative as default_lie_derivative,
)
from jacopy.calculus.musical import Sharp
from jacopy.core.expr import Symbol
from jacopy.library.triangular_lie_bialgebroid import (
    TriangularLieBialgebroid,
    triangular_lie_bialgebroid,
)


# --------------------------------------------------------------------- #
# Fixtures                                                               #
# --------------------------------------------------------------------- #


@pytest.fixture
def pi():
    return Symbol("π")


@pytest.fixture
def tlb(pi):
    return TriangularLieBialgebroid(pi)


# --------------------------------------------------------------------- #
# Construction                                                           #
# --------------------------------------------------------------------- #


class TestConstruction:
    def test_smoke(self, tlb):
        assert isinstance(tlb, TriangularLieBialgebroid)

    def test_pi_stored(self, tlb, pi):
        assert tlb.pi is pi

    def test_default_name(self, tlb):
        assert "TLB" in tlb.name
        assert "π" in tlb.name

    def test_custom_name(self, pi):
        T = TriangularLieBialgebroid(pi, name="Trial")
        assert T.name == "Trial"

    def test_repr(self, tlb):
        assert "TriangularLieBialgebroid" in repr(tlb)
        assert "π" in repr(tlb)

    def test_rejects_non_expr_pi(self):
        with pytest.raises(TypeError, match="Expr"):
            TriangularLieBialgebroid("π")

    def test_factory(self, pi):
        T = triangular_lie_bialgebroid(pi)
        assert isinstance(T, TriangularLieBialgebroid)


# --------------------------------------------------------------------- #
# Sharp                                                                  #
# --------------------------------------------------------------------- #


class TestSharp:
    def test_sharp_type(self, tlb):
        assert isinstance(tlb.sharp, Sharp)

    def test_sharp_carries_pi(self, tlb, pi):
        assert tlb.sharp.bivector is pi


# --------------------------------------------------------------------- #
# TM side accessors                                                      #
# --------------------------------------------------------------------- #


class TestTMSide:
    def test_tm_bracket_is_lie(self, tlb):
        assert isinstance(tlb.tm_bracket, LieBracket)

    def test_tm_d_is_default_singleton(self, tlb):
        assert tlb.tm_d is default_d

    def test_tm_lie_derivative_is_default(self, tlb):
        assert tlb.tm_lie_derivative is default_lie_derivative

    def test_tm_interior_is_default(self, tlb):
        assert tlb.tm_interior is default_interior


# --------------------------------------------------------------------- #
# T*M side accessors                                                     #
# --------------------------------------------------------------------- #


class TestTStarMSide:
    def test_koszul_type(self, tlb):
        assert isinstance(tlb.koszul, KoszulBracket)

    def test_koszul_anchor_is_sharp(self, tlb):
        # The Koszul bracket holds the anchor as its `_anchor` slot;
        # we check the anchor identity through Sharp equality.
        assert tlb.koszul._anchor is tlb.sharp

    def test_koszul_name_marks_tstar(self, tlb):
        assert "T*M" in tlb.koszul.name

    def test_tilde_d_type(self, tlb):
        assert isinstance(tlb.tilde_d, ExteriorDerivative)

    def test_tilde_d_is_fresh(self, tlb):
        """``d̃`` must be distinct from the default ``d`` singleton."""
        assert tlb.tilde_d is not default_d

    def test_tilde_d_name(self, tlb):
        assert tlb.tilde_d.name == "d̃"

    def test_tilde_lie_factory_returns_lie_derivative(self, tlb):
        omega = Symbol("ω")
        L = tlb.tilde_lie_derivative(omega)
        assert isinstance(L, LieDerivative)

    def test_tilde_lie_carries_tilde_name(self, tlb):
        omega = Symbol("ω")
        L = tlb.tilde_lie_derivative(omega)
        assert "L̃" in L.name

    def test_tilde_lie_threads_tilde_d(self, tlb):
        """``L̃_ω`` must carry ``d̃`` so its Cartan magic stays inside T*M."""
        omega = Symbol("ω")
        L = tlb.tilde_lie_derivative(omega)
        assert L.d is tlb.tilde_d

    def test_tilde_lie_threads_tilde_iota(self, tlb):
        omega = Symbol("ω")
        mu = Symbol("μ")
        L = tlb.tilde_lie_derivative(omega)
        # Build the iota via the same factory and check name.
        i = L.iota_factory(mu)
        assert "ι̃" in i.name

    def test_tilde_interior_factory_returns_interior_product(self, tlb):
        omega = Symbol("ω")
        i = tlb.tilde_interior(omega)
        assert isinstance(i, InteriorProduct)

    def test_tilde_interior_carries_tilde_name(self, tlb):
        omega = Symbol("ω")
        i = tlb.tilde_interior(omega)
        assert "ι̃" in i.name

    def test_tilde_interior_rejects_non_expr(self, tlb):
        with pytest.raises(TypeError):
            tlb.tilde_interior("not_expr")

    def test_tilde_lie_rejects_non_expr(self, tlb):
        with pytest.raises(TypeError):
            tlb.tilde_lie_derivative("not_expr")


# --------------------------------------------------------------------- #
# Application semantics                                                  #
# --------------------------------------------------------------------- #


class TestApplicationSemantics:
    def test_tilde_lie_applies_to_form(self, tlb):
        omega, mu = Symbol("ω"), Symbol("μ")
        L = tlb.tilde_lie_derivative(omega)
        result = Act(L, mu)
        rendered = result._repr_inner()
        assert "L̃_ω" in rendered
        assert "μ" in rendered

    def test_tilde_iota_applies_to_form(self, tlb):
        omega, mu = Symbol("ω"), Symbol("μ")
        i = tlb.tilde_interior(omega)
        result = Act(i, mu)
        rendered = result._repr_inner()
        assert "ι̃_ω" in rendered
        assert "μ" in rendered

    def test_distinct_omegas_distinct_factories(self, tlb):
        a, b = Symbol("a"), Symbol("b")
        L_a = tlb.tilde_lie_derivative(a)
        L_b = tlb.tilde_lie_derivative(b)
        assert L_a != L_b
