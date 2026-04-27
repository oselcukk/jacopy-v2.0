"""Tests for the Dorfman-Courant bridge helper."""

import pytest

from jacopy.algebra.derivation import Act, Derivation
from jacopy.brackets.courant import CourantBracket
from jacopy.brackets.dorfman import DorfmanBracket, SectionPair
from jacopy.brackets.dorfman_courant import (
    dorfman_courant_correction,
    prove_dorfman_courant_bridge,
)
from jacopy.calculus.exterior_d import d as default_d
from jacopy.calculus.interior import interior as default_interior
from jacopy.core.expr import (
    Integer,
    Neg,
    Product,
    Rational,
    Sum,
    Symbol,
    Zero,
)
from jacopy.core.properties import Graded
from jacopy.core.registry import PropertyRegistry
from jacopy.proof.chain import ProofChain
from jacopy.proof.strategies import ProofFailure


# --------------------------------------------------------------------- #
# Fixtures                                                              #
# --------------------------------------------------------------------- #


@pytest.fixture
def reg():
    return PropertyRegistry()


@pytest.fixture
def section_pair_data(reg):
    X = Derivation("X", 0)
    Y = Derivation("Y", 0)
    alpha = Symbol("α")
    beta = Symbol("β")
    reg.declare(alpha, Graded(degree=1))
    reg.declare(beta, Graded(degree=1))
    a = SectionPair(X, alpha)
    b = SectionPair(Y, beta)
    return X, Y, alpha, beta, a, b


# --------------------------------------------------------------------- #
# dorfman_courant_correction                                             #
# --------------------------------------------------------------------- #


class TestCorrection:
    def test_returns_section_pair(self, section_pair_data):
        _, _, _, _, a, b = section_pair_data
        corr = dorfman_courant_correction(a, b)
        assert isinstance(corr, SectionPair)

    def test_vector_part_is_zero(self, section_pair_data):
        _, _, _, _, a, b = section_pair_data
        corr = dorfman_courant_correction(a, b)
        assert corr.vector is Zero

    def test_form_part_is_half_d_symmetric_inner(self, section_pair_data):
        X, Y, alpha, beta, a, b = section_pair_data
        corr = dorfman_courant_correction(a, b)
        expected = Product(
            Rational(1, 2),
            Act(default_d, Sum(
                Act(default_interior(X), beta),
                Act(default_interior(Y), alpha),
            )),
        )
        assert corr.form == expected

    def test_form_inner_is_symmetric_under_swap(self, section_pair_data):
        """ι_X β + ι_Y α is symmetric under (a,b) ↔ (b,a)."""
        X, Y, alpha, beta, a, b = section_pair_data
        corr_ab = dorfman_courant_correction(a, b)
        corr_ba = dorfman_courant_correction(b, a)
        # Inner sums should match modulo Sum's child ordering.
        inner_ab = set(corr_ab.form.children[1].arg.children)
        inner_ba = set(corr_ba.form.children[1].arg.children)
        assert inner_ab == inner_ba

    def test_rejects_non_section_pair_left(self):
        with pytest.raises(TypeError):
            dorfman_courant_correction(Symbol("X"), SectionPair(Symbol("Y"), Symbol("β")))  # type: ignore[arg-type]

    def test_rejects_non_section_pair_right(self):
        with pytest.raises(TypeError):
            dorfman_courant_correction(SectionPair(Symbol("X"), Symbol("α")), Symbol("Y"))  # type: ignore[arg-type]


# --------------------------------------------------------------------- #
# prove_dorfman_courant_bridge                                           #
# --------------------------------------------------------------------- #


class TestBridgeProof:
    def test_returns_proof_chain(self, reg, section_pair_data):
        _, _, _, _, a, b = section_pair_data
        D, C = DorfmanBracket(), CourantBracket()
        chain = prove_dorfman_courant_bridge(D, C, a, b, registry=reg)
        assert isinstance(chain, ProofChain)

    def test_chain_is_non_empty(self, reg, section_pair_data):
        _, _, _, _, a, b = section_pair_data
        D, C = DorfmanBracket(), CourantBracket()
        chain = prove_dorfman_courant_bridge(D, C, a, b, registry=reg)
        assert len(chain) > 0

    def test_closes_for_default_brackets(self, reg, section_pair_data):
        """Smoke test: default Cartan operators carry magic-formula
        rewrite, so the bridge identity reduces."""
        _, _, _, _, a, b = section_pair_data
        D, C = DorfmanBracket(), CourantBracket()
        # No exception → both halves cancelled to zero.
        prove_dorfman_courant_bridge(D, C, a, b, registry=reg)

    def test_rejects_twisted_courant(self, reg, section_pair_data):
        _, _, _, _, a, b = section_pair_data
        H = Symbol("H")
        reg.declare(H, Graded(degree=3))
        D, C = DorfmanBracket(), CourantBracket(background_H=H)
        with pytest.raises(TypeError, match="untwisted"):
            prove_dorfman_courant_bridge(D, C, a, b, registry=reg)

    def test_rejects_non_dorfman_first_arg(self, reg, section_pair_data):
        _, _, _, _, a, b = section_pair_data
        C = CourantBracket()
        with pytest.raises(TypeError):
            prove_dorfman_courant_bridge(C, C, a, b, registry=reg)  # type: ignore[arg-type]

    def test_rejects_non_courant_second_arg(self, reg, section_pair_data):
        _, _, _, _, a, b = section_pair_data
        D = DorfmanBracket()
        with pytest.raises(TypeError):
            prove_dorfman_courant_bridge(D, D, a, b, registry=reg)  # type: ignore[arg-type]

    def test_rejects_non_section_pair_operands(self, reg):
        D, C = DorfmanBracket(), CourantBracket()
        X, Y = Derivation("X", 0), Derivation("Y", 0)
        with pytest.raises(TypeError):
            prove_dorfman_courant_bridge(D, C, X, Y, registry=reg)  # type: ignore[arg-type]
