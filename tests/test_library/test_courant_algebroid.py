"""Tests for ``jacopy.library.courant_algebroid``."""

from __future__ import annotations

import pytest

from jacopy.brackets.base import GradedBracket
from jacopy.brackets.courant import CourantBracket
from jacopy.brackets.derived import VanishingCondition
from jacopy.brackets.dorfman import DorfmanBracket, SectionPair
from jacopy.brackets.lie import LieBracket
from jacopy.core.expr import Expr, Integer, Symbol
from jacopy.core.properties import Graded
from jacopy.core.registry import PropertyRegistry
from jacopy.library import theorem_book
from jacopy.library.courant_algebroid import (
    THEOREM_COURANT_DORFMAN_BRIDGE,
    THEOREM_COURANT_JACOBI_TWIST,
    CourantAlgebroid,
    courant_algebroid,
)
from jacopy.proof.chain import ProofChain


# --------------------------------------------------------------------- #
# Fixtures                                                               #
# --------------------------------------------------------------------- #


@pytest.fixture
def algebroid():
    return CourantAlgebroid()


@pytest.fixture
def twisted():
    return CourantAlgebroid(background_H=Symbol("H"))


@pytest.fixture
def registry():
    reg = PropertyRegistry()
    X, Y = Symbol("X"), Symbol("Y")
    alpha, beta = Symbol("α"), Symbol("β")
    reg.declare(X, Graded(degree=0))
    reg.declare(Y, Graded(degree=0))
    reg.declare(alpha, Graded(degree=1))
    reg.declare(beta, Graded(degree=1))
    return reg


@pytest.fixture
def ab():
    a = SectionPair(Symbol("X"), Symbol("α"))
    b = SectionPair(Symbol("Y"), Symbol("β"))
    return a, b


# --------------------------------------------------------------------- #
# Construction                                                           #
# --------------------------------------------------------------------- #


class TestConstruction:
    def test_default_courant_type(self, algebroid):
        assert isinstance(algebroid.courant, CourantBracket)

    def test_default_dorfman_type(self, algebroid):
        assert isinstance(algebroid.dorfman, DorfmanBracket)

    def test_untwisted_by_default(self, algebroid):
        assert algebroid.is_twisted is False
        assert algebroid.background_H is None

    def test_twisted_is_twisted(self, twisted):
        assert twisted.is_twisted is True
        assert twisted.background_H == Symbol("H")

    def test_courant_and_dorfman_share_d(self, algebroid):
        assert algebroid.courant._d is algebroid.dorfman._d

    def test_courant_and_dorfman_share_interior(self, algebroid):
        assert algebroid.courant._interior is algebroid.dorfman._interior

    def test_courant_and_dorfman_share_lie(self, algebroid):
        assert (
            algebroid.courant._lie_derivative
            is algebroid.dorfman._lie_derivative
        )

    def test_dorfman_is_never_twisted(self, twisted):
        """The bridge identity only closes on untwisted Dorfman; the
        wrapper never twists the Dorfman twin even when the Courant is
        H-twisted."""
        assert not hasattr(twisted.dorfman, "_background_H") or \
            getattr(twisted.dorfman, "_background_H", None) is None

    def test_default_vector_bracket(self, algebroid):
        assert isinstance(algebroid.vector_bracket, LieBracket)

    def test_custom_vector_bracket(self):
        custom = LieBracket(name="[·,·]_custom")
        A = CourantAlgebroid(vector_bracket=custom)
        assert A.vector_bracket is custom

    def test_name_untwisted(self, algebroid):
        assert "TM⊕T*M" in algebroid.name
        assert "H" not in algebroid.name

    def test_name_twisted(self, twisted):
        assert "H" in twisted.name

    def test_custom_name(self):
        A = CourantAlgebroid(name="MyCourant")
        assert A.name == "MyCourant"

    def test_rejects_non_expr_H(self):
        with pytest.raises(TypeError, match="Expr"):
            CourantAlgebroid(background_H="H")  # type: ignore[arg-type]

    def test_factory(self):
        A = courant_algebroid()
        assert isinstance(A, CourantAlgebroid)

    def test_factory_with_twist(self):
        H = Symbol("H")
        A = courant_algebroid(background_H=H)
        assert A.is_twisted is True


# --------------------------------------------------------------------- #
# Bracket views                                                          #
# --------------------------------------------------------------------- #


class TestBracketViews:
    def test_expand_returns_section_pair(self, algebroid, ab, registry):
        a, b = ab
        out = algebroid.expand(a, b, registry)
        assert isinstance(out, SectionPair)

    def test_expand_dorfman_returns_section_pair(self, algebroid, ab, registry):
        a, b = ab
        out = algebroid.expand_dorfman(a, b, registry)
        assert isinstance(out, SectionPair)

    def test_expand_matches_courant_bracket(self, algebroid, ab, registry):
        a, b = ab
        assert (
            algebroid.expand(a, b, registry)
            == algebroid.courant.expand(a, b, registry)
        )

    def test_expand_dorfman_matches_dorfman_bracket(self, algebroid, ab, registry):
        a, b = ab
        assert (
            algebroid.expand_dorfman(a, b, registry)
            == algebroid.dorfman.expand(a, b, registry)
        )


# --------------------------------------------------------------------- #
# Jacobi                                                                 #
# --------------------------------------------------------------------- #


class TestJacobi:
    def test_untwisted_condition_is_vanishing(self, algebroid):
        cond = algebroid.jacobi_condition()
        assert isinstance(cond, VanishingCondition)

    def test_untwisted_obstruction_is_zero(self, algebroid):
        cond = algebroid.jacobi_condition()
        assert cond.obstruction == Integer(0)

    def test_twisted_condition_obstruction_involves_H(self, twisted):
        cond = twisted.jacobi_condition()
        # The obstruction is Act(d, H), assert H appears in it.
        assert "H" in repr(cond.obstruction)

    def test_prove_jacobi_reduction_untwisted(self, algebroid):
        chain = algebroid.prove_jacobi_reduction()
        assert isinstance(chain, ProofChain)
        assert len(chain) == 1
        step = chain.steps[0]
        assert step.rule == "CourantAlgebroidJacobi"
        assert step.provenance_tag == "axiom"

    def test_prove_jacobi_reduction_twisted(self, twisted):
        chain = twisted.prove_jacobi_reduction()
        assert len(chain) == 1
        step = chain.steps[0]
        assert step.rule == "CourantAlgebroidJacobi"
        assert step.provenance_tag == "axiom"


# --------------------------------------------------------------------- #
# Courant–Dorfman bridge                                                 #
# --------------------------------------------------------------------- #


class TestCourantDorfmanBridge:
    def test_obstruction_returns_section_pair(self, algebroid, ab, registry):
        a, b = ab
        obs = algebroid.courant_dorfman_obstruction(a, b, registry)
        assert isinstance(obs, SectionPair)

    def test_correction_returns_section_pair(self, algebroid, ab):
        a, b = ab
        corr = algebroid.bridge_correction(a, b)
        assert isinstance(corr, SectionPair)

    def test_correction_vector_is_zero(self, algebroid, ab):
        a, b = ab
        corr = algebroid.bridge_correction(a, b)
        assert corr.vector == Integer(0)

    def test_correction_form_involves_d(self, algebroid, ab):
        """The form half is ``½ d(ι_X β + ι_Y α)``, ``d`` and both
        interior products must show up."""
        a, b = ab
        corr = algebroid.bridge_correction(a, b)
        text = repr(corr.form)
        assert "d" in text
        assert "ι" in text

    def test_prove_bridge_returns_proof_chain(self, algebroid, ab, registry):
        a, b = ab
        chain = algebroid.prove_courant_dorfman_bridge(a, b, registry=registry)
        assert isinstance(chain, ProofChain)

    def test_prove_bridge_single_theorem_step(self, algebroid, ab, registry):
        a, b = ab
        chain = algebroid.prove_courant_dorfman_bridge(a, b, registry=registry)
        assert len(chain) == 1
        step = chain.steps[0]
        assert step.rule == "CourantDorfmanBridge"
        assert step.provenance_tag == "theorem"

    def test_prove_bridge_before_matches_obstruction(
        self, algebroid, ab, registry
    ):
        a, b = ab
        chain = algebroid.prove_courant_dorfman_bridge(a, b, registry=registry)
        expected = algebroid.courant_dorfman_obstruction(a, b, registry)
        assert chain.steps[0].before == expected

    def test_prove_bridge_after_matches_correction(
        self, algebroid, ab, registry
    ):
        a, b = ab
        chain = algebroid.prove_courant_dorfman_bridge(a, b, registry=registry)
        expected = algebroid.bridge_correction(a, b)
        assert chain.steps[0].after == expected

    def test_obstruction_rejects_non_section_pair(self, algebroid):
        with pytest.raises(TypeError, match="SectionPair"):
            algebroid.courant_dorfman_obstruction(Symbol("X"), Symbol("Y"))  # type: ignore[arg-type]

    def test_correction_rejects_non_section_pair(self, algebroid):
        with pytest.raises(TypeError, match="SectionPair"):
            algebroid.bridge_correction(Symbol("X"), Symbol("Y"))  # type: ignore[arg-type]


# --------------------------------------------------------------------- #
# Seeded theorems                                                        #
# --------------------------------------------------------------------- #


class TestSeededTheorems:
    def test_courant_jacobi_twist_registered(self):
        assert "courant_jacobi_twist" in theorem_book
        assert (
            theorem_book.get("courant_jacobi_twist")
            is THEOREM_COURANT_JACOBI_TWIST
        )

    def test_courant_dorfman_bridge_registered(self):
        assert "courant_dorfman_bridge" in theorem_book
        assert (
            theorem_book.get("courant_dorfman_bridge")
            is THEOREM_COURANT_DORFMAN_BRIDGE
        )

    def test_jacobi_twist_proof_is_single_axiom_step(self):
        thm = THEOREM_COURANT_JACOBI_TWIST
        assert isinstance(thm.proof, ProofChain)
        assert len(thm.proof) == 1
        step = thm.proof.steps[0]
        assert step.rule == "CourantAlgebroidJacobi"
        assert step.provenance_tag == "axiom"

    def test_jacobi_twist_from_axioms_mentions_dH(self):
        thm = THEOREM_COURANT_JACOBI_TWIST
        assert any("dH" in ax for ax in thm.from_axioms)

    def test_bridge_proof_is_single_theorem_step(self):
        thm = THEOREM_COURANT_DORFMAN_BRIDGE
        assert isinstance(thm.proof, ProofChain)
        assert len(thm.proof) == 1
        step = thm.proof.steps[0]
        assert step.rule == "CourantDorfmanBridge"
        assert step.provenance_tag == "theorem"

    def test_bridge_from_axioms_cites_cartan(self):
        thm = THEOREM_COURANT_DORFMAN_BRIDGE
        assert any("Cartan" in ax or "magic" in ax for ax in thm.from_axioms)

    def test_bridge_statement_carries_half_d(self):
        thm = THEOREM_COURANT_DORFMAN_BRIDGE
        assert "½" in thm.statement
        assert "d(" in thm.statement
