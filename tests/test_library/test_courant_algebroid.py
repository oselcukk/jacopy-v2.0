"""Tests for ``jacopy.library.courant_algebroid``."""

from __future__ import annotations

import pytest

from jacopy.algebra.derivation import Act
from jacopy.brackets.base import GradedBracket
from jacopy.brackets.courant import CourantBracket
from jacopy.brackets.courant_anchor_d import CourantAnchor, DOperator
from jacopy.brackets.courant_inner_product import CourantInnerProduct
from jacopy.brackets.derived import VanishingCondition
from jacopy.brackets.dorfman import DorfmanBracket, SectionPair
from jacopy.brackets.lie import LieBracket
from jacopy.calculus.exterior_d import d as default_d
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


# --------------------------------------------------------------------- #
# Stage E.0 wiring: inner_product / D / anchor_of                        #
# --------------------------------------------------------------------- #


class TestStageEStructuralOperators:
    """``inner_product``, ``D``, ``anchor_of`` helpers on ``CourantAlgebroid``."""

    def test_inner_product_returns_cip(self):
        C = CourantAlgebroid()
        X, alpha = Symbol("X"), Symbol("α")
        Y, beta = Symbol("Y"), Symbol("β")
        ip = C.inner_product(SectionPair(X, alpha), SectionPair(Y, beta))
        assert isinstance(ip, CourantInnerProduct)
        assert ip.left == SectionPair(X, alpha)
        assert ip.right == SectionPair(Y, beta)

    def test_inner_product_rejects_non_section_pair(self):
        C = CourantAlgebroid()
        with pytest.raises(TypeError):
            C.inner_product(Symbol("a"), Symbol("b"))

    def test_D_returns_d_operator_with_algebroid_d(self):
        C = CourantAlgebroid()
        f = Symbol("f")
        Df = C.D(f)
        assert isinstance(Df, DOperator)
        assert Df.f is f
        assert Df.d_op is C.d

    def test_D_rejects_non_expr(self):
        C = CourantAlgebroid()
        with pytest.raises(TypeError):
            C.D("not_expr")

    def test_anchor_of_returns_courant_anchor(self):
        C = CourantAlgebroid()
        X, alpha = Symbol("X"), Symbol("α")
        a = C.anchor_of(SectionPair(X, alpha))
        assert isinstance(a, CourantAnchor)
        assert a.section == SectionPair(X, alpha)

    def test_anchor_of_accepts_opaque_section(self):
        C = CourantAlgebroid()
        e = Symbol("e")
        a = C.anchor_of(e)
        assert isinstance(a, CourantAnchor)
        assert a.section is e


# --------------------------------------------------------------------- #
# Stage E.1: prove_D_compat                                              #
# --------------------------------------------------------------------- #


class TestStageEProveDCompat:
    """Definitional proof of ``anchor(D f) = 0``."""

    def test_returns_proof_chain(self):
        C = CourantAlgebroid()
        chain = C.prove_D_compat(Symbol("f"))
        assert isinstance(chain, ProofChain)

    def test_two_definitional_steps(self):
        """Chain has *exactly* two atomic axiom steps (no theorem citation)."""
        C = CourantAlgebroid()
        chain = C.prove_D_compat(Symbol("f"))
        assert len(chain) == 2

    def test_both_steps_axiom_tagged(self):
        """Neither step is a ``theorem`` citation — full definitional unfold."""
        C = CourantAlgebroid()
        chain = C.prove_D_compat(Symbol("f"))
        for step in chain.steps:
            assert step.provenance_tag == "axiom"

    def test_step_zero_unfolds_D(self):
        C = CourantAlgebroid()
        chain = C.prove_D_compat(Symbol("f"))
        step = chain.steps[0]
        assert step.rule == "DOperatorDefinition"
        assert isinstance(step.before, CourantAnchor)
        assert isinstance(step.before.section, DOperator)
        assert isinstance(step.after, CourantAnchor)
        assert isinstance(step.after.section, SectionPair)

    def test_step_one_extracts_vector(self):
        C = CourantAlgebroid()
        f = Symbol("f")
        chain = C.prove_D_compat(f)
        step = chain.steps[1]
        assert step.rule == "CourantAnchorDefinition"
        assert step.after == Integer(0)

    def test_initial_is_anchor_of_D(self):
        C = CourantAlgebroid()
        f = Symbol("f")
        chain = C.prove_D_compat(f)
        initial = chain.initial
        assert isinstance(initial, CourantAnchor)
        assert isinstance(initial.section, DOperator)
        assert initial.section.f is f

    def test_final_is_zero(self):
        C = CourantAlgebroid()
        chain = C.prove_D_compat(Symbol("f"))
        assert chain.final == Integer(0)

    def test_d_operator_propagates_algebroid_d(self):
        """If the algebroid carries a custom ``d``, ``D f`` uses *that* d."""
        C = CourantAlgebroid()
        f = Symbol("f")
        chain = C.prove_D_compat(f)
        step = chain.steps[0]
        section_after = step.after.section
        assert isinstance(section_after, SectionPair)
        assert section_after.form == Act(C.d, f)

    def test_rejects_non_expr_argument(self):
        C = CourantAlgebroid()
        with pytest.raises(TypeError):
            C.prove_D_compat("not_an_expr")

    def test_chain_consistent(self):
        """Step 1's ``before`` matches step 0's ``after``."""
        C = CourantAlgebroid()
        chain = C.prove_D_compat(Symbol("f"))
        assert chain.steps[1].before == chain.steps[0].after


# --------------------------------------------------------------------- #
# Stage E.2: prove_anchor_compat                                         #
# --------------------------------------------------------------------- #


class TestStageEProveAnchorCompat:
    """Definitional proof of ``anchor([e1, e2]_C) = [X, Y]_VF``."""

    def test_returns_proof_chain(self, algebroid, ab, registry):
        a, b = ab
        chain = algebroid.prove_anchor_compat(a, b, registry=registry)
        assert isinstance(chain, ProofChain)

    def test_two_definitional_steps(self, algebroid, ab, registry):
        a, b = ab
        chain = algebroid.prove_anchor_compat(a, b, registry=registry)
        assert len(chain) == 2

    def test_both_steps_axiom_tagged(self, algebroid, ab, registry):
        a, b = ab
        chain = algebroid.prove_anchor_compat(a, b, registry=registry)
        for step in chain.steps:
            assert step.provenance_tag == "axiom"

    def test_step_zero_courant_definition(self, algebroid, ab, registry):
        a, b = ab
        chain = algebroid.prove_anchor_compat(a, b, registry=registry)
        step = chain.steps[0]
        assert step.rule == "CourantBracketDefinition"
        assert isinstance(step.before, CourantAnchor)
        assert isinstance(step.after, CourantAnchor)
        assert isinstance(step.after.section, SectionPair)

    def test_step_one_anchor_projection(self, algebroid, ab, registry):
        a, b = ab
        chain = algebroid.prove_anchor_compat(a, b, registry=registry)
        step = chain.steps[1]
        assert step.rule == "CourantAnchorDefinition"

    def test_initial_is_anchor_of_courant_apply(self, algebroid, ab, registry):
        a, b = ab
        chain = algebroid.prove_anchor_compat(a, b, registry=registry)
        initial = chain.initial
        assert isinstance(initial, CourantAnchor)

    def test_final_is_inert_vector_bracket(self, algebroid, ab, registry):
        """Final form is ``BracketApply(vector_bracket, X, Y)``."""
        from jacopy.brackets.base import BracketApply
        a, b = ab
        chain = algebroid.prove_anchor_compat(a, b, registry=registry)
        final = chain.final
        assert isinstance(final, BracketApply)
        assert final.bracket is algebroid.vector_bracket
        assert final.a == a.vector
        assert final.b == b.vector

    def test_chain_consistent(self, algebroid, ab, registry):
        a, b = ab
        chain = algebroid.prove_anchor_compat(a, b, registry=registry)
        assert chain.steps[1].before == chain.steps[0].after

    def test_section_after_step_zero_carries_inert_vector(
        self, algebroid, ab, registry
    ):
        """Step 0 produces a SectionPair whose vector half is inert."""
        from jacopy.brackets.base import BracketApply
        a, b = ab
        chain = algebroid.prove_anchor_compat(a, b, registry=registry)
        section = chain.steps[0].after.section
        assert isinstance(section, SectionPair)
        assert isinstance(section.vector, BracketApply)

    def test_form_half_includes_lie_terms(self, algebroid, ab, registry):
        """Form half of the unfold contains L_X β / L_Y α / d(ι terms."""
        a, b = ab
        chain = algebroid.prove_anchor_compat(a, b, registry=registry)
        section = chain.steps[0].after.section
        assert "L_X" in repr(section.form) or "L_" in repr(section.form)

    def test_twisted_justification_mentions_iY_iX_H(self, twisted, ab, registry):
        a, b = ab
        chain = twisted.prove_anchor_compat(a, b, registry=registry)
        # Justification on step 0 includes the twist contribution
        assert "ι_Y ι_X H" in chain.steps[0].justification

    def test_twisted_final_still_inert_vector_bracket(self, twisted, ab, registry):
        """H-twist only affects the form half; vector projection unchanged."""
        from jacopy.brackets.base import BracketApply
        a, b = ab
        chain = twisted.prove_anchor_compat(a, b, registry=registry)
        assert isinstance(chain.final, BracketApply)
        assert chain.final.bracket is twisted.vector_bracket

    def test_rejects_non_section_pair_left(self, algebroid, registry):
        with pytest.raises(TypeError):
            algebroid.prove_anchor_compat(
                Symbol("e1"),
                SectionPair(Symbol("Y"), Symbol("β")),
                registry=registry,
            )

    def test_rejects_non_section_pair_right(self, algebroid, registry):
        with pytest.raises(TypeError):
            algebroid.prove_anchor_compat(
                SectionPair(Symbol("X"), Symbol("α")),
                Symbol("e2"),
                registry=registry,
            )


# --------------------------------------------------------------------- #
# Stage E.3: prove_leibniz                                               #
# --------------------------------------------------------------------- #


class TestStageEProveLeibniz:
    """Definitional proof of Vaisman Leibniz on Courant bracket."""

    @pytest.fixture
    def f_sym(self):
        return Symbol("f")

    def test_returns_proof_chain(self, algebroid, ab, f_sym, registry):
        a, b = ab
        chain = algebroid.prove_leibniz(a, b, f_sym, registry=registry)
        assert isinstance(chain, ProofChain)

    def test_eight_axiom_steps(self, algebroid, ab, f_sym, registry):
        """Granular 8-step chain (Vaisman atomic axioms)."""
        a, b = ab
        chain = algebroid.prove_leibniz(a, b, f_sym, registry=registry)
        assert len(chain) == 8

    def test_all_steps_axiom_tagged(self, algebroid, ab, f_sym, registry):
        """No theorem-citation shortcut; all eight steps are axioms."""
        a, b = ab
        chain = algebroid.prove_leibniz(a, b, f_sym, registry=registry)
        for step in chain.steps:
            assert step.provenance_tag == "axiom"

    def test_step_rules_in_expected_order(self, algebroid, ab, f_sym, registry):
        a, b = ab
        chain = algebroid.prove_leibniz(a, b, f_sym, registry=registry)
        rules = [s.rule for s in chain.steps]
        assert rules == [
            "CourantBracketDefinition",
            "LieBracketLeibnizSecondSlot",
            "LieDerivativeProductRule",
            "LieRescaling",
            "InteriorScalarLinearity",
            "InteriorPairing",
            "ExteriorDProductRule",
            "VaismanLeibnizRegroup",
        ]

    def test_initial_is_bracket_apply(self, algebroid, ab, f_sym, registry):
        from jacopy.brackets.base import BracketApply
        a, b = ab
        chain = algebroid.prove_leibniz(a, b, f_sym, registry=registry)
        assert isinstance(chain.initial, BracketApply)
        assert chain.initial.bracket is algebroid.courant

    def test_final_is_section_pair(self, algebroid, ab, f_sym, registry):
        a, b = ab
        chain = algebroid.prove_leibniz(a, b, f_sym, registry=registry)
        assert isinstance(chain.final, SectionPair)

    def test_chain_consistency(self, algebroid, ab, f_sym, registry):
        """Each step's before equals the previous step's after."""
        a, b = ab
        chain = algebroid.prove_leibniz(a, b, f_sym, registry=registry)
        for i in range(1, len(chain)):
            assert chain.steps[i].before == chain.steps[i - 1].after

    def test_step_zero_unfolds_to_section_pair(
        self, algebroid, ab, f_sym, registry
    ):
        """After Courant def, current state is a SectionPair."""
        a, b = ab
        chain = algebroid.prove_leibniz(a, b, f_sym, registry=registry)
        assert isinstance(chain.steps[0].after, SectionPair)

    def test_twisted_chain_has_same_length(self, twisted, ab, f_sym, registry):
        a, b = ab
        chain = twisted.prove_leibniz(a, b, f_sym, registry=registry)
        assert len(chain) == 8

    def test_twisted_step_zero_mentions_iY_iX_H(
        self, twisted, ab, f_sym, registry
    ):
        a, b = ab
        chain = twisted.prove_leibniz(a, b, f_sym, registry=registry)
        # The Courant def justification mentions the H twist contraction
        # (in the f-scaled form: ι_{fY} ι_X H).
        assert "ι_{fY}" in chain.steps[0].justification

    def test_twisted_step_four_factors_h_via_iota_linearity(
        self, twisted, ab, f_sym, registry
    ):
        """Step 4 (InteriorScalarLinearity) factors f out of ι_{fY} ι_X H."""
        a, b = ab
        chain = twisted.prove_leibniz(a, b, f_sym, registry=registry)
        assert "f ι_Y ι_X H" in chain.steps[4].justification

    def test_final_form_carries_inner_product_node(
        self, algebroid, ab, f_sym, registry
    ):
        """The − ⟨e1, e2⟩ Df term appears via a CourantInnerProduct node."""
        a, b = ab
        chain = algebroid.prove_leibniz(a, b, f_sym, registry=registry)
        final_repr = repr(chain.final)
        assert "⟨" in final_repr or "CourantInnerProduct" in final_repr

    def test_final_form_carries_d_of_f(self, algebroid, ab, f_sym, registry):
        """The Df term shows up in the form half via d(f)."""
        a, b = ab
        chain = algebroid.prove_leibniz(a, b, f_sym, registry=registry)
        # df should appear in the form half.
        assert "d" in repr(chain.final.form)

    def test_rejects_non_section_pair_e1(self, algebroid, f_sym, registry):
        with pytest.raises(TypeError):
            algebroid.prove_leibniz(
                Symbol("e1"),
                SectionPair(Symbol("Y"), Symbol("β")),
                f_sym,
                registry=registry,
            )

    def test_rejects_non_section_pair_e2(self, algebroid, f_sym, registry):
        with pytest.raises(TypeError):
            algebroid.prove_leibniz(
                SectionPair(Symbol("X"), Symbol("α")),
                Symbol("e2"),
                f_sym,
                registry=registry,
            )

    def test_rejects_non_expr_f(self, algebroid, ab, registry):
        a, b = ab
        with pytest.raises(TypeError):
            algebroid.prove_leibniz(a, b, "not_an_expr", registry=registry)


# --------------------------------------------------------------------- #
# Stage E.4: prove_inner_compat                                          #
# --------------------------------------------------------------------- #


class TestStageEProveInnerCompat:
    """Vaisman inner-product compatibility on Courant bracket."""

    @pytest.fixture
    def e3(self):
        return SectionPair(Symbol("Z"), Symbol("γ"))

    @pytest.fixture
    def registry3(self, registry):
        Z, gamma = Symbol("Z"), Symbol("γ")
        registry.declare(Z, Graded(degree=0))
        registry.declare(gamma, Graded(degree=1))
        return registry

    def test_returns_proof_chain(self, algebroid, ab, e3, registry3):
        a, b = ab
        chain = algebroid.prove_inner_compat(a, b, e3, registry=registry3)
        assert isinstance(chain, ProofChain)

    def test_seven_axiom_steps(self, algebroid, ab, e3, registry3):
        a, b = ab
        chain = algebroid.prove_inner_compat(a, b, e3, registry=registry3)
        assert len(chain) == 7

    def test_all_steps_axiom_tagged(self, algebroid, ab, e3, registry3):
        """No theorem-citation shortcut; all seven steps are axioms."""
        a, b = ab
        chain = algebroid.prove_inner_compat(a, b, e3, registry=registry3)
        for step in chain.steps:
            assert step.provenance_tag == "axiom"

    def test_step_rules_in_expected_order(self, algebroid, ab, e3, registry3):
        a, b = ab
        chain = algebroid.prove_inner_compat(a, b, e3, registry=registry3)
        rules = [s.rule for s in chain.steps]
        assert rules == [
            "CourantInnerProductDefinition",
            "PairingLieLeibniz",
            "VectorLieDerivativeIsBracket",
            "DAlphaAntisymmetry",
            "CourantInnerProductDefinition",
            "DorfmanBracketDefinition",
            "CourantDorfmanBridge",
        ]

    def test_chain_consistency(self, algebroid, ab, e3, registry3):
        """Each step's before equals the previous step's after."""
        a, b = ab
        chain = algebroid.prove_inner_compat(a, b, e3, registry=registry3)
        for i in range(1, len(chain)):
            assert chain.steps[i].before == chain.steps[i - 1].after

    def test_initial_is_lie_derivative_on_inner_product(
        self, algebroid, ab, e3, registry3
    ):
        from jacopy.algebra.derivation import Act
        from jacopy.brackets.courant_inner_product import CourantInnerProduct
        a, b = ab
        chain = algebroid.prove_inner_compat(a, b, e3, registry=registry3)
        initial = chain.initial
        assert isinstance(initial, Act)
        assert isinstance(initial.arg, CourantInnerProduct)

    def test_final_is_sum_of_two_inner_products(
        self, algebroid, ab, e3, registry3
    ):
        """RHS = ⟨[e1,e2]_C + D⟨e1,e2⟩, e3⟩ + ⟨e2, [e1,e3]_C + D⟨e1,e3⟩⟩."""
        from jacopy.brackets.courant_inner_product import CourantInnerProduct
        from jacopy.core.expr import Sum
        a, b = ab
        chain = algebroid.prove_inner_compat(a, b, e3, registry=registry3)
        final = chain.final
        assert isinstance(final, Sum)
        assert len(final.children) == 2
        for child in final.children:
            assert isinstance(child, CourantInnerProduct)

    def test_dα_antisymmetry_step_uses_pairings(
        self, algebroid, ab, e3, registry3
    ):
        """Step 3 (DAlphaAntisymmetry) introduces ι dα pairings."""
        a, b = ab
        chain = algebroid.prove_inner_compat(a, b, e3, registry=registry3)
        step3_repr = repr(chain.steps[3].after)
        # The state should now contain ι_? dα-related Pairings.
        assert "d" in step3_repr  # d operator present

    def test_courant_dorfman_bridge_step_carries_d(
        self, algebroid, ab, e3, registry3
    ):
        """Step 6 (CourantDorfmanBridge reverse) introduces D operator."""
        from jacopy.brackets.courant_anchor_d import DOperator
        a, b = ab
        chain = algebroid.prove_inner_compat(a, b, e3, registry=registry3)
        # The final state's Sum operands should each carry a Sum of
        # BracketApply + DOperator on the inner-product e_j operands.
        assert "D(" in repr(chain.final)

    def test_twisted_same_chain_length(self, twisted, ab, e3, registry3):
        a, b = ab
        chain = twisted.prove_inner_compat(a, b, e3, registry=registry3)
        assert len(chain) == 7

    def test_rejects_non_section_pair_e1(self, algebroid, ab, e3, registry3):
        a, b = ab
        with pytest.raises(TypeError):
            algebroid.prove_inner_compat(
                Symbol("not_pair"), b, e3, registry=registry3
            )

    def test_rejects_non_section_pair_e2(self, algebroid, ab, e3, registry3):
        a, b = ab
        with pytest.raises(TypeError):
            algebroid.prove_inner_compat(
                a, Symbol("not_pair"), e3, registry=registry3
            )

    def test_rejects_non_section_pair_e3(self, algebroid, ab, registry3):
        a, b = ab
        with pytest.raises(TypeError):
            algebroid.prove_inner_compat(
                a, b, Symbol("not_pair"), registry=registry3
            )


# --------------------------------------------------------------------- #
# Stage E.5: prove_jacobi_by_definitions                                 #
# --------------------------------------------------------------------- #


class TestStageEProveJacobiByDefinitions:
    """Definitional alternative to prove_jacobi_reduction (cyclic Jacobi)."""

    @pytest.fixture
    def e3(self):
        return SectionPair(Symbol("Z"), Symbol("γ"))

    def test_returns_proof_chain(self, algebroid, ab, e3):
        a, b = ab
        chain = algebroid.prove_jacobi_by_definitions(a, b, e3)
        assert isinstance(chain, ProofChain)

    def test_four_axiom_steps(self, algebroid, ab, e3):
        a, b = ab
        chain = algebroid.prove_jacobi_by_definitions(a, b, e3)
        assert len(chain) == 4

    def test_all_steps_axiom_tagged(self, algebroid, ab, e3):
        """No theorem-citation shortcut; all four steps are axioms."""
        a, b = ab
        chain = algebroid.prove_jacobi_by_definitions(a, b, e3)
        for step in chain.steps:
            assert step.provenance_tag == "axiom"

    def test_step_rules_in_expected_order(self, algebroid, ab, e3):
        a, b = ab
        chain = algebroid.prove_jacobi_by_definitions(a, b, e3)
        rules = [s.rule for s in chain.steps]
        assert rules == [
            "CyclicCourantJacobiatorDefinition",
            "CourantDorfmanBridge",
            "CyclicDInnerProductCancellation",
            "DorfmanLodayClosure",
        ]

    def test_chain_consistency(self, algebroid, ab, e3):
        a, b = ab
        chain = algebroid.prove_jacobi_by_definitions(a, b, e3)
        for i in range(1, len(chain)):
            assert chain.steps[i].before == chain.steps[i - 1].after

    def test_initial_is_sum_of_three_outer_brackets(self, algebroid, ab, e3):
        from jacopy.brackets.base import BracketApply
        from jacopy.core.expr import Sum
        a, b = ab
        chain = algebroid.prove_jacobi_by_definitions(a, b, e3)
        initial = chain.initial
        assert isinstance(initial, Sum)
        # Three cyclic summands.
        assert len(initial.children) == 3
        for child in initial.children:
            assert isinstance(child, BracketApply)
            assert child.bracket is algebroid.courant

    def test_untwisted_final_is_zero(self, algebroid, ab, e3):
        a, b = ab
        chain = algebroid.prove_jacobi_by_definitions(a, b, e3)
        assert chain.final == Integer(0)

    def test_twisted_final_carries_dH(self, twisted, ab, e3):
        a, b = ab
        chain = twisted.prove_jacobi_by_definitions(a, b, e3)
        assert "H" in repr(chain.final)
        assert "d" in repr(chain.final)

    def test_twisted_chain_same_length(self, twisted, ab, e3):
        """Both untwisted and twisted produce 4-step chains."""
        a, b = ab
        chain = twisted.prove_jacobi_by_definitions(a, b, e3)
        assert len(chain) == 4

    def test_step_one_brings_in_dorfman(self, algebroid, ab, e3):
        """Step 1 (bridge) introduces Dorfman bracket and D operator."""
        a, b = ab
        chain = algebroid.prove_jacobi_by_definitions(a, b, e3)
        step1_repr = repr(chain.steps[1].after)
        assert "_D" in step1_repr  # Dorfman bracket
        assert "D(" in step1_repr  # D operator

    def test_step_three_uses_jacobi_condition_obstruction(
        self, algebroid, twisted, ab, e3
    ):
        """Step 3's after equals jacobi_condition().obstruction."""
        a, b = ab
        chain_u = algebroid.prove_jacobi_by_definitions(a, b, e3)
        assert chain_u.final == algebroid.jacobi_condition().obstruction
        chain_t = twisted.prove_jacobi_by_definitions(a, b, e3)
        assert chain_t.final == twisted.jacobi_condition().obstruction

    def test_rejects_non_section_pair_e1(self, algebroid, ab, e3):
        a, b = ab
        with pytest.raises(TypeError):
            algebroid.prove_jacobi_by_definitions(Symbol("nope"), b, e3)

    def test_rejects_non_section_pair_e2(self, algebroid, ab, e3):
        a, b = ab
        with pytest.raises(TypeError):
            algebroid.prove_jacobi_by_definitions(a, Symbol("nope"), e3)

    def test_rejects_non_section_pair_e3(self, algebroid, ab):
        a, b = ab
        with pytest.raises(TypeError):
            algebroid.prove_jacobi_by_definitions(a, b, Symbol("nope"))

    def test_does_not_replace_prove_jacobi_reduction(self, algebroid):
        """prove_jacobi_reduction (existing seeded version) still works."""
        chain = algebroid.prove_jacobi_reduction()
        assert isinstance(chain, ProofChain)
        # The two methods coexist; prove_jacobi_by_definitions is the
        # definitional alternative.
