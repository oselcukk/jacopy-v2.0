"""Tests for ``jacopy.library.dirac``."""

from __future__ import annotations

import pytest

from jacopy.brackets.derived import VanishingCondition
from jacopy.brackets.dorfman import SectionPair
from jacopy.core.expr import Expr, Integer, Symbol
from jacopy.core.properties import Graded
from jacopy.core.registry import PropertyRegistry
from jacopy.library import theorem_book
from jacopy.library.courant_algebroid import CourantAlgebroid
from jacopy.library.dirac import (
    THEOREM_DIRAC_INVOLUTIVITY,
    THEOREM_DIRAC_ISOTROPY,
    DiracStructure,
    poisson_dirac,
    presymplectic_dirac,
)
from jacopy.proof.chain import ProofChain


# --------------------------------------------------------------------- #
# Fixtures                                                               #
# --------------------------------------------------------------------- #


@pytest.fixture
def courant():
    return CourantAlgebroid()


@pytest.fixture
def subbundle():
    return Symbol("L")


@pytest.fixture
def dirac(courant, subbundle):
    return DiracStructure(courant, subbundle)


@pytest.fixture
def ab():
    return (
        SectionPair(Symbol("X"), Symbol("α")),
        SectionPair(Symbol("Y"), Symbol("β")),
    )


# --------------------------------------------------------------------- #
# Construction                                                           #
# --------------------------------------------------------------------- #


class TestConstruction:
    def test_basic(self, dirac, courant, subbundle):
        assert dirac.courant is courant
        assert dirac.subbundle is subbundle

    def test_default_name_carries_subbundle(self, dirac):
        assert "L" in dirac.name

    def test_custom_name(self, courant, subbundle):
        D = DiracStructure(courant, subbundle, name="MyDirac")
        assert D.name == "MyDirac"

    def test_rejects_non_courant(self, subbundle):
        with pytest.raises(TypeError, match="CourantAlgebroid"):
            DiracStructure("not courant", subbundle)  # type: ignore[arg-type]

    def test_rejects_non_expr_subbundle(self, courant):
        with pytest.raises(TypeError, match="Expr"):
            DiracStructure(courant, "L")  # type: ignore[arg-type]


# --------------------------------------------------------------------- #
# Pairing + isotropy                                                     #
# --------------------------------------------------------------------- #


class TestPairing:
    def test_pairing_returns_expr(self, dirac, ab):
        a, b = ab
        p = dirac.pairing(a, b)
        assert isinstance(p, Expr)

    def test_pairing_involves_both_interiors(self, dirac, ab):
        """``½(ι_X β + ι_Y α)``: both ``ι`` applications must show up."""
        a, b = ab
        text = repr(dirac.pairing(a, b))
        assert "ι_X" in text
        assert "ι_Y" in text

    def test_pairing_has_half_factor(self, dirac, ab):
        a, b = ab
        text = repr(dirac.pairing(a, b))
        assert "1/2" in text

    def test_pairing_rejects_non_section_pair(self, dirac):
        with pytest.raises(TypeError, match="SectionPair"):
            dirac.pairing(Symbol("a"), Symbol("b"))  # type: ignore[arg-type]

    def test_isotropy_obstruction_is_iota_x_alpha(self, dirac, ab):
        a, _ = ab
        obs = dirac.isotropy_obstruction(a)
        text = repr(obs)
        assert "ι_X" in text
        assert "α" in text

    def test_isotropy_obstruction_rejects_non_section_pair(self, dirac):
        with pytest.raises(TypeError, match="SectionPair"):
            dirac.isotropy_obstruction(Symbol("a"))  # type: ignore[arg-type]

    def test_isotropy_condition_type(self, dirac, ab):
        a, _ = ab
        cond = dirac.isotropy_condition(a)
        assert isinstance(cond, VanishingCondition)

    def test_isotropy_condition_name_mentions_isotropy(self, dirac, ab):
        a, _ = ab
        cond = dirac.isotropy_condition(a)
        assert "isotropy" in cond.name


# --------------------------------------------------------------------- #
# Prove isotropy                                                         #
# --------------------------------------------------------------------- #


class TestProveIsotropy:
    def test_returns_proof_chain(self, dirac, ab):
        a, _ = ab
        chain = dirac.prove_isotropy(a)
        assert isinstance(chain, ProofChain)

    def test_single_axiom_step(self, dirac, ab):
        a, _ = ab
        chain = dirac.prove_isotropy(a)
        assert len(chain) == 1
        step = chain.steps[0]
        assert step.rule == "DiracIsotropyAxiom"
        assert step.provenance_tag == "axiom"

    def test_step_discharges_to_zero(self, dirac, ab):
        a, _ = ab
        chain = dirac.prove_isotropy(a)
        assert chain.steps[0].after == Integer(0)

    def test_step_starts_on_obstruction(self, dirac, ab):
        a, _ = ab
        chain = dirac.prove_isotropy(a)
        assert chain.steps[0].before == dirac.isotropy_obstruction(a)


# --------------------------------------------------------------------- #
# Involutivity                                                           #
# --------------------------------------------------------------------- #


class TestInvolutivity:
    def test_condition_type(self, dirac, ab):
        a, b = ab
        cond = dirac.involutivity_condition(a, b)
        assert isinstance(cond, VanishingCondition)

    def test_condition_name_mentions_involutivity(self, dirac, ab):
        a, b = ab
        cond = dirac.involutivity_condition(a, b)
        assert "involutivity" in cond.name

    def test_prove_returns_proof_chain(self, dirac, ab):
        a, b = ab
        chain = dirac.prove_involutivity(a, b)
        assert isinstance(chain, ProofChain)

    def test_prove_single_axiom_step(self, dirac, ab):
        a, b = ab
        chain = dirac.prove_involutivity(a, b)
        assert len(chain) == 1
        step = chain.steps[0]
        assert step.rule == "DiracInvolutivityAxiom"
        assert step.provenance_tag == "axiom"

    def test_prove_discharges_to_zero(self, dirac, ab):
        a, b = ab
        chain = dirac.prove_involutivity(a, b)
        assert chain.steps[0].after == Integer(0)

    def test_prove_rejects_non_section_pair(self, dirac, ab):
        a, _ = ab
        with pytest.raises(TypeError, match="SectionPair"):
            dirac.prove_involutivity(a, Symbol("b"))  # type: ignore[arg-type]


# --------------------------------------------------------------------- #
# Factories: poisson_dirac + presymplectic_dirac                         #
# --------------------------------------------------------------------- #


class TestPoissonDirac:
    def test_returns_dirac_structure(self):
        pi = Symbol("π")
        D = poisson_dirac(pi)
        assert isinstance(D, DiracStructure)

    def test_default_courant_is_untwisted(self):
        D = poisson_dirac(Symbol("π"))
        assert D.courant.is_twisted is False

    def test_custom_courant(self):
        H = Symbol("H")
        C = CourantAlgebroid(background_H=H)
        D = poisson_dirac(Symbol("π"), courant=C)
        assert D.courant is C

    def test_subbundle_tagged_with_pi(self):
        D = poisson_dirac(Symbol("π"))
        assert "π" in D.subbundle._repr_inner()

    def test_default_name_tags_pi(self):
        D = poisson_dirac(Symbol("π"))
        assert "π" in D.name

    def test_rejects_non_expr_pi(self):
        with pytest.raises(TypeError, match="Expr"):
            poisson_dirac("π")  # type: ignore[arg-type]

    def test_rejects_non_courant_ambient(self):
        with pytest.raises(TypeError, match="CourantAlgebroid"):
            poisson_dirac(Symbol("π"), courant="not courant")  # type: ignore[arg-type]


class TestPresymplecticDirac:
    def test_returns_dirac_structure(self):
        D = presymplectic_dirac(Symbol("ω"))
        assert isinstance(D, DiracStructure)

    def test_subbundle_tagged_with_omega(self):
        D = presymplectic_dirac(Symbol("ω"))
        assert "ω" in D.subbundle._repr_inner()

    def test_default_name_tags_omega(self):
        D = presymplectic_dirac(Symbol("ω"))
        assert "ω" in D.name

    def test_rejects_non_expr_omega(self):
        with pytest.raises(TypeError, match="Expr"):
            presymplectic_dirac("ω")  # type: ignore[arg-type]

    def test_rejects_non_courant_ambient(self):
        with pytest.raises(TypeError, match="CourantAlgebroid"):
            presymplectic_dirac(Symbol("ω"), courant=42)  # type: ignore[arg-type]


# --------------------------------------------------------------------- #
# Seeded theorems                                                        #
# --------------------------------------------------------------------- #


class TestSeededTheorems:
    def test_dirac_isotropy_registered(self):
        assert "dirac_isotropy" in theorem_book
        assert theorem_book.get("dirac_isotropy") is THEOREM_DIRAC_ISOTROPY

    def test_dirac_involutivity_registered(self):
        assert "dirac_involutivity" in theorem_book
        assert (
            theorem_book.get("dirac_involutivity")
            is THEOREM_DIRAC_INVOLUTIVITY
        )

    def test_isotropy_proof_is_single_axiom_step(self):
        thm = THEOREM_DIRAC_ISOTROPY
        assert len(thm.proof) == 1
        step = thm.proof.steps[0]
        assert step.rule == "DiracIsotropyAxiom"
        assert step.provenance_tag == "axiom"
        assert step.after == Integer(0)

    def test_involutivity_proof_is_single_axiom_step(self):
        thm = THEOREM_DIRAC_INVOLUTIVITY
        assert len(thm.proof) == 1
        step = thm.proof.steps[0]
        assert step.rule == "DiracInvolutivityAxiom"
        assert step.provenance_tag == "axiom"
        assert step.after == Integer(0)

    def test_isotropy_from_axioms_mentions_isotropy(self):
        thm = THEOREM_DIRAC_ISOTROPY
        assert any("isotropy" in ax.lower() for ax in thm.from_axioms)

    def test_involutivity_from_axioms_mentions_involutivity(self):
        thm = THEOREM_DIRAC_INVOLUTIVITY
        assert any("involutivity" in ax.lower() for ax in thm.from_axioms)
