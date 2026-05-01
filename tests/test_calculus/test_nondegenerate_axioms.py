"""Tests for the Non-degeneracy injectivity axiom (Faz 12.B #9)."""

import pytest

from jacopy.algebra.derivation import Act, Derivation
from jacopy.calculus.exterior_d import d as default_d
from jacopy.calculus.interior import interior
from jacopy.calculus.nondegenerate_axioms import (
    NonDegenerateInteriorEqualityDefinition,
)
from jacopy.core.expr import Integer, Neg, Sum, Symbol
from jacopy.core.properties import (
    NonDegenerate,
    Provenance,
    ProofRef,
)
from jacopy.core.registry import PropertyRegistry
from jacopy.proof.expansion import ExpansionEngine


# --------------------------------------------------------------------- #
# NonDegenerate property                                                #
# --------------------------------------------------------------------- #


class TestNonDegenerateProperty:
    def test_construction_axiom_default(self):
        prop = NonDegenerate()
        assert prop.is_axiom
        assert prop.proof is None

    def test_derived_requires_proof_ref(self):
        ref = ProofRef(rule="symplectic 2-form", sources=("ω",))
        prop = NonDegenerate(provenance=Provenance.DERIVED, proof=ref)
        assert prop.is_derived
        assert prop.proof is ref

    def test_derived_without_proof_raises(self):
        with pytest.raises(ValueError):
            NonDegenerate(provenance=Provenance.DERIVED)

    def test_axiom_with_proof_ref_raises(self):
        ref = ProofRef(rule="x", sources=())
        with pytest.raises(ValueError):
            NonDegenerate(provenance=Provenance.AXIOM, proof=ref)

    def test_registry_roundtrip(self):
        reg = PropertyRegistry()
        omega = Symbol("ω")
        reg.declare(omega, NonDegenerate())
        assert reg.has(omega, NonDegenerate)
        eta = Symbol("η")
        assert not reg.has(eta, NonDegenerate)


# --------------------------------------------------------------------- #
# Rule match logic                                                      #
# --------------------------------------------------------------------- #


@pytest.fixture
def reg_omega():
    reg = PropertyRegistry()
    omega = Symbol("ω")
    reg.declare(omega, NonDegenerate())
    return reg, omega


class TestRuleMatches:
    def test_matches_iota_y_minus_iota_z(self, reg_omega):
        reg, omega = reg_omega
        Y = Derivation("Y", 0)
        Z = Derivation("Z", 0)
        rule = NonDegenerateInteriorEqualityDefinition(registry=reg)
        expr = Sum(
            Act(interior(Y), omega),
            Neg(Act(interior(Z), omega)),
        )
        assert rule.matches(expr)

    def test_matches_with_swapped_order(self, reg_omega):
        reg, omega = reg_omega
        Y = Derivation("Y", 0)
        Z = Derivation("Z", 0)
        rule = NonDegenerateInteriorEqualityDefinition(registry=reg)
        # Negated child first, positive child second.
        expr = Sum(
            Neg(Act(interior(Z), omega)),
            Act(interior(Y), omega),
        )
        assert rule.matches(expr)

    def test_no_match_when_omega_not_declared(self):
        reg = PropertyRegistry()
        omega = Symbol("ω")  # not declared NonDegenerate
        Y = Derivation("Y", 0)
        Z = Derivation("Z", 0)
        rule = NonDegenerateInteriorEqualityDefinition(registry=reg)
        expr = Sum(
            Act(interior(Y), omega),
            Neg(Act(interior(Z), omega)),
        )
        assert not rule.matches(expr)

    def test_no_match_when_registry_none(self):
        omega = Symbol("ω")
        Y = Derivation("Y", 0)
        Z = Derivation("Z", 0)
        rule = NonDegenerateInteriorEqualityDefinition()
        expr = Sum(
            Act(interior(Y), omega),
            Neg(Act(interior(Z), omega)),
        )
        assert not rule.matches(expr)

    def test_no_match_when_both_positive(self, reg_omega):
        reg, omega = reg_omega
        Y = Derivation("Y", 0)
        Z = Derivation("Z", 0)
        rule = NonDegenerateInteriorEqualityDefinition(registry=reg)
        # Sum(ι_Y ω, ι_Z ω), same sign, doesn't match injectivity shape.
        expr = Sum(
            Act(interior(Y), omega),
            Act(interior(Z), omega),
        )
        assert not rule.matches(expr)

    def test_no_match_when_different_omegas(self, reg_omega):
        reg, omega = reg_omega
        eta = Symbol("η")
        reg.declare(eta, NonDegenerate())
        Y = Derivation("Y", 0)
        Z = Derivation("Z", 0)
        rule = NonDegenerateInteriorEqualityDefinition(registry=reg)
        expr = Sum(
            Act(interior(Y), omega),
            Neg(Act(interior(Z), eta)),
        )
        assert not rule.matches(expr)

    def test_no_match_on_three_term_sum(self, reg_omega):
        reg, omega = reg_omega
        Y = Derivation("Y", 0)
        Z = Derivation("Z", 0)
        W = Derivation("W", 0)
        rule = NonDegenerateInteriorEqualityDefinition(registry=reg)
        expr = Sum(
            Act(interior(Y), omega),
            Neg(Act(interior(Z), omega)),
            Act(interior(W), omega),
        )
        assert not rule.matches(expr)

    def test_no_match_on_non_sum(self, reg_omega):
        reg, omega = reg_omega
        Y = Derivation("Y", 0)
        rule = NonDegenerateInteriorEqualityDefinition(registry=reg)
        # Just a single ι_Y ω, no Sum wrapper.
        assert not rule.matches(Act(interior(Y), omega))

    def test_no_match_when_arg_is_not_omega(self, reg_omega):
        reg, omega = reg_omega
        eta = Symbol("η")  # NOT declared NonDegenerate
        Y = Derivation("Y", 0)
        Z = Derivation("Z", 0)
        rule = NonDegenerateInteriorEqualityDefinition(registry=reg)
        expr = Sum(
            Act(interior(Y), eta),
            Neg(Act(interior(Z), eta)),
        )
        assert not rule.matches(expr)


# --------------------------------------------------------------------- #
# Rule rewrite                                                          #
# --------------------------------------------------------------------- #


class TestRuleRewrite:
    def test_rewrites_to_y_minus_z(self, reg_omega):
        reg, omega = reg_omega
        Y = Derivation("Y", 0)
        Z = Derivation("Z", 0)
        rule = NonDegenerateInteriorEqualityDefinition(registry=reg)
        expr = Sum(
            Act(interior(Y), omega),
            Neg(Act(interior(Z), omega)),
        )
        out = rule.rewrite(expr)
        assert out == Sum(Y, Neg(Z))

    def test_rewrites_with_swapped_order(self, reg_omega):
        reg, omega = reg_omega
        Y = Derivation("Y", 0)
        Z = Derivation("Z", 0)
        rule = NonDegenerateInteriorEqualityDefinition(registry=reg)
        # Sum(Neg(ι_Z ω), ι_Y ω), negated first; rewrite still maps to
        # Y − Z (the positive child names Y, the negated names Z).
        expr = Sum(
            Neg(Act(interior(Z), omega)),
            Act(interior(Y), omega),
        )
        out = rule.rewrite(expr)
        assert out == Sum(Y, Neg(Z))


# --------------------------------------------------------------------- #
# Engine integration                                                    #
# --------------------------------------------------------------------- #


class TestEngineIntegration:
    def test_engine_collapses_when_y_equals_z(self, reg_omega):
        reg, omega = reg_omega
        Y = Derivation("Y", 0)
        engine = ExpansionEngine(
            [NonDegenerateInteriorEqualityDefinition(registry=reg)]
        )
        # Sum(ι_Y ω, Neg(ι_Y ω)), even before the rule fires the Sum
        # may collapse via canonicalization, but the rule firing is
        # the proof step we want to observe.
        expr = Sum(
            Act(interior(Y), omega),
            Neg(Act(interior(Y), omega)),
        )
        out, _steps = engine.expand(expr)
        # After rule fires: Sum(Y, Neg(Y)), the canonical-form
        # pipeline reduces this to 0 via simplify, but engine.expand
        # only fires definitions; check it's at least at the
        # peeled-off shape.
        from jacopy.algorithms.simplify import simplify

        final = simplify(out, reg)
        assert final == Integer(0)

    def test_engine_peels_off_for_distinct_y_z(self, reg_omega):
        reg, omega = reg_omega
        Y = Derivation("Y", 0)
        Z = Derivation("Z", 0)
        engine = ExpansionEngine(
            [NonDegenerateInteriorEqualityDefinition(registry=reg)]
        )
        expr = Sum(
            Act(interior(Y), omega),
            Neg(Act(interior(Z), omega)),
        )
        out, steps = engine.expand(expr)
        # Should rewrite to Sum(Y, Neg(Z)).
        assert out == Sum(Y, Neg(Z))
        assert any(
            "NonDegenerate" in str(s.rule) or "ι_Y ω − ι_Z ω" in str(s.rule)
            for s in steps
        )

    def test_engine_no_op_when_omega_not_nondegenerate(self):
        reg = PropertyRegistry()
        omega = Symbol("ω")
        Y = Derivation("Y", 0)
        Z = Derivation("Z", 0)
        engine = ExpansionEngine(
            [NonDegenerateInteriorEqualityDefinition(registry=reg)]
        )
        expr = Sum(
            Act(interior(Y), omega),
            Neg(Act(interior(Z), omega)),
        )
        out, steps = engine.expand(expr)
        assert out == expr
        assert steps == []
