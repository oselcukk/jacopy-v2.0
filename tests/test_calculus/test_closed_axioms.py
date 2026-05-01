"""Tests for the Closed-form axiom (Faz 12.B #7)."""

import pytest

from jacopy.algebra.derivation import Act, Derivation
from jacopy.calculus.closed_axioms import ClosedFormDefinition
from jacopy.calculus.exterior_d import d as default_d
from jacopy.calculus.interior import interior
from jacopy.calculus.lie_derivative import lie_derivative
from jacopy.core.expr import Integer, Sum, Symbol
from jacopy.core.properties import Closed, Provenance, ProofRef
from jacopy.core.registry import PropertyRegistry
from jacopy.proof.expansion import ExpansionEngine


# --------------------------------------------------------------------- #
# Closed property                                                        #
# --------------------------------------------------------------------- #


class TestClosedProperty:
    def test_construction_axiom_default(self):
        prop = Closed()
        assert prop.is_axiom
        assert prop.proof is None

    def test_derived_requires_proof_ref(self):
        ref = ProofRef(rule="d²=0 chain", sources=("ExteriorDerivative",))
        prop = Closed(provenance=Provenance.DERIVED, proof=ref)
        assert prop.is_derived
        assert prop.proof is ref

    def test_derived_without_proof_raises(self):
        with pytest.raises(ValueError):
            Closed(provenance=Provenance.DERIVED)

    def test_axiom_with_proof_ref_raises(self):
        ref = ProofRef(rule="x", sources=())
        with pytest.raises(ValueError):
            Closed(provenance=Provenance.AXIOM, proof=ref)

    def test_registry_roundtrip(self):
        reg = PropertyRegistry()
        omega = Symbol("ω")
        reg.declare(omega, Closed())
        assert reg.has(omega, Closed)
        eta = Symbol("η")
        assert not reg.has(eta, Closed)


# --------------------------------------------------------------------- #
# ClosedFormDefinition, match logic                                     #
# --------------------------------------------------------------------- #


class TestClosedFormDefinitionMatches:
    def test_matches_d_omega_when_closed(self):
        reg = PropertyRegistry()
        omega = Symbol("ω")
        reg.declare(omega, Closed())
        rule = ClosedFormDefinition(registry=reg)
        assert rule.matches(Act(default_d, omega))

    def test_no_match_when_not_declared(self):
        reg = PropertyRegistry()
        rule = ClosedFormDefinition(registry=reg)
        eta = Symbol("η")
        assert not rule.matches(Act(default_d, eta))

    def test_no_match_when_registry_none(self):
        omega = Symbol("ω")
        rule = ClosedFormDefinition()
        # Even if some other registry declared it, this rule has no view.
        assert not rule.matches(Act(default_d, omega))

    def test_no_match_non_d_outer(self):
        reg = PropertyRegistry()
        omega = Symbol("ω")
        reg.declare(omega, Closed())
        rule = ClosedFormDefinition(registry=reg)
        X = Derivation("X", 0)
        # ι_X(ω), interior product, not d
        assert not rule.matches(Act(interior(X), omega))
        # L_X(ω), Lie derivative
        assert not rule.matches(Act(lie_derivative(X), omega))

    def test_no_match_when_only_subterm_closed(self):
        # Closed declared on ω, but the expression is d(ω + η) with η
        # not declared. Equality checks the *whole* arg, so a Sum
        # containing a closed form doesn't itself look closed,
        # downstream linearity rules can split it first.
        reg = PropertyRegistry()
        omega = Symbol("ω")
        eta = Symbol("η")
        reg.declare(omega, Closed())
        rule = ClosedFormDefinition(registry=reg)
        assert not rule.matches(Act(default_d, Sum(omega, eta)))


class TestClosedFormDefinitionRewrite:
    def test_rewrites_to_integer_zero(self):
        reg = PropertyRegistry()
        omega = Symbol("ω")
        reg.declare(omega, Closed())
        rule = ClosedFormDefinition(registry=reg)
        out = rule.rewrite(Act(default_d, omega))
        assert out == Integer(0)


# --------------------------------------------------------------------- #
# Engine integration                                                     #
# --------------------------------------------------------------------- #


class TestClosedFormEngineIntegration:
    def test_engine_closes_d_omega(self):
        reg = PropertyRegistry()
        omega = Symbol("ω")
        reg.declare(omega, Closed())
        engine = ExpansionEngine([ClosedFormDefinition(registry=reg)])
        out, steps = engine.expand(Act(default_d, omega))
        assert out == Integer(0)
        assert len(steps) == 1
        assert steps[0].rule == "Closed: d(ω) = 0 when ω is declared closed"

    def test_engine_closes_after_linearity_split(self):
        # d(ω + η) where ω is closed, η is closed → d-linearity splits,
        # then both children collapse to 0.
        from jacopy.calculus.linearity_axioms import (
            ExteriorDerivativeLinearityDefinition,
        )

        reg = PropertyRegistry()
        omega = Symbol("ω")
        eta = Symbol("η")
        reg.declare(omega, Closed())
        reg.declare(eta, Closed())
        engine = ExpansionEngine(
            [
                ExteriorDerivativeLinearityDefinition(),
                ClosedFormDefinition(registry=reg),
            ]
        )
        out, steps = engine.expand(Act(default_d, Sum(omega, eta)))
        # After linearity split + two closed-collapse passes, the Sum of
        # two zeros should reduce; engine.expand may not run a final
        # ``simplify``, but the children must each become Integer(0).
        # Reach final-form by running simplify.
        from jacopy.algorithms.simplify import simplify

        final = simplify(out, reg)
        assert final == Integer(0)

    def test_engine_no_op_when_arg_not_closed(self):
        reg = PropertyRegistry()
        eta = Symbol("η")  # not declared
        engine = ExpansionEngine([ClosedFormDefinition(registry=reg)])
        expr = Act(default_d, eta)
        out, steps = engine.expand(expr)
        assert out == expr
        assert steps == []
