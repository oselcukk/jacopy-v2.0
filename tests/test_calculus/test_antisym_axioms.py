"""Tests for registry-driven antisymmetry (Faz 12.B #11)."""

import pytest

from jacopy.calculus.antisym_axioms import RegistryAntiSymCanonicalDefinition
from jacopy.core.expr import Neg, Symbol
from jacopy.core.multi_eval import multi_eval
from jacopy.core.properties import Antisymmetric
from jacopy.core.registry import PropertyRegistry
from jacopy.proof.expansion import ExpansionEngine


# --------------------------------------------------------------------- #
# Match logic                                                            #
# --------------------------------------------------------------------- #


class TestRegistryAntiSymMatches:
    def test_matches_unsorted_args(self):
        reg = PropertyRegistry()
        pi = Symbol("π")
        reg.declare(pi, Antisymmetric())
        rule = RegistryAntiSymCanonicalDefinition(registry=reg)
        a, b = Symbol("a"), Symbol("b")
        # b > a in repr → unsorted when args are (b, a).
        expr = multi_eval(pi, b, a, alternating=False, slot_kind="covector")
        assert rule.matches(expr)

    def test_no_match_sorted_args(self):
        reg = PropertyRegistry()
        pi = Symbol("π")
        reg.declare(pi, Antisymmetric())
        rule = RegistryAntiSymCanonicalDefinition(registry=reg)
        a, b = Symbol("a"), Symbol("b")
        expr = multi_eval(pi, a, b, alternating=False, slot_kind="covector")
        assert not rule.matches(expr)

    def test_no_match_when_head_not_declared(self):
        reg = PropertyRegistry()
        rule = RegistryAntiSymCanonicalDefinition(registry=reg)
        eta = Symbol("η")  # not declared
        a, b = Symbol("a"), Symbol("b")
        expr = multi_eval(eta, b, a, alternating=False)
        assert not rule.matches(expr)

    def test_no_match_when_registry_none(self):
        rule = RegistryAntiSymCanonicalDefinition()
        pi = Symbol("π")
        a, b = Symbol("a"), Symbol("b")
        expr = multi_eval(pi, b, a, alternating=False)
        assert not rule.matches(expr)

    def test_no_match_arity_three(self):
        # Antisymmetric is binary; arity-3 should be left to the
        # alternating=True flag + MultiEvalAlternatingNormalDefinition.
        reg = PropertyRegistry()
        pi = Symbol("π")
        reg.declare(pi, Antisymmetric())
        rule = RegistryAntiSymCanonicalDefinition(registry=reg)
        a, b, c = Symbol("a"), Symbol("b"), Symbol("c")
        expr = multi_eval(pi, c, b, a, alternating=False)
        assert not rule.matches(expr)

    def test_no_match_non_multi_eval(self):
        reg = PropertyRegistry()
        pi = Symbol("π")
        reg.declare(pi, Antisymmetric())
        rule = RegistryAntiSymCanonicalDefinition(registry=reg)
        # A bare Symbol can't carry slots; rule shouldn't fire.
        assert not rule.matches(pi)


# --------------------------------------------------------------------- #
# Rewrite                                                                #
# --------------------------------------------------------------------- #


class TestRegistryAntiSymRewrite:
    def test_rewrites_to_neg_swap(self):
        reg = PropertyRegistry()
        pi = Symbol("π")
        reg.declare(pi, Antisymmetric())
        rule = RegistryAntiSymCanonicalDefinition(registry=reg)
        a, b = Symbol("a"), Symbol("b")
        expr = multi_eval(pi, b, a, alternating=False, slot_kind="covector")
        out = rule.rewrite(expr)
        expected = Neg(
            multi_eval(pi, a, b, alternating=False, slot_kind="covector")
        )
        assert out == expected

    def test_preserves_alternating_and_slot_kind(self):
        reg = PropertyRegistry()
        pi = Symbol("π")
        reg.declare(pi, Antisymmetric())
        rule = RegistryAntiSymCanonicalDefinition(registry=reg)
        a, b = Symbol("a"), Symbol("b")
        # Even with alternating=True the rule still rewrites, it
        # produces the same canonical form as the alt-flag rule.
        expr = multi_eval(pi, b, a, alternating=True, slot_kind="vector")
        out = rule.rewrite(expr)
        inner = out.arg
        assert inner.alternating is True
        assert inner.slot_kind == "vector"


# --------------------------------------------------------------------- #
# Engine integration                                                     #
# --------------------------------------------------------------------- #


class TestRegistryAntiSymEngineIntegration:
    def test_engine_canonicalizes(self):
        reg = PropertyRegistry()
        pi = Symbol("π")
        reg.declare(pi, Antisymmetric())
        engine = ExpansionEngine(
            [RegistryAntiSymCanonicalDefinition(registry=reg)]
        )
        a, b = Symbol("a"), Symbol("b")
        expr = multi_eval(pi, b, a, alternating=False, slot_kind="covector")
        out, steps = engine.expand(expr)
        assert out == Neg(
            multi_eval(pi, a, b, alternating=False, slot_kind="covector")
        )
        assert len(steps) == 1
        assert steps[0].rule == "Antisymmetric (registry): π(α, β) = -π(β, α)"

    def test_engine_terminates_on_sorted(self):
        reg = PropertyRegistry()
        pi = Symbol("π")
        reg.declare(pi, Antisymmetric())
        engine = ExpansionEngine(
            [RegistryAntiSymCanonicalDefinition(registry=reg)]
        )
        a, b = Symbol("a"), Symbol("b")
        expr = multi_eval(pi, a, b, alternating=False)
        out, steps = engine.expand(expr)
        assert out == expr
        assert steps == []


# --------------------------------------------------------------------- #
# Bivector helper auto-declares Antisymmetric                            #
# --------------------------------------------------------------------- #


class TestBivectorHelperAutoDeclaresAntisymmetric:
    def test_bivector_helper_declares_antisymmetric(self):
        from jacopy.library.declarations import Bivector

        reg = PropertyRegistry()
        pi = Bivector("π", registry=reg)
        assert reg.has(pi, Antisymmetric)

    def test_bivector_helper_still_declares_graded(self):
        from jacopy.core.properties import Graded
        from jacopy.library.declarations import Bivector

        reg = PropertyRegistry()
        pi = Bivector("π", registry=reg)
        # Still SN-graded (degree=1 for 2-vector).
        graded = reg.get(pi, Graded)
        assert graded is not None
