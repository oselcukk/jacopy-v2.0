"""Tests for tilde-calculus defining axioms (Faz 14.B)."""

import pytest

from jacopy.algebra.derivation import Act, Derivation
from jacopy.brackets.base import BracketApply
from jacopy.brackets.schouten import sn
from jacopy.calculus.interior import InteriorProduct
from jacopy.calculus.tilde import (
    TildeExteriorDLichnerowiczDefinition,
    TildeExteriorDerivative,
    TildeInteriorProduct,
    TildeIotaSwapDefinition,
    TildeLieDerivative,
    TildeLieMagicDefinition,
    tilde_d,
    tilde_interior,
    tilde_lie,
)
from jacopy.core.expr import Symbol, Sum
from jacopy.proof.expansion import Definition, ExpansionEngine


# --------------------------------------------------------------------- #
# TildeIotaSwapDefinition                                               #
# --------------------------------------------------------------------- #


class TestIotaSwap:
    def test_is_definition(self):
        rule = TildeIotaSwapDefinition()
        assert isinstance(rule, Definition)

    def test_matches_act_with_tilde_interior(self):
        omega = Symbol("ω")
        V = Symbol("V")
        rule = TildeIotaSwapDefinition()
        assert rule.matches(Act(tilde_interior(omega), V))

    def test_no_match_on_plain_act(self):
        X = Symbol("X")
        omega = Symbol("ω")
        rule = TildeIotaSwapDefinition()
        # Standard ι_X(ω), not the tilde shape.
        assert not rule.matches(Act(InteriorProduct(X), omega))

    def test_no_match_on_atom(self):
        rule = TildeIotaSwapDefinition()
        assert not rule.matches(Symbol("ω"))

    def test_rewrite_swaps_roles(self):
        omega = Symbol("ω")
        V = Symbol("V")
        rule = TildeIotaSwapDefinition()
        out = rule.rewrite(Act(tilde_interior(omega), V))
        assert out == Act(InteriorProduct(V), omega)

    def test_rewrite_with_derivation_vector(self):
        omega = Symbol("ω")
        X = Derivation("X", 0)
        rule = TildeIotaSwapDefinition()
        out = rule.rewrite(Act(tilde_interior(omega), X))
        assert out == Act(InteriorProduct(X), omega)

    def test_rewrite_preserves_form_identity(self):
        omega = Symbol("ω")
        eta = Symbol("η")
        V = Symbol("V")
        rule = TildeIotaSwapDefinition()
        # Two distinct forms produce two distinct results.
        out1 = rule.rewrite(Act(tilde_interior(omega), V))
        out2 = rule.rewrite(Act(tilde_interior(eta), V))
        assert out1 != out2
        assert out1 == Act(InteriorProduct(V), omega)
        assert out2 == Act(InteriorProduct(V), eta)


# --------------------------------------------------------------------- #
# TildeExteriorDLichnerowiczDefinition                                  #
# --------------------------------------------------------------------- #


class TestExteriorDLichnerowicz:
    def test_is_definition(self):
        pi = Symbol("π")
        rule = TildeExteriorDLichnerowiczDefinition(pi)
        assert isinstance(rule, Definition)

    def test_carries_pi(self):
        pi = Symbol("π")
        rule = TildeExteriorDLichnerowiczDefinition(pi)
        assert rule.pi is pi

    def test_rejects_non_expr_pi(self):
        with pytest.raises(TypeError):
            TildeExteriorDLichnerowiczDefinition("π")  # type: ignore[arg-type]

    def test_matches_for_correct_pi(self):
        pi = Symbol("π")
        V = Symbol("V")
        rule = TildeExteriorDLichnerowiczDefinition(pi)
        assert rule.matches(Act(tilde_d(pi), V))

    def test_no_match_for_distinct_pi(self):
        pi1 = Symbol("π1")
        pi2 = Symbol("π2")
        V = Symbol("V")
        rule = TildeExteriorDLichnerowiczDefinition(pi1)
        # Same V, but tilde-d for a different bivector, must not match.
        assert not rule.matches(Act(tilde_d(pi2), V))

    def test_no_match_on_standard_d(self):
        from jacopy.calculus.exterior_d import d as default_d

        pi = Symbol("π")
        f = Symbol("f")
        rule = TildeExteriorDLichnerowiczDefinition(pi)
        assert not rule.matches(Act(default_d, f))

    def test_no_match_on_atom(self):
        pi = Symbol("π")
        rule = TildeExteriorDLichnerowiczDefinition(pi)
        assert not rule.matches(Symbol("V"))

    def test_rewrite_emits_sn_bracket_apply(self):
        pi = Symbol("π")
        V = Symbol("V")
        rule = TildeExteriorDLichnerowiczDefinition(pi)
        out = rule.rewrite(Act(tilde_d(pi), V))
        assert isinstance(out, BracketApply)
        assert out.bracket is sn
        assert out.a is pi
        assert out.b is V


# --------------------------------------------------------------------- #
# TildeLieMagicDefinition                                               #
# --------------------------------------------------------------------- #


class TestLieMagic:
    def test_is_definition(self):
        pi = Symbol("π")
        rule = TildeLieMagicDefinition(pi)
        assert isinstance(rule, Definition)

    def test_carries_pi(self):
        pi = Symbol("π")
        rule = TildeLieMagicDefinition(pi)
        assert rule.pi is pi

    def test_rejects_non_expr_pi(self):
        with pytest.raises(TypeError):
            TildeLieMagicDefinition("π")  # type: ignore[arg-type]

    def test_matches_for_correct_pi(self):
        omega, pi = Symbol("ω"), Symbol("π")
        V = Symbol("V")
        rule = TildeLieMagicDefinition(pi)
        assert rule.matches(Act(tilde_lie(omega, pi), V))

    def test_no_match_for_distinct_pi(self):
        omega = Symbol("ω")
        pi1 = Symbol("π1")
        pi2 = Symbol("π2")
        V = Symbol("V")
        rule = TildeLieMagicDefinition(pi1)
        # Same form, but L̃ for a different bivector, must not match.
        assert not rule.matches(Act(tilde_lie(omega, pi2), V))

    def test_no_match_on_standard_lie(self):
        from jacopy.calculus.lie_derivative import lie_derivative

        pi = Symbol("π")
        X = Derivation("X", 0)
        omega = Symbol("ω")
        rule = TildeLieMagicDefinition(pi)
        assert not rule.matches(Act(lie_derivative(X), omega))

    def test_rewrite_emits_cartan_magic_sum(self):
        omega, pi = Symbol("ω"), Symbol("π")
        V = Symbol("V")
        rule = TildeLieMagicDefinition(pi)
        out = rule.rewrite(Act(tilde_lie(omega, pi), V))
        d_t = TildeExteriorDerivative(pi)
        iota_t = TildeInteriorProduct(omega)
        expected = Sum(
            Act(d_t, Act(iota_t, V)),
            Act(iota_t, Act(d_t, V)),
        )
        assert out == expected


# --------------------------------------------------------------------- #
# Engine integration                                                    #
# --------------------------------------------------------------------- #


class TestEngineIntegration:
    def test_swap_fires_on_engine(self):
        omega = Symbol("ω")
        V = Symbol("V")
        engine = ExpansionEngine([TildeIotaSwapDefinition()])
        out, steps = engine.expand(Act(tilde_interior(omega), V))
        assert out == Act(InteriorProduct(V), omega)
        assert len(steps) == 1

    def test_lichnerowicz_fires_on_engine(self):
        pi = Symbol("π")
        V = Symbol("V")
        engine = ExpansionEngine([TildeExteriorDLichnerowiczDefinition(pi)])
        out, steps = engine.expand(Act(tilde_d(pi), V))
        assert out == BracketApply(sn, pi, V)
        assert len(steps) == 1

    def test_magic_then_swap_then_lichnerowicz_unfolds(self):
        """Full chain: L̃_ω X → magic → swap + Lichnerowicz."""
        omega, pi = Symbol("ω"), Symbol("π")
        X = Derivation("X", 0)
        engine = ExpansionEngine(
            [
                TildeIotaSwapDefinition(),
                TildeExteriorDLichnerowiczDefinition(pi),
                TildeLieMagicDefinition(pi),
            ]
        )
        out, steps = engine.expand(Act(tilde_lie(omega, pi), X))
        # After the fix-point: both sides of the magic Sum have been
        # rewritten, neither tilde-d nor tilde-iota survives.
        assert "L̃_ω V" in steps[0].rule
        assert any("ι̃_ω V" in s.rule for s in steps)
        assert any("d̃ V" in s.rule for s in steps)
        # Final form is a Sum of two SN/iota-mixed terms; the outer
        # heads are no longer tilde operators.
        assert isinstance(out, Sum)
        for child in out.children:
            assert "ι̃" not in str(child)
            assert "d̃" not in str(child)

    def test_two_tildes_with_distinct_pi_do_not_alias(self):
        omega = Symbol("ω")
        pi1, pi2 = Symbol("π1"), Symbol("π2")
        V = Symbol("V")
        # Engine carries d̃ rules for both π1 and π2.
        engine = ExpansionEngine(
            [
                TildeExteriorDLichnerowiczDefinition(pi1),
                TildeExteriorDLichnerowiczDefinition(pi2),
            ]
        )
        out1, steps1 = engine.expand(Act(tilde_d(pi1), V))
        out2, steps2 = engine.expand(Act(tilde_d(pi2), V))
        assert out1 == BracketApply(sn, pi1, V)
        assert out2 == BracketApply(sn, pi2, V)
        # Each fired exactly once, and from the correct rule.
        assert len(steps1) == 1 and len(steps2) == 1
        assert pi1._repr_inner() in steps1[0].rule
        assert pi2._repr_inner() in steps2[0].rule
