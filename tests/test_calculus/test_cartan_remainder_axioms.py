"""Tests for Cartan-remainder defining axioms (Faz 15.B)."""

import pytest

from jacopy.algebra.derivation import Act, Derivation
from jacopy.calculus.cartan_remainder import K
from jacopy.calculus.cartan_remainder_axioms import (
    CartanRemainderDefinition,
    TildeCartanRemainderDefinition,
)
from jacopy.calculus.exterior_d import d as default_d
from jacopy.calculus.interior import InteriorProduct, interior as default_interior
from jacopy.calculus.lie_derivative import (
    LieDerivative,
    lie_derivative as default_lie_derivative,
)
from jacopy.calculus.tilde import (
    K_tilde,
    TildeExteriorDerivative,
    TildeInteriorProduct,
    TildeLieDerivative,
    tilde_d,
    tilde_interior,
    tilde_lie,
)
from jacopy.core.expr import Expr, Neg, Sum, Symbol


# --------------------------------------------------------------------- #
# Standard CartanRemainderDefinition                                     #
# --------------------------------------------------------------------- #


class TestCartanRemainderDefinition:
    def test_matches_act_on_K_atom(self):
        rule = CartanRemainderDefinition()
        V, omega = Symbol("V"), Symbol("ω")
        expr = Act(K(V), omega)
        assert rule.matches(expr)

    def test_does_not_match_non_K_act(self):
        rule = CartanRemainderDefinition()
        V, omega = Symbol("V"), Symbol("ω")
        expr = Act(default_lie_derivative(V), omega)
        assert not rule.matches(expr)

    def test_does_not_match_bare_K_atom(self):
        rule = CartanRemainderDefinition()
        V = Symbol("V")
        assert not rule.matches(K(V))

    def test_rewrite_produces_two_term_sum(self):
        rule = CartanRemainderDefinition()
        V, omega = Symbol("V"), Symbol("ω")
        expr = Act(K(V), omega)
        result = rule.rewrite(expr)
        assert isinstance(result, Sum)
        assert len(result.children) == 2

    def test_first_term_is_neg_lie(self):
        rule = CartanRemainderDefinition()
        V, omega = Symbol("V"), Symbol("ω")
        result = rule.rewrite(Act(K(V), omega))
        first = result.children[0]
        assert isinstance(first, Neg)
        inner = first.arg
        assert isinstance(inner, Act)
        assert inner.op == default_lie_derivative(V)
        assert inner.arg is omega

    def test_second_term_is_d_iota(self):
        rule = CartanRemainderDefinition()
        V, omega = Symbol("V"), Symbol("ω")
        result = rule.rewrite(Act(K(V), omega))
        second = result.children[1]
        assert isinstance(second, Act)
        assert second.op is default_d
        inner = second.arg
        assert isinstance(inner, Act)
        assert inner.op == default_interior(V)
        assert inner.arg is omega

    def test_custom_d_lie_interior_overrides_used(self):
        d_custom = default_d
        L_factory = default_lie_derivative
        i_factory = default_interior
        rule = CartanRemainderDefinition(
            d=d_custom, lie_derivative=L_factory, interior=i_factory
        )
        V, omega = Symbol("V"), Symbol("ω")
        result = rule.rewrite(Act(K(V), omega))
        assert isinstance(result, Sum)


# --------------------------------------------------------------------- #
# TildeCartanRemainderDefinition                                         #
# --------------------------------------------------------------------- #


class TestTildeCartanRemainderDefinition:
    def test_matches_act_on_tilde_K_atom(self):
        rule = TildeCartanRemainderDefinition()
        eta, pi, V = Symbol("η"), Symbol("π"), Symbol("V")
        expr = Act(K_tilde(eta, pi), V)
        assert rule.matches(expr)

    def test_does_not_match_standard_K_act(self):
        rule = TildeCartanRemainderDefinition()
        V, omega = Symbol("V"), Symbol("ω")
        assert not rule.matches(Act(K(V), omega))

    def test_does_not_match_non_K_tilde_act(self):
        rule = TildeCartanRemainderDefinition()
        eta, pi, V = Symbol("η"), Symbol("π"), Symbol("V")
        assert not rule.matches(Act(tilde_lie(eta, pi), V))

    def test_rewrite_produces_two_term_sum(self):
        rule = TildeCartanRemainderDefinition()
        eta, pi, V = Symbol("η"), Symbol("π"), Symbol("V")
        result = rule.rewrite(Act(K_tilde(eta, pi), V))
        assert isinstance(result, Sum)
        assert len(result.children) == 2

    def test_first_term_is_neg_tilde_lie(self):
        rule = TildeCartanRemainderDefinition()
        eta, pi, V = Symbol("η"), Symbol("π"), Symbol("V")
        result = rule.rewrite(Act(K_tilde(eta, pi), V))
        first = result.children[0]
        assert isinstance(first, Neg)
        inner = first.arg
        assert isinstance(inner, Act)
        assert inner.op == tilde_lie(eta, pi)
        assert inner.arg is V

    def test_second_term_is_tilde_d_tilde_iota(self):
        rule = TildeCartanRemainderDefinition()
        eta, pi, V = Symbol("η"), Symbol("π"), Symbol("V")
        result = rule.rewrite(Act(K_tilde(eta, pi), V))
        second = result.children[1]
        assert isinstance(second, Act)
        assert second.op == tilde_d(pi)
        inner = second.arg
        assert isinstance(inner, Act)
        assert inner.op == tilde_interior(eta)
        assert inner.arg is V

    def test_pi_independence(self):
        """Two K̃ rules built on distinct π's stay distinct on rewrite."""
        rule = TildeCartanRemainderDefinition()
        eta = Symbol("η")
        pi1, pi2 = Symbol("π_1"), Symbol("π_2")
        V = Symbol("V")
        r1 = rule.rewrite(Act(K_tilde(eta, pi1), V))
        r2 = rule.rewrite(Act(K_tilde(eta, pi2), V))
        # The d̃ heads encode their own π, distinct π's give distinct
        # rewritten subtrees.
        assert r1 != r2


# --------------------------------------------------------------------- #
# Engine integration smoke                                               #
# --------------------------------------------------------------------- #


class TestEngineIntegrationSmoke:
    """Both rules slot into the engine without exception when registered."""

    def test_standard_rule_fires_in_engine(self):
        from jacopy.proof.expansion import ExpansionEngine

        engine = ExpansionEngine([CartanRemainderDefinition()])
        V, omega = Symbol("V"), Symbol("ω")
        expr = Act(K(V), omega)
        rewritten, steps = engine.expand(expr)
        assert len(steps) >= 1
        assert isinstance(rewritten, Sum)

    def test_tilde_rule_fires_in_engine(self):
        from jacopy.proof.expansion import ExpansionEngine

        engine = ExpansionEngine([TildeCartanRemainderDefinition()])
        eta, pi, V = Symbol("η"), Symbol("π"), Symbol("V")
        expr = Act(K_tilde(eta, pi), V)
        rewritten, steps = engine.expand(expr)
        assert len(steps) >= 1
        assert isinstance(rewritten, Sum)

    def test_rules_coexist_without_collision(self):
        """Standard and tilde rules don't fight over the same shape."""
        from jacopy.proof.expansion import ExpansionEngine

        engine = ExpansionEngine([
            CartanRemainderDefinition(),
            TildeCartanRemainderDefinition(),
        ])
        V, omega = Symbol("V"), Symbol("ω")
        eta, pi, W = Symbol("η"), Symbol("π"), Symbol("W")

        std_result, std_steps = engine.expand(Act(K(V), omega))
        tilde_result, tilde_steps = engine.expand(Act(K_tilde(eta, pi), W))
        # Each rule fires only on its own shape.
        assert isinstance(std_result, Sum)
        assert isinstance(tilde_result, Sum)
