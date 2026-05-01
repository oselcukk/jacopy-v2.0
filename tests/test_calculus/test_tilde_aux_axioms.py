"""Tests for tilde-calculus auxiliary axioms (Faz 14.D)."""

import pytest

from jacopy.algebra.derivation import Act, Derivation
from jacopy.calculus.exterior_d import d as default_d
from jacopy.calculus.hamiltonian_vf import HamiltonianVectorField
from jacopy.calculus.musical import Sharp
from jacopy.calculus.tilde import (
    TildeDOfFunctionDefinition,
    TildeDSquaredPoissonDefinition,
    TildeIotaOnZeroVectorDefinition,
    TildeIotaSquaredZeroDefinition,
    TildeLieOnZeroVectorDefinition,
    tilde_d,
    tilde_interior,
    tilde_lie,
)
from jacopy.core.expr import Neg, Symbol, Zero
from jacopy.core.properties import Graded, Poisson
from jacopy.core.registry import PropertyRegistry
from jacopy.proof.expansion import Definition, ExpansionEngine


@pytest.fixture
def reg():
    return PropertyRegistry()


@pytest.fixture
def pi():
    return Symbol("π")


@pytest.fixture
def omega():
    return Symbol("ω")


@pytest.fixture
def f(reg):
    f = Symbol("f")
    reg.declare(f, Graded(degree=0))
    return f


# --------------------------------------------------------------------- #
# Aux-1, TildeIotaOnZeroVectorDefinition                               #
# --------------------------------------------------------------------- #


class TestIotaOnZeroVector:
    def test_is_definition(self, reg):
        rule = TildeIotaOnZeroVectorDefinition(registry=reg)
        assert isinstance(rule, Definition)

    def test_matches_on_zero_vector(self, reg, omega, f):
        rule = TildeIotaOnZeroVectorDefinition(registry=reg)
        assert rule.matches(Act(tilde_interior(omega), f))

    def test_no_match_without_degree(self, reg, omega):
        # V has no Graded property → degree_of fails → no match.
        V = Symbol("V")
        rule = TildeIotaOnZeroVectorDefinition(registry=reg)
        assert not rule.matches(Act(tilde_interior(omega), V))

    def test_no_match_on_higher_degree(self, reg, omega):
        X = Symbol("X")
        reg.declare(X, Graded(degree=1))
        rule = TildeIotaOnZeroVectorDefinition(registry=reg)
        assert not rule.matches(Act(tilde_interior(omega), X))

    def test_no_match_without_registry(self, omega, f):
        # Without registry, degree_of can't resolve f's degree.
        rule = TildeIotaOnZeroVectorDefinition()
        assert not rule.matches(Act(tilde_interior(omega), f))

    def test_no_match_on_plain_act(self, reg, f):
        # A plain Act with a non-tilde operator must not match.
        X = Derivation("X", 0)
        rule = TildeIotaOnZeroVectorDefinition(registry=reg)
        assert not rule.matches(Act(X, f))

    def test_rewrite_returns_zero(self, reg, omega, f):
        rule = TildeIotaOnZeroVectorDefinition(registry=reg)
        out = rule.rewrite(Act(tilde_interior(omega), f))
        assert out is Zero


# --------------------------------------------------------------------- #
# Aux-2, TildeIotaSquaredZeroDefinition                                #
# --------------------------------------------------------------------- #


class TestIotaSquaredZero:
    def test_is_definition(self):
        rule = TildeIotaSquaredZeroDefinition()
        assert isinstance(rule, Definition)

    def test_matches_same_form_squared(self, omega):
        V = Symbol("V")
        rule = TildeIotaSquaredZeroDefinition()
        assert rule.matches(
            Act(tilde_interior(omega), Act(tilde_interior(omega), V))
        )

    def test_no_match_distinct_forms(self, omega):
        eta = Symbol("η")
        V = Symbol("V")
        rule = TildeIotaSquaredZeroDefinition()
        # ι̃_ω(ι̃_η V) is a distinct shape, anti-commute owns it.
        assert not rule.matches(
            Act(tilde_interior(omega), Act(tilde_interior(eta), V))
        )

    def test_no_match_single_iota(self, omega):
        V = Symbol("V")
        rule = TildeIotaSquaredZeroDefinition()
        assert not rule.matches(Act(tilde_interior(omega), V))

    def test_no_match_on_atom(self):
        rule = TildeIotaSquaredZeroDefinition()
        assert not rule.matches(Symbol("ω"))

    def test_rewrite_returns_zero(self, omega):
        V = Symbol("V")
        rule = TildeIotaSquaredZeroDefinition()
        out = rule.rewrite(
            Act(tilde_interior(omega), Act(tilde_interior(omega), V))
        )
        assert out is Zero


# --------------------------------------------------------------------- #
# Aux-3, TildeLieOnZeroVectorDefinition                                #
# --------------------------------------------------------------------- #


class TestLieOnZeroVector:
    def test_is_definition(self, reg, pi):
        rule = TildeLieOnZeroVectorDefinition(pi, registry=reg)
        assert isinstance(rule, Definition)

    def test_carries_pi(self, reg, pi):
        rule = TildeLieOnZeroVectorDefinition(pi, registry=reg)
        assert rule.pi is pi

    def test_rejects_non_expr_pi(self, reg):
        with pytest.raises(TypeError):
            TildeLieOnZeroVectorDefinition("π", registry=reg)  # type: ignore[arg-type]

    def test_matches_on_zero_vector(self, reg, pi, omega, f):
        rule = TildeLieOnZeroVectorDefinition(pi, registry=reg)
        assert rule.matches(Act(tilde_lie(omega, pi), f))

    def test_no_match_for_distinct_pi(self, reg, omega, f):
        pi1, pi2 = Symbol("π1"), Symbol("π2")
        rule = TildeLieOnZeroVectorDefinition(pi1, registry=reg)
        assert not rule.matches(Act(tilde_lie(omega, pi2), f))

    def test_no_match_on_higher_degree(self, reg, pi, omega):
        X = Symbol("X")
        reg.declare(X, Graded(degree=1))
        rule = TildeLieOnZeroVectorDefinition(pi, registry=reg)
        assert not rule.matches(Act(tilde_lie(omega, pi), X))

    def test_rewrite_emits_anchor_action(self, reg, pi, omega, f):
        rule = TildeLieOnZeroVectorDefinition(pi, registry=reg)
        out = rule.rewrite(Act(tilde_lie(omega, pi), f))
        assert out == Act(Act(Sharp(pi), omega), f)


# --------------------------------------------------------------------- #
# Aux-4, TildeDOfFunctionDefinition                                    #
# --------------------------------------------------------------------- #


class TestDOfFunction:
    def test_is_definition(self, reg, pi):
        rule = TildeDOfFunctionDefinition(pi, registry=reg)
        assert isinstance(rule, Definition)

    def test_carries_pi_and_d(self, reg, pi):
        rule = TildeDOfFunctionDefinition(pi, registry=reg)
        assert rule.pi is pi
        assert rule.d is default_d

    def test_rejects_non_expr_pi(self, reg):
        with pytest.raises(TypeError):
            TildeDOfFunctionDefinition("π", registry=reg)  # type: ignore[arg-type]

    def test_matches_on_zero_vector(self, reg, pi, f):
        rule = TildeDOfFunctionDefinition(pi, registry=reg)
        assert rule.matches(Act(tilde_d(pi), f))

    def test_no_match_for_distinct_pi(self, reg, f):
        pi1, pi2 = Symbol("π1"), Symbol("π2")
        rule = TildeDOfFunctionDefinition(pi1, registry=reg)
        assert not rule.matches(Act(tilde_d(pi2), f))

    def test_no_match_on_higher_degree(self, reg, pi):
        X = Symbol("X")
        reg.declare(X, Graded(degree=1))
        rule = TildeDOfFunctionDefinition(pi, registry=reg)
        assert not rule.matches(Act(tilde_d(pi), X))

    def test_rewrite_emits_neg_sharp_d(self, reg, pi, f):
        rule = TildeDOfFunctionDefinition(pi, registry=reg)
        out = rule.rewrite(Act(tilde_d(pi), f))
        assert out == Neg(Act(Sharp(pi), Act(default_d, f)))


# --------------------------------------------------------------------- #
# Aux-5, TildeDSquaredPoissonDefinition                                #
# --------------------------------------------------------------------- #


class TestDSquaredPoisson:
    def test_is_definition(self, reg, pi):
        rule = TildeDSquaredPoissonDefinition(pi, registry=reg)
        assert isinstance(rule, Definition)

    def test_rejects_non_expr_pi(self, reg):
        with pytest.raises(TypeError):
            TildeDSquaredPoissonDefinition("π", registry=reg)  # type: ignore[arg-type]

    def test_rejects_missing_registry(self, pi):
        with pytest.raises(TypeError, match="PropertyRegistry"):
            TildeDSquaredPoissonDefinition(pi, registry=None)  # type: ignore[arg-type]

    def test_no_match_without_poisson_flag(self, reg, pi):
        V = Symbol("V")
        rule = TildeDSquaredPoissonDefinition(pi, registry=reg)
        assert not rule.matches(Act(tilde_d(pi), Act(tilde_d(pi), V)))

    def test_matches_after_poisson_declared(self, reg, pi):
        V = Symbol("V")
        rule = TildeDSquaredPoissonDefinition(pi, registry=reg)
        reg.declare(pi, Poisson())
        assert rule.matches(Act(tilde_d(pi), Act(tilde_d(pi), V)))

    def test_no_match_on_single_d(self, reg, pi):
        V = Symbol("V")
        reg.declare(pi, Poisson())
        rule = TildeDSquaredPoissonDefinition(pi, registry=reg)
        assert not rule.matches(Act(tilde_d(pi), V))

    def test_no_match_for_distinct_pi(self, reg):
        pi1, pi2 = Symbol("π1"), Symbol("π2")
        V = Symbol("V")
        reg.declare(pi1, Poisson())
        reg.declare(pi2, Poisson())
        rule = TildeDSquaredPoissonDefinition(pi1, registry=reg)
        # Outer head is for pi2, rule scoped to pi1 must not match.
        assert not rule.matches(Act(tilde_d(pi2), Act(tilde_d(pi2), V)))

    def test_rewrite_returns_zero(self, reg, pi):
        V = Symbol("V")
        reg.declare(pi, Poisson())
        rule = TildeDSquaredPoissonDefinition(pi, registry=reg)
        out = rule.rewrite(Act(tilde_d(pi), Act(tilde_d(pi), V)))
        assert out is Zero


# --------------------------------------------------------------------- #
# Engine integration, the four "single-shape" auxiliaries that fire    #
# under the engine's leftmost-innermost traversal.                       #
# --------------------------------------------------------------------- #


class TestEngineIntegration:
    def test_iota0_fires_alone(self, reg, omega, f):
        engine = ExpansionEngine(
            [TildeIotaOnZeroVectorDefinition(registry=reg)]
        )
        out, steps = engine.expand(Act(tilde_interior(omega), f))
        assert out is Zero
        assert len(steps) == 1

    def test_lie0_fires_alone(self, reg, pi, omega, f):
        engine = ExpansionEngine(
            [TildeLieOnZeroVectorDefinition(pi, registry=reg)]
        )
        out, steps = engine.expand(Act(tilde_lie(omega, pi), f))
        assert out == Act(Act(Sharp(pi), omega), f)
        assert len(steps) == 1

    def test_d_func_fires_alone(self, reg, pi, f):
        engine = ExpansionEngine(
            [TildeDOfFunctionDefinition(pi, registry=reg)]
        )
        out, steps = engine.expand(Act(tilde_d(pi), f))
        assert out == Neg(Act(Sharp(pi), Act(default_d, f)))
        assert len(steps) == 1


# --------------------------------------------------------------------- #
# Engine traversal limitation note, "double-shape" Aux-2 and Aux-5     #
# can't fire under the engine's leftmost-innermost traversal because    #
# the inner subterm is rewritten by a defining axiom (swap or           #
# Lichnerowicz) before the parent rule sees the original shape.         #
# These rules are still semantically correct: they fire when applied    #
# directly to the matching shape (verified above). Stage E (proof       #
# closure) will either rely on direct rule application or extend the    #
# engine with a parent-first traversal.                                 #
# --------------------------------------------------------------------- #


class TestDoubleShapeIsolatedOnly:
    def test_iota_squared_applies_directly(self, omega):
        V = Symbol("V")
        rule = TildeIotaSquaredZeroDefinition()
        target = Act(tilde_interior(omega), Act(tilde_interior(omega), V))
        # Rule fires when invoked directly on the matching shape.
        assert rule.matches(target)
        assert rule.rewrite(target) is Zero

    def test_d_squared_applies_directly_when_poisson(self, reg, pi):
        V = Symbol("V")
        reg.declare(pi, Poisson())
        rule = TildeDSquaredPoissonDefinition(pi, registry=reg)
        target = Act(tilde_d(pi), Act(tilde_d(pi), V))
        assert rule.matches(target)
        assert rule.rewrite(target) is Zero
