"""Tests for Faz 13.E, function-level Poisson axioms."""

import pytest

from jacopy.algebra.derivation import Act
from jacopy.brackets.base import BracketApply
from jacopy.brackets.derived import DerivedBracket
from jacopy.brackets.schouten import sn as default_sn
from jacopy.calculus.hamiltonian_vf import hamiltonian_vf
from jacopy.calculus.poisson_axioms import (
    HamiltonianCyclicSnFormulaDefinition,
    PoissonAsHamiltonianDefinition,
)
from jacopy.core.expr import Integer, Neg, Sum, Symbol
from jacopy.library.poisson import PoissonBracket
from jacopy.proof.expansion import ExpansionEngine


# --------------------------------------------------------------------- #
# Axiom 2g-1, PoissonAsHamiltonianDefinition                            #
# --------------------------------------------------------------------- #


def _ham(f, pi):
    return hamiltonian_vf(f, bivector=pi)


class TestPoissonAsHamiltonianConstruction:
    def test_requires_derived_bracket(self):
        with pytest.raises(TypeError):
            PoissonAsHamiltonianDefinition("not a bracket")  # type: ignore[arg-type]

    def test_default_bivector_from_bracket(self):
        pi = Symbol("π")
        P = PoissonBracket.from_bivector(pi).derived
        rule = PoissonAsHamiltonianDefinition(P)
        assert rule._bivector == pi

    def test_explicit_bivector_override(self):
        pi = Symbol("π")
        pi_alt = Symbol("π'")
        P = PoissonBracket.from_bivector(pi).derived
        rule = PoissonAsHamiltonianDefinition(P, bivector=pi_alt)
        assert rule._bivector == pi_alt

    def test_name_includes_bracket(self):
        pi = Symbol("π")
        P = PoissonBracket.from_bivector(pi).derived
        rule = PoissonAsHamiltonianDefinition(P)
        assert P.name in rule.name


class TestPoissonAsHamiltonianMatches:
    def test_matches_pinned_bracket(self):
        pi = Symbol("π")
        f, g = Symbol("f"), Symbol("g")
        P = PoissonBracket.from_bivector(pi).derived
        rule = PoissonAsHamiltonianDefinition(P)
        assert rule.matches(BracketApply(P, f, g))

    def test_no_match_on_unrelated_bracket(self):
        pi = Symbol("π")
        pi2 = Symbol("π2")
        f, g = Symbol("f"), Symbol("g")
        P1 = PoissonBracket.from_bivector(pi).derived
        P2 = PoissonBracket.from_bivector(pi2).derived
        rule = PoissonAsHamiltonianDefinition(P1)
        assert not rule.matches(BracketApply(P2, f, g))

    def test_no_match_on_atomic(self):
        pi = Symbol("π")
        P = PoissonBracket.from_bivector(pi).derived
        rule = PoissonAsHamiltonianDefinition(P)
        assert not rule.matches(Symbol("ω"))


class TestPoissonAsHamiltonianRewrite:
    def test_rewrite_to_hamiltonian_act(self):
        pi = Symbol("π")
        f, g = Symbol("f"), Symbol("g")
        P = PoissonBracket.from_bivector(pi).derived
        rule = PoissonAsHamiltonianDefinition(P)
        out = rule.rewrite(BracketApply(P, f, g))
        assert out == Act(_ham(f, pi), g)

    def test_rewrite_uses_explicit_bivector(self):
        pi = Symbol("π")
        pi_alt = Symbol("π_alt")
        f, g = Symbol("f"), Symbol("g")
        P = PoissonBracket.from_bivector(pi).derived
        rule = PoissonAsHamiltonianDefinition(P, bivector=pi_alt)
        out = rule.rewrite(BracketApply(P, f, g))
        assert out == Act(_ham(f, pi_alt), g)


class TestPoissonAsHamiltonianEngine:
    def test_engine_collapses_nested_bracket(self):
        pi = Symbol("π")
        f, g, h = Symbol("f"), Symbol("g"), Symbol("h")
        P = PoissonBracket.from_bivector(pi).derived
        engine = ExpansionEngine([PoissonAsHamiltonianDefinition(P)])
        nested = BracketApply(P, f, BracketApply(P, g, h))
        result, _ = engine.expand(nested)
        # Both inner and outer collapse to Hamiltonian acts.
        assert result == Act(_ham(f, pi), Act(_ham(g, pi), h))


# --------------------------------------------------------------------- #
# Axiom 2g-2, HamiltonianCyclicSnFormulaDefinition                      #
# --------------------------------------------------------------------- #


def _ham_term(pi, a, b, c):
    """Build ``Act(X_a, Act(X_b, c))``."""
    return Act(_ham(a, pi), Act(_ham(b, pi), c))


class TestHamiltonianCyclicSnConstruction:
    def test_requires_expr_bivector(self):
        with pytest.raises(TypeError):
            HamiltonianCyclicSnFormulaDefinition("not an expr")  # type: ignore[arg-type]

    def test_default_sn_bracket(self):
        pi = Symbol("π")
        rule = HamiltonianCyclicSnFormulaDefinition(pi)
        assert rule._sn is default_sn

    def test_custom_sn_bracket(self):
        from jacopy.brackets.schouten import SchoutenBracket

        pi = Symbol("π")
        custom = SchoutenBracket(name="[·,·]_SN'")
        rule = HamiltonianCyclicSnFormulaDefinition(pi, sn_bracket=custom)
        assert rule._sn is custom

    def test_name(self):
        pi = Symbol("π")
        rule = HamiltonianCyclicSnFormulaDefinition(pi)
        assert rule.name == "[π,π]_SN function-level formula"


class TestHamiltonianCyclicSnMatches:
    def test_matches_canonical_cyclic_triple(self):
        pi = Symbol("π")
        f, g, h = Symbol("f"), Symbol("g"), Symbol("h")
        rule = HamiltonianCyclicSnFormulaDefinition(pi)
        s = Sum(
            _ham_term(pi, f, g, h),
            _ham_term(pi, g, h, f),
            _ham_term(pi, h, f, g),
        )
        assert rule.matches(s)

    def test_matches_negated_cyclic_triple(self):
        # Koszul-signed Jacobi obstruction on functions of shifted
        # SN-degree ``-1`` enters the chain with Neg-wrapped terms.
        pi = Symbol("π")
        f, g, h = Symbol("f"), Symbol("g"), Symbol("h")
        rule = HamiltonianCyclicSnFormulaDefinition(pi)
        s = Sum(
            Neg(_ham_term(pi, f, g, h)),
            Neg(_ham_term(pi, g, h, f)),
            Neg(_ham_term(pi, h, f, g)),
        )
        assert rule.matches(s)

    def test_matches_reversed_cycle(self):
        pi = Symbol("π")
        f, g, h = Symbol("f"), Symbol("g"), Symbol("h")
        rule = HamiltonianCyclicSnFormulaDefinition(pi)
        s = Sum(
            _ham_term(pi, f, h, g),
            _ham_term(pi, h, g, f),
            _ham_term(pi, g, f, h),
        )
        assert rule.matches(s)

    def test_no_match_on_two_term_subset(self):
        pi = Symbol("π")
        f, g, h = Symbol("f"), Symbol("g"), Symbol("h")
        rule = HamiltonianCyclicSnFormulaDefinition(pi)
        s = Sum(
            _ham_term(pi, f, g, h),
            _ham_term(pi, g, h, f),
        )
        assert not rule.matches(s)

    def test_no_match_on_non_cyclic_triple(self):
        pi = Symbol("π")
        f, g, h = Symbol("f"), Symbol("g"), Symbol("h")
        rule = HamiltonianCyclicSnFormulaDefinition(pi)
        s = Sum(
            _ham_term(pi, f, g, h),
            _ham_term(pi, f, g, h),
            _ham_term(pi, g, h, f),
        )
        assert not rule.matches(s)

    def test_no_match_on_mixed_polarity(self):
        # One Neg-wrapped + two positive, polarity mismatch.
        pi = Symbol("π")
        f, g, h = Symbol("f"), Symbol("g"), Symbol("h")
        rule = HamiltonianCyclicSnFormulaDefinition(pi)
        s = Sum(
            Neg(_ham_term(pi, f, g, h)),
            _ham_term(pi, g, h, f),
            _ham_term(pi, h, f, g),
        )
        assert not rule.matches(s)

    def test_no_match_with_different_bivector(self):
        pi1, pi2 = Symbol("π1"), Symbol("π2")
        f, g, h = Symbol("f"), Symbol("g"), Symbol("h")
        rule = HamiltonianCyclicSnFormulaDefinition(pi1)
        # Hamiltonians built over a different bivector.
        s = Sum(
            _ham_term(pi2, f, g, h),
            _ham_term(pi2, g, h, f),
            _ham_term(pi2, h, f, g),
        )
        assert not rule.matches(s)

    def test_no_match_when_inner_not_hamiltonian(self):
        pi = Symbol("π")
        f, g, h = Symbol("f"), Symbol("g"), Symbol("h")
        rule = HamiltonianCyclicSnFormulaDefinition(pi)
        # Bare-symbol inner, not Act(X_b, c).
        bad = Act(_ham(f, pi), Act(Symbol("Y"), h))
        s = Sum(
            bad,
            _ham_term(pi, g, h, f),
            _ham_term(pi, h, f, g),
        )
        assert not rule.matches(s)

    def test_no_match_atomic_term(self):
        pi = Symbol("π")
        rule = HamiltonianCyclicSnFormulaDefinition(pi)
        assert not rule.matches(Symbol("ω"))


class TestHamiltonianCyclicSnRewrite:
    def test_collapses_positive_triple_to_sn_bracket_apply(self):
        pi = Symbol("π")
        f, g, h = Symbol("f"), Symbol("g"), Symbol("h")
        rule = HamiltonianCyclicSnFormulaDefinition(pi)
        s = Sum(
            _ham_term(pi, f, g, h),
            _ham_term(pi, g, h, f),
            _ham_term(pi, h, f, g),
        )
        out = rule.rewrite(s)
        assert out == BracketApply(default_sn, pi, pi)

    def test_collapses_negated_triple_to_negated_sn_bracket(self):
        pi = Symbol("π")
        f, g, h = Symbol("f"), Symbol("g"), Symbol("h")
        rule = HamiltonianCyclicSnFormulaDefinition(pi)
        s = Sum(
            Neg(_ham_term(pi, f, g, h)),
            Neg(_ham_term(pi, g, h, f)),
            Neg(_ham_term(pi, h, f, g)),
        )
        out = rule.rewrite(s)
        assert out == Neg(BracketApply(default_sn, pi, pi))

    def test_keeps_residue_terms(self):
        pi = Symbol("π")
        f, g, h = Symbol("f"), Symbol("g"), Symbol("h")
        residue = Symbol("residue")
        rule = HamiltonianCyclicSnFormulaDefinition(pi)
        s = Sum(
            _ham_term(pi, f, g, h),
            _ham_term(pi, g, h, f),
            _ham_term(pi, h, f, g),
            residue,
        )
        out = rule.rewrite(s)
        assert out == Sum(BracketApply(default_sn, pi, pi), residue)

    def test_uses_supplied_sn_bracket(self):
        from jacopy.brackets.schouten import SchoutenBracket

        pi = Symbol("π")
        f, g, h = Symbol("f"), Symbol("g"), Symbol("h")
        custom = SchoutenBracket(name="[·,·]_SN'")
        rule = HamiltonianCyclicSnFormulaDefinition(pi, sn_bracket=custom)
        s = Sum(
            _ham_term(pi, f, g, h),
            _ham_term(pi, g, h, f),
            _ham_term(pi, h, f, g),
        )
        out = rule.rewrite(s)
        assert out == BracketApply(custom, pi, pi)


# --------------------------------------------------------------------- #
# Engine integration                                                     #
# --------------------------------------------------------------------- #


class TestPoissonAxiomsEngine:
    def test_engine_chain_function_level(self):
        # Stitch the two axioms together: the cyclic Poisson Jacobi
        # obstruction collapses end-to-end through 2g-1 (replaces
        # nested {·,·}_π with iterated X_a) and 2g-2 (cyclic triple
        # of Hamiltonian acts → SN handle).
        pi = Symbol("π")
        f, g, h = Symbol("f"), Symbol("g"), Symbol("h")
        from jacopy.core.properties import Graded
        from jacopy.core.registry import PropertyRegistry

        reg = PropertyRegistry()
        reg.declare(pi, Graded(degree=1))
        reg.declare(f, Graded(degree=-1))
        reg.declare(g, Graded(degree=-1))
        reg.declare(h, Graded(degree=-1))

        P = PoissonBracket.from_bivector(pi).derived
        lhs = P.graded_jacobi_obstruction(f, g, h, registry=reg)

        engine = ExpansionEngine(
            [
                PoissonAsHamiltonianDefinition(P),
                HamiltonianCyclicSnFormulaDefinition(pi),
            ]
        )
        result, steps = engine.expand(lhs)
        # The Koszul-signed cyclic obstruction collapses to a Neg-
        # wrapped SN handle (functions are SN-degree -1, so each cyclic
        # term enters with a Neg).
        assert result == Neg(BracketApply(default_sn, pi, pi))
        # Both axioms appear in the proof transcript.
        rules = [step.rule for step in steps]
        assert any("X_f(g)" in r for r in rules)
        assert any("function-level" in r for r in rules)

    def test_engine_preserves_other_terms(self):
        pi = Symbol("π")
        f, g, h = Symbol("f"), Symbol("g"), Symbol("h")
        residue = Symbol("residue")
        rule = HamiltonianCyclicSnFormulaDefinition(pi)
        engine = ExpansionEngine([rule])
        s = Sum(
            _ham_term(pi, f, g, h),
            _ham_term(pi, g, h, f),
            _ham_term(pi, h, f, g),
            residue,
        )
        result, _ = engine.expand(s)
        assert result == Sum(BracketApply(default_sn, pi, pi), residue)
