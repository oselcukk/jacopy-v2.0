"""Tests for jacopy.proof.strategies."""

import pytest

from jacopy.algebra.derivation import Act, compose
from jacopy.calculus.exterior_d import d
from jacopy.calculus.interior import interior
from jacopy.calculus.lie_derivative import lie_derivative
from jacopy.core.expr import Integer, Sum, Symbol
from jacopy.proof.strategies import (
    ExpandAndSimplify,
    ProofFailure,
)


class TestReflexive:
    def test_identical_expressions_close_with_one_step(self):
        a = Symbol("a")
        chain = ExpandAndSimplify().prove(a, a)
        assert len(chain) == 1
        assert chain.steps[0].rule == "reflexive"
        assert chain.final == a


class TestCartanTautology:
    def test_cartan_mode_tautology_closes(self):
        """With cartan definition, L_X(ω) ≡ (d∘ι_X + ι_X∘d)(ω)."""
        X = Symbol("X")
        L = lie_derivative(X, definition="cartan")
        omega = Symbol("omega")
        iota_X = interior(X)
        lhs = L(omega)
        rhs = Sum(
            Act(compose(d, iota_X), omega),
            Act(compose(iota_X, d), omega),
        )
        chain = ExpandAndSimplify().prove(lhs, rhs)
        assert chain.final == Integer(0)
        assert len(chain) >= 1

    def test_cartan_tautology_with_unfolded_rhs(self):
        """Same identity, rhs given with composition unfolded into nested Act."""
        X = Symbol("X")
        L = lie_derivative(X, definition="cartan")
        omega = Symbol("omega")
        iota_X = interior(X)
        lhs = L(omega)
        rhs = Sum(
            Act(d, Act(iota_X, omega)),
            Act(iota_X, Act(d, omega)),
        )
        chain = ExpandAndSimplify().prove(lhs, rhs)
        assert chain.final == Integer(0)


class TestAlgebraicCancellation:
    def test_trivial_cancellation_closes(self):
        """Sum(a, b) == Sum(a, b) via trivial expansion, no engine fires."""
        a, b = Symbol("a"), Symbol("b")
        lhs = Sum(a, b)
        rhs = Sum(a, b)
        chain = ExpandAndSimplify().prove(lhs, rhs)
        assert chain.final == Sum(a, b)
        # Reflexive path; not the obstruction path.
        assert chain.steps[0].rule == "reflexive"

    def test_structurally_distinct_but_algebraically_equal(self):
        """Sum(a, b) vs Sum(b, a), cancels under collect_terms."""
        a, b = Symbol("a"), Symbol("b")
        chain = ExpandAndSimplify().prove(Sum(a, b), Sum(b, a))
        assert chain.final == Integer(0)


class TestFailure:
    def test_unrelated_expressions_raise(self):
        a, b = Symbol("a"), Symbol("b")
        with pytest.raises(ProofFailure):
            ExpandAndSimplify().prove(a, b)

    def test_failure_message_includes_residual(self):
        a, b = Symbol("a"), Symbol("b")
        try:
            ExpandAndSimplify().prove(a, b)
        except ProofFailure as e:
            msg = str(e)
            assert "a" in msg
            assert "b" in msg


class TestChainContract:
    def test_final_is_zero_for_successful_obstruction(self):
        X = Symbol("X")
        L = lie_derivative(X, definition="cartan")
        omega = Symbol("omega")
        iota_X = interior(X)
        chain = ExpandAndSimplify().prove(
            L(omega),
            Sum(
                Act(compose(d, iota_X), omega),
                Act(compose(iota_X, d), omega),
            ),
        )
        assert chain.final == Integer(0)

    def test_chain_has_expansion_step(self):
        X = Symbol("X")
        L = lie_derivative(X, definition="cartan")
        omega = Symbol("omega")
        iota_X = interior(X)
        chain = ExpandAndSimplify().prove(
            L(omega),
            Sum(
                Act(compose(d, iota_X), omega),
                Act(compose(iota_X, d), omega),
            ),
        )
        rules = [s.rule for s in chain]
        # The Cartan definition should show up.
        assert any("Cartan" in r for r in rules)
