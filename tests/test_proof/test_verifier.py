"""Tests for jacopy.proof.verifier."""

import pytest

from jacopy.algebra.derivation import Act, compose
from jacopy.calculus.exterior_d import d
from jacopy.calculus.interior import interior
from jacopy.calculus.lie_derivative import lie_derivative
from jacopy.core.expr import Integer, Sum, Symbol
from jacopy.proof import (
    ExpandAndSimplify,
    ProofChain,
    ProofFailure,
    Strategy,
    show_equal,
)


class _EchoStrategy(Strategy):
    """Minimal strategy for testing strategy substitution, always closes."""

    name = "echo"

    def prove(self, lhs, rhs, *, registry=None, engine=None):
        from jacopy.proof.step import ProofStep

        chain = ProofChain()
        chain.append(ProofStep(lhs, rhs, rule="echo", justification="unconditional"))
        return chain


class TestShowEqual:
    def test_returns_proof_chain(self):
        a = Symbol("a")
        chain = show_equal(a, a)
        assert isinstance(chain, ProofChain)

    def test_default_strategy_is_expand_and_simplify(self):
        X = Symbol("X")
        L = lie_derivative(X, definition="cartan")
        omega = Symbol("omega")
        iota_X = interior(X)
        chain = show_equal(
            L(omega),
            Sum(
                Act(compose(d, iota_X), omega),
                Act(compose(iota_X, d), omega),
            ),
        )
        assert chain.final == Integer(0)

    def test_custom_strategy_used(self):
        a, b = Symbol("a"), Symbol("b")
        chain = show_equal(a, b, strategy=_EchoStrategy())
        assert len(chain) == 1
        assert chain.steps[0].rule == "echo"

    def test_proof_failure_propagates(self):
        a, b = Symbol("a"), Symbol("b")
        with pytest.raises(ProofFailure):
            show_equal(a, b)

    def test_explicit_expand_and_simplify(self):
        a = Symbol("a")
        chain = show_equal(a, a, strategy=ExpandAndSimplify())
        assert chain.final == a
