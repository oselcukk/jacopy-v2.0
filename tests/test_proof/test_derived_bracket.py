"""Tests for jacopy.proof.strategies.DerivedBracketStrategy."""

import pytest

from jacopy.brackets.derived import DerivedBracket
from jacopy.brackets.lie import LieBracket
from jacopy.core.expr import Integer, Symbol
from jacopy.core.properties import Graded
from jacopy.core.registry import PropertyRegistry
from jacopy.proof.strategies import (
    DerivedBracketStrategy,
    ProofFailure,
)


# --------------------------------------------------------------------- #
# Fixtures                                                               #
# --------------------------------------------------------------------- #


@pytest.fixture
def reg():
    r = PropertyRegistry()
    for s in ("a", "b", "c"):
        r.declare(Symbol(s), Graded(degree=0))
    return r


# --------------------------------------------------------------------- #
# Input validation                                                       #
# --------------------------------------------------------------------- #


class TestInputValidation:
    def test_requires_derived_bracket(self):
        lie = LieBracket()
        a, b, c = Symbol("a"), Symbol("b"), Symbol("c")
        with pytest.raises(TypeError, match="DerivedBracket"):
            DerivedBracketStrategy().prove_jacobi(lie, a, b, c)


# --------------------------------------------------------------------- #
# Closure: [Q, Q]_Lie = Q·Q − Q·Q = 0                                    #
# --------------------------------------------------------------------- #


class TestSuccessfulClosure:
    def test_lie_base_closes_jacobi(self, reg):
        """DerivedBracket(LieBracket, Q) has [Q, Q]_Lie = Q·Q − Q·Q = 0."""
        lie = LieBracket()
        Q = Symbol("Q")
        reg.declare(Q, Graded(degree=0))
        derived = DerivedBracket(lie, Q, degree_Q=0)

        a, b, c = Symbol("a"), Symbol("b"), Symbol("c")
        chain = DerivedBracketStrategy().prove_jacobi(
            derived, a, b, c, registry=reg,
        )
        assert chain.final == Integer(0)

    def test_first_step_is_theorem(self, reg):
        lie = LieBracket()
        Q = Symbol("Q")
        reg.declare(Q, Graded(degree=0))
        derived = DerivedBracket(lie, Q, degree_Q=0)

        a, b, c = Symbol("a"), Symbol("b"), Symbol("c")
        chain = DerivedBracketStrategy().prove_jacobi(
            derived, a, b, c, registry=reg,
        )
        first = chain.steps[0]
        assert first.rule == "DerivedBracketTheorem"
        assert first.provenance_tag == "theorem"

    def test_chain_is_non_trivial(self, reg):
        lie = LieBracket()
        Q = Symbol("Q")
        reg.declare(Q, Graded(degree=0))
        derived = DerivedBracket(lie, Q, degree_Q=0)

        a, b, c = Symbol("a"), Symbol("b"), Symbol("c")
        chain = DerivedBracketStrategy().prove_jacobi(
            derived, a, b, c, registry=reg,
        )
        # Expect at least theorem-step + simplification closing at 0.
        assert len(chain) >= 2


# --------------------------------------------------------------------- #
# Failure propagation                                                    #
# --------------------------------------------------------------------- #


class TestFailure:
    def test_symbolic_parity_raises(self):
        """No registry → operand degrees are undecidable → ProofFailure."""
        lie = LieBracket()
        Q = Symbol("Q")
        derived = DerivedBracket(lie, Q, degree_Q=0)

        a, b, c = Symbol("a"), Symbol("b"), Symbol("c")
        with pytest.raises(ProofFailure, match="ill-formed"):
            DerivedBracketStrategy().prove_jacobi(derived, a, b, c)
