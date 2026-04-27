"""Tests for prove_derivator_identity driver and KoszulProblem
engine accessors (Faz 15.B)."""

import pytest

from jacopy.algebra.derivation import Act, Derivation
from jacopy.brackets.koszul import KoszulBracket
from jacopy.calculus.cartan_remainder import K
from jacopy.calculus.cartan_remainder_axioms import (
    CartanRemainderDefinition,
    TildeCartanRemainderDefinition,
)
from jacopy.calculus.derivator import derivator, prove_derivator_identity
from jacopy.calculus.exterior_d import d as default_d
from jacopy.calculus.interior import interior
from jacopy.calculus.lie_derivative import lie_derivative
from jacopy.calculus.tilde import K_tilde, tilde_d, tilde_interior, tilde_lie
from jacopy.core.expr import Neg, Sum, Symbol
from jacopy.core.properties import Graded
from jacopy.core.registry import PropertyRegistry
from jacopy.library.koszul_problem import KoszulProblem
from jacopy.proof.chain import ProofChain
from jacopy.proof.expansion import ExpansionEngine
from jacopy.proof.strategies import ProofFailure


@pytest.fixture
def problem():
    reg = PropertyRegistry()
    pi = Symbol("π")
    omega, eta = Symbol("ω"), Symbol("η")
    reg.declare(omega, Graded(degree=1))
    reg.declare(eta, Graded(degree=1))
    return KoszulProblem(pi, (omega, eta), registry=reg)


# --------------------------------------------------------------------- #
# prove_derivator_identity driver                                        #
# --------------------------------------------------------------------- #


class TestProveDerivatorIdentityType:
    def test_rejects_non_expr_lhs(self):
        engine = ExpansionEngine([CartanRemainderDefinition()])
        with pytest.raises(TypeError):
            prove_derivator_identity(
                "not-an-expr", Symbol("μ"),  # type: ignore[arg-type]
                engine=engine, eval_args=(Symbol("Y"),),
            )

    def test_rejects_non_expr_rhs(self):
        engine = ExpansionEngine([CartanRemainderDefinition()])
        with pytest.raises(TypeError):
            prove_derivator_identity(
                Symbol("μ"), "not-an-expr",  # type: ignore[arg-type]
                engine=engine, eval_args=(Symbol("Y"),),
            )

    def test_rejects_non_engine(self):
        with pytest.raises(TypeError):
            prove_derivator_identity(
                Symbol("μ"), Symbol("μ"),
                engine="not-an-engine",  # type: ignore[arg-type]
                eval_args=(Symbol("Y"),),
            )

    def test_rejects_empty_eval_args(self):
        engine = ExpansionEngine([CartanRemainderDefinition()])
        with pytest.raises(TypeError):
            prove_derivator_identity(
                Symbol("μ"), Symbol("μ"),
                engine=engine, eval_args=(),
            )

    def test_rejects_invalid_slot_kind(self):
        engine = ExpansionEngine([CartanRemainderDefinition()])
        with pytest.raises(ValueError):
            prove_derivator_identity(
                Symbol("μ"), Symbol("μ"),
                engine=engine,
                eval_args=(Symbol("Y"),),
                slot_kind="invalid",
            )


# --------------------------------------------------------------------- #
# KoszulProblem engine accessors                                         #
# --------------------------------------------------------------------- #


class TestKoszulProblemEngines:
    def test_form_engine_returns_expansion_engine(self, problem):
        engine = problem.derivator_form_engine()
        assert isinstance(engine, ExpansionEngine)

    def test_form_engine_includes_cartan_remainder_rules(self, problem):
        engine = problem.derivator_form_engine()
        rules = list(engine.definitions)
        assert any(isinstance(r, CartanRemainderDefinition) for r in rules)
        assert any(isinstance(r, TildeCartanRemainderDefinition) for r in rules)

    def test_multivector_engine_returns_expansion_engine(self, problem):
        engine = problem.derivator_multivector_engine()
        assert isinstance(engine, ExpansionEngine)

    def test_multivector_engine_includes_cartan_remainder_rules(self, problem):
        engine = problem.derivator_multivector_engine()
        rules = list(engine.definitions)
        assert any(isinstance(r, CartanRemainderDefinition) for r in rules)
        assert any(isinstance(r, TildeCartanRemainderDefinition) for r in rules)

    def test_form_and_multivector_engines_are_independent(self, problem):
        e1 = problem.derivator_form_engine()
        e2 = problem.derivator_form_engine()
        assert e1 is not e2  # fresh each call


# --------------------------------------------------------------------- #
# Tautology smoke proofs                                                #
# --------------------------------------------------------------------- #


class TestTautologySmoke:
    def test_K_definition_closes_form_side(self, problem):
        """``K_V ω == −L_V ω + d ι_V ω`` via form engine."""
        V, Y = Symbol("V"), Symbol("Y")
        problem.registry.declare(V, Graded(degree=0))
        problem.registry.declare(Y, Graded(degree=0))
        omega = problem.forms[0]
        lhs = Act(K(V), omega)
        rhs = Sum(
            Neg(Act(lie_derivative(V), omega)),
            Act(default_d, Act(interior(V), omega)),
        )
        chain = problem.prove_derivator(
            lhs, rhs, eval_args=(Y,), side="form",
        )
        assert isinstance(chain, ProofChain)
        assert len(chain.steps) > 0

    def test_K_tilde_definition_closes_multivector_side(self, problem):
        """``K̃_η V == −L̃_η V + d̃ ι̃_η V`` via multivector engine."""
        V = Symbol("V")
        problem.registry.declare(V, Graded(degree=1))
        eta = problem.forms[1]
        xi = Symbol("ξ")
        problem.registry.declare(xi, Graded(degree=1))
        pi = problem.pi
        lhs = Act(K_tilde(eta, pi), V)
        rhs = Sum(
            Neg(Act(tilde_lie(eta, pi), V)),
            Act(tilde_d(pi), Act(tilde_interior(eta), V)),
        )
        chain = problem.prove_derivator(
            lhs, rhs, eval_args=(xi,), side="multivector",
        )
        assert isinstance(chain, ProofChain)
        assert len(chain.steps) > 0

    def test_reflexive_pair_closes(self, problem):
        V, Y = Symbol("V"), Symbol("Y")
        problem.registry.declare(V, Graded(degree=0))
        problem.registry.declare(Y, Graded(degree=0))
        omega = problem.forms[0]
        head = Act(K(V), omega)
        chain = problem.prove_derivator(
            head, head, eval_args=(Y,), side="form",
        )
        assert isinstance(chain, ProofChain)


# --------------------------------------------------------------------- #
# prove_derivator side guard                                            #
# --------------------------------------------------------------------- #


class TestProveDerivatorSideGuard:
    def test_invalid_side_raises(self, problem):
        with pytest.raises(ValueError):
            problem.prove_derivator(
                Symbol("a"), Symbol("a"),
                eval_args=(Symbol("Y"),), side="invalid",
            )
