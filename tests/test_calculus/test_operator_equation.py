"""Tests for OperatorEquation — the object wrapper around prove_operator_equation."""

import pytest

from gradalg.algebra.derivation import Derivation, compose
from gradalg.calculus.exterior_algebra import ExteriorAlgebra
from gradalg.calculus.exterior_d import d
from gradalg.calculus.interior import interior
from gradalg.calculus.lie_derivative import lie_derivative
from gradalg.calculus.operator_equation import OperatorEquation
from gradalg.core.expr import Sum, Symbol
from gradalg.core.properties import Graded
from gradalg.core.registry import PropertyRegistry
from gradalg.proof import ProofChain, ProofFailure
from gradalg.proof.expansion import default_engine


def _make_algebra():
    reg = PropertyRegistry()
    f = Symbol("f")
    reg.declare(f, Graded(degree=0))
    return reg, f, ExteriorAlgebra((f,))


class TestOperatorEquationProve:
    def test_cartan_magic_closes(self):
        reg, f, algebra = _make_algebra()
        X = Derivation("X", degree=0)
        iota_X = interior(X)
        eq = OperatorEquation(
            lhs=lie_derivative(X, definition="cartan"),
            rhs=Sum(compose(d, iota_X), compose(iota_X, d)),
            algebra=algebra,
        )
        chain = eq.prove(
            registry=reg,
            engine=default_engine(registry=reg),
        )
        assert isinstance(chain, ProofChain)
        assert chain.steps[0].rule == "AgreementOnGenerators"

    def test_degree_mismatch_raises(self):
        algebra = ExteriorAlgebra((Symbol("f"),))
        X = Derivation("X", degree=0)
        eq = OperatorEquation(lhs=d, rhs=interior(X), algebra=algebra)
        with pytest.raises(ProofFailure, match="distinct degrees"):
            eq.prove()

    def test_is_frozen_dataclass(self):
        algebra = ExteriorAlgebra((Symbol("f"),))
        eq = OperatorEquation(lhs=d, rhs=d, algebra=algebra)
        with pytest.raises(Exception):
            eq.lhs = d  # type: ignore[misc]

    def test_is_hashable(self):
        algebra = ExteriorAlgebra((Symbol("f"),))
        eq1 = OperatorEquation(lhs=d, rhs=d, algebra=algebra)
        eq2 = OperatorEquation(lhs=d, rhs=d, algebra=algebra)
        assert eq1 == eq2
        assert hash(eq1) == hash(eq2)

    def test_rejects_algebra_without_generators(self):
        class _NoGenerators:
            pass

        eq = OperatorEquation(lhs=d, rhs=d, algebra=_NoGenerators())
        with pytest.raises(TypeError, match="generators"):
            eq.prove()
