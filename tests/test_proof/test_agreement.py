"""Tests for jacopy.proof.strategies.AgreementOnGenerators."""

import pytest

from jacopy.algebra.derivation import Act, Derivation, compose
from jacopy.calculus.exterior_algebra import ExteriorAlgebra
from jacopy.calculus.exterior_d import d
from jacopy.calculus.interior import interior
from jacopy.calculus.lie_derivative import lie_derivative
from jacopy.core.expr import Sum, Symbol
from jacopy.core.properties import Graded
from jacopy.core.registry import PropertyRegistry
from jacopy.proof.expansion import default_engine
from jacopy.proof.strategies import (
    AgreementOnGenerators,
    ExpandAndSimplify,
    ProofFailure,
    Strategy,
)


# --------------------------------------------------------------------- #
# Construction and guardrails                                            #
# --------------------------------------------------------------------- #


class TestConstruction:
    def test_requires_algebra_with_generators(self):
        class _NoGenerators:
            pass

        with pytest.raises(TypeError, match="generators"):
            AgreementOnGenerators(_NoGenerators())

    def test_accepts_exterior_algebra(self):
        algebra = ExteriorAlgebra((Symbol("f"),))
        strat = AgreementOnGenerators(algebra)
        assert isinstance(strat, Strategy)


# --------------------------------------------------------------------- #
# Degree well-formedness                                                 #
# --------------------------------------------------------------------- #


class TestDegreeChecks:
    def test_distinct_operator_degrees_fail(self):
        algebra = ExteriorAlgebra((Symbol("f"),))
        iota_X = interior(Symbol("X"))
        strat = AgreementOnGenerators(algebra)
        # |d| = 1, |ι_X| = -1, clearly distinct.
        with pytest.raises(ProofFailure, match="distinct degrees"):
            strat.prove(d, iota_X)

    def test_inhomogeneous_sum_fails(self):
        algebra = ExteriorAlgebra((Symbol("f"),))
        iota_X = interior(Symbol("X"))
        # Sum(d, ι_X) has summands of different degree, ill-formed.
        strat = AgreementOnGenerators(algebra)
        with pytest.raises(ProofFailure, match="determinable degrees"):
            strat.prove(Sum(d, iota_X), d)

    def test_undeclared_operator_fails(self):
        algebra = ExteriorAlgebra((Symbol("f"),))
        strat = AgreementOnGenerators(algebra)
        bare = Symbol("D")  # not graded in any registry
        with pytest.raises(ProofFailure, match="determinable degrees"):
            strat.prove(bare, bare)


# --------------------------------------------------------------------- #
# Reflexive operator agreement                                           #
# --------------------------------------------------------------------- #


class TestReflexive:
    def test_same_operator_agrees_on_all_generators(self):
        reg = PropertyRegistry()
        f = Symbol("f")
        reg.declare(f, Graded(degree=0))
        algebra = ExteriorAlgebra((f,))
        strat = AgreementOnGenerators(algebra)
        chain = strat.prove(d, d, registry=reg, engine=default_engine(registry=reg))
        assert len(chain) == 1
        parent = chain.steps[0]
        assert parent.rule == "AgreementOnGenerators"
        # One generator-child per generator (f, df): 2 children.
        assert len(parent.children) == len(algebra.generators)


# --------------------------------------------------------------------- #
# Cartan magic formula, [d, ι_X] = L_X in cartan mode                   #
# --------------------------------------------------------------------- #


class TestCartanMagicFormula:
    """In cartan-mode the formula is tautological, the proof closes
    structurally, generator by generator, using only the Cartan
    definition rewrite and Act-over-Sum linearity.
    """

    def test_closes_on_single_function_algebra(self):
        reg = PropertyRegistry()
        f = Symbol("f")
        reg.declare(f, Graded(degree=0))
        algebra = ExteriorAlgebra((f,))

        X_deriv = Derivation("X", degree=0)
        iota_X = interior(X_deriv)
        L = lie_derivative(X_deriv, definition="cartan")

        # [d, ι_X]_graded = d∘ι_X + ι_X∘d (both odd-degree operators).
        rhs = Sum(compose(d, iota_X), compose(iota_X, d))

        strat = AgreementOnGenerators(algebra)
        chain = strat.prove(
            L, rhs,
            registry=reg,
            engine=default_engine(registry=reg),
        )
        assert len(chain) == 1
        parent = chain.steps[0]
        assert parent.rule == "AgreementOnGenerators"
        # Exterior algebra generators: {f, df} → 2 per-generator children.
        assert len(parent.children) == 2

    def test_chain_structure_has_nested_sub_proofs(self):
        reg = PropertyRegistry()
        f = Symbol("f")
        reg.declare(f, Graded(degree=0))
        algebra = ExteriorAlgebra((f,))

        X_deriv = Derivation("X", degree=0)
        iota_X = interior(X_deriv)
        L = lie_derivative(X_deriv, definition="cartan")
        rhs = Sum(compose(d, iota_X), compose(iota_X, d))

        strat = AgreementOnGenerators(algebra)
        chain = strat.prove(
            L, rhs,
            registry=reg,
            engine=default_engine(registry=reg),
        )
        parent = chain.steps[0]
        # Each per-generator child carries the ExpandAndSimplify sub-proof.
        for gen_step in parent.children:
            assert gen_step.rule == "check-on-generator"
            assert len(gen_step.children) >= 1


# --------------------------------------------------------------------- #
# Failure propagation                                                    #
# --------------------------------------------------------------------- #


class TestFailurePropagation:
    def test_disagreement_on_generator_raises(self):
        reg = PropertyRegistry()
        f = Symbol("f")
        reg.declare(f, Graded(degree=0))
        algebra = ExteriorAlgebra((f,))
        X_deriv = Derivation("X", degree=0)
        Y_deriv = Derivation("Y", degree=0)

        strat = AgreementOnGenerators(algebra)
        with pytest.raises(ProofFailure, match="failed on generator"):
            strat.prove(
                X_deriv, Y_deriv,
                registry=reg,
                engine=default_engine(registry=reg),
            )


# --------------------------------------------------------------------- #
# Custom sub-strategy                                                    #
# --------------------------------------------------------------------- #


class _AlwaysCloseStrategy(Strategy):
    name = "always-close"

    def prove(self, lhs, rhs, *, registry=None, engine=None):
        from jacopy.proof.chain import ProofChain
        from jacopy.proof.step import ProofStep

        chain = ProofChain()
        chain.append(ProofStep(lhs, rhs, rule=self.name, justification="stub"))
        return chain


class TestCustomSubStrategy:
    def test_custom_sub_strategy_is_used(self):
        reg = PropertyRegistry()
        f = Symbol("f")
        reg.declare(f, Graded(degree=0))
        algebra = ExteriorAlgebra((f,))
        X_deriv = Derivation("X", degree=0)
        Y_deriv = Derivation("Y", degree=0)

        strat = AgreementOnGenerators(algebra, sub_strategy=_AlwaysCloseStrategy())
        # Normally unequal derivations would fail, here the sub-strategy
        # rubber-stamps each generator so the outer proof succeeds.
        chain = strat.prove(X_deriv, Y_deriv, registry=reg)
        parent = chain.steps[0]
        for gen_step in parent.children:
            assert any(c.rule == "always-close" for c in gen_step.children)
