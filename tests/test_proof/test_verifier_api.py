"""Tests for the Faz 7 C verifier API: prove_jacobi, prove_operator_equation, unroll_property."""

import pytest

from jacopy.algebra.derivation import Act, Derivation, compose
from jacopy.brackets.derived import DerivedBracket
from jacopy.brackets.lie import LieBracket
from jacopy.calculus.exterior_algebra import ExteriorAlgebra
from jacopy.calculus.exterior_d import d
from jacopy.calculus.interior import interior
from jacopy.calculus.lie_derivative import lie_derivative
from jacopy.core.expr import Integer, Sum, Symbol
from jacopy.core.properties import (
    Graded,
    ProofRef,
    Provenance,
)
from jacopy.core.registry import PropertyRegistry
from jacopy.brackets.derived import VanishingCondition
from jacopy.proof import (
    ProofChain,
    ProofFailure,
    prove_equivalence,
    prove_jacobi,
    prove_operator_equation,
    unroll_property,
)
from jacopy.proof.expansion import default_engine


# --------------------------------------------------------------------- #
# prove_jacobi                                                           #
# --------------------------------------------------------------------- #


class TestProveJacobiOnDerivedBracket:
    def test_derived_bracket_closes(self):
        reg = PropertyRegistry()
        for s in ("a", "b", "c", "Q"):
            reg.declare(Symbol(s), Graded(degree=0))
        lie = LieBracket()
        derived = DerivedBracket(lie, Symbol("Q"), degree_Q=0)
        chain = prove_jacobi(
            derived,
            Symbol("a"), Symbol("b"), Symbol("c"),
            registry=reg,
        )
        assert isinstance(chain, ProofChain)
        assert chain.final == Integer(0)

    def test_derived_bracket_routes_through_theorem(self):
        reg = PropertyRegistry()
        for s in ("a", "b", "c", "Q"):
            reg.declare(Symbol(s), Graded(degree=0))
        lie = LieBracket()
        derived = DerivedBracket(lie, Symbol("Q"), degree_Q=0)
        chain = prove_jacobi(
            derived,
            Symbol("a"), Symbol("b"), Symbol("c"),
            registry=reg,
        )
        rules = [s.rule for s in chain]
        assert "DerivedBracketTheorem" in rules


class TestProveJacobiOnGenericBracket:
    def test_lie_bracket_closes(self):
        """LieBracket [X, Y] := X·Y − Y·X satisfies Jacobi after simplify."""
        reg = PropertyRegistry()
        for s in ("a", "b", "c"):
            reg.declare(Symbol(s), Graded(degree=0))
        lie = LieBracket()
        chain = prove_jacobi(
            lie,
            Symbol("a"), Symbol("b"), Symbol("c"),
            registry=reg,
        )
        assert chain.final == Integer(0)


# --------------------------------------------------------------------- #
# prove_operator_equation                                                #
# --------------------------------------------------------------------- #


class TestProveOperatorEquation:
    def test_cartan_magic_formula_closes_on_function_algebra(self):
        """L_X = d∘ι_X + ι_X∘d as operators on Ω*({f})."""
        reg = PropertyRegistry()
        f = Symbol("f")
        reg.declare(f, Graded(degree=0))
        algebra = ExteriorAlgebra((f,))

        X = Derivation("X", degree=0)
        iota_X = interior(X)
        L = lie_derivative(X, definition="cartan")
        rhs = Sum(compose(d, iota_X), compose(iota_X, d))

        chain = prove_operator_equation(
            L, rhs, algebra,
            registry=reg,
            engine=default_engine(registry=reg),
        )
        assert isinstance(chain, ProofChain)
        parent = chain.steps[0]
        assert parent.rule == "AgreementOnGenerators"
        assert len(parent.children) == len(algebra.generators)

    def test_rejects_object_without_generators(self):
        class _NoGenerators:
            pass

        op = Symbol("X")
        with pytest.raises(TypeError, match="generators"):
            prove_operator_equation(op, op, _NoGenerators())

    def test_degree_mismatch_raises(self):
        algebra = ExteriorAlgebra((Symbol("f"),))
        with pytest.raises(ProofFailure, match="distinct degrees"):
            prove_operator_equation(
                d, interior(Symbol("X")), algebra,
            )


# --------------------------------------------------------------------- #
# unroll_property                                                        #
# --------------------------------------------------------------------- #


class TestUnrollProperty:
    def test_axiom_property_produces_one_step_chain(self):
        prop = Graded(degree=1)  # default Provenance.AXIOM
        chain = unroll_property(prop)
        assert len(chain) == 1
        step = chain.steps[0]
        assert step.rule == "axiom"
        assert step.provenance_tag == "axiom"
        assert "Graded" in step.justification

    def test_derived_property_records_rule_and_sources(self):
        prop = Graded(
            degree=2,
            provenance=Provenance.DERIVED,
            proof=ProofRef(rule="derived-bracket-theorem",
                           sources=("SchoutenNijenhuis", "Q")),
        )
        chain = unroll_property(prop)
        assert len(chain) == 1
        step = chain.steps[0]
        assert step.rule == "derived-bracket-theorem"
        assert step.provenance_tag == "theorem"
        assert "SchoutenNijenhuis" in step.justification
        assert "Q" in step.justification

    def test_derived_property_with_no_sources_is_still_valid(self):
        prop = Graded(
            degree=0,
            provenance=Provenance.DERIVED,
            proof=ProofRef(rule="placeholder-rule"),
        )
        chain = unroll_property(prop)
        step = chain.steps[0]
        assert step.rule == "placeholder-rule"
        assert "no recorded sources" in step.justification

    def test_rejects_non_property(self):
        with pytest.raises(TypeError, match="Property"):
            unroll_property("not a property")  # type: ignore[arg-type]


# --------------------------------------------------------------------- #
# prove_equivalence                                                      #
# --------------------------------------------------------------------- #


class TestProveEquivalence:
    def test_expr_pair_delegates_to_show_equal(self):
        reg = PropertyRegistry()
        for s in ("a", "b"):
            reg.declare(Symbol(s), Graded(degree=0))
        chain = prove_equivalence(Symbol("a"), Symbol("a"), registry=reg)
        assert isinstance(chain, ProofChain)
        assert chain.final == Symbol("a")

    def test_vanishing_conditions_with_equal_obstructions(self):
        obs = Integer(0)
        c1 = VanishingCondition(obstruction=obs, name="cond 1")
        c2 = VanishingCondition(obstruction=obs, name="cond 2")
        chain = prove_equivalence(c1, c2)
        assert isinstance(chain, ProofChain)
        assert chain.final == Integer(0)

    def test_bracket_pair_on_same_bracket_closes(self):
        reg = PropertyRegistry()
        for s in ("a", "b"):
            reg.declare(Symbol(s), Graded(degree=0))
        lie = LieBracket()
        chain = prove_equivalence(
            lie, lie, Symbol("a"), Symbol("b"),
            registry=reg,
        )
        assert isinstance(chain, ProofChain)

    def test_bracket_pair_requires_two_operands(self):
        lie = LieBracket()
        with pytest.raises(TypeError, match="two trailing"):
            prove_equivalence(lie, lie, Symbol("a"))

    def test_unsupported_pair_raises(self):
        with pytest.raises(TypeError, match="unsupported pair"):
            prove_equivalence("not-an-expr", "also-not")  # type: ignore[arg-type]

    def test_vanishing_condition_with_operands_raises(self):
        c = VanishingCondition(obstruction=Integer(0))
        with pytest.raises(TypeError, match="no operands"):
            prove_equivalence(c, c, Symbol("a"))
