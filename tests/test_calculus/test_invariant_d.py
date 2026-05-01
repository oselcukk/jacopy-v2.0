"""Tests for the Cartan–Koszul invariant formula for ``d`` on 1-forms."""

import pytest

from jacopy.algebra.derivation import Act, Derivation
from jacopy.brackets.base import GradedBracket
from jacopy.brackets.lie import LieBracket, lie
from jacopy.calculus.exterior_d import ExteriorDerivative, d as default_d
from jacopy.calculus.interior import InteriorProduct, interior
from jacopy.calculus.invariant_d import (
    INVARIANT_D_CLASSIFICATIONS,
    InvariantDOneFormDefinition,
    invariant_d_one_form,
)
from jacopy.core.expr import Expr, Neg, Sum, Symbol
from jacopy.core.properties import Graded
from jacopy.core.registry import PropertyRegistry
from jacopy.proof.chain import ProofChain
from jacopy.proof.expansion import ExpansionEngine


# --------------------------------------------------------------------- #
# Fixtures                                                               #
# --------------------------------------------------------------------- #


@pytest.fixture
def one_form_registry():
    reg = PropertyRegistry()
    omega = Symbol("ω")
    reg.declare(omega, Graded(degree=1))
    return reg, omega


# --------------------------------------------------------------------- #
# invariant_d_one_form, RHS builder                                     #
# --------------------------------------------------------------------- #


class TestInvariantDOneFormBuilder:
    def test_returns_three_term_sum(self):
        omega = Symbol("ω")
        X, Y = Symbol("X"), Symbol("Y")
        out = invariant_d_one_form(omega, X, Y, bracket=lie)
        assert isinstance(out, Sum)
        assert len(out.children) == 3

    def test_first_term_is_X_of_iota_Y_omega(self):
        """``Act(X, Act(ι_Y, ω))`` with no outer Neg."""
        omega = Symbol("ω")
        X, Y = Symbol("X"), Symbol("Y")
        out = invariant_d_one_form(omega, X, Y, bracket=lie)
        first = out.children[0]
        assert isinstance(first, Act)
        assert first.op is X
        inner = first.arg
        assert isinstance(inner, Act)
        assert isinstance(inner.op, InteriorProduct)
        assert inner.op.vector_field is Y
        assert inner.arg is omega

    def test_second_term_is_neg_Y_of_iota_X_omega(self):
        omega = Symbol("ω")
        X, Y = Symbol("X"), Symbol("Y")
        out = invariant_d_one_form(omega, X, Y, bracket=lie)
        second = out.children[1]
        assert isinstance(second, Neg)
        inner = second.arg
        assert isinstance(inner, Act)
        assert inner.op is Y
        inner2 = inner.arg
        assert isinstance(inner2, Act)
        assert isinstance(inner2.op, InteriorProduct)
        assert inner2.op.vector_field is X
        assert inner2.arg is omega

    def test_third_term_is_neg_iota_bracket_omega(self):
        """``Neg(Act(ι_{[X, Y]}, ω))``, bracket supplied by the bracket arg."""
        omega = Symbol("ω")
        X, Y = Symbol("X"), Symbol("Y")
        out = invariant_d_one_form(omega, X, Y, bracket=lie)
        third = out.children[2]
        assert isinstance(third, Neg)
        inner = third.arg
        assert isinstance(inner, Act)
        assert isinstance(inner.op, InteriorProduct)
        # ι's vector field equals the bracket's own expansion on (X, Y).
        assert inner.op.vector_field == lie.expand(X, Y)
        assert inner.arg is omega

    def test_uses_supplied_bracket_expansion(self):
        """Swapping the bracket changes the third term's ι vector field."""
        omega = Symbol("ω")
        X, Y = Symbol("X"), Symbol("Y")
        alt = LieBracket(name="[·,·]_alt")
        out = invariant_d_one_form(omega, X, Y, bracket=alt)
        third = out.children[2]
        assert third.arg.op.vector_field == alt.expand(X, Y)

    def test_custom_interior_factory_honoured(self):
        omega = Symbol("ω")
        X, Y = Symbol("X"), Symbol("Y")
        sentinel_X = Derivation("ι_custom_X", degree=-1)
        sentinel_Y = Derivation("ι_custom_Y", degree=-1)
        sentinel_XY = Derivation("ι_custom_[X,Y]", degree=-1)
        calls = []

        def factory(V: Expr) -> Derivation:
            calls.append(V)
            if V is X:
                return sentinel_X
            if V is Y:
                return sentinel_Y
            return sentinel_XY

        out = invariant_d_one_form(omega, X, Y, bracket=lie, interior=factory)
        assert out.children[0].arg.op is sentinel_Y
        assert out.children[1].arg.arg.op is sentinel_X
        assert out.children[2].arg.op is sentinel_XY

    def test_rejects_non_expr_omega(self):
        X, Y = Symbol("X"), Symbol("Y")
        with pytest.raises(TypeError, match="ω"):
            invariant_d_one_form("ω", X, Y, bracket=lie)  # type: ignore[arg-type]

    def test_rejects_non_expr_X(self):
        omega, Y = Symbol("ω"), Symbol("Y")
        with pytest.raises(TypeError, match="X"):
            invariant_d_one_form(omega, "X", Y, bracket=lie)  # type: ignore[arg-type]

    def test_rejects_non_expr_Y(self):
        omega, X = Symbol("ω"), Symbol("X")
        with pytest.raises(TypeError, match="Y"):
            invariant_d_one_form(omega, X, "Y", bracket=lie)  # type: ignore[arg-type]

    def test_rejects_non_bracket(self):
        omega = Symbol("ω")
        X, Y = Symbol("X"), Symbol("Y")
        with pytest.raises(TypeError, match="bracket"):
            invariant_d_one_form(omega, X, Y, bracket="lie")  # type: ignore[arg-type]


# --------------------------------------------------------------------- #
# InvariantDOneFormDefinition, construction                             #
# --------------------------------------------------------------------- #


class TestConstruction:
    def test_default_classification_is_theorem(self):
        defn = InvariantDOneFormDefinition(lie)
        assert defn.classification == "theorem"
        assert defn.is_theorem is True

    def test_axiom_classification(self):
        defn = InvariantDOneFormDefinition(lie, classification="axiom")
        assert defn.classification == "axiom"
        assert defn.is_theorem is False
        assert defn.theorem_proof_builder() is None

    def test_rejects_unknown_classification(self):
        with pytest.raises(ValueError, match="classification"):
            InvariantDOneFormDefinition(lie, classification="lemma")

    def test_rejects_non_bracket(self):
        with pytest.raises(TypeError, match="GradedBracket"):
            InvariantDOneFormDefinition("lie")  # type: ignore[arg-type]

    def test_exposes_bracket_property(self):
        defn = InvariantDOneFormDefinition(lie)
        assert defn.bracket is lie


# --------------------------------------------------------------------- #
# matches / rewrite                                                      #
# --------------------------------------------------------------------- #


class TestMatches:
    def test_matches_iota_iota_d_on_one_form(self, one_form_registry):
        reg, omega = one_form_registry
        X, Y = Symbol("X"), Symbol("Y")
        expr = Act(interior(Y), Act(interior(X), Act(default_d, omega)))
        defn = InvariantDOneFormDefinition(lie, registry=reg)
        assert defn.matches(expr)

    def test_requires_one_form_degree(self):
        """Without the registry declaring ``|ω| = 1`` the rule stays inert."""
        omega = Symbol("ω")  # no declaration
        X, Y = Symbol("X"), Symbol("Y")
        expr = Act(interior(Y), Act(interior(X), Act(default_d, omega)))
        defn = InvariantDOneFormDefinition(lie)  # no registry
        assert not defn.matches(expr)

    def test_rejects_wrong_degree(self):
        reg = PropertyRegistry()
        omega = Symbol("ω")
        reg.declare(omega, Graded(degree=2))  # 2-form, not 1-form
        X, Y = Symbol("X"), Symbol("Y")
        expr = Act(interior(Y), Act(interior(X), Act(default_d, omega)))
        defn = InvariantDOneFormDefinition(lie, registry=reg)
        assert not defn.matches(expr)

    def test_rejects_single_interior(self, one_form_registry):
        """``ι_X(dω)`` alone, only one interior product, does not match."""
        reg, omega = one_form_registry
        X = Symbol("X")
        expr = Act(interior(X), Act(default_d, omega))
        defn = InvariantDOneFormDefinition(lie, registry=reg)
        assert not defn.matches(expr)

    def test_rejects_missing_d(self, one_form_registry):
        reg, omega = one_form_registry
        X, Y = Symbol("X"), Symbol("Y")
        expr = Act(interior(Y), Act(interior(X), omega))
        defn = InvariantDOneFormDefinition(lie, registry=reg)
        assert not defn.matches(expr)

    def test_rejects_non_interior_outer(self, one_form_registry):
        reg, omega = one_form_registry
        X = Symbol("X")
        expr = Act(default_d, Act(interior(X), Act(default_d, omega)))
        defn = InvariantDOneFormDefinition(lie, registry=reg)
        assert not defn.matches(expr)

    def test_respects_d_target(self, one_form_registry):
        """A distinct ``d_E`` instance should not match the default ``d`` rule."""
        reg, omega = one_form_registry
        d_E = ExteriorDerivative(name="d_E")
        X, Y = Symbol("X"), Symbol("Y")
        expr = Act(interior(Y), Act(interior(X), Act(d_E, omega)))
        defn = InvariantDOneFormDefinition(lie, registry=reg)  # default d
        assert not defn.matches(expr)
        defn_E = InvariantDOneFormDefinition(lie, d=d_E, registry=reg)
        assert defn_E.matches(expr)


class TestRewrite:
    def test_rewrite_equals_builder_rhs(self, one_form_registry):
        reg, omega = one_form_registry
        X, Y = Symbol("X"), Symbol("Y")
        expr = Act(interior(Y), Act(interior(X), Act(default_d, omega)))
        defn = InvariantDOneFormDefinition(lie, registry=reg)
        out = defn.rewrite(expr)
        expected = invariant_d_one_form(omega, X, Y, bracket=lie)
        assert out == expected

    def test_rewrite_uses_definition_bracket(self, one_form_registry):
        reg, omega = one_form_registry
        X, Y = Symbol("X"), Symbol("Y")
        expr = Act(interior(Y), Act(interior(X), Act(default_d, omega)))
        alt = LieBracket(name="[·,·]_alt")
        defn = InvariantDOneFormDefinition(alt, registry=reg)
        out = defn.rewrite(expr)
        # Third term's ι_{[X,Y]} uses the alt bracket's expansion.
        third = out.children[2].arg
        assert third.op.vector_field == alt.expand(X, Y)


# --------------------------------------------------------------------- #
# theorem_proof_builder                                                  #
# --------------------------------------------------------------------- #


class TestTheoremProofBuilder:
    def test_theorem_builder_returns_single_step_chain(self, one_form_registry):
        reg, omega = one_form_registry
        X, Y = Symbol("X"), Symbol("Y")
        expr = Act(interior(Y), Act(interior(X), Act(default_d, omega)))
        defn = InvariantDOneFormDefinition(lie, registry=reg)
        builder = defn.theorem_proof_builder()
        assert builder is not None
        chain = builder(expr)
        assert isinstance(chain, ProofChain)
        assert len(chain) == 1

    def test_sub_step_cites_cartan_foundations(self, one_form_registry):
        reg, omega = one_form_registry
        X, Y = Symbol("X"), Symbol("Y")
        expr = Act(interior(Y), Act(interior(X), Act(default_d, omega)))
        defn = InvariantDOneFormDefinition(lie, registry=reg)
        chain = defn.theorem_proof_builder()(expr)
        step = chain.steps[0]
        assert step.before == expr
        assert step.after == defn.rewrite(expr)
        assert "magic" in step.rule.lower() or "cartan" in step.rule.lower()
        assert step.provenance_tag == "axiom"

    def test_axiom_classification_has_no_builder(self):
        defn = InvariantDOneFormDefinition(lie, classification="axiom")
        assert defn.theorem_proof_builder() is None


# --------------------------------------------------------------------- #
# ExpansionEngine integration                                            #
# --------------------------------------------------------------------- #


class TestEngineIntegration:
    def test_efficient_mode_fires_and_tags_theorem(self, one_form_registry):
        reg, omega = one_form_registry
        X, Y = Symbol("X"), Symbol("Y")
        expr = Act(interior(Y), Act(interior(X), Act(default_d, omega)))
        engine = ExpansionEngine(
            [InvariantDOneFormDefinition(lie, registry=reg)],
            mode="efficient",
        )
        out, step = engine.expand_once(expr)
        assert step is not None
        assert step.provenance_tag == "theorem"
        assert step.children == ()
        assert out == invariant_d_one_form(omega, X, Y, bracket=lie)

    def test_foundational_mode_attaches_sub_proof(self, one_form_registry):
        reg, omega = one_form_registry
        X, Y = Symbol("X"), Symbol("Y")
        expr = Act(interior(Y), Act(interior(X), Act(default_d, omega)))
        engine = ExpansionEngine(
            [InvariantDOneFormDefinition(lie, registry=reg)],
            mode="foundational",
        )
        _, step = engine.expand_once(expr)
        assert step is not None
        assert step.provenance_tag == "theorem"
        assert len(step.children) == 1
        sub = step.children[0]
        assert sub.provenance_tag == "axiom"

    def test_axiom_classification_no_children_under_foundational(
        self, one_form_registry
    ):
        reg, omega = one_form_registry
        X, Y = Symbol("X"), Symbol("Y")
        expr = Act(interior(Y), Act(interior(X), Act(default_d, omega)))
        engine = ExpansionEngine(
            [
                InvariantDOneFormDefinition(
                    lie, registry=reg, classification="axiom"
                )
            ],
            mode="foundational",
        )
        _, step = engine.expand_once(expr)
        assert step is not None
        assert step.provenance_tag == "axiom"
        assert step.children == ()

    def test_stays_inert_without_registry(self):
        """Without ``|ω| = 1`` the rule cannot fire, no step produced."""
        omega = Symbol("ω")
        X, Y = Symbol("X"), Symbol("Y")
        expr = Act(interior(Y), Act(interior(X), Act(default_d, omega)))
        engine = ExpansionEngine(
            [InvariantDOneFormDefinition(lie)],  # no registry
            mode="efficient",
        )
        out, step = engine.expand_once(expr)
        assert step is None
        assert out == expr


# --------------------------------------------------------------------- #
# Classifications tuple                                                  #
# --------------------------------------------------------------------- #


class TestClassificationsTuple:
    def test_exposes_exactly_two_labels(self):
        assert INVARIANT_D_CLASSIFICATIONS == ("axiom", "theorem")
