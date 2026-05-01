"""Tests for ``jacopy.library.lie_algebroid``."""

from __future__ import annotations

import pytest

from jacopy.algebra.derivation import Derivation
from jacopy.brackets.base import GradedBracket
from jacopy.brackets.derived import VanishingCondition
from jacopy.brackets.lie import LieBracket, lie
from jacopy.calculus.anchor import Anchor
from jacopy.calculus.cartan import CartanCalculus
from jacopy.calculus.exterior_algebra import ExteriorAlgebra
from jacopy.calculus.exterior_d import ExteriorDerivative
from jacopy.calculus.interior import InteriorProduct
from jacopy.calculus.lie_derivative import LieDerivative
from jacopy.core.expr import Expr, Integer, Sum, Symbol
from jacopy.core.properties import Graded
from jacopy.core.registry import PropertyRegistry
from jacopy.library import theorem_book
from jacopy.library.lie_algebroid import (
    THEOREM_LIE_ALGEBROID_ANCHOR_COMPAT,
    LieAlgebroid,
    lie_algebroid,
)
from jacopy.proof.chain import ProofChain


# --------------------------------------------------------------------- #
# Fixtures                                                               #
# --------------------------------------------------------------------- #


@pytest.fixture
def bundle():
    return Symbol("E")


@pytest.fixture
def bracket_E():
    return LieBracket(name="[·,·]_E")


@pytest.fixture
def rho():
    return Anchor(name="ρ")


@pytest.fixture
def algebroid(bundle, bracket_E, rho):
    return LieAlgebroid(bundle, bracket=bracket_E, anchor=rho)


@pytest.fixture
def registry():
    reg = PropertyRegistry()
    X, Y = Symbol("X"), Symbol("Y")
    reg.declare(X, Graded(degree=0))
    reg.declare(Y, Graded(degree=0))
    return reg


# --------------------------------------------------------------------- #
# Construction                                                           #
# --------------------------------------------------------------------- #


class TestConstruction:
    def test_basic(self, algebroid, bundle, bracket_E, rho):
        assert algebroid.bundle is bundle
        assert algebroid.bracket is bracket_E
        assert algebroid.anchor is rho

    def test_default_vector_bracket_is_tm_lie(self, algebroid):
        assert algebroid.vector_bracket is lie

    def test_custom_vector_bracket(self, bundle, bracket_E, rho):
        custom = LieBracket(name="[·,·]_{TM}")
        A = LieAlgebroid(
            bundle, bracket=bracket_E, anchor=rho, vector_bracket=custom,
        )
        assert A.vector_bracket is custom

    def test_default_name_carries_bundle(self, algebroid):
        assert "E" in algebroid.name

    def test_custom_name(self, bundle, bracket_E, rho):
        A = LieAlgebroid(
            bundle, bracket=bracket_E, anchor=rho, name="TpoiM",
        )
        assert A.name == "TpoiM"

    def test_factory(self, bundle, bracket_E, rho):
        A = lie_algebroid(bundle, bracket=bracket_E, anchor=rho)
        assert isinstance(A, LieAlgebroid)
        assert A.bundle is bundle

    def test_rejects_non_expr_bundle(self, bracket_E, rho):
        with pytest.raises(TypeError, match="Expr"):
            LieAlgebroid("E", bracket=bracket_E, anchor=rho)  # type: ignore[arg-type]

    def test_rejects_non_bracket(self, bundle, rho):
        with pytest.raises(TypeError, match="GradedBracket"):
            LieAlgebroid(bundle, bracket="not a bracket", anchor=rho)  # type: ignore[arg-type]

    def test_rejects_non_anchor(self, bundle, bracket_E):
        with pytest.raises(TypeError, match="Anchor"):
            LieAlgebroid(bundle, bracket=bracket_E, anchor="ρ")  # type: ignore[arg-type]

    def test_rejects_non_bracket_vector_bracket(self, bundle, bracket_E, rho):
        with pytest.raises(TypeError, match="GradedBracket"):
            LieAlgebroid(
                bundle,
                bracket=bracket_E,
                anchor=rho,
                vector_bracket="lie",  # type: ignore[arg-type]
            )


# --------------------------------------------------------------------- #
# Algebroid exterior derivative + Cartan bundle                           #
# --------------------------------------------------------------------- #


class TestAlgebroidCartanBundle:
    def test_d_is_named_for_bundle(self, algebroid):
        """``d_E`` carries the bundle tag so it's distinguishable from
        the ambient ``d`` on TM."""
        assert isinstance(algebroid.d, ExteriorDerivative)
        assert algebroid.d.name == "d_E"

    def test_d_is_distinct_from_tm_d(self, algebroid):
        from jacopy.calculus.exterior_d import d as d_TM
        assert algebroid.d != d_TM

    def test_cartan_type(self, algebroid):
        assert isinstance(algebroid.cartan, CartanCalculus)

    def test_cartan_d_is_algebroid_d(self, algebroid):
        assert algebroid.cartan.d is algebroid.d

    def test_cartan_vector_bracket_is_algebroid_bracket(self, algebroid):
        assert algebroid.cartan.vector_bracket is algebroid.bracket

    def test_lie_factory_names_carry_bundle_tag(self, algebroid):
        X = Symbol("X")
        L = algebroid.cartan.lie_derivative(X)
        assert isinstance(L, LieDerivative)
        assert L.name == "L_E,X"

    def test_lie_factory_plumbs_bundle_d_and_iota(self, algebroid):
        """The algebroid ``L_{E,X}`` carries its bundle's ``d_E`` and
        ``ι_E`` factory on its slots so the Cartan expansion engine can
        keep the operator names aligned."""
        X = Symbol("X")
        L = algebroid.cartan.lie_derivative(X)
        assert L.d is algebroid.d
        # The factory is the one stored on the algebroid's Cartan bundle,
        # call it on a fresh field and check the resulting ι carries the
        # bundle tag.
        assert L.iota_factory is not None
        iota_X = L.iota_factory(X)
        assert iota_X.name == "ι_E,X"

    def test_interior_factory_names_carry_bundle_tag(self, algebroid):
        X = Symbol("X")
        iota = algebroid.cartan.interior(X)
        assert isinstance(iota, InteriorProduct)
        assert iota.name == "ι_E,X"

    def test_factory_rejects_non_expr(self, algebroid):
        with pytest.raises(TypeError, match="Expr"):
            algebroid.cartan.lie_derivative("X")  # type: ignore[arg-type]

    def test_d_squared_zero_relation_builds(self, algebroid):
        """``d_E² = 0`` is an :class:`OperatorEquation` on the algebroid
        Cartan bundle, same relation API as the TM Cartan calculus.
        We don't run it through :meth:`verify` because the equation
        sides disagree on degree (``|d² | = 2`` vs ``|0| = 0``); the
        axiomatic rewrite path in
        :mod:`jacopy.calculus.exterior_d.apply_d_squared_zero` is how
        downstream code actually discharges it."""
        f = Symbol("f")
        reg = PropertyRegistry()
        reg.declare(f, Graded(degree=0))
        algebra = ExteriorAlgebra((f,))
        eq = algebroid.cartan.relation("d_squared_zero", algebra=algebra)
        assert eq.rhs == Integer(0)

    def test_cartan_magic_relation_builds(self, algebroid):
        """Magic-formula :class:`OperatorEquation` is buildable on the
        algebroid Cartan bundle, same API as the TM Cartan calculus."""
        X = Derivation("X", degree=0)
        eq = algebroid.cartan.relation("cartan_magic", X=X)
        # LHS is the commutator of d_E with ι_{E,X}; RHS is L_{E,X}.
        assert eq.rhs.name == "L_E,X"

    def test_cartan_magic_verify_closes_on_algebroid(self, algebroid):
        """The magic formula verifies as a :class:`ProofChain` on the
        algebroid bundle, ``L_{E,X}`` carries its own ``d_E`` and
        ``ι_E`` factory, so the expansion engine's Cartan rewrite lines
        up operator names on both sides and the residual collapses."""
        X = Symbol("X")
        f = Symbol("f")
        reg = PropertyRegistry()
        reg.declare(X, Graded(degree=0))
        reg.declare(f, Graded(degree=0))
        algebra = ExteriorAlgebra((f,), d=algebroid.d)
        chain = algebroid.cartan.verify(
            "cartan_magic", algebra=algebra, X=X, registry=reg
        )
        assert isinstance(chain, ProofChain)
        assert len(chain.steps) >= 1

    def test_d_squared_zero_verify_closes_on_algebroid(self, algebroid):
        """``d_E² = 0`` closes on the algebroid because
        :meth:`CartanCalculus.verify` now threads its own ``d`` into the
        default engine, ``DSquaredZeroDefinition`` gets pinned to
        ``d_E`` rather than silently to the TM default."""
        f = Symbol("f")
        reg = PropertyRegistry()
        reg.declare(f, Graded(degree=0))
        algebra = ExteriorAlgebra((f,), d=algebroid.d)
        chain = algebroid.cartan.verify(
            "d_squared_zero", algebra=algebra, registry=reg
        )
        assert isinstance(chain, ProofChain)

    def test_d_lie_verify_closes_on_algebroid(self, algebroid):
        """``[d_E, L_{E,X}] = 0`` on the algebroid."""
        X = Symbol("X")
        f = Symbol("f")
        reg = PropertyRegistry()
        reg.declare(X, Graded(degree=0))
        reg.declare(f, Graded(degree=0))
        algebra = ExteriorAlgebra((f,), d=algebroid.d)
        chain = algebroid.cartan.verify(
            "d_lie", algebra=algebra, X=X, registry=reg
        )
        assert isinstance(chain, ProofChain)

    def test_verify_all_closes_on_algebroid(self, algebroid):
        """Every Cartan relation closes on the algebroid bundle in one
        ``verify_all`` sweep, parity with the TM ``CartanCalculus``.
        Vector fields are declared as :class:`Derivation` instances so
        the generator-level Leibniz reductions (``ι_X(df) = X(f)``)
        fire; with plain :class:`Symbol` sections ``lie_lie`` and
        ``lie_iota`` leave a residual on *both* TM and algebroid sides,
        which is a universal limitation of the pairing rule rather
        than an algebroid-specific gap."""
        X = Derivation("X", degree=0)
        Y = Derivation("Y", degree=0)
        f = Symbol("f")
        reg = PropertyRegistry()
        reg.declare(f, Graded(degree=0))
        algebra = ExteriorAlgebra((f,), d=algebroid.d)
        results = algebroid.cartan.verify_all(
            algebra=algebra, X=X, Y=Y, registry=reg
        )
        assert set(results.keys()) == {
            "d_squared_zero",
            "cartan_magic",
            "d_lie",
            "lie_lie",
            "lie_iota",
        }
        for name, chain in results.items():
            assert isinstance(chain, ProofChain), name


# --------------------------------------------------------------------- #
# Anchor compatibility                                                   #
# --------------------------------------------------------------------- #


class TestAnchorCompatibilityObstruction:
    def test_returns_sum_expr(self, algebroid, registry):
        X, Y = Symbol("X"), Symbol("Y")
        ob = algebroid.anchor_compatibility_obstruction(X, Y, registry)
        assert isinstance(ob, Sum)

    def test_matches_helper_output(self, algebroid, registry):
        """The method is a thin forwarder to the calculus-level
        helper; equality on the same inputs is the direct check."""
        from jacopy.calculus.anchor import bracket_compatibility_obstruction
        X, Y = Symbol("X"), Symbol("Y")
        helper_ob = bracket_compatibility_obstruction(
            algebroid.anchor,
            algebroid.bracket,
            algebroid.vector_bracket,
            X, Y, registry,
        )
        assert algebroid.anchor_compatibility_obstruction(X, Y, registry) \
            == helper_ob

    def test_rejects_non_expr(self, algebroid, registry):
        with pytest.raises(TypeError, match="Expr"):
            algebroid.anchor_compatibility_obstruction("X", Symbol("Y"), registry)  # type: ignore[arg-type]


class TestAnchorCompatibilityCondition:
    def test_returns_vanishing_condition(self, algebroid, registry):
        X, Y = Symbol("X"), Symbol("Y")
        cond = algebroid.anchor_compatibility_condition(X, Y, registry)
        assert isinstance(cond, VanishingCondition)

    def test_name_mentions_compatibility(self, algebroid, registry):
        X, Y = Symbol("X"), Symbol("Y")
        cond = algebroid.anchor_compatibility_condition(X, Y, registry)
        assert "compatibility" in cond.name

    def test_obstruction_matches_direct_call(self, algebroid, registry):
        X, Y = Symbol("X"), Symbol("Y")
        cond = algebroid.anchor_compatibility_condition(X, Y, registry)
        assert cond.obstruction \
            == algebroid.anchor_compatibility_obstruction(X, Y, registry)


class TestProveAnchorCompatibility:
    def test_returns_proof_chain(self, algebroid, registry):
        X, Y = Symbol("X"), Symbol("Y")
        chain = algebroid.prove_anchor_compatibility(X, Y, registry=registry)
        assert isinstance(chain, ProofChain)

    def test_single_axiom_step(self, algebroid, registry):
        X, Y = Symbol("X"), Symbol("Y")
        chain = algebroid.prove_anchor_compatibility(X, Y, registry=registry)
        assert len(chain) == 1
        step = chain.steps[0]
        assert step.rule == "LieAlgebroidAnchorCompat"
        assert step.provenance_tag == "axiom"

    def test_step_discharges_to_zero(self, algebroid, registry):
        """The axiom step lands on :class:`Integer` ``0``, the whole
        point of citing the axiom is to close out the obstruction
        without running the ambient expansion pipeline."""
        X, Y = Symbol("X"), Symbol("Y")
        chain = algebroid.prove_anchor_compatibility(X, Y, registry=registry)
        assert chain.steps[0].after == Integer(0)

    def test_step_starts_on_obstruction(self, algebroid, registry):
        X, Y = Symbol("X"), Symbol("Y")
        chain = algebroid.prove_anchor_compatibility(X, Y, registry=registry)
        expected_start = algebroid.anchor_compatibility_obstruction(
            X, Y, registry
        )
        assert chain.steps[0].before == expected_start


# --------------------------------------------------------------------- #
# Seeded theorem                                                         #
# --------------------------------------------------------------------- #


class TestSeededAnchorCompatTheorem:
    def test_theorem_registered(self):
        assert "lie_algebroid_anchor_compat" in theorem_book
        assert (
            theorem_book.get("lie_algebroid_anchor_compat")
            is THEOREM_LIE_ALGEBROID_ANCHOR_COMPAT
        )

    def test_theorem_from_axioms_names_compatibility(self):
        thm = THEOREM_LIE_ALGEBROID_ANCHOR_COMPAT
        assert any("compatibility" in ax for ax in thm.from_axioms)

    def test_theorem_proof_is_single_axiom_step(self):
        thm = THEOREM_LIE_ALGEBROID_ANCHOR_COMPAT
        assert isinstance(thm.proof, ProofChain)
        assert len(thm.proof) == 1
        step = thm.proof.steps[0]
        assert step.rule == "LieAlgebroidAnchorCompat"
        assert step.provenance_tag == "axiom"
        assert step.after == Integer(0)
