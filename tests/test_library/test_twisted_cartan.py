"""Tests for ``jacopy.library.twisted_cartan``."""

from __future__ import annotations

import pytest

from jacopy.algebra.derivation import Derivation
from jacopy.brackets.base import GradedBracket
from jacopy.brackets.lie import LieBracket, lie
from jacopy.calculus.cartan import RELATIONS, CartanCalculus
from jacopy.calculus.exterior_algebra import ExteriorAlgebra
from jacopy.calculus.exterior_d import ExteriorDerivative
from jacopy.calculus.interior import InteriorProduct
from jacopy.calculus.lie_derivative import LieDerivative
from jacopy.core.expr import Symbol
from jacopy.core.properties import Graded
from jacopy.core.registry import PropertyRegistry
from jacopy.library import TwistedCartanBundle, twisted_cartan_bundle
from jacopy.proof.chain import ProofChain


# --------------------------------------------------------------------- #
# Fixtures                                                               #
# --------------------------------------------------------------------- #


@pytest.fixture
def H():
    return Symbol("H")


@pytest.fixture
def bundle(H):
    return TwistedCartanBundle(H)


@pytest.fixture
def context(bundle):
    reg = PropertyRegistry()
    H_sym = bundle.twist_form
    reg.declare(H_sym, Graded(degree=3))
    f = Symbol("f")
    reg.declare(f, Graded(degree=0))
    algebra = ExteriorAlgebra((f,), d=bundle.d)
    return reg, algebra, f


@pytest.fixture
def XY():
    X = Derivation("X", degree=0)
    Y = Derivation("Y", degree=0)
    return X, Y


# --------------------------------------------------------------------- #
# Construction                                                           #
# --------------------------------------------------------------------- #


class TestConstruction:
    def test_rejects_non_expr_twist_form(self):
        with pytest.raises(TypeError, match="twist_form must be an Expr"):
            TwistedCartanBundle("H")  # type: ignore[arg-type]

    def test_rejects_non_bracket_vector_bracket(self, H):
        with pytest.raises(TypeError, match="vector_bracket"):
            TwistedCartanBundle(H, vector_bracket="not a bracket")  # type: ignore[arg-type]

    def test_default_vector_bracket_is_lie_TM(self, H):
        bundle = TwistedCartanBundle(H)
        assert bundle.vector_bracket is lie

    def test_custom_vector_bracket_is_retained(self, H):
        lb = LieBracket(name="[·,·]_custom")
        bundle = TwistedCartanBundle(H, vector_bracket=lb)
        assert bundle.vector_bracket is lb

    def test_d_is_fresh_exterior_derivative(self, bundle):
        assert isinstance(bundle.d, ExteriorDerivative)
        assert bundle.d.degree == 1
        # Name carries the twist tag so transcripts stay distinguishable.
        assert "H" in bundle.d._repr_inner()

    def test_cartan_wires_bundle_d(self, bundle):
        assert isinstance(bundle.cartan, CartanCalculus)
        assert bundle.cartan.d is bundle.d

    def test_factory_alias_matches_class(self, H):
        a = twisted_cartan_bundle(H)
        b = TwistedCartanBundle(H)
        assert isinstance(a, TwistedCartanBundle)
        # twist_form is the same Expr; d_H is a fresh instance per call.
        assert a.twist_form == b.twist_form

    def test_repr_mentions_twist(self, bundle):
        r = repr(bundle)
        assert "TwistedCartanBundle" in r
        assert "H" in r

    def test_custom_name_is_honoured(self, H):
        bundle = TwistedCartanBundle(H, name="B-field bundle")
        assert bundle.name == "B-field bundle"

    def test_default_name_tags_twist(self, bundle):
        assert "H" in bundle.name


# --------------------------------------------------------------------- #
# Factories, L_{H,X} carries d_H in its bundle slot                      #
# --------------------------------------------------------------------- #


class TestFactories:
    def test_interior_factory_returns_interior_product(self, bundle):
        X = Symbol("X")
        iota_X = bundle.cartan.interior(X)
        assert isinstance(iota_X, InteriorProduct)
        assert iota_X.vector_field is X

    def test_lie_derivative_factory_plumbs_d_H(self, bundle):
        X = Symbol("X")
        L_X = bundle.cartan.lie_derivative(X)
        assert isinstance(L_X, LieDerivative)
        # Bundle slot threads d_H, not the default TM d.
        assert L_X.d is bundle.d
        assert L_X.iota_factory is not None

    def test_lie_derivative_factory_rejects_non_expr(self, bundle):
        with pytest.raises(TypeError, match="Expr"):
            bundle.cartan.lie_derivative("X")  # type: ignore[arg-type]


# --------------------------------------------------------------------- #
# Cartan relations close                                                 #
# --------------------------------------------------------------------- #


class TestCartanClosure:
    """Smoke tests, the five Cartan relations close on the twisted bundle.

    The bundle treats ``d_H`` as a fresh degree-+1 derivation with
    ``d_H² = 0``; instantiating a :class:`TwistedCartanBundle` is the
    act of asserting ``dH = 0``. With that single assumption baked in,
    agreement-on-generators closes every relation exactly as it does on
    the untwisted TM bundle.
    """

    def test_d_squared_zero_closes(self, bundle, context):
        reg, algebra, _ = context
        chain = bundle.cartan.verify(
            "d_squared_zero", algebra=algebra, registry=reg
        )
        assert isinstance(chain, ProofChain)

    def test_cartan_magic_closes(self, bundle, context, XY):
        reg, algebra, _ = context
        X, _ = XY
        chain = bundle.cartan.verify(
            "cartan_magic", algebra=algebra, X=X, registry=reg
        )
        assert isinstance(chain, ProofChain)

    def test_d_lie_closes(self, bundle, context, XY):
        reg, algebra, _ = context
        X, _ = XY
        chain = bundle.cartan.verify(
            "d_lie", algebra=algebra, X=X, registry=reg
        )
        assert isinstance(chain, ProofChain)

    def test_lie_lie_closes(self, bundle, context, XY):
        reg, algebra, _ = context
        X, Y = XY
        chain = bundle.cartan.verify(
            "lie_lie", algebra=algebra, X=X, Y=Y, registry=reg
        )
        assert isinstance(chain, ProofChain)

    def test_lie_iota_closes(self, bundle, context, XY):
        reg, algebra, _ = context
        X, Y = XY
        chain = bundle.cartan.verify(
            "lie_iota", algebra=algebra, X=X, Y=Y, registry=reg
        )
        assert isinstance(chain, ProofChain)

    def test_verify_all_closes_every_relation(self, bundle, context, XY):
        reg, algebra, _ = context
        X, Y = XY
        results = bundle.cartan.verify_all(
            algebra=algebra, X=X, Y=Y, registry=reg
        )
        assert set(results) == set(RELATIONS)
        for chain in results.values():
            assert isinstance(chain, ProofChain)


# --------------------------------------------------------------------- #
# Distinct twists don't collide                                          #
# --------------------------------------------------------------------- #


class TestIsolation:
    def test_two_bundles_have_distinct_d_instances(self):
        H1 = Symbol("H1")
        H2 = Symbol("H2")
        b1 = TwistedCartanBundle(H1)
        b2 = TwistedCartanBundle(H2)
        assert b1.d is not b2.d
        assert b1.d != b2.d

    def test_twisted_d_does_not_match_untwisted_d_squared_zero(self, bundle):
        """A twisted ``d_H`` is a separate instance from the TM default,
        so the TM-pinned ``DSquaredZeroDefinition(target=d)`` does not fire
        on ``d_H²``, the twisted bundle brings its own pin."""
        from jacopy.algebra.derivation import Act
        from jacopy.core.expr import Symbol as S
        from jacopy.proof.expansion import DSquaredZeroDefinition
        from jacopy.calculus.exterior_d import d as default_d

        f = S("f")
        tm_rule = DSquaredZeroDefinition(target=default_d)
        assert not tm_rule.matches(Act(bundle.d, Act(bundle.d, f)))
