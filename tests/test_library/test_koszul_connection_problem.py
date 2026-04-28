"""Tests for the Q9 Stage 9.F :class:`KoszulConnectionProblem` umbrella."""

from __future__ import annotations

import pytest

from jacopy.algebra.derivation import Derivation
from jacopy.brackets.koszul import KoszulBracket
from jacopy.calculus.anchor import Anchor
from jacopy.calculus.connection import connection, koszul_connection
from jacopy.calculus.local_frame import local_frame
from jacopy.calculus.metric import metric as metric_factory
from jacopy.core.expr import Symbol, Zero
from jacopy.library.bianchi_problem import BianchiProblem
from jacopy.library.cartan_form_property import CartanFormPropertyProblem
from jacopy.library.cartan_structure import CartanStructureProblem
from jacopy.library.koszul_connection_problem import KoszulConnectionProblem


# --------------------------------------------------------------------- #
# Fixtures                                                               #
# --------------------------------------------------------------------- #


def _problem():
    anchor = Anchor(name="ρ")
    bracket = KoszulBracket(anchor)
    nabla = koszul_connection("∇̃", anchor=anchor, bracket=bracket)
    F = local_frame("F̃", dim=3)
    return KoszulConnectionProblem(nabla, F)


def _problem_with_metric():
    anchor = Anchor(name="ρ")
    bracket = KoszulBracket(anchor)
    nabla = koszul_connection("∇̃", anchor=anchor, bracket=bracket)
    F = local_frame("F̃", dim=3)
    g = metric_factory("g̃")
    return KoszulConnectionProblem(nabla, F, metric=g)


# --------------------------------------------------------------------- #
# Construction / validation                                              #
# --------------------------------------------------------------------- #


class TestConstruction:
    def test_accepts_koszul_connection(self):
        P = _problem()
        assert P.connection.bracket is not None
        assert isinstance(P.frame.name, str)

    def test_rejects_bracket_free_connection(self):
        nabla = connection("∇")
        F = local_frame("F", dim=3)
        with pytest.raises(ValueError):
            KoszulConnectionProblem(nabla, F)

    def test_rejects_non_connection(self):
        F = local_frame("F", dim=3)
        with pytest.raises(TypeError):
            KoszulConnectionProblem("nabla", F)  # type: ignore[arg-type]

    def test_rejects_non_frame(self):
        anchor = Anchor(name="ρ")
        bracket = KoszulBracket(anchor)
        nabla = koszul_connection("∇̃", anchor=anchor, bracket=bracket)
        with pytest.raises(TypeError):
            KoszulConnectionProblem(nabla, "F")  # type: ignore[arg-type]

    def test_default_name_includes_connection_and_frame(self):
        P = _problem()
        assert "∇̃" in P.name and "F̃" in P.name

    def test_custom_name_overrides_default(self):
        anchor = Anchor(name="ρ")
        bracket = KoszulBracket(anchor)
        nabla = koszul_connection("∇̃", anchor=anchor, bracket=bracket)
        F = local_frame("F̃", dim=3)
        P = KoszulConnectionProblem(nabla, F, name="custom")
        assert P.name == "custom"

    def test_metric_defaults_to_none(self):
        P = _problem()
        assert P.metric is None

    def test_metric_kwarg_is_stored(self):
        P = _problem_with_metric()
        assert P.metric is not None
        assert P.metric.name == "g̃"

    def test_rejects_non_metric(self):
        anchor = Anchor(name="ρ")
        bracket = KoszulBracket(anchor)
        nabla = koszul_connection("∇̃", anchor=anchor, bracket=bracket)
        F = local_frame("F̃", dim=3)
        with pytest.raises(TypeError):
            KoszulConnectionProblem(nabla, F, metric="g")  # type: ignore[arg-type]

    def test_metric_forwarded_to_form_property_facet(self):
        P = _problem_with_metric()
        assert P.form_property._metric is P.metric


# --------------------------------------------------------------------- #
# Lazy facets                                                            #
# --------------------------------------------------------------------- #


class TestFacetTypes:
    def test_bianchi_facet_is_a_bianchi_problem(self):
        P = _problem()
        assert isinstance(P.bianchi, BianchiProblem)

    def test_form_property_facet_is_a_cartan_form_property_problem(self):
        P = _problem()
        assert isinstance(P.form_property, CartanFormPropertyProblem)

    def test_cartan_structure_facet_is_a_cartan_structure_problem(self):
        P = _problem()
        assert isinstance(P.cartan_structure, CartanStructureProblem)

    def test_facets_are_cached(self):
        P = _problem()
        assert P.bianchi is P.bianchi
        assert P.form_property is P.form_property
        assert P.cartan_structure is P.cartan_structure


# --------------------------------------------------------------------- #
# End-to-end mechanised proofs through the umbrella                      #
# --------------------------------------------------------------------- #


class TestMechanisedProofsClose:
    """The capstone test: every facet's headline proof closes through
    the umbrella, with no per-facet wiring required at the call site.
    """

    def test_bianchi_first_closes(self):
        P = _problem()
        U, V, W = Symbol("U"), Symbol("V"), Symbol("W")
        result = P.bianchi.prove_first_bianchi(U, V, W)
        assert result.ok

    def test_bianchi_second_closes(self):
        P = _problem()
        U, V, W, Z = Symbol("U"), Symbol("V"), Symbol("W"), Symbol("Z")
        result = P.bianchi.prove_second_bianchi(U, V, W, Z)
        assert result.ok

    def test_form_property_T_antisymmetric_closes(self):
        P = _problem()
        U, V = Symbol("U"), Symbol("V")
        assert P.form_property.prove_T_antisymmetric("a", U, V).ok

    def test_form_property_R_bilinearity_closes(self):
        P = _problem()
        f, U, V = Symbol("f"), Symbol("U"), Symbol("V")
        assert P.form_property.prove_R_scalar_linear_in_first(
            "a", "b", f, U, V
        ).ok

    def test_first_cartan_closes(self):
        P = _problem()
        U, V = Derivation("U", 0), Derivation("V", 0)
        result = P.cartan_structure.prove_first_cartan(U, V, "a")
        assert result.ok
        assert result.final == Zero

    def test_second_cartan_closes(self):
        P = _problem()
        U, V = Derivation("U", 0), Derivation("V", 0)
        result = P.cartan_structure.prove_second_cartan(U, V, "a", "b")
        assert result.ok
        assert result.final == Zero

    def test_form_property_Q_scalar_linear_closes_with_metric(self):
        """Q̃_{ab} V-scalar-linear closes once a metric is supplied."""
        P = _problem_with_metric()
        f, V = Symbol("f"), Symbol("V")
        result = P.form_property.prove_Q_scalar_linear_in_V("a", "b", f, V)
        assert result.ok
        assert result.final == Zero

    def test_form_property_Q_additive_closes_with_metric(self):
        """Q̃_{ab} V-additive closes once a metric is supplied."""
        P = _problem_with_metric()
        V1, V2 = Symbol("V1"), Symbol("V2")
        result = P.form_property.prove_Q_additive_in_V("a", "b", V1, V2)
        assert result.ok
        assert result.final == Zero
