"""Tests for ∇-on-tensor Leibniz rules, Faz 16.C."""

from __future__ import annotations

import pytest

from jacopy.algebra.derivation import Derivation
from jacopy.calculus.connection import ConnectionEvalExpr, connection
from jacopy.calculus.torsion_curvature import (
    Curvature,
    CurvatureCovariantDerivative,
    CurvatureCovariantDerivativeDefinition,
    Torsion,
    TorsionCovariantDerivative,
    TorsionCovariantDerivativeDefinition,
)
from jacopy.core.expr import Neg, Sum


# --------------------------------------------------------------------- #
# TorsionCovariantDerivative node                                         #
# --------------------------------------------------------------------- #


def test_torsion_cov_deriv_children_and_key():
    nabla = connection()
    U = Derivation("U", 0)
    V = Derivation("V", 0)
    W = Derivation("W", 0)
    e = TorsionCovariantDerivative(nabla, U, V, W)
    assert e.children == (U, V, W)
    assert e.U is U
    assert e.V is V
    assert e.W is W
    assert e.connection == nabla


def test_torsion_cov_deriv_structural_equality():
    nabla = connection()
    other = connection("∇'")
    U = Derivation("U", 0)
    V = Derivation("V", 0)
    W = Derivation("W", 0)
    a = TorsionCovariantDerivative(nabla, U, V, W)
    b = TorsionCovariantDerivative(nabla, U, V, W)
    c = TorsionCovariantDerivative(nabla, V, U, W)
    d = TorsionCovariantDerivative(other, U, V, W)
    assert a == b
    assert hash(a) == hash(b)
    assert a != c
    assert a != d


def test_torsion_cov_deriv_rebuild_preserves_connection():
    nabla = connection()
    U = Derivation("U", 0)
    V = Derivation("V", 0)
    W = Derivation("W", 0)
    X = Derivation("X", 0)
    e = TorsionCovariantDerivative(nabla, U, V, W)
    rebuilt = e._rebuild((U, V, X))
    assert isinstance(rebuilt, TorsionCovariantDerivative)
    assert rebuilt.connection == nabla
    assert rebuilt.children == (U, V, X)
    with pytest.raises(ValueError):
        e._rebuild((U, V))


def test_torsion_cov_deriv_walks_into_slots():
    nabla = connection()
    U = Derivation("U", 0)
    V = Derivation("V", 0)
    W = Derivation("W", 0)
    e = TorsionCovariantDerivative(nabla, U, V, W)
    walked = list(e.walk())
    assert e in walked
    assert U in walked
    assert V in walked
    assert W in walked


def test_torsion_cov_deriv_type_errors():
    nabla = connection()
    U = Derivation("U", 0)
    with pytest.raises(TypeError):
        TorsionCovariantDerivative("not-a-conn", U, U, U)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        TorsionCovariantDerivative(nabla, "u", U, U)  # type: ignore[arg-type]


# --------------------------------------------------------------------- #
# CurvatureCovariantDerivative node                                      #
# --------------------------------------------------------------------- #


def test_curvature_cov_deriv_children_and_key():
    nabla = connection()
    U = Derivation("U", 0)
    V = Derivation("V", 0)
    W = Derivation("W", 0)
    Z = Derivation("Z", 0)
    e = CurvatureCovariantDerivative(nabla, U, V, W, Z)
    assert e.children == (U, V, W, Z)
    assert e.U is U
    assert e.V is V
    assert e.W is W
    assert e.Z is Z
    assert e.connection == nabla


def test_curvature_cov_deriv_structural_equality():
    nabla = connection()
    other = connection("∇'")
    U = Derivation("U", 0)
    V = Derivation("V", 0)
    W = Derivation("W", 0)
    Z = Derivation("Z", 0)
    a = CurvatureCovariantDerivative(nabla, U, V, W, Z)
    b = CurvatureCovariantDerivative(nabla, U, V, W, Z)
    c = CurvatureCovariantDerivative(nabla, V, U, W, Z)
    d = CurvatureCovariantDerivative(other, U, V, W, Z)
    assert a == b
    assert hash(a) == hash(b)
    assert a != c
    assert a != d


def test_curvature_cov_deriv_rebuild_preserves_connection():
    nabla = connection()
    U = Derivation("U", 0)
    V = Derivation("V", 0)
    W = Derivation("W", 0)
    Z = Derivation("Z", 0)
    X = Derivation("X", 0)
    e = CurvatureCovariantDerivative(nabla, U, V, W, Z)
    rebuilt = e._rebuild((U, V, W, X))
    assert isinstance(rebuilt, CurvatureCovariantDerivative)
    assert rebuilt.connection == nabla
    assert rebuilt.children == (U, V, W, X)
    with pytest.raises(ValueError):
        e._rebuild((U, V, W))


def test_curvature_cov_deriv_type_errors():
    nabla = connection()
    U = Derivation("U", 0)
    with pytest.raises(TypeError):
        CurvatureCovariantDerivative("not-a-conn", U, U, U, U)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        CurvatureCovariantDerivative(nabla, U, U, U, "z")  # type: ignore[arg-type]


# --------------------------------------------------------------------- #
# Torsion-Leibniz axiom                                                  #
# --------------------------------------------------------------------- #


def test_torsion_cov_deriv_definition_rewrites():
    nabla = connection()
    U = Derivation("U", 0)
    V = Derivation("V", 0)
    W = Derivation("W", 0)
    rule = TorsionCovariantDerivativeDefinition(nabla)
    e = TorsionCovariantDerivative(nabla, U, V, W)
    assert rule.matches(e)
    rhs = rule.rewrite(e)
    expected = Sum.make(
        ConnectionEvalExpr(nabla, U, Torsion(nabla, V, W)),
        Neg(Torsion(nabla, ConnectionEvalExpr(nabla, U, V), W)),
        Neg(Torsion(nabla, V, ConnectionEvalExpr(nabla, U, W))),
    )
    assert rhs == expected


def test_torsion_cov_deriv_definition_scoped_to_connection():
    a = connection("∇1")
    b = connection("∇2")
    U = Derivation("U", 0)
    V = Derivation("V", 0)
    W = Derivation("W", 0)
    rule_a = TorsionCovariantDerivativeDefinition(a)
    e = TorsionCovariantDerivative(b, U, V, W)
    assert not rule_a.matches(e)


def test_torsion_cov_deriv_definition_does_not_match_torsion():
    nabla = connection()
    U = Derivation("U", 0)
    V = Derivation("V", 0)
    rule = TorsionCovariantDerivativeDefinition(nabla)
    assert not rule.matches(Torsion(nabla, U, V))


def test_torsion_cov_deriv_definition_rejects_non_connection():
    with pytest.raises(TypeError):
        TorsionCovariantDerivativeDefinition("not-a-conn")  # type: ignore[arg-type]


# --------------------------------------------------------------------- #
# Curvature-Leibniz axiom                                                #
# --------------------------------------------------------------------- #


def test_curvature_cov_deriv_definition_rewrites():
    nabla = connection()
    U = Derivation("U", 0)
    V = Derivation("V", 0)
    W = Derivation("W", 0)
    Z = Derivation("Z", 0)
    rule = CurvatureCovariantDerivativeDefinition(nabla)
    e = CurvatureCovariantDerivative(nabla, U, V, W, Z)
    assert rule.matches(e)
    rhs = rule.rewrite(e)
    expected = Sum.make(
        ConnectionEvalExpr(nabla, U, Curvature(nabla, V, W, Z)),
        Neg(Curvature(nabla, ConnectionEvalExpr(nabla, U, V), W, Z)),
        Neg(Curvature(nabla, V, ConnectionEvalExpr(nabla, U, W), Z)),
        Neg(Curvature(nabla, V, W, ConnectionEvalExpr(nabla, U, Z))),
    )
    assert rhs == expected


def test_curvature_cov_deriv_definition_scoped_to_connection():
    a = connection("∇1")
    b = connection("∇2")
    U = Derivation("U", 0)
    V = Derivation("V", 0)
    W = Derivation("W", 0)
    Z = Derivation("Z", 0)
    rule_a = CurvatureCovariantDerivativeDefinition(a)
    e = CurvatureCovariantDerivative(b, U, V, W, Z)
    assert not rule_a.matches(e)


def test_curvature_cov_deriv_definition_does_not_match_curvature():
    nabla = connection()
    U = Derivation("U", 0)
    V = Derivation("V", 0)
    W = Derivation("W", 0)
    rule = CurvatureCovariantDerivativeDefinition(nabla)
    assert not rule.matches(Curvature(nabla, U, V, W))


def test_curvature_cov_deriv_definition_rejects_non_connection():
    with pytest.raises(TypeError):
        CurvatureCovariantDerivativeDefinition("not-a-conn")  # type: ignore[arg-type]


# --------------------------------------------------------------------- #
# Engine integration                                                     #
# --------------------------------------------------------------------- #


def test_engine_expands_torsion_cov_deriv():
    """Single rule fires, producing the 3-term tensor-Leibniz expansion."""
    from jacopy.proof.expansion import ExpansionEngine

    nabla = connection()
    U = Derivation("U", 0)
    V = Derivation("V", 0)
    W = Derivation("W", 0)
    engine = ExpansionEngine([TorsionCovariantDerivativeDefinition(nabla)])
    final, steps = engine.expand(TorsionCovariantDerivative(nabla, U, V, W))
    expected = Sum.make(
        ConnectionEvalExpr(nabla, U, Torsion(nabla, V, W)),
        Neg(Torsion(nabla, ConnectionEvalExpr(nabla, U, V), W)),
        Neg(Torsion(nabla, V, ConnectionEvalExpr(nabla, U, W))),
    )
    assert final == expected
    assert len(steps) == 1


def test_engine_expands_curvature_cov_deriv():
    """Single rule fires, producing the 4-term tensor-Leibniz expansion."""
    from jacopy.proof.expansion import ExpansionEngine

    nabla = connection()
    U = Derivation("U", 0)
    V = Derivation("V", 0)
    W = Derivation("W", 0)
    Z = Derivation("Z", 0)
    engine = ExpansionEngine(
        [CurvatureCovariantDerivativeDefinition(nabla)]
    )
    final, steps = engine.expand(
        CurvatureCovariantDerivative(nabla, U, V, W, Z)
    )
    expected = Sum.make(
        ConnectionEvalExpr(nabla, U, Curvature(nabla, V, W, Z)),
        Neg(Curvature(nabla, ConnectionEvalExpr(nabla, U, V), W, Z)),
        Neg(Curvature(nabla, V, ConnectionEvalExpr(nabla, U, W), Z)),
        Neg(Curvature(nabla, V, W, ConnectionEvalExpr(nabla, U, Z))),
    )
    assert final == expected
    assert len(steps) == 1


def test_engine_expands_torsion_cov_deriv_then_torsion_definitions():
    """Composing tensor-Leibniz with the Torsion-definition rule keeps
    walking, the inner ``T(V,W)``, ``T(∇_U V, W)``, ``T(V, ∇_U W)`` all
    further unfold into ∇-commutator + LBVF terms."""
    from jacopy.proof.expansion import ExpansionEngine
    from jacopy.calculus.torsion_curvature import (
        TorsionDefinitionDefinition,
    )

    nabla = connection()
    U = Derivation("U", 0)
    V = Derivation("V", 0)
    W = Derivation("W", 0)
    engine = ExpansionEngine(
        [
            TorsionCovariantDerivativeDefinition(nabla),
            TorsionDefinitionDefinition(nabla),
        ]
    )
    final, steps = engine.expand(TorsionCovariantDerivative(nabla, U, V, W))
    # No Torsion / TorsionCovariantDerivative nodes should remain.
    leftovers = list(
        final.find(
            lambda n: isinstance(n, (Torsion, TorsionCovariantDerivative))
        )
    )
    assert leftovers == []
    # Steps: one for the tensor-Leibniz rule, then one per leftover Torsion.
    assert len(steps) >= 2
