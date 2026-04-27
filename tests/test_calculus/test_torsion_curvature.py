"""Tests for torsion / curvature tensors — Faz 16.B."""

from __future__ import annotations

import pytest

from jacopy.algebra.derivation import Derivation
from jacopy.algebra.lie_bracket_vf import LieBracketVF
from jacopy.calculus.connection import (
    AffineConnection,
    ConnectionEvalExpr,
    connection,
)
from jacopy.calculus.torsion_curvature import (
    Curvature,
    CurvatureDefinitionDefinition,
    Torsion,
    TorsionDefinitionDefinition,
)
from jacopy.core.expr import Neg, Sum


# --------------------------------------------------------------------- #
# Torsion node                                                            #
# --------------------------------------------------------------------- #


def test_torsion_children_and_key():
    nabla = connection()
    X = Derivation("X", 0)
    Y = Derivation("Y", 0)
    t = Torsion(nabla, X, Y)
    assert t.children == (X, Y)
    assert t.X is X
    assert t.Y is Y
    assert t.connection == nabla


def test_torsion_structural_equality():
    nabla = connection()
    other = connection("∇'")
    X = Derivation("X", 0)
    Y = Derivation("Y", 0)
    a = Torsion(nabla, X, Y)
    b = Torsion(nabla, X, Y)
    c = Torsion(nabla, Y, X)
    d = Torsion(other, X, Y)
    assert a == b
    assert hash(a) == hash(b)
    assert a != c
    assert a != d


def test_torsion_rebuild_preserves_connection():
    nabla = connection()
    X = Derivation("X", 0)
    Y = Derivation("Y", 0)
    Z = Derivation("Z", 0)
    t = Torsion(nabla, X, Y)
    rebuilt = t._rebuild((X, Z))
    assert isinstance(rebuilt, Torsion)
    assert rebuilt.connection == nabla
    assert rebuilt.children == (X, Z)
    with pytest.raises(ValueError):
        t._rebuild((X,))


def test_torsion_walks_into_slots():
    nabla = connection()
    X = Derivation("X", 0)
    Y = Derivation("Y", 0)
    t = Torsion(nabla, X, Y)
    walked = list(t.walk())
    assert t in walked
    assert X in walked
    assert Y in walked


def test_torsion_type_errors():
    nabla = connection()
    X = Derivation("X", 0)
    with pytest.raises(TypeError):
        Torsion("not-a-connection", X, X)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        Torsion(nabla, "not-an-expr", X)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        Torsion(nabla, X, "not-an-expr")  # type: ignore[arg-type]


# --------------------------------------------------------------------- #
# Curvature node                                                          #
# --------------------------------------------------------------------- #


def test_curvature_children_and_key():
    nabla = connection()
    X = Derivation("X", 0)
    Y = Derivation("Y", 0)
    Z = Derivation("Z", 0)
    r = Curvature(nabla, X, Y, Z)
    assert r.children == (X, Y, Z)
    assert r.X is X
    assert r.Y is Y
    assert r.Z is Z
    assert r.connection == nabla


def test_curvature_structural_equality():
    nabla = connection()
    other = connection("∇'")
    X = Derivation("X", 0)
    Y = Derivation("Y", 0)
    Z = Derivation("Z", 0)
    a = Curvature(nabla, X, Y, Z)
    b = Curvature(nabla, X, Y, Z)
    c = Curvature(nabla, Y, X, Z)
    d = Curvature(other, X, Y, Z)
    assert a == b
    assert hash(a) == hash(b)
    assert a != c
    assert a != d


def test_curvature_rebuild_preserves_connection():
    nabla = connection()
    X = Derivation("X", 0)
    Y = Derivation("Y", 0)
    Z = Derivation("Z", 0)
    W = Derivation("W", 0)
    r = Curvature(nabla, X, Y, Z)
    rebuilt = r._rebuild((X, Y, W))
    assert isinstance(rebuilt, Curvature)
    assert rebuilt.connection == nabla
    assert rebuilt.children == (X, Y, W)
    with pytest.raises(ValueError):
        r._rebuild((X, Y))


def test_curvature_walks_into_slots():
    nabla = connection()
    X = Derivation("X", 0)
    Y = Derivation("Y", 0)
    Z = Derivation("Z", 0)
    r = Curvature(nabla, X, Y, Z)
    walked = list(r.walk())
    assert r in walked
    assert X in walked
    assert Y in walked
    assert Z in walked


def test_curvature_type_errors():
    nabla = connection()
    X = Derivation("X", 0)
    with pytest.raises(TypeError):
        Curvature("not-a-connection", X, X, X)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        Curvature(nabla, "x", X, X)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        Curvature(nabla, X, "y", X)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        Curvature(nabla, X, X, "z")  # type: ignore[arg-type]


# --------------------------------------------------------------------- #
# Torsion definition axiom                                                #
# --------------------------------------------------------------------- #


def test_torsion_definition_rewrites():
    nabla = connection()
    X = Derivation("X", 0)
    Y = Derivation("Y", 0)
    rule = TorsionDefinitionDefinition(nabla)
    t = Torsion(nabla, X, Y)
    assert rule.matches(t)
    rhs = rule.rewrite(t)
    expected = Sum.make(
        ConnectionEvalExpr(nabla, X, Y),
        Neg(ConnectionEvalExpr(nabla, Y, X)),
        Neg(LieBracketVF(X, Y)),
    )
    assert rhs == expected


def test_torsion_definition_scoped_to_connection():
    a = connection("∇1")
    b = connection("∇2")
    X = Derivation("X", 0)
    Y = Derivation("Y", 0)
    rule_a = TorsionDefinitionDefinition(a)
    t = Torsion(b, X, Y)
    assert not rule_a.matches(t)


def test_torsion_definition_does_not_match_random_expr():
    nabla = connection()
    X = Derivation("X", 0)
    Y = Derivation("Y", 0)
    rule = TorsionDefinitionDefinition(nabla)
    assert not rule.matches(ConnectionEvalExpr(nabla, X, Y))


def test_torsion_definition_rejects_non_connection():
    with pytest.raises(TypeError):
        TorsionDefinitionDefinition("not-a-connection")  # type: ignore[arg-type]


# --------------------------------------------------------------------- #
# Curvature definition axiom                                              #
# --------------------------------------------------------------------- #


def test_curvature_definition_rewrites():
    nabla = connection()
    X = Derivation("X", 0)
    Y = Derivation("Y", 0)
    Z = Derivation("Z", 0)
    rule = CurvatureDefinitionDefinition(nabla)
    r = Curvature(nabla, X, Y, Z)
    assert rule.matches(r)
    rhs = rule.rewrite(r)
    expected = Sum.make(
        ConnectionEvalExpr(nabla, X, ConnectionEvalExpr(nabla, Y, Z)),
        Neg(ConnectionEvalExpr(nabla, Y, ConnectionEvalExpr(nabla, X, Z))),
        Neg(ConnectionEvalExpr(nabla, LieBracketVF(X, Y), Z)),
    )
    assert rhs == expected


def test_curvature_definition_scoped_to_connection():
    a = connection("∇1")
    b = connection("∇2")
    X = Derivation("X", 0)
    Y = Derivation("Y", 0)
    Z = Derivation("Z", 0)
    rule_a = CurvatureDefinitionDefinition(a)
    r = Curvature(b, X, Y, Z)
    assert not rule_a.matches(r)


def test_curvature_definition_rejects_non_connection():
    with pytest.raises(TypeError):
        CurvatureDefinitionDefinition("not-a-connection")  # type: ignore[arg-type]


# --------------------------------------------------------------------- #
# Engine integration                                                      #
# --------------------------------------------------------------------- #


def test_engine_expands_torsion():
    """``T(∇)(X, Y)`` rewrites to its 3-term definition under the engine."""
    from jacopy.proof.expansion import ExpansionEngine

    nabla = connection()
    X = Derivation("X", 0)
    Y = Derivation("Y", 0)
    engine = ExpansionEngine([TorsionDefinitionDefinition(nabla)])
    final, steps = engine.expand(Torsion(nabla, X, Y))
    expected = Sum.make(
        ConnectionEvalExpr(nabla, X, Y),
        Neg(ConnectionEvalExpr(nabla, Y, X)),
        Neg(LieBracketVF(X, Y)),
    )
    assert final == expected
    assert len(steps) == 1


def test_engine_expands_curvature():
    """``R(∇)(X, Y) Z`` rewrites to its 3-term commutator definition."""
    from jacopy.proof.expansion import ExpansionEngine

    nabla = connection()
    X = Derivation("X", 0)
    Y = Derivation("Y", 0)
    Z = Derivation("Z", 0)
    engine = ExpansionEngine([CurvatureDefinitionDefinition(nabla)])
    final, steps = engine.expand(Curvature(nabla, X, Y, Z))
    expected = Sum.make(
        ConnectionEvalExpr(nabla, X, ConnectionEvalExpr(nabla, Y, Z)),
        Neg(ConnectionEvalExpr(nabla, Y, ConnectionEvalExpr(nabla, X, Z))),
        Neg(ConnectionEvalExpr(nabla, LieBracketVF(X, Y), Z)),
    )
    assert final == expected
    assert len(steps) == 1
