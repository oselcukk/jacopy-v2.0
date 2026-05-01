"""Tests for torsion / curvature tensors, Faz 16.B."""

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
    CurvatureXLinearityDefinition,
    CurvatureXScalarPullDefinition,
    CurvatureXYAntiSymmetryDefinition,
    CurvatureYLinearityDefinition,
    CurvatureYScalarPullDefinition,
    Torsion,
    TorsionAntiSymmetryDefinition,
    TorsionDefinitionDefinition,
    TorsionXLinearityDefinition,
    TorsionXScalarPullDefinition,
    TorsionYLinearityDefinition,
    TorsionYScalarPullDefinition,
)
from jacopy.core.expr import Neg, Product, Sum, Symbol


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


# --------------------------------------------------------------------- #
# Torsion C∞-bilinearity + antisymmetry, Faz 17.D                       #
# --------------------------------------------------------------------- #


def test_torsion_x_linearity_distributes_sum():
    nabla = connection("∇")
    rule = TorsionXLinearityDefinition(nabla)
    A, B, Y = Symbol("A"), Symbol("B"), Symbol("Y")
    e = Torsion(nabla, Sum.make(A, B), Y)
    assert rule.matches(e)
    out = rule.rewrite(e)
    assert isinstance(out, Sum)
    assert Torsion(nabla, A, Y) in out.children
    assert Torsion(nabla, B, Y) in out.children


def test_torsion_x_linearity_distributes_neg():
    nabla = connection("∇")
    rule = TorsionXLinearityDefinition(nabla)
    A, Y = Symbol("A"), Symbol("Y")
    e = Torsion(nabla, Neg(A), Y)
    out = rule.rewrite(e)
    assert isinstance(out, Neg)
    assert out.arg == Torsion(nabla, A, Y)


def test_torsion_y_linearity_distributes_sum():
    nabla = connection("∇")
    rule = TorsionYLinearityDefinition(nabla)
    X, A, B = Symbol("X"), Symbol("A"), Symbol("B")
    e = Torsion(nabla, X, Sum.make(A, B))
    out = rule.rewrite(e)
    assert isinstance(out, Sum)


def test_torsion_x_scalar_pull_pulls_two_factor_product():
    nabla = connection("∇")
    rule = TorsionXScalarPullDefinition(nabla)
    f, X, Y = Symbol("f"), Symbol("X"), Symbol("Y")
    e = Torsion(nabla, Product(f, X), Y)
    assert rule.matches(e)
    out = rule.rewrite(e)
    assert isinstance(out, Product)
    inner = out.children[1]
    assert isinstance(inner, Torsion)
    assert inner.X == X and inner.Y == Y


def test_torsion_y_scalar_pull_pulls_two_factor_product():
    nabla = connection("∇")
    rule = TorsionYScalarPullDefinition(nabla)
    f, X, Y = Symbol("f"), Symbol("X"), Symbol("Y")
    e = Torsion(nabla, X, Product(f, Y))
    out = rule.rewrite(e)
    assert isinstance(out, Product)
    inner = out.children[1]
    assert isinstance(inner, Torsion)
    assert inner.Y == Y


def test_torsion_x_scalar_pull_does_not_match_singleton():
    nabla = connection("∇")
    rule = TorsionXScalarPullDefinition(nabla)
    X, Y = Symbol("X"), Symbol("Y")
    assert not rule.matches(Torsion(nabla, X, Y))


def test_torsion_antisymmetry_canonicalizes_pair():
    nabla = connection("∇")
    rule = TorsionAntiSymmetryDefinition(nabla)
    # repr-canonicalize: pick names so repr(X) > repr(Y).
    Y = Symbol("a")
    X = Symbol("z")
    e = Torsion(nabla, X, Y)
    assert rule.matches(e)
    out = rule.rewrite(e)
    assert isinstance(out, Neg)
    assert out.arg == Torsion(nabla, Y, X)


def test_torsion_antisymmetry_does_not_loop():
    """Already in canonical order, rule should not match."""
    nabla = connection("∇")
    rule = TorsionAntiSymmetryDefinition(nabla)
    Y = Symbol("z")
    X = Symbol("a")
    e = Torsion(nabla, X, Y)
    assert not rule.matches(e)


def test_torsion_axioms_scoped_to_specific_connection():
    n1 = connection("∇1")
    n2 = connection("∇2")
    rule = TorsionXLinearityDefinition(n1)
    A, B, Y = Symbol("A"), Symbol("B"), Symbol("Y")
    assert not rule.matches(Torsion(n2, Sum.make(A, B), Y))


# --------------------------------------------------------------------- #
# Curvature C∞-bilinearity + antisymmetry, Faz 17.D                     #
# --------------------------------------------------------------------- #


def test_curvature_x_linearity_distributes_sum():
    nabla = connection("∇")
    rule = CurvatureXLinearityDefinition(nabla)
    A, B, Y, Z = Symbol("A"), Symbol("B"), Symbol("Y"), Symbol("Z")
    e = Curvature(nabla, Sum.make(A, B), Y, Z)
    out = rule.rewrite(e)
    assert isinstance(out, Sum)
    assert Curvature(nabla, A, Y, Z) in out.children


def test_curvature_y_linearity_distributes_neg():
    nabla = connection("∇")
    rule = CurvatureYLinearityDefinition(nabla)
    X, A, Z = Symbol("X"), Symbol("A"), Symbol("Z")
    e = Curvature(nabla, X, Neg(A), Z)
    out = rule.rewrite(e)
    assert isinstance(out, Neg)


def test_curvature_x_scalar_pull():
    nabla = connection("∇")
    rule = CurvatureXScalarPullDefinition(nabla)
    f, X, Y, Z = Symbol("f"), Symbol("X"), Symbol("Y"), Symbol("Z")
    e = Curvature(nabla, Product(f, X), Y, Z)
    out = rule.rewrite(e)
    assert isinstance(out, Product)
    inner = out.children[1]
    assert isinstance(inner, Curvature)
    assert inner.X == X


def test_curvature_y_scalar_pull():
    nabla = connection("∇")
    rule = CurvatureYScalarPullDefinition(nabla)
    f, X, Y, Z = Symbol("f"), Symbol("X"), Symbol("Y"), Symbol("Z")
    e = Curvature(nabla, X, Product(f, Y), Z)
    out = rule.rewrite(e)
    assert isinstance(out, Product)
    inner = out.children[1]
    assert isinstance(inner, Curvature)
    assert inner.Y == Y


def test_curvature_xy_antisymmetry_canonicalizes_first_two_slots():
    nabla = connection("∇")
    rule = CurvatureXYAntiSymmetryDefinition(nabla)
    Y = Symbol("a")
    X = Symbol("z")
    Z = Symbol("c")
    e = Curvature(nabla, X, Y, Z)
    assert rule.matches(e)
    out = rule.rewrite(e)
    assert isinstance(out, Neg)
    inner = out.arg
    assert isinstance(inner, Curvature)
    # X / Y swapped, Z untouched.
    assert inner.X == Y and inner.Y == X and inner.Z == Z


def test_curvature_xy_antisymmetry_leaves_z_slot_untouched():
    """Even with Z out of repr-order, only (X, Y) swap is considered."""
    nabla = connection("∇")
    rule = CurvatureXYAntiSymmetryDefinition(nabla)
    X = Symbol("a")
    Y = Symbol("b")
    Z = Symbol("z")  # out of order but Z slot doesn't trigger
    e = Curvature(nabla, X, Y, Z)
    # X < Y in repr order, rule should not match.
    assert not rule.matches(e)


def test_curvature_axioms_scoped_to_specific_connection():
    n1 = connection("∇1")
    n2 = connection("∇2")
    rule = CurvatureXScalarPullDefinition(n1)
    f, X, Y, Z = Symbol("f"), Symbol("X"), Symbol("Y"), Symbol("Z")
    e = Curvature(n2, Product(f, X), Y, Z)
    assert not rule.matches(e)


# --------------------------------------------------------------------- #
# Bracket-parametrized Torsion / Curvature, Q9 / Math 595 algebroid    #
# --------------------------------------------------------------------- #


class TestBracketParametrization:
    """``T(∇̃)(ω, η)`` and ``R(∇̃)(ω, η) ζ`` over a Koszul-bracket connection."""

    def test_torsion_default_uses_lie_bracket_vf(self):
        nabla = connection()
        rule = TorsionDefinitionDefinition(nabla)
        X = Derivation("X", 0)
        Y = Derivation("Y", 0)
        out = rule.rewrite(Torsion(nabla, X, Y))
        # Third term must wrap a LieBracketVF.
        assert isinstance(out, Sum)
        third = out.children[2]
        assert isinstance(third, Neg)
        assert isinstance(third.arg, LieBracketVF)

    def test_torsion_custom_bracket_uses_bracket_apply(self):
        from jacopy.brackets.base import BracketApply
        from jacopy.brackets.koszul import KoszulBracket
        from jacopy.calculus.anchor import Anchor
        from jacopy.calculus.connection import koszul_connection

        nabla = koszul_connection()
        rule = TorsionDefinitionDefinition(nabla)
        omega = Symbol("ω")
        eta = Symbol("η")
        out = rule.rewrite(Torsion(nabla, omega, eta))
        assert isinstance(out, Sum)
        third = out.children[2]
        assert isinstance(third, Neg)
        # No more LieBracketVF, should now be a BracketApply on Koszul.
        assert isinstance(third.arg, BracketApply)
        assert isinstance(third.arg.bracket, KoszulBracket)

    def test_curvature_default_uses_lie_bracket_vf(self):
        nabla = connection()
        rule = CurvatureDefinitionDefinition(nabla)
        X = Derivation("X", 0)
        Y = Derivation("Y", 0)
        Z = Derivation("Z", 0)
        out = rule.rewrite(Curvature(nabla, X, Y, Z))
        assert isinstance(out, Sum)
        third = out.children[2]
        assert isinstance(third, Neg)
        # third = -∇_[X,Y]_VF Z
        third_eval = third.arg
        assert isinstance(third_eval, ConnectionEvalExpr)
        assert isinstance(third_eval.X, LieBracketVF)

    def test_curvature_custom_bracket_uses_bracket_apply(self):
        from jacopy.brackets.base import BracketApply
        from jacopy.brackets.koszul import KoszulBracket
        from jacopy.calculus.connection import koszul_connection

        nabla = koszul_connection()
        rule = CurvatureDefinitionDefinition(nabla)
        omega = Symbol("ω")
        eta = Symbol("η")
        zeta = Symbol("ζ")
        out = rule.rewrite(Curvature(nabla, omega, eta, zeta))
        assert isinstance(out, Sum)
        third = out.children[2]
        assert isinstance(third, Neg)
        third_eval = third.arg
        assert isinstance(third_eval, ConnectionEvalExpr)
        # X-slot should now be a BracketApply over Koszul.
        assert isinstance(third_eval.X, BracketApply)
        assert isinstance(third_eval.X.bracket, KoszulBracket)

    def test_connection_bracket_slot_participates_in_equality(self):
        from jacopy.brackets.koszul import KoszulBracket
        from jacopy.calculus.anchor import Anchor

        rho = Anchor("π^♯")
        a = AffineConnection("∇", anchor=rho, bracket=KoszulBracket(rho))
        b = AffineConnection("∇", anchor=rho, bracket=KoszulBracket(rho))
        c = AffineConnection("∇", anchor=rho)
        assert a == b
        assert a != c

    def test_connection_rejects_non_graded_bracket(self):
        with pytest.raises(TypeError):
            AffineConnection(
                "∇", bracket="not-a-bracket"
            )  # type: ignore[arg-type]
