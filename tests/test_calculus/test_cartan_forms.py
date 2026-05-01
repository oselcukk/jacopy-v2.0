"""Tests for Cartan-form Expr nodes, Faz 17.C."""

from __future__ import annotations

import pytest

from jacopy.algebra.derivation import Derivation
from jacopy.calculus.cartan_forms import (
    ConnectionForm,
    ConnectionFormDefinition,
    CurvatureForm,
    CurvatureFormDefinition,
    NonMetricityForm,
    NonMetricityFormDefinition,
    TorsionForm,
    TorsionFormDefinition,
)
from jacopy.calculus.connection import ConnectionEvalExpr, connection
from jacopy.calculus.local_frame import (
    FrameCovector,
    FrameIndex,
    FrameVectorField,
    local_frame,
)
from jacopy.calculus.metric import metric
from jacopy.calculus.non_metricity import NonMetricityEvalExpr
from jacopy.calculus.pairing import Pairing
from jacopy.calculus.torsion_curvature import Curvature, Torsion
from jacopy.core.expr import Atom, Expr
from jacopy.core.multi_eval import MultiEval
from jacopy.proof.expansion import ExpansionEngine


# --------------------------------------------------------------------- #
# ConnectionForm                                                        #
# --------------------------------------------------------------------- #


def test_connection_form_carries_params():
    nabla = connection()
    F = local_frame()
    omega = ConnectionForm(nabla, F, "a", "b")
    assert omega.connection == nabla
    assert omega.frame == F
    assert omega.upper == FrameIndex("a")
    assert omega.lower == FrameIndex("b")
    assert isinstance(omega, Atom)


def test_connection_form_accepts_frame_index_objects():
    nabla = connection()
    F = local_frame()
    a = F.index("a")
    b = F.index("b")
    omega = ConnectionForm(nabla, F, a, b)
    assert omega.upper is a
    assert omega.lower is b


def test_connection_form_equality_includes_all_params():
    nabla = connection()
    F = local_frame()
    a, b = ConnectionForm(nabla, F, "a", "b"), ConnectionForm(
        nabla, F, "a", "b"
    )
    assert a == b
    assert hash(a) == hash(b)
    assert ConnectionForm(nabla, F, "a", "b") != ConnectionForm(
        nabla, F, "b", "a"
    )
    nabla2 = connection("∇'")
    assert ConnectionForm(nabla, F, "a", "b") != ConnectionForm(
        nabla2, F, "a", "b"
    )
    F2 = local_frame("F2")
    assert ConnectionForm(nabla, F, "a", "b") != ConnectionForm(
        nabla, F2, "a", "b"
    )


def test_connection_form_repr_shape():
    nabla = connection("∇")
    F = local_frame()
    s = repr(ConnectionForm(nabla, F, "a", "b"))
    assert "ω" in s
    assert "a" in s
    assert "b" in s
    assert "∇" in s


def test_connection_form_validation():
    nabla = connection()
    F = local_frame()
    with pytest.raises(TypeError):
        ConnectionForm("not conn", F, "a", "b")  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        ConnectionForm(nabla, "not frame", "a", "b")  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        ConnectionForm(nabla, F, 42, "b")  # type: ignore[arg-type]


# --------------------------------------------------------------------- #
# ConnectionFormDefinition                                              #
# --------------------------------------------------------------------- #


def test_connection_form_definition_unfolds_pairing():
    nabla = connection()
    F = local_frame()
    rule = ConnectionFormDefinition(nabla, F)
    omega = ConnectionForm(nabla, F, "a", "b")
    V = Derivation("V")
    pairing = Pairing(omega, V)
    assert rule.matches(pairing)
    res = rule.rewrite(pairing)
    assert isinstance(res, Pairing)
    # alpha is the coframe, X is ∇_V X_b
    assert isinstance(res.alpha, FrameCovector)
    assert res.alpha.idx == FrameIndex("a")
    assert isinstance(res.X, ConnectionEvalExpr)
    assert res.X.X == V  # ∇_X Y reading: V is in .X
    assert isinstance(res.X.Y, FrameVectorField)
    assert res.X.Y.idx == FrameIndex("b")


def test_connection_form_definition_no_match_other_pairing():
    nabla = connection()
    F = local_frame()
    rule = ConnectionFormDefinition(nabla, F)
    V = Derivation("V")
    # Pairing whose first slot is not a ConnectionForm.
    e_a = F.coframe("a")
    assert not rule.matches(Pairing(e_a, V))


def test_connection_form_definition_scoped_to_pair():
    nabla1 = connection("∇1")
    nabla2 = connection("∇2")
    F = local_frame()
    F2 = local_frame("F2")
    rule = ConnectionFormDefinition(nabla1, F)
    V = Derivation("V")
    omega_other_conn = ConnectionForm(nabla2, F, "a", "b")
    omega_other_frame = ConnectionForm(nabla1, F2, "a", "b")
    assert not rule.matches(Pairing(omega_other_conn, V))
    assert not rule.matches(Pairing(omega_other_frame, V))


def test_connection_form_definition_validation():
    nabla = connection()
    F = local_frame()
    with pytest.raises(TypeError):
        ConnectionFormDefinition("not conn", F)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        ConnectionFormDefinition(nabla, "not frame")  # type: ignore[arg-type]


def test_connection_form_definition_engine_run():
    nabla = connection()
    F = local_frame()
    engine = ExpansionEngine([ConnectionFormDefinition(nabla, F)])
    omega = ConnectionForm(nabla, F, "a", "b")
    V = Derivation("V")
    final, steps = engine.expand(Pairing(omega, V))
    assert len(steps) == 1
    assert isinstance(final, Pairing)
    # The form is gone from the expanded tree.
    def has_form(e):
        if isinstance(e, ConnectionForm):
            return True
        return any(has_form(c) for c in e.children)
    assert not has_form(final)


# --------------------------------------------------------------------- #
# NonMetricityForm                                                      #
# --------------------------------------------------------------------- #


def test_non_metricity_form_carries_params():
    nabla = connection()
    g = metric()
    F = local_frame()
    Q = NonMetricityForm(nabla, g, F, "a", "b")
    assert Q.connection == nabla
    assert Q.metric == g
    assert Q.frame == F
    assert Q.lower_a == FrameIndex("a")
    assert Q.lower_b == FrameIndex("b")
    assert isinstance(Q, Atom)


def test_non_metricity_form_equality():
    nabla = connection()
    g = metric()
    g2 = metric("g2")
    F = local_frame()
    a = NonMetricityForm(nabla, g, F, "a", "b")
    b = NonMetricityForm(nabla, g, F, "a", "b")
    assert a == b
    assert hash(a) == hash(b)
    assert NonMetricityForm(nabla, g, F, "a", "b") != NonMetricityForm(
        nabla, g2, F, "a", "b"
    )


def test_non_metricity_form_repr_shape():
    nabla = connection("∇")
    g = metric("g")
    F = local_frame()
    s = repr(NonMetricityForm(nabla, g, F, "a", "b"))
    assert "Q" in s and "a" in s and "b" in s and "g" in s and "∇" in s


def test_non_metricity_form_validation():
    nabla = connection()
    g = metric()
    F = local_frame()
    with pytest.raises(TypeError):
        NonMetricityForm("nope", g, F, "a", "b")  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        NonMetricityForm(nabla, "nope", F, "a", "b")  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        NonMetricityForm(nabla, g, "nope", "a", "b")  # type: ignore[arg-type]


# --------------------------------------------------------------------- #
# NonMetricityFormDefinition                                            #
# --------------------------------------------------------------------- #


def test_non_metricity_form_definition_unfolds():
    nabla = connection()
    g = metric()
    F = local_frame()
    rule = NonMetricityFormDefinition(nabla, g, F)
    Q = NonMetricityForm(nabla, g, F, "a", "b")
    V = Derivation("V")
    pairing = Pairing(Q, V)
    assert rule.matches(pairing)
    res = rule.rewrite(pairing)
    assert isinstance(res, NonMetricityEvalExpr)
    assert res.connection == nabla
    assert res.metric == g
    assert res.V == V
    assert isinstance(res.X, FrameVectorField)
    assert res.X.idx == FrameIndex("a")
    assert isinstance(res.Y, FrameVectorField)
    assert res.Y.idx == FrameIndex("b")


def test_non_metricity_form_definition_scoped():
    nabla = connection()
    g1 = metric("g1")
    g2 = metric("g2")
    F = local_frame()
    F2 = local_frame("F2")
    rule = NonMetricityFormDefinition(nabla, g1, F)
    V = Derivation("V")
    Q_other_metric = NonMetricityForm(nabla, g2, F, "a", "b")
    Q_other_frame = NonMetricityForm(nabla, g1, F2, "a", "b")
    assert not rule.matches(Pairing(Q_other_metric, V))
    assert not rule.matches(Pairing(Q_other_frame, V))


def test_non_metricity_form_definition_validation():
    nabla = connection()
    g = metric()
    F = local_frame()
    with pytest.raises(TypeError):
        NonMetricityFormDefinition("nope", g, F)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        NonMetricityFormDefinition(nabla, "nope", F)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        NonMetricityFormDefinition(nabla, g, "nope")  # type: ignore[arg-type]


def test_non_metricity_form_definition_engine_run():
    nabla = connection()
    g = metric()
    F = local_frame()
    engine = ExpansionEngine([NonMetricityFormDefinition(nabla, g, F)])
    Q = NonMetricityForm(nabla, g, F, "a", "b")
    V = Derivation("V")
    final, steps = engine.expand(Pairing(Q, V))
    assert len(steps) == 1
    assert isinstance(final, NonMetricityEvalExpr)


# --------------------------------------------------------------------- #
# TorsionForm                                                           #
# --------------------------------------------------------------------- #


def test_torsion_form_carries_params():
    nabla = connection()
    F = local_frame()
    T = TorsionForm(nabla, F, "a")
    assert T.connection == nabla
    assert T.frame == F
    assert T.upper == FrameIndex("a")
    assert isinstance(T, Atom)


def test_torsion_form_equality():
    nabla = connection()
    F = local_frame()
    assert TorsionForm(nabla, F, "a") == TorsionForm(nabla, F, "a")
    assert TorsionForm(nabla, F, "a") != TorsionForm(nabla, F, "b")


def test_torsion_form_repr_shape():
    nabla = connection("∇")
    F = local_frame()
    s = repr(TorsionForm(nabla, F, "a"))
    assert "T" in s and "a" in s and "∇" in s


def test_torsion_form_validation():
    nabla = connection()
    F = local_frame()
    with pytest.raises(TypeError):
        TorsionForm("nope", F, "a")  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        TorsionForm(nabla, "nope", "a")  # type: ignore[arg-type]


# --------------------------------------------------------------------- #
# TorsionFormDefinition                                                 #
# --------------------------------------------------------------------- #


def test_torsion_form_definition_unfolds():
    nabla = connection()
    F = local_frame()
    rule = TorsionFormDefinition(nabla, F)
    T = TorsionForm(nabla, F, "a")
    U = Derivation("U")
    V = Derivation("V")
    me = MultiEval(T, U, V)
    assert rule.matches(me)
    res = rule.rewrite(me)
    assert isinstance(res, Pairing)
    assert isinstance(res.alpha, FrameCovector)
    assert res.alpha.idx == FrameIndex("a")
    assert isinstance(res.X, Torsion)
    assert res.X.X == U
    assert res.X.Y == V


def test_torsion_form_definition_arity_check():
    nabla = connection()
    F = local_frame()
    rule = TorsionFormDefinition(nabla, F)
    T = TorsionForm(nabla, F, "a")
    U = Derivation("U")
    V = Derivation("V")
    W = Derivation("W")
    # Arity 3, does not match.
    me = MultiEval(T, U, V, W)
    assert not rule.matches(me)


def test_torsion_form_definition_no_match_when_head_is_other_form():
    nabla = connection()
    F = local_frame()
    rule = TorsionFormDefinition(nabla, F)
    R = CurvatureForm(nabla, F, "a", "b")
    U = Derivation("U")
    V = Derivation("V")
    assert not rule.matches(MultiEval(R, U, V))


def test_torsion_form_definition_scoped():
    nabla1 = connection("∇1")
    nabla2 = connection("∇2")
    F = local_frame()
    F2 = local_frame("F2")
    rule = TorsionFormDefinition(nabla1, F)
    U = Derivation("U")
    V = Derivation("V")
    assert not rule.matches(
        MultiEval(TorsionForm(nabla2, F, "a"), U, V)
    )
    assert not rule.matches(
        MultiEval(TorsionForm(nabla1, F2, "a"), U, V)
    )


def test_torsion_form_definition_validation():
    nabla = connection()
    F = local_frame()
    with pytest.raises(TypeError):
        TorsionFormDefinition("nope", F)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        TorsionFormDefinition(nabla, "nope")  # type: ignore[arg-type]


def test_torsion_form_definition_engine_run():
    nabla = connection()
    F = local_frame()
    engine = ExpansionEngine([TorsionFormDefinition(nabla, F)])
    T = TorsionForm(nabla, F, "a")
    U = Derivation("U")
    V = Derivation("V")
    final, steps = engine.expand(MultiEval(T, U, V))
    assert len(steps) == 1
    assert isinstance(final, Pairing)


# --------------------------------------------------------------------- #
# CurvatureForm                                                         #
# --------------------------------------------------------------------- #


def test_curvature_form_carries_params():
    nabla = connection()
    F = local_frame()
    R = CurvatureForm(nabla, F, "a", "b")
    assert R.connection == nabla
    assert R.frame == F
    assert R.upper == FrameIndex("a")
    assert R.lower == FrameIndex("b")
    assert isinstance(R, Atom)


def test_curvature_form_equality():
    nabla = connection()
    F = local_frame()
    assert CurvatureForm(nabla, F, "a", "b") == CurvatureForm(
        nabla, F, "a", "b"
    )
    assert CurvatureForm(nabla, F, "a", "b") != CurvatureForm(
        nabla, F, "b", "a"
    )


def test_curvature_form_repr_shape():
    nabla = connection("∇")
    F = local_frame()
    s = repr(CurvatureForm(nabla, F, "a", "b"))
    assert "R" in s and "a" in s and "b" in s and "∇" in s


def test_curvature_form_validation():
    nabla = connection()
    F = local_frame()
    with pytest.raises(TypeError):
        CurvatureForm("nope", F, "a", "b")  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        CurvatureForm(nabla, "nope", "a", "b")  # type: ignore[arg-type]


# --------------------------------------------------------------------- #
# CurvatureFormDefinition                                               #
# --------------------------------------------------------------------- #


def test_curvature_form_definition_unfolds():
    nabla = connection()
    F = local_frame()
    rule = CurvatureFormDefinition(nabla, F)
    R = CurvatureForm(nabla, F, "a", "b")
    U = Derivation("U")
    V = Derivation("V")
    me = MultiEval(R, U, V)
    assert rule.matches(me)
    res = rule.rewrite(me)
    assert isinstance(res, Pairing)
    assert isinstance(res.alpha, FrameCovector)
    assert res.alpha.idx == FrameIndex("a")
    assert isinstance(res.X, Curvature)
    assert res.X.X == U
    assert res.X.Y == V
    assert isinstance(res.X.Z, FrameVectorField)
    assert res.X.Z.idx == FrameIndex("b")


def test_curvature_form_definition_arity_check():
    nabla = connection()
    F = local_frame()
    rule = CurvatureFormDefinition(nabla, F)
    R = CurvatureForm(nabla, F, "a", "b")
    U = Derivation("U")
    V = Derivation("V")
    W = Derivation("W")
    assert not rule.matches(MultiEval(R, U, V, W))


def test_curvature_form_definition_scoped():
    nabla1 = connection("∇1")
    nabla2 = connection("∇2")
    F = local_frame()
    F2 = local_frame("F2")
    rule = CurvatureFormDefinition(nabla1, F)
    U = Derivation("U")
    V = Derivation("V")
    assert not rule.matches(
        MultiEval(CurvatureForm(nabla2, F, "a", "b"), U, V)
    )
    assert not rule.matches(
        MultiEval(CurvatureForm(nabla1, F2, "a", "b"), U, V)
    )


def test_curvature_form_definition_validation():
    nabla = connection()
    F = local_frame()
    with pytest.raises(TypeError):
        CurvatureFormDefinition("nope", F)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        CurvatureFormDefinition(nabla, "nope")  # type: ignore[arg-type]


def test_curvature_form_definition_engine_run():
    nabla = connection()
    F = local_frame()
    engine = ExpansionEngine([CurvatureFormDefinition(nabla, F)])
    R = CurvatureForm(nabla, F, "a", "b")
    U = Derivation("U")
    V = Derivation("V")
    final, steps = engine.expand(MultiEval(R, U, V))
    assert len(steps) == 1
    assert isinstance(final, Pairing)


# --------------------------------------------------------------------- #
# Cross-form isolation                                                  #
# --------------------------------------------------------------------- #


def test_form_axioms_do_not_cross_fire():
    nabla = connection()
    g = metric()
    F = local_frame()
    engine = ExpansionEngine(
        [
            ConnectionFormDefinition(nabla, F),
            NonMetricityFormDefinition(nabla, g, F),
            TorsionFormDefinition(nabla, F),
            CurvatureFormDefinition(nabla, F),
        ]
    )
    V = Derivation("V")
    U = Derivation("U")

    # Each axiom fires exactly once on its own form.
    omega = ConnectionForm(nabla, F, "a", "b")
    final, steps = engine.expand(Pairing(omega, V))
    assert len(steps) == 1

    Q = NonMetricityForm(nabla, g, F, "a", "b")
    final, steps = engine.expand(Pairing(Q, V))
    assert len(steps) == 1

    T = TorsionForm(nabla, F, "a")
    final, steps = engine.expand(MultiEval(T, U, V))
    assert len(steps) == 1

    R = CurvatureForm(nabla, F, "a", "b")
    final, steps = engine.expand(MultiEval(R, U, V))
    assert len(steps) == 1
