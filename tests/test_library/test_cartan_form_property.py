"""Tests for the Cartan form-property wrapper."""

from __future__ import annotations

import pytest

from jacopy.algebra.derivation import Derivation
from jacopy.calculus.cartan_forms import (
    ConnectionForm,
    CurvatureForm,
    NonMetricityForm,
    TorsionForm,
)
from jacopy.calculus.connection import connection
from jacopy.calculus.local_frame import local_frame
from jacopy.calculus.metric import metric
from jacopy.calculus.pairing import Pairing
from jacopy.core.expr import Symbol, Zero
from jacopy.core.multi_eval import MultiEval
from jacopy.library.cartan_form_property import (
    CartanFormPropertyProblem,
    CartanFormPropertyProofResult,
)
from jacopy.proof.expansion import ExpansionEngine


# --------------------------------------------------------------------- #
# construction / validation                                              #
# --------------------------------------------------------------------- #


def test_construct_with_metric_records_all_three():
    nabla = connection("∇")
    F = local_frame("F", dim=3)
    g = metric("g")
    P = CartanFormPropertyProblem(nabla, F, metric=g)
    assert P.connection is nabla
    assert P.frame is F
    assert P.metric is g
    assert isinstance(P.engine, ExpansionEngine)


def test_construct_without_metric_metric_is_none():
    nabla = connection("∇")
    F = local_frame("F", dim=3)
    P = CartanFormPropertyProblem(nabla, F)
    assert P.metric is None


def test_construct_rejects_non_connection():
    F = local_frame("F", dim=3)
    with pytest.raises(TypeError):
        CartanFormPropertyProblem("nabla", F)  # type: ignore[arg-type]


def test_construct_rejects_non_frame():
    nabla = connection("∇")
    with pytest.raises(TypeError):
        CartanFormPropertyProblem(nabla, "F")  # type: ignore[arg-type]


def test_construct_rejects_non_metric():
    nabla = connection("∇")
    F = local_frame("F", dim=3)
    with pytest.raises(TypeError):
        CartanFormPropertyProblem(nabla, F, metric="g")  # type: ignore[arg-type]


def test_default_name_includes_connection_and_frame():
    nabla = connection("∇")
    F = local_frame("F", dim=3)
    P = CartanFormPropertyProblem(nabla, F)
    assert "∇" in P.name and "F" in P.name


def test_custom_name_overrides_default():
    nabla = connection("∇")
    F = local_frame("F", dim=3)
    P = CartanFormPropertyProblem(nabla, F, name="custom")
    assert P.name == "custom"


def test_repr_with_metric_contains_metric_name():
    nabla = connection("∇")
    F = local_frame("F", dim=3)
    g = metric("g")
    P = CartanFormPropertyProblem(nabla, F, metric=g)
    assert "g" in repr(P)


# --------------------------------------------------------------------- #
# builders                                                               #
# --------------------------------------------------------------------- #


def test_omega_builder_returns_connection_form():
    nabla = connection("∇")
    F = local_frame("F", dim=3)
    P = CartanFormPropertyProblem(nabla, F)
    w = P.omega("a", "b")
    assert isinstance(w, ConnectionForm)
    assert w.connection is nabla
    assert w.frame is F


def test_T_builder_returns_torsion_form():
    nabla = connection("∇")
    F = local_frame("F", dim=3)
    P = CartanFormPropertyProblem(nabla, F)
    t = P.T("a")
    assert isinstance(t, TorsionForm)
    assert t.connection is nabla


def test_R_builder_returns_curvature_form():
    nabla = connection("∇")
    F = local_frame("F", dim=3)
    P = CartanFormPropertyProblem(nabla, F)
    r = P.R("a", "b")
    assert isinstance(r, CurvatureForm)


def test_Q_builder_returns_non_metricity_form():
    nabla = connection("∇")
    F = local_frame("F", dim=3)
    g = metric("g")
    P = CartanFormPropertyProblem(nabla, F, metric=g)
    q = P.Q("a", "b")
    assert isinstance(q, NonMetricityForm)
    assert q.metric is g


def test_Q_builder_without_metric_raises():
    nabla = connection("∇")
    F = local_frame("F", dim=3)
    P = CartanFormPropertyProblem(nabla, F)
    with pytest.raises(ValueError):
        P.Q("a", "b")


def test_omega_eval_returns_pairing():
    nabla = connection("∇")
    F = local_frame("F", dim=3)
    P = CartanFormPropertyProblem(nabla, F)
    V = Symbol("V")
    pe = P.omega_eval("a", "b", V)
    assert isinstance(pe, Pairing)
    assert isinstance(pe.alpha, ConnectionForm)
    assert pe.X == V


def test_T_eval_returns_multi_eval_arity_two():
    nabla = connection("∇")
    F = local_frame("F", dim=3)
    P = CartanFormPropertyProblem(nabla, F)
    U, V = Symbol("U"), Symbol("V")
    me = P.T_eval("a", U, V)
    assert isinstance(me, MultiEval)
    assert me.arity == 2


# --------------------------------------------------------------------- #
# ω 1-form proofs                                                        #
# --------------------------------------------------------------------- #


def _problem_with_metric():
    nabla = connection("∇")
    F = local_frame("F", dim=3)
    g = metric("g")
    return CartanFormPropertyProblem(nabla, F, metric=g)


def test_prove_omega_scalar_linear_in_V_closes():
    P = _problem_with_metric()
    f, V = Symbol("f"), Symbol("V")
    result = P.prove_omega_scalar_linear_in_V("a", "b", f, V)
    assert isinstance(result, CartanFormPropertyProofResult)
    assert result.ok
    assert result.final == Zero


def test_prove_omega_additive_in_V_closes():
    P = _problem_with_metric()
    V1, V2 = Symbol("V1"), Symbol("V2")
    result = P.prove_omega_additive_in_V("a", "b", V1, V2)
    assert result.ok
    assert result.final == Zero


def test_prove_omega_records_lhs_and_rhs():
    P = _problem_with_metric()
    f, V = Symbol("f"), Symbol("V")
    result = P.prove_omega_scalar_linear_in_V("a", "b", f, V)
    assert result.lhs_initial != result.rhs_initial
    assert len(result.steps) > 0


# --------------------------------------------------------------------- #
# Q 1-form proofs                                                        #
# --------------------------------------------------------------------- #


def test_prove_Q_scalar_linear_in_V_closes():
    P = _problem_with_metric()
    f, V = Symbol("f"), Symbol("V")
    result = P.prove_Q_scalar_linear_in_V("a", "b", f, V)
    assert result.ok
    assert result.final == Zero


def test_prove_Q_additive_in_V_closes():
    P = _problem_with_metric()
    V1, V2 = Symbol("V1"), Symbol("V2")
    result = P.prove_Q_additive_in_V("a", "b", V1, V2)
    assert result.ok
    assert result.final == Zero


# --------------------------------------------------------------------- #
# T^a 2-form proofs                                                      #
# --------------------------------------------------------------------- #


def test_prove_T_scalar_linear_in_first_closes():
    P = _problem_with_metric()
    f, U, V = Symbol("f"), Symbol("U"), Symbol("V")
    result = P.prove_T_scalar_linear_in_first("a", f, U, V)
    assert result.ok
    assert result.final == Zero


def test_prove_T_scalar_linear_in_second_closes():
    P = _problem_with_metric()
    f, U, V = Symbol("f"), Symbol("U"), Symbol("V")
    result = P.prove_T_scalar_linear_in_second("a", f, U, V)
    assert result.ok
    assert result.final == Zero


def test_prove_T_additive_in_first_closes():
    P = _problem_with_metric()
    U1, U2, V = Symbol("U1"), Symbol("U2"), Symbol("V")
    result = P.prove_T_additive_in_first("a", U1, U2, V)
    assert result.ok
    assert result.final == Zero


def test_prove_T_additive_in_second_closes():
    P = _problem_with_metric()
    U, V1, V2 = Symbol("U"), Symbol("V1"), Symbol("V2")
    result = P.prove_T_additive_in_second("a", U, V1, V2)
    assert result.ok
    assert result.final == Zero


def test_prove_T_antisymmetric_closes():
    P = _problem_with_metric()
    U, V = Symbol("U"), Symbol("V")
    result = P.prove_T_antisymmetric("a", U, V)
    assert result.ok
    assert result.final == Zero


def test_prove_T_antisymmetric_with_derivations():
    """Antisymmetry should also close with Derivation-typed args."""
    P = _problem_with_metric()
    U = Derivation("U", 0)
    V = Derivation("V", 0)
    result = P.prove_T_antisymmetric("a", U, V)
    assert result.ok


# --------------------------------------------------------------------- #
# R^a_b 2-form proofs                                                    #
# --------------------------------------------------------------------- #


def test_prove_R_scalar_linear_in_first_closes():
    P = _problem_with_metric()
    f, U, V = Symbol("f"), Symbol("U"), Symbol("V")
    result = P.prove_R_scalar_linear_in_first("a", "b", f, U, V)
    assert result.ok
    assert result.final == Zero


def test_prove_R_scalar_linear_in_second_closes():
    P = _problem_with_metric()
    f, U, V = Symbol("f"), Symbol("U"), Symbol("V")
    result = P.prove_R_scalar_linear_in_second("a", "b", f, U, V)
    assert result.ok
    assert result.final == Zero


def test_prove_R_additive_in_first_closes():
    P = _problem_with_metric()
    U1, U2, V = Symbol("U1"), Symbol("U2"), Symbol("V")
    result = P.prove_R_additive_in_first("a", "b", U1, U2, V)
    assert result.ok
    assert result.final == Zero


def test_prove_R_additive_in_second_closes():
    P = _problem_with_metric()
    U, V1, V2 = Symbol("U"), Symbol("V1"), Symbol("V2")
    result = P.prove_R_additive_in_second("a", "b", U, V1, V2)
    assert result.ok
    assert result.final == Zero


def test_prove_R_antisymmetric_closes():
    P = _problem_with_metric()
    U, V = Symbol("U"), Symbol("V")
    result = P.prove_R_antisymmetric("a", "b", U, V)
    assert result.ok
    assert result.final == Zero


# --------------------------------------------------------------------- #
# scoping: two connections / frames don't cross-fire                     #
# --------------------------------------------------------------------- #


def test_proofs_for_one_connection_do_not_open_another_connection():
    """Engine bundle is scoped to (∇, F); a second ∇' form left untouched."""
    nabla1 = connection("∇1")
    nabla2 = connection("∇2")
    F = local_frame("F", dim=3)
    P = CartanFormPropertyProblem(nabla1, F)
    # An ω-shape over nabla2 shouldn't be opened by P's engine.
    other_omega_eval = Pairing(
        ConnectionForm(nabla2, F, "a", "b"), Symbol("V")
    )
    expanded, _ = P.engine.expand(other_omega_eval, max_steps=64)
    # Engine doesn't unfold the other connection's ω.
    assert isinstance(expanded, Pairing)
    assert isinstance(expanded.alpha, ConnectionForm)
    assert expanded.alpha.connection is nabla2


def test_proofs_for_one_frame_do_not_open_another_frame():
    nabla = connection("∇")
    F1 = local_frame("F1", dim=3)
    F2 = local_frame("F2", dim=3)
    P = CartanFormPropertyProblem(nabla, F1)
    other_omega_eval = Pairing(
        ConnectionForm(nabla, F2, "a", "b"), Symbol("V")
    )
    expanded, _ = P.engine.expand(other_omega_eval, max_steps=64)
    assert isinstance(expanded, Pairing)
    assert isinstance(expanded.alpha, ConnectionForm)
    assert expanded.alpha.frame is F2


# --------------------------------------------------------------------- #
# proof-result shape                                                     #
# --------------------------------------------------------------------- #


def test_proof_result_is_frozen():
    P = _problem_with_metric()
    U, V = Symbol("U"), Symbol("V")
    result = P.prove_T_antisymmetric("a", U, V)
    with pytest.raises(Exception):
        result.ok = False  # type: ignore[misc]


def test_steps_are_non_empty_for_non_trivial_proof():
    P = _problem_with_metric()
    f, U, V = Symbol("f"), Symbol("U"), Symbol("V")
    result = P.prove_R_scalar_linear_in_first("a", "b", f, U, V)
    assert len(result.steps) > 0


# --------------------------------------------------------------------- #
# Q9 Stage 9.D, Koszul-connection mode                                  #
# --------------------------------------------------------------------- #


class TestKoszulConnectionFormProperties:
    """Form-degree property props on a Koszul-bracket connection.

    The Cartan form components ``ω̃^a_b``, ``T̃^a``, ``R̃^a_b`` of a
    Koszul-equipped affine connection on T*M obey the same
    ``C^∞``-bilinearity + antisymmetry shape as their Lie-algebroid /
    smooth-manifold counterparts: the property proofs only push scalars
    and Sums through the connection slots and pull them out of the
    pairing, they don't open the bracket. Confirming the existing
    :class:`CartanFormPropertyProblem` engine bundle closes all 12
    pure-V / V-additive / antisym claims unchanged when the connection
    carries a Koszul bracket. This is the "Q9 Stage 9.D" certification
    that no T*M-specific form-property axioms are needed.
    """

    def _setup(self):
        from jacopy.calculus.anchor import Anchor
        from jacopy.calculus.connection import koszul_connection
        from jacopy.brackets.koszul import KoszulBracket

        anchor = Anchor(name="ρ")
        bracket = KoszulBracket(anchor)
        nabla = koszul_connection("∇̃", anchor=anchor, bracket=bracket)
        F = local_frame("F̃", dim=3)
        return CartanFormPropertyProblem(nabla, F)

    def test_omega_scalar_linear_closes(self):
        P = self._setup()
        f, V = Symbol("f"), Symbol("V")
        assert P.prove_omega_scalar_linear_in_V("a", "b", f, V).ok

    def test_omega_additive_closes(self):
        P = self._setup()
        V1, V2 = Symbol("V1"), Symbol("V2")
        assert P.prove_omega_additive_in_V("a", "b", V1, V2).ok

    def test_T_bilinearity_closes(self):
        P = self._setup()
        f, U, V, W = Symbol("f"), Symbol("U"), Symbol("V"), Symbol("W")
        assert P.prove_T_scalar_linear_in_first("a", f, U, V).ok
        assert P.prove_T_scalar_linear_in_second("a", U, f, V).ok
        assert P.prove_T_additive_in_first("a", U, V, W).ok
        assert P.prove_T_additive_in_second("a", U, V, W).ok

    def test_T_antisymmetric_closes(self):
        P = self._setup()
        U, V = Symbol("U"), Symbol("V")
        assert P.prove_T_antisymmetric("a", U, V).ok

    def test_R_bilinearity_closes(self):
        P = self._setup()
        f, U, V, W = Symbol("f"), Symbol("U"), Symbol("V"), Symbol("W")
        assert P.prove_R_scalar_linear_in_first("a", "b", f, U, V).ok
        assert P.prove_R_scalar_linear_in_second("a", "b", U, f, V).ok
        assert P.prove_R_additive_in_first("a", "b", U, V, W).ok
        assert P.prove_R_additive_in_second("a", "b", U, V, W).ok

    def test_R_antisymmetric_closes(self):
        P = self._setup()
        U, V = Symbol("U"), Symbol("V")
        assert P.prove_R_antisymmetric("a", "b", U, V).ok
