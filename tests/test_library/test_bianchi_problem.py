"""Tests for the Bianchi-identity wrapper — Faz 16.D."""

from __future__ import annotations

import pytest

from jacopy.algebra.derivation import Derivation
from jacopy.calculus.connection import (
    AffineConnection,
    ConnectionEvalExpr,
    connection,
)
from jacopy.calculus.torsion_curvature import (
    Curvature,
    CurvatureCovariantDerivative,
    Torsion,
    TorsionCovariantDerivative,
)
from jacopy.core.expr import Expr, Sum, Zero
from jacopy.library.bianchi_problem import (
    BianchiProblem,
    BianchiProofResult,
    cyclic_sum_3,
    cyclic_sum_3_fixed_last,
)


# --------------------------------------------------------------------- #
# cyclic-sum helpers                                                    #
# --------------------------------------------------------------------- #


def test_cyclic_sum_3_rotates_args():
    nabla = connection()

    def factory(a, b, c):
        return Curvature(nabla, a, b, c)

    U = Derivation("U", 0)
    V = Derivation("V", 0)
    W = Derivation("W", 0)
    s = cyclic_sum_3(factory, U, V, W)
    assert isinstance(s, Sum)
    assert s.children == (
        Curvature(nabla, U, V, W),
        Curvature(nabla, V, W, U),
        Curvature(nabla, W, U, V),
    )


def test_cyclic_sum_3_fixed_last_holds_final_slot():
    nabla = connection()

    def factory(a, b, c, last):
        return CurvatureCovariantDerivative(nabla, a, b, c, last)

    U = Derivation("U", 0)
    V = Derivation("V", 0)
    W = Derivation("W", 0)
    Z = Derivation("Z", 0)
    s = cyclic_sum_3_fixed_last(factory, U, V, W, Z)
    assert isinstance(s, Sum)
    assert s.children == (
        CurvatureCovariantDerivative(nabla, U, V, W, Z),
        CurvatureCovariantDerivative(nabla, V, W, U, Z),
        CurvatureCovariantDerivative(nabla, W, U, V, Z),
    )


# --------------------------------------------------------------------- #
# BianchiProblem construction + accessors                                #
# --------------------------------------------------------------------- #


def test_bianchi_problem_carries_connection_and_engine():
    nabla = connection()
    prob = BianchiProblem(nabla)
    assert prob.connection == nabla
    assert prob.registry is None
    assert prob.engine is not None
    assert prob.name == f"BianchiProblem({nabla._repr_inner()})"


def test_bianchi_problem_repr_includes_connection_name():
    nabla = connection("∇*")
    prob = BianchiProblem(nabla)
    assert "∇*" in repr(prob)


def test_bianchi_problem_rejects_non_connection():
    with pytest.raises(TypeError):
        BianchiProblem("not-a-conn")  # type: ignore[arg-type]


def test_bianchi_problem_rejects_bad_registry():
    nabla = connection()
    with pytest.raises(TypeError):
        BianchiProblem(nabla, registry="not-a-registry")  # type: ignore[arg-type]


def test_bianchi_problem_builders_are_connection_bound():
    nabla = connection()
    prob = BianchiProblem(nabla)
    U = Derivation("U", 0)
    V = Derivation("V", 0)
    W = Derivation("W", 0)
    Z = Derivation("Z", 0)
    assert prob.torsion(U, V) == Torsion(nabla, U, V)
    assert prob.curvature(U, V, W) == Curvature(nabla, U, V, W)
    assert prob.cov_deriv_torsion(U, V, W) == TorsionCovariantDerivative(
        nabla, U, V, W
    )
    assert prob.cov_deriv_curvature(
        U, V, W, Z
    ) == CurvatureCovariantDerivative(nabla, U, V, W, Z)


# --------------------------------------------------------------------- #
# Bianchi I / II construction shapes                                     #
# --------------------------------------------------------------------- #


def test_first_bianchi_lhs_is_cyclic_curvature_sum():
    nabla = connection()
    prob = BianchiProblem(nabla)
    U = Derivation("U", 0)
    V = Derivation("V", 0)
    W = Derivation("W", 0)
    lhs = prob.first_bianchi_lhs(U, V, W)
    assert isinstance(lhs, Sum)
    assert lhs.children == (
        Curvature(nabla, U, V, W),
        Curvature(nabla, V, W, U),
        Curvature(nabla, W, U, V),
    )


def test_first_bianchi_rhs_carries_six_terms():
    nabla = connection()
    prob = BianchiProblem(nabla)
    U = Derivation("U", 0)
    V = Derivation("V", 0)
    W = Derivation("W", 0)
    rhs = prob.first_bianchi_rhs(U, V, W)
    assert isinstance(rhs, Sum)
    # 3 cycles × 2 terms (covariant-derivative-of-T + T(T(·,·),·))
    assert len(rhs.children) == 6
    nablaT = [
        c for c in rhs.children
        if isinstance(c, TorsionCovariantDerivative)
    ]
    TofT = [c for c in rhs.children if isinstance(c, Torsion)]
    assert len(nablaT) == 3
    assert len(TofT) == 3
    # Each T(T(...)) has Torsion(...) inside one of its slots.
    for t in TofT:
        assert isinstance(t.X, Torsion) or isinstance(t.Y, Torsion)


def test_second_bianchi_lhs_is_cyclic_cov_deriv_curvature_sum():
    nabla = connection()
    prob = BianchiProblem(nabla)
    U = Derivation("U", 0)
    V = Derivation("V", 0)
    W = Derivation("W", 0)
    Z = Derivation("Z", 0)
    lhs = prob.second_bianchi_lhs(U, V, W, Z)
    assert isinstance(lhs, Sum)
    for c in lhs.children:
        assert isinstance(c, CurvatureCovariantDerivative)
        assert c.Z == Z


def test_second_bianchi_rhs_has_torsion_in_curvature_y_slot():
    nabla = connection()
    prob = BianchiProblem(nabla)
    U = Derivation("U", 0)
    V = Derivation("V", 0)
    W = Derivation("W", 0)
    Z = Derivation("Z", 0)
    rhs = prob.second_bianchi_rhs(U, V, W, Z)
    assert isinstance(rhs, Sum)
    assert len(rhs.children) == 3
    for c in rhs.children:
        assert isinstance(c, Curvature)
        assert isinstance(c.Y, Torsion)
        assert c.Z == Z


# --------------------------------------------------------------------- #
# Mechanical closures                                                    #
# --------------------------------------------------------------------- #


def test_prove_first_bianchi_closes_to_zero():
    """``cycl R(U,V)W − cycl[(∇_U T)(V,W) + T(T(U,V),W)] → 0``."""
    nabla = connection()
    prob = BianchiProblem(nabla)
    U = Derivation("U", 0)
    V = Derivation("V", 0)
    W = Derivation("W", 0)
    result = prob.prove_first_bianchi(U, V, W)
    assert isinstance(result, BianchiProofResult)
    assert result.ok is True
    assert result.lhs_final == Zero
    assert result.rhs_final == Zero
    assert len(result.lhs_steps) >= 10  # non-trivial proof


def test_prove_second_bianchi_closes_to_zero():
    """``cycl (∇_U R)(V,W)Z − cycl R(U, T(V,W)) Z → 0``."""
    nabla = connection()
    prob = BianchiProblem(nabla)
    U = Derivation("U", 0)
    V = Derivation("V", 0)
    W = Derivation("W", 0)
    Z = Derivation("Z", 0)
    result = prob.prove_second_bianchi(U, V, W, Z)
    assert isinstance(result, BianchiProofResult)
    assert result.ok is True
    assert result.lhs_final == Zero
    assert result.rhs_final == Zero
    assert len(result.lhs_steps) >= 10


def test_prove_first_bianchi_records_initial_shapes():
    nabla = connection()
    prob = BianchiProblem(nabla)
    U = Derivation("U", 0)
    V = Derivation("V", 0)
    W = Derivation("W", 0)
    result = prob.prove_first_bianchi(U, V, W)
    assert isinstance(result.lhs_initial, Sum)
    assert isinstance(result.rhs_initial, Sum)
    assert len(result.lhs_initial.children) == 3
    assert len(result.rhs_initial.children) == 6


def test_two_problems_dont_cross_fire():
    """A problem on ``∇1`` shouldn't mechanically prove identities on ``∇2``."""
    nabla1 = connection("∇1")
    nabla2 = connection("∇2")
    prob1 = BianchiProblem(nabla1)
    U = Derivation("U", 0)
    V = Derivation("V", 0)
    W = Derivation("W", 0)
    # A Curvature node bound to ∇2 doesn't get expanded by prob1's engine.
    foreign = Curvature(nabla2, U, V, W)
    final, steps = prob1.engine.expand(foreign)
    assert final == foreign
    assert len(steps) == 0


def test_proof_result_is_frozen():
    nabla = connection()
    prob = BianchiProblem(nabla)
    U = Derivation("U", 0)
    V = Derivation("V", 0)
    W = Derivation("W", 0)
    result = prob.prove_first_bianchi(U, V, W)
    with pytest.raises(Exception):
        result.ok = False  # type: ignore[misc]
