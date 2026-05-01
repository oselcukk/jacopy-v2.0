"""Tests for the Bianchi-identity wrapper, Faz 16.D."""

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


# --------------------------------------------------------------------- #
# Q9 Stage 9.C, Koszul-bracket-mode closures                            #
# --------------------------------------------------------------------- #


class TestKoszulBracketBianchi:
    """BianchiProblem on a Koszul-bracket-equipped connection.

    The connection's vector_bracket emits ``BracketApply(koszul, …)``
    instead of ``LieBracketVF`` from Torsion/Curvature unfolds; the
    engine swaps in the BracketApply closure family
    (jacopy.calculus.bracket_apply_axioms) and the same Bianchi I/II
    sums close to zero.
    """

    @staticmethod
    def _setup():
        from jacopy.brackets.koszul import KoszulBracket
        from jacopy.calculus.anchor import Anchor
        from jacopy.calculus.connection import koszul_connection
        from jacopy.core.expr import Symbol
        from jacopy.core.properties import Graded
        from jacopy.core.registry import PropertyRegistry
        from jacopy.core.symbolic_degree import Degree

        rho = Anchor(name="ρ")
        bracket = KoszulBracket(rho)
        conn = koszul_connection("∇̃", anchor=rho, bracket=bracket)
        reg = PropertyRegistry()
        alpha, beta, gamma, delta = (
            Symbol("α"), Symbol("β"), Symbol("γ"), Symbol("δ"),
        )
        for s in (alpha, beta, gamma, delta):
            reg.declare(s, Graded(degree=Degree.const(1)))
        return conn, reg, (alpha, beta, gamma, delta), bracket

    def test_engine_picks_bracket_apply_rules(self):
        from jacopy.calculus.bracket_apply_axioms import (
            BracketApplyJacobiDefinition,
            BracketApplySumLinearityDefinition,
        )
        from jacopy.calculus.closure_axioms import (
            LieBracketVfJacobiDefinition,
        )

        conn, reg, _, _ = self._setup()
        prob = BianchiProblem(conn, registry=reg)
        rule_types = {type(r) for r in prob.engine.definitions}
        assert BracketApplySumLinearityDefinition in rule_types
        assert BracketApplyJacobiDefinition in rule_types
        # LBVF rules must NOT be bundled (they'd be inert here, but the
        # swap is the whole point of the Stage 9.C contract).
        assert LieBracketVfJacobiDefinition not in rule_types

    def test_first_bianchi_closes_for_koszul_connection(self):
        conn, reg, (alpha, beta, gamma, _), _ = self._setup()
        prob = BianchiProblem(conn, registry=reg)
        result = prob.prove_first_bianchi(alpha, beta, gamma)
        assert result.ok is True
        assert result.lhs_final == Zero

    def test_second_bianchi_closes_for_koszul_connection(self):
        conn, reg, (alpha, beta, gamma, delta), _ = self._setup()
        prob = BianchiProblem(conn, registry=reg)
        result = prob.prove_second_bianchi(alpha, beta, gamma, delta)
        assert result.ok is True
        assert result.lhs_final == Zero

    def test_koszul_engine_does_not_carry_lbvf_rules(self):
        # Same content as ``test_engine_picks_bracket_apply_rules`` but
        # checks every LBVF rule type is absent, guards against
        # accidental double-bundling.
        from jacopy.calculus.bracket_apply_axioms import (
            BracketApplyAntiSymmetryDefinition,
            BracketApplyArgAntisymmetryDefinition,
            BracketApplyJacobiDefinition,
            BracketApplyNegLinearityDefinition,
            BracketApplySumLinearityDefinition,
        )
        from jacopy.calculus.closure_axioms import (
            LieBracketVfAntiSymmetryDefinition,
            LieBracketVfJacobiDefinition,
        )
        from jacopy.calculus.sn_function_axiom import (
            LieBracketVfAntisymmetryDefinition as LbvfArgAntisym,
            LieBracketVfNegLinearityDefinition,
            LieBracketVfSumLinearityDefinition,
        )

        conn, reg, _, _ = self._setup()
        prob = BianchiProblem(conn, registry=reg)
        rule_types = {type(r) for r in prob.engine.definitions}
        for cls in (
            BracketApplySumLinearityDefinition,
            BracketApplyNegLinearityDefinition,
            BracketApplyArgAntisymmetryDefinition,
            BracketApplyAntiSymmetryDefinition,
            BracketApplyJacobiDefinition,
        ):
            assert cls in rule_types
        for cls in (
            LieBracketVfSumLinearityDefinition,
            LieBracketVfNegLinearityDefinition,
            LbvfArgAntisym,
            LieBracketVfAntiSymmetryDefinition,
            LieBracketVfJacobiDefinition,
        ):
            assert cls not in rule_types

    def test_lbvf_engine_is_unchanged_for_default_connection(self):
        """Default :func:`connection()` (no bracket) keeps the LBVF rule set."""
        from jacopy.calculus.bracket_apply_axioms import (
            BracketApplyJacobiDefinition,
        )
        from jacopy.calculus.closure_axioms import (
            LieBracketVfJacobiDefinition,
        )

        nabla = connection()
        prob = BianchiProblem(nabla)
        rule_types = {type(r) for r in prob.engine.definitions}
        assert LieBracketVfJacobiDefinition in rule_types
        assert BracketApplyJacobiDefinition not in rule_types
