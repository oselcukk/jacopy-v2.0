"""Tests for Faz 17.E.7 frame-decomposition axioms."""

from __future__ import annotations

import pytest

from jacopy.algebra.derivation import Derivation
from jacopy.algorithms.canonicalize import canonicalize
from jacopy.calculus.cartan_forms import (
    ConnectionForm,
    ConnectionFormDefinition,
)
from jacopy.calculus.connection import (
    ConnectionEvalExpr,
    ConnectionXLinearityDefinition,
    ConnectionXScalarPullDefinition,
    connection,
)
from jacopy.calculus.frame_decomposition import (
    ConnectionEvalYFrameDecompositionDefinition,
    ConnectionFormDecompositionDefinition,
    FrameDecompositionDefinition,
)
from jacopy.calculus.indexed_sum_axioms import (
    IndexedSumKroneckerContractDefinition,
    IndexedSumPairingPushInLeftDefinition,
    IndexedSumPairingPushInRightDefinition,
    IndexedSumScalarPullDefinition,
    IndexedSumSumDistributeDefinition,
)
from jacopy.calculus.local_frame import (
    FrameIndex,
    FrameVectorField,
    LocalFrame,
)
from jacopy.calculus.pairing import Pairing
from jacopy.calculus.pairing_axioms import PairingLinearityDefinition
from jacopy.calculus import PairingScalarPullDefinition
from jacopy.core.expr import Product, Symbol
from jacopy.core.indexed_sum import IndexedSum, dummy_in
from jacopy.proof.expansion import ExpansionEngine


# --------------------------------------------------------------------- #
# FrameDecompositionDefinition                                           #
# --------------------------------------------------------------------- #


def test_frame_decomp_matches_outside_derivation():
    F = LocalFrame("F")
    V = Derivation("V", 0)
    rule = FrameDecompositionDefinition(F)
    assert rule.matches(V)


def test_frame_decomp_does_not_match_own_frame_vf():
    F = LocalFrame("F")
    b = F.index("b")
    rule = FrameDecompositionDefinition(F)
    assert not rule.matches(F.X(b))


def test_frame_decomp_matches_other_frame_vf():
    F = LocalFrame("F")
    G = LocalFrame("G")
    a = G.index("a")
    rule = FrameDecompositionDefinition(F)
    assert rule.matches(G.X(a))


def test_frame_decomp_does_not_match_non_derivation():
    F = LocalFrame("F")
    rule = FrameDecompositionDefinition(F)
    assert not rule.matches(Symbol("V"))


def test_frame_decomp_rewrite_shape():
    F = LocalFrame("F")
    V = Derivation("V", 0)
    out = FrameDecompositionDefinition(F).rewrite(V)
    assert isinstance(out, IndexedSum)
    assert out.range_ is F
    assert isinstance(out.dummy, FrameIndex)
    assert out.dummy.is_bound
    body = out.body
    assert isinstance(body, Product)
    # body must contain a Pairing(coframe(a), V) and X_a.
    assert any(isinstance(c, Pairing) for c in body.children)
    assert any(isinstance(c, FrameVectorField) for c in body.children)


def test_frame_decomp_dummy_is_alpha_fresh_against_existing():
    F = LocalFrame("F")
    a = F.index("a")  # free outer index
    # Build a Derivation whose frame is *not* F, but craft something
    # that already has FrameIndex named "a" inside it.
    # A FrameVectorField of another frame with index name "a":
    G = LocalFrame("G")
    Va = G.X(a)
    out = FrameDecompositionDefinition(F).rewrite(Va)
    assert isinstance(out, IndexedSum)
    # Dummy must not collide with the free "a".
    assert out.dummy.name != "a" or out.dummy.is_bound


# --------------------------------------------------------------------- #
# ConnectionFormDecompositionDefinition                                  #
# --------------------------------------------------------------------- #


def test_conn_decomp_matches_nabla_v_xb():
    nabla = connection("∇")
    F = LocalFrame("F")
    V = Derivation("V", 0)
    b = F.index("b")
    expr = ConnectionEvalExpr(nabla, V, F.X(b))
    rule = ConnectionFormDecompositionDefinition(nabla, F)
    assert rule.matches(expr)


def test_conn_decomp_does_not_match_when_y_is_arbitrary_vf():
    nabla = connection("∇")
    F = LocalFrame("F")
    V = Derivation("V", 0)
    W = Derivation("W", 0)
    expr = ConnectionEvalExpr(nabla, V, W)
    assert not ConnectionFormDecompositionDefinition(nabla, F).matches(expr)


def test_conn_decomp_does_not_match_other_connection():
    nabla = connection("∇")
    other = connection("∇'")
    F = LocalFrame("F")
    V = Derivation("V", 0)
    b = F.index("b")
    expr = ConnectionEvalExpr(other, V, F.X(b))
    assert not ConnectionFormDecompositionDefinition(nabla, F).matches(expr)


def test_conn_decomp_does_not_match_other_frame():
    nabla = connection("∇")
    F = LocalFrame("F")
    G = LocalFrame("G")
    V = Derivation("V", 0)
    b = G.index("b")
    expr = ConnectionEvalExpr(nabla, V, G.X(b))
    assert not ConnectionFormDecompositionDefinition(nabla, F).matches(expr)


def test_conn_decomp_matches_when_b_is_bound():
    """Relaxed guard: bound ``b`` is the inner shape that arises inside
    a frame-decomposition IS body (Cartan I/II reductions). Fresh dummy
    ``d`` collision-avoidance keeps the rewrite shadow-safe.
    """
    nabla = connection("∇")
    F = LocalFrame("F")
    V = Derivation("V", 0)
    b = F.index("b", bound=True)
    expr = ConnectionEvalExpr(nabla, V, F.X(b))
    rule = ConnectionFormDecompositionDefinition(nabla, F)
    assert rule.matches(expr)
    out = rule.rewrite(expr)
    assert isinstance(out, IndexedSum)
    # The new dummy must not collide with the outer-bound ``b``.
    assert out.dummy.name != "b"


def test_conn_decomp_rewrite_shape():
    nabla = connection("∇")
    F = LocalFrame("F")
    V = Derivation("V", 0)
    b = F.index("b")
    expr = ConnectionEvalExpr(nabla, V, F.X(b))
    out = ConnectionFormDecompositionDefinition(nabla, F).rewrite(expr)
    assert isinstance(out, IndexedSum)
    assert out.range_ is F
    assert out.dummy.is_bound
    body = out.body
    assert isinstance(body, Product)
    # Locate the Pairing factor, should wrap a ConnectionForm.
    pairing_factor = next(c for c in body.children if isinstance(c, Pairing))
    assert isinstance(pairing_factor.alpha, ConnectionForm)
    assert pairing_factor.alpha.connection == nabla
    assert pairing_factor.alpha.frame == F
    assert pairing_factor.alpha.lower == b
    assert pairing_factor.alpha.upper == out.dummy
    assert pairing_factor.X is V
    # The X_c factor uses the same dummy.
    vf_factor = next(c for c in body.children if isinstance(c, FrameVectorField))
    assert vf_factor.idx == out.dummy


def test_conn_decomp_alpha_fresh_avoids_collision_with_b():
    nabla = connection("∇")
    F = LocalFrame("F")
    V = Derivation("V", 0)
    # If b is named "c", the alpha-fresh dummy must not also be "c".
    b = FrameIndex("c", "free")
    expr = ConnectionEvalExpr(nabla, V, F.X(b))
    out = ConnectionFormDecompositionDefinition(nabla, F).rewrite(expr)
    assert isinstance(out, IndexedSum)
    assert out.dummy.name != "c"


# --------------------------------------------------------------------- #
# ConnectionEvalYFrameDecompositionDefinition (Faz 17.F.2)                #
# --------------------------------------------------------------------- #


def test_eval_y_frame_decomp_matches_outside_derivation_y():
    nabla = connection("∇")
    F = LocalFrame("F")
    U = Derivation("U", 0)
    V = Derivation("V", 0)
    expr = ConnectionEvalExpr(nabla, U, V)
    rule = ConnectionEvalYFrameDecompositionDefinition(nabla, F)
    assert rule.matches(expr)


def test_eval_y_frame_decomp_does_not_match_frame_vf_y():
    """Y is X_b of THIS frame, handled by ConnectionFormDecomposition instead."""
    nabla = connection("∇")
    F = LocalFrame("F")
    U = Derivation("U", 0)
    b = F.index("b")
    expr = ConnectionEvalExpr(nabla, U, F.X(b))
    rule = ConnectionEvalYFrameDecompositionDefinition(nabla, F)
    assert not rule.matches(expr)


def test_eval_y_frame_decomp_matches_other_frame_vf_y():
    """Y is X_b of a DIFFERENT frame, decompose into THIS frame."""
    nabla = connection("∇")
    F = LocalFrame("F")
    G = LocalFrame("G")
    U = Derivation("U", 0)
    b = G.index("b")
    expr = ConnectionEvalExpr(nabla, U, G.X(b))
    rule = ConnectionEvalYFrameDecompositionDefinition(nabla, F)
    assert rule.matches(expr)


def test_eval_y_frame_decomp_does_not_match_other_connection():
    nabla = connection("∇")
    other = connection("∇'")
    F = LocalFrame("F")
    U = Derivation("U", 0)
    V = Derivation("V", 0)
    expr = ConnectionEvalExpr(other, U, V)
    rule = ConnectionEvalYFrameDecompositionDefinition(nabla, F)
    assert not rule.matches(expr)


def test_eval_y_frame_decomp_does_not_match_non_derivation_y():
    """Symbol-as-VF is not a Derivation; rule should skip."""
    nabla = connection("∇")
    F = LocalFrame("F")
    U = Derivation("U", 0)
    expr = ConnectionEvalExpr(nabla, U, Symbol("V"))
    rule = ConnectionEvalYFrameDecompositionDefinition(nabla, F)
    assert not rule.matches(expr)


def test_eval_y_frame_decomp_does_not_match_non_connection_eval():
    """Bare Derivation, no ConnectionEval wrapper, rule skips."""
    nabla = connection("∇")
    F = LocalFrame("F")
    V = Derivation("V", 0)
    rule = ConnectionEvalYFrameDecompositionDefinition(nabla, F)
    assert not rule.matches(V)


def test_eval_y_frame_decomp_rewrite_shape():
    nabla = connection("∇")
    F = LocalFrame("F")
    U = Derivation("U", 0)
    V = Derivation("V", 0)
    expr = ConnectionEvalExpr(nabla, U, V)
    out = ConnectionEvalYFrameDecompositionDefinition(nabla, F).rewrite(expr)
    # Outside is still ∇_U(...) with the same X.
    assert isinstance(out, ConnectionEvalExpr)
    assert out.connection == nabla
    assert out.X is U
    # Y is now an IndexedSum.
    assert isinstance(out.Y, IndexedSum)
    assert out.Y.range_ is F
    assert out.Y.dummy.is_bound
    body = out.Y.body
    assert isinstance(body, Product)
    pairing_factor = next(c for c in body.children if isinstance(c, Pairing))
    assert pairing_factor.X is V
    vf_factor = next(c for c in body.children if isinstance(c, FrameVectorField))
    assert vf_factor.idx == out.Y.dummy


def test_eval_y_frame_decomp_does_not_re_fire_after_rewrite():
    """Loop-safety: after the rewrite, Y is an IndexedSum, not a Derivation."""
    nabla = connection("∇")
    F = LocalFrame("F")
    U = Derivation("U", 0)
    V = Derivation("V", 0)
    expr = ConnectionEvalExpr(nabla, U, V)
    rule = ConnectionEvalYFrameDecompositionDefinition(nabla, F)
    out = rule.rewrite(expr)
    assert not rule.matches(out)


# --------------------------------------------------------------------- #
# Pipeline: contraction collapses ⟨e^a, decomposed VF⟩ via δ            #
# --------------------------------------------------------------------- #


def test_frame_decomp_loops_when_unbounded():
    """Standalone FrameDecompositionDefinition is *not* idempotent.

    The rewrite produces ``Σ_a e^a(V)·X_a`` where ``V`` is still an
    outside derivation; bottom-up walking re-matches ``V`` inside the
    body and expands again, ad infinitum. The Cartan-structure problem
    wrappers (Faz 17.F/G) tame this by registering the rule only on a
    one-shot pass against an explicit target, never as part of the
    fix-point bundle. This test pins that contract.
    """
    F = LocalFrame("F")
    V = Derivation("V", 0)
    engine = ExpansionEngine()
    engine.register(FrameDecompositionDefinition(F))
    with pytest.raises(RuntimeError):
        engine.expand(V, max_steps=8)


def test_pipeline_conn_decomp_pairing_collapse_to_omega():
    """⟨e^a, ∇_V X_b⟩ → ω^a_b(∇)(V) via decomposition + contraction."""
    nabla = connection("∇")
    F = LocalFrame("F")
    a = F.index("a")
    b = F.index("b")
    V = Derivation("V", 0)
    expr = Pairing(F.coframe(a), ConnectionEvalExpr(nabla, V, F.X(b)))
    engine = ExpansionEngine()
    engine.register(ConnectionFormDecompositionDefinition(nabla, F))
    engine.register(IndexedSumPairingPushInRightDefinition())
    engine.register(IndexedSumScalarPullDefinition())
    engine.register(IndexedSumKroneckerContractDefinition())
    engine.register(F.duality_definition())
    engine.register(PairingLinearityDefinition())
    engine.register(PairingScalarPullDefinition())
    result, _ = engine.expand(expr, max_steps=128)
    final = canonicalize(result)
    expected = Pairing(ConnectionForm(nabla, F, a, b), V)
    assert final == expected


def test_pipeline_conn_decomp_does_not_loop_with_form_definition_absent():
    """The conn-decomp rule does not loop on its own."""
    nabla = connection("∇")
    F = LocalFrame("F")
    b = F.index("b")
    V = Derivation("V", 0)
    expr = ConnectionEvalExpr(nabla, V, F.X(b))
    engine = ExpansionEngine()
    engine.register(ConnectionFormDecompositionDefinition(nabla, F))
    result, _ = engine.expand(expr, max_steps=32)
    # Single decomposition, produces an IS containing ω·X_c, then no
    # rule applies.
    assert isinstance(result, IndexedSum)


# --------------------------------------------------------------------- #
# Helpers                                                                #
# --------------------------------------------------------------------- #


def _has_indexed_sum(expr) -> bool:
    if isinstance(expr, IndexedSum):
        return True
    if expr.is_atom:
        return False
    return any(_has_indexed_sum(c) for c in expr.children)
