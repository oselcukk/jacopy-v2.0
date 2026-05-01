"""Tests for the IndexedSum engine axioms, Faz 17.E.3-E.6 + Faz 17.F.1."""

from __future__ import annotations

import pytest

from jacopy.algebra.derivation import Derivation
from jacopy.algorithms.canonicalize import canonicalize
from jacopy.calculus.cartan_forms import ConnectionForm
from jacopy.calculus.connection import ConnectionEvalExpr, connection
from jacopy.calculus.indexed_sum_axioms import (
    ConnectionEvalIndexedSumPushInDefinition,
    IndexedSumKroneckerContractDefinition,
    IndexedSumNegPullDefinition,
    IndexedSumPairingPushInLeftDefinition,
    IndexedSumPairingPushInRightDefinition,
    IndexedSumScalarPullDefinition,
    IndexedSumSumDistributeDefinition,
    MultiEvalIndexedSumPushInDefinition,
)
from jacopy.calculus.local_frame import KroneckerDelta, LocalFrame
from jacopy.calculus.pairing import Pairing
from jacopy.calculus.pairing_axioms import PairingLinearityDefinition
from jacopy.calculus import PairingScalarPullDefinition
from jacopy.core.expr import Neg, One, Product, Sum, Symbol
from jacopy.core.indexed_sum import IndexedSum
from jacopy.core.multi_eval import MultiEval
from jacopy.core.wedge import Wedge
from jacopy.proof.expansion import ExpansionEngine


# --------------------------------------------------------------------- #
# 17.E.3, Sum / Neg distribute                                          #
# --------------------------------------------------------------------- #


def test_sum_distribute_matches_only_sum_body():
    F = LocalFrame("F")
    b = F.index("b", bound=True)
    rule = IndexedSumSumDistributeDefinition()
    s_with_sum = IndexedSum(b, F, Sum.make(F.X(b), F.coframe(b)))
    s_no_sum = IndexedSum(b, F, F.X(b))
    assert rule.matches(s_with_sum)
    assert not rule.matches(s_no_sum)


def test_sum_distribute_rewrites_to_sum_of_indexed_sums():
    F = LocalFrame("F")
    b = F.index("b", bound=True)
    body = Sum.make(F.X(b), F.coframe(b))
    s = IndexedSum(b, F, body)
    out = IndexedSumSumDistributeDefinition().rewrite(s)
    assert isinstance(out, Sum)
    assert all(isinstance(c, IndexedSum) for c in out.children)


def test_neg_pull_rewrites_correctly():
    F = LocalFrame("F")
    b = F.index("b", bound=True)
    s = IndexedSum(b, F, Neg(F.X(b)))
    out = IndexedSumNegPullDefinition().rewrite(s)
    assert isinstance(out, Neg)
    assert isinstance(out.arg, IndexedSum)


def test_neg_pull_does_not_match_when_body_is_not_neg():
    F = LocalFrame("F")
    b = F.index("b", bound=True)
    s = IndexedSum(b, F, F.X(b))
    assert not IndexedSumNegPullDefinition().matches(s)


# --------------------------------------------------------------------- #
# Scalar pull                                                            #
# --------------------------------------------------------------------- #


def test_scalar_pull_separates_dummy_free_factor():
    F = LocalFrame("F")
    b = F.index("b", bound=True)
    c = Symbol("c")
    s = IndexedSum(b, F, Product.make(c, F.X(b)))
    rule = IndexedSumScalarPullDefinition()
    assert rule.matches(s)
    out = rule.rewrite(s)
    assert isinstance(out, Product)
    inner = out.children[-1]
    assert isinstance(inner, IndexedSum)
    assert inner.body == F.X(b)


def test_scalar_pull_does_not_match_when_all_factors_dummy_dependent():
    F = LocalFrame("F")
    b = F.index("b", bound=True)
    s = IndexedSum(b, F, Product.make(F.X(b), F.coframe(b)))
    rule = IndexedSumScalarPullDefinition()
    assert not rule.matches(s)


def test_scalar_pull_pulls_multiple_factors_at_once():
    F = LocalFrame("F")
    b = F.index("b", bound=True)
    c = Symbol("c")
    d = Symbol("d")
    s = IndexedSum(b, F, Product.make(c, d, F.X(b)))
    out = IndexedSumScalarPullDefinition().rewrite(s)
    # Both c and d should be on the outside.
    assert isinstance(out, Product)
    factors_outside = list(out.children[:-1])
    assert c in factors_outside
    assert d in factors_outside


def test_scalar_pull_when_body_has_no_dependent_factor_leaves_one_inside():
    F = LocalFrame("F")
    b = F.index("b", bound=True)
    c = Symbol("c")
    d = Symbol("d")
    s = IndexedSum(b, F, Product.make(c, d))
    out = IndexedSumScalarPullDefinition().rewrite(s)
    inner = out.children[-1]
    assert isinstance(inner, IndexedSum)
    assert inner.body == One


# --------------------------------------------------------------------- #
# 17.E.5, Pairing push-in                                               #
# --------------------------------------------------------------------- #


def test_pairing_push_in_right_when_alpha_dummy_free():
    F = LocalFrame("F")
    a = F.index("a")
    b = F.index("b", bound=True)
    e_a = F.coframe(a)
    s = IndexedSum(b, F, F.X(b))
    expr = Pairing(e_a, s)
    rule = IndexedSumPairingPushInRightDefinition()
    assert rule.matches(expr)
    out = rule.rewrite(expr)
    assert isinstance(out, IndexedSum)
    assert isinstance(out.body, Pairing)
    assert out.body.alpha == e_a


def test_pairing_push_in_right_does_not_match_when_alpha_uses_dummy():
    F = LocalFrame("F")
    b = F.index("b", bound=True)
    s = IndexedSum(b, F, F.X(b))
    # alpha references the dummy via a coframe of the same name
    expr = Pairing(F.coframe(b), s)
    assert not IndexedSumPairingPushInRightDefinition().matches(expr)


def test_pairing_push_in_left_when_X_dummy_free():
    F = LocalFrame("F")
    a = F.index("a")
    b = F.index("b", bound=True)
    s = IndexedSum(b, F, F.coframe(b))
    expr = Pairing(s, F.X(a))
    rule = IndexedSumPairingPushInLeftDefinition()
    assert rule.matches(expr)
    out = rule.rewrite(expr)
    assert isinstance(out, IndexedSum)
    assert isinstance(out.body, Pairing)


# --------------------------------------------------------------------- #
# 17.E.6, Kronecker contraction                                         #
# --------------------------------------------------------------------- #


def test_kronecker_contract_body_just_delta_a_b():
    F = LocalFrame("F")
    a = F.index("a")
    b = F.index("b", bound=True)
    s = IndexedSum(b, F, KroneckerDelta(a, b))
    out = IndexedSumKroneckerContractDefinition().rewrite(s)
    assert out == One


def test_kronecker_contract_body_just_delta_b_a():
    F = LocalFrame("F")
    a = F.index("a")
    b = F.index("b", bound=True)
    s = IndexedSum(b, F, KroneckerDelta(b, a))
    out = IndexedSumKroneckerContractDefinition().rewrite(s)
    assert out == One


def test_kronecker_contract_substitutes_dummy_in_rest():
    F = LocalFrame("F")
    a = F.index("a")
    b = F.index("b", bound=True)
    s = IndexedSum(b, F, Product.make(KroneckerDelta(a, b), F.X(b)))
    raw = IndexedSumKroneckerContractDefinition().rewrite(s)
    assert canonicalize(raw) == F.X(a)


def test_kronecker_contract_substitutes_in_connection_form_lower():
    nabla = connection("∇")
    F = LocalFrame("F")
    a = F.index("a")
    c = F.index("c")
    b = F.index("b", bound=True)
    body = Product.make(KroneckerDelta(c, b), ConnectionForm(nabla, F, a, b))
    s = IndexedSum(b, F, body)
    out = canonicalize(IndexedSumKroneckerContractDefinition().rewrite(s))
    assert out == ConnectionForm(nabla, F, a, c)


def test_kronecker_contract_does_not_match_without_delta():
    F = LocalFrame("F")
    b = F.index("b", bound=True)
    s = IndexedSum(b, F, F.X(b))
    assert not IndexedSumKroneckerContractDefinition().matches(s)


def test_kronecker_contract_does_not_match_when_both_indices_bound():
    F = LocalFrame("F")
    b = F.index("b", bound=True)
    c = F.index("c", bound=True)
    s = IndexedSum(b, F, KroneckerDelta(c, b))
    # Partner would be the bound `c`, which isn't free → no match.
    assert not IndexedSumKroneckerContractDefinition().matches(s)


# --------------------------------------------------------------------- #
# end-to-end pipeline                                                    #
# --------------------------------------------------------------------- #


def test_pipeline_scalar_paired_indexed_sum_reduces_to_scalar():
    F = LocalFrame("F")
    a = F.index("a")
    b = F.index("b", bound=True)
    f = Symbol("f")
    expr = Pairing(F.coframe(a), IndexedSum(b, F, Product.make(f, F.X(b))))
    engine = ExpansionEngine()
    engine.register(IndexedSumPairingPushInRightDefinition())
    engine.register(IndexedSumScalarPullDefinition())
    engine.register(IndexedSumKroneckerContractDefinition())
    engine.register(F.duality_definition())
    engine.register(PairingLinearityDefinition())
    engine.register(PairingScalarPullDefinition())
    result, _ = engine.expand(expr, max_steps=64)
    assert canonicalize(result) == f


# --------------------------------------------------------------------- #
# 17.F.1, ConnectionEval push-in over IndexedSum                        #
# --------------------------------------------------------------------- #


def test_conn_eval_pushin_matches_is_in_y_slot():
    nabla = connection("∇")
    F = LocalFrame("F")
    V = Derivation("V", 0)
    b = F.index("b", bound=True)
    body = Product.make(F.coframe(b), F.X(b))
    expr = ConnectionEvalExpr(nabla, V, IndexedSum(b, F, body))
    rule = ConnectionEvalIndexedSumPushInDefinition(nabla)
    assert rule.matches(expr)


def test_conn_eval_pushin_does_not_match_other_connection():
    nabla = connection("∇")
    other = connection("∇'")
    F = LocalFrame("F")
    V = Derivation("V", 0)
    b = F.index("b", bound=True)
    expr = ConnectionEvalExpr(other, V, IndexedSum(b, F, F.X(b)))
    assert not ConnectionEvalIndexedSumPushInDefinition(nabla).matches(expr)


def test_conn_eval_pushin_does_not_match_non_is_y():
    nabla = connection("∇")
    F = LocalFrame("F")
    V = Derivation("V", 0)
    W = Derivation("W", 0)
    expr = ConnectionEvalExpr(nabla, V, W)
    assert not ConnectionEvalIndexedSumPushInDefinition(nabla).matches(expr)


def test_conn_eval_pushin_rewrite_shape():
    nabla = connection("∇")
    F = LocalFrame("F")
    V = Derivation("V", 0)
    b = F.index("b", bound=True)
    body = F.X(b)
    expr = ConnectionEvalExpr(nabla, V, IndexedSum(b, F, body))
    out = ConnectionEvalIndexedSumPushInDefinition(nabla).rewrite(expr)
    assert isinstance(out, IndexedSum)
    assert out.dummy == b
    assert out.range_ is F
    inner = out.body
    assert isinstance(inner, ConnectionEvalExpr)
    assert inner.connection == nabla
    assert inner.X is V
    assert inner.Y == F.X(b)


def test_conn_eval_pushin_does_not_match_when_x_uses_dummy():
    """Guard: ``X`` must be dummy-free in the IS dummy. If ``X`` itself
    contains the dummy index, pushing ``∇_X`` past the binder is a
    capture violation."""
    nabla = connection("∇")
    F = LocalFrame("F")
    b = F.index("b", bound=True)
    # X uses the same bound name; ∇_{X_b}(IS_b body), pushing in would capture.
    expr = ConnectionEvalExpr(nabla, F.X(b), IndexedSum(b, F, F.X(b)))
    assert not ConnectionEvalIndexedSumPushInDefinition(nabla).matches(expr)


# --------------------------------------------------------------------- #
# 17.F.2, MultiEval push-in over IndexedSum                             #
# --------------------------------------------------------------------- #


def test_multieval_pushin_matches_is_in_head():
    F = LocalFrame("F")
    b = F.index("b", bound=True)
    U = Symbol("U")
    V = Symbol("V")
    body = Wedge(F.coframe(b), F.coframe(b))  # structural; degree-2-ish, doesn't matter for shape
    expr = MultiEval(IndexedSum(b, F, body), U, V)
    assert MultiEvalIndexedSumPushInDefinition().matches(expr)


def test_multieval_pushin_does_not_match_non_is_head():
    a = Symbol("α")
    expr = MultiEval(a, Symbol("U"), Symbol("V"))
    assert not MultiEvalIndexedSumPushInDefinition().matches(expr)


def test_multieval_pushin_does_not_match_when_arg_uses_dummy():
    F = LocalFrame("F")
    b = F.index("b", bound=True)
    body = F.coframe(b)
    # arg uses the same bound name; pushing in would capture.
    expr = MultiEval(IndexedSum(b, F, body), F.X(b))
    assert not MultiEvalIndexedSumPushInDefinition().matches(expr)


def test_multieval_pushin_rewrite_shape():
    F = LocalFrame("F")
    b = F.index("b", bound=True)
    U = Symbol("U")
    V = Symbol("V")
    body = Wedge(F.coframe(b), F.coframe(b))
    expr = MultiEval(IndexedSum(b, F, body), U, V)
    out = MultiEvalIndexedSumPushInDefinition().rewrite(expr)
    assert isinstance(out, IndexedSum)
    assert out.dummy == b
    assert out.range_ is F
    inner = out.body
    assert isinstance(inner, MultiEval)
    assert inner.head == body
    assert inner.args == (U, V)


def test_multieval_pushin_preserves_alternating_and_slot_kind():
    F = LocalFrame("F")
    b = F.index("b", bound=True)
    body = F.coframe(b)
    expr = MultiEval(
        IndexedSum(b, F, body),
        Symbol("U"),
        alternating=False,
        slot_kind="covector",
    )
    out = MultiEvalIndexedSumPushInDefinition().rewrite(expr)
    assert isinstance(out, IndexedSum)
    inner = out.body
    assert isinstance(inner, MultiEval)
    assert inner.alternating is False
    assert inner.slot_kind == "covector"


def test_multieval_pushin_does_not_match_plain_multieval():
    """Sanity: a MultiEval whose head is not an IndexedSum stays put."""
    F = LocalFrame("F")
    a = F.index("a")
    expr = MultiEval(F.coframe(a), Symbol("U"))
    assert not MultiEvalIndexedSumPushInDefinition().matches(expr)
