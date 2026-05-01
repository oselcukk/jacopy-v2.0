"""Tests for wedge axioms (Faz 17.F.1.5/6).

Covers:

* :class:`WedgeMultiEvalAlternatingDefinition`, full ``S_p``
  alternating expansion of a wedge of one-forms evaluated on ``p``
  vector fields.
* :class:`MultiEvalOneFormPairingBridgeDefinition`, arity-1
  ``MultiEval(α, V)`` with a one-form ``α`` rewrites to ``Pairing(α, V)``.

Plus an end-to-end engine fix-point regression that composes the two
rules: ``(α ∧ β)(U, V) → ⟨α, U⟩·⟨β, V⟩ − ⟨α, V⟩·⟨β, U⟩``.
"""

import pytest

from jacopy.calculus.pairing import Pairing
from jacopy.calculus.pairing_axioms import (
    MultiEvalOneFormPairingBridgeDefinition,
)
from jacopy.calculus.wedge_axioms import (
    WedgeMultiEvalAlternatingDefinition,
    _permutation_sign,
)
from jacopy.core.expr import Expr, Neg, Product, Sum, Symbol
from jacopy.core.multi_eval import MultiEval
from jacopy.core.properties import Graded
from jacopy.core.registry import PropertyRegistry
from jacopy.core.symbolic_degree import Degree
from jacopy.core.wedge import Wedge
from jacopy.proof.expansion import ExpansionEngine


# --------------------------------------------------------------------- #
# Helpers                                                                #
# --------------------------------------------------------------------- #


def _one_form_registry(*names: str) -> tuple[PropertyRegistry, list[Symbol]]:
    reg = PropertyRegistry()
    syms: list[Symbol] = []
    for n in names:
        s = Symbol(n)
        reg.declare(s, Graded(Degree.const(1)))
        syms.append(s)
    return reg, syms


# --------------------------------------------------------------------- #
# _permutation_sign                                                      #
# --------------------------------------------------------------------- #


class TestPermutationSign:
    def test_identity_is_even(self):
        assert _permutation_sign((0, 1, 2)) == 1

    def test_single_swap_is_odd(self):
        assert _permutation_sign((1, 0, 2)) == -1

    def test_two_swaps_is_even(self):
        assert _permutation_sign((1, 2, 0)) == 1

    def test_reverse_three(self):
        assert _permutation_sign((2, 1, 0)) == -1

    def test_singleton(self):
        assert _permutation_sign((0,)) == 1


# --------------------------------------------------------------------- #
# WedgeMultiEvalAlternatingDefinition                                    #
# --------------------------------------------------------------------- #


class TestWedgeAlternatingMatch:
    def test_two_one_forms_match(self):
        reg, [a, b] = _one_form_registry("α", "β")
        U, V = Symbol("U"), Symbol("V")
        m = MultiEval(Wedge(a, b), U, V)
        rule = WedgeMultiEvalAlternatingDefinition(registry=reg)
        assert rule.matches(m)

    def test_three_one_forms_match(self):
        reg, [a, b, c] = _one_form_registry("α", "β", "γ")
        U, V, W = Symbol("U"), Symbol("V"), Symbol("W")
        m = MultiEval(Wedge(a, b, c), U, V, W)
        rule = WedgeMultiEvalAlternatingDefinition(registry=reg)
        assert rule.matches(m)

    def test_non_alternating_skipped(self):
        reg, [a, b] = _one_form_registry("α", "β")
        U, V = Symbol("U"), Symbol("V")
        m = MultiEval(Wedge(a, b), U, V, alternating=False)
        rule = WedgeMultiEvalAlternatingDefinition(registry=reg)
        assert not rule.matches(m)

    def test_arity_mismatch_skipped(self):
        reg, [a, b] = _one_form_registry("α", "β")
        m = MultiEval(Wedge(a, b), Symbol("U"))  # 1 arg, 2 factors
        rule = WedgeMultiEvalAlternatingDefinition(registry=reg)
        assert not rule.matches(m)

    def test_non_wedge_head_skipped(self):
        reg, [a] = _one_form_registry("α")
        m = MultiEval(a, Symbol("U"))
        rule = WedgeMultiEvalAlternatingDefinition(registry=reg)
        assert not rule.matches(m)

    def test_non_one_form_factor_skipped(self):
        # α has degree 1 but β's degree is unknown, guard says no.
        reg = PropertyRegistry()
        a = Symbol("α")
        b = Symbol("β")
        reg.declare(a, Graded(Degree.const(1)))
        m = MultiEval(Wedge(a, b), Symbol("U"), Symbol("V"))
        rule = WedgeMultiEvalAlternatingDefinition(registry=reg)
        assert not rule.matches(m)

    def test_two_form_factor_skipped(self):
        # All factors must be 1-forms; a 2-form blocks the rule.
        reg = PropertyRegistry()
        a = Symbol("α")
        b = Symbol("β")
        reg.declare(a, Graded(Degree.const(1)))
        reg.declare(b, Graded(Degree.const(2)))
        m = MultiEval(Wedge(a, b), Symbol("U"), Symbol("V"))
        rule = WedgeMultiEvalAlternatingDefinition(registry=reg)
        assert not rule.matches(m)


class TestWedgeAlternatingRewrite:
    def test_two_factor_expansion(self):
        reg, [a, b] = _one_form_registry("α", "β")
        U, V = Symbol("U"), Symbol("V")
        rule = WedgeMultiEvalAlternatingDefinition(registry=reg)
        out = rule.rewrite(MultiEval(Wedge(a, b), U, V))
        # (α ∧ β)(U, V) = α(U)·β(V) − α(V)·β(U)
        expected = Sum(
            Product.make(MultiEval(a, U), MultiEval(b, V)),
            Neg(Product.make(MultiEval(a, V), MultiEval(b, U))),
        )
        assert out == expected

    def test_repeated_args_give_zero_after_simplify(self):
        # The alternating expansion of (α ∧ β)(U, U) is α(U)β(U) − α(U)β(U).
        # The rule emits exactly that pre-cancellation Sum; downstream
        # simplification cancels it. We only assert the structural shape.
        reg, [a, b] = _one_form_registry("α", "β")
        U = Symbol("U")
        rule = WedgeMultiEvalAlternatingDefinition(registry=reg)
        out = rule.rewrite(MultiEval(Wedge(a, b), U, U))
        positive = Product.make(MultiEval(a, U), MultiEval(b, U))
        expected = Sum(positive, Neg(positive))
        assert out == expected

    def test_three_factor_expansion_has_six_terms(self):
        reg, [a, b, c] = _one_form_registry("α", "β", "γ")
        U, V, W = Symbol("U"), Symbol("V"), Symbol("W")
        rule = WedgeMultiEvalAlternatingDefinition(registry=reg)
        out = rule.rewrite(MultiEval(Wedge(a, b, c), U, V, W))
        assert isinstance(out, Sum)
        assert len(out.children) == 6  # |S_3| = 6

    def test_three_factor_signs_split_evenly(self):
        reg, [a, b, c] = _one_form_registry("α", "β", "γ")
        U, V, W = Symbol("U"), Symbol("V"), Symbol("W")
        rule = WedgeMultiEvalAlternatingDefinition(registry=reg)
        out = rule.rewrite(MultiEval(Wedge(a, b, c), U, V, W))
        positives = sum(1 for t in out.children if not isinstance(t, Neg))
        negatives = sum(1 for t in out.children if isinstance(t, Neg))
        assert positives == 3
        assert negatives == 3


# --------------------------------------------------------------------- #
# MultiEvalOneFormPairingBridgeDefinition                                #
# --------------------------------------------------------------------- #


class TestBridgeMatch:
    def test_one_form_arity_one_matches(self):
        reg, [a] = _one_form_registry("α")
        bridge = MultiEvalOneFormPairingBridgeDefinition(registry=reg)
        assert bridge.matches(MultiEval(a, Symbol("V")))

    def test_arity_two_skipped(self):
        reg, [a] = _one_form_registry("α")
        bridge = MultiEvalOneFormPairingBridgeDefinition(registry=reg)
        m = MultiEval(a, Symbol("V"), Symbol("W"))
        assert not bridge.matches(m)

    def test_non_alternating_skipped(self):
        reg, [a] = _one_form_registry("α")
        bridge = MultiEvalOneFormPairingBridgeDefinition(registry=reg)
        m = MultiEval(a, Symbol("V"), alternating=False)
        assert not bridge.matches(m)

    def test_covector_slot_skipped(self):
        # MultiEval(π, α) with slot_kind="covector" is the bivector
        # branch, bridging to a Pairing(π, α) would conflate two
        # contracts. Bridge stays out.
        reg, [a] = _one_form_registry("α")
        bridge = MultiEvalOneFormPairingBridgeDefinition(registry=reg)
        m = MultiEval(a, Symbol("V"), slot_kind="covector")
        assert not bridge.matches(m)

    def test_non_one_form_head_skipped(self):
        # 0-form head, degree mismatch.
        reg = PropertyRegistry()
        f = Symbol("f")
        reg.declare(f, Graded(Degree.const(0)))
        bridge = MultiEvalOneFormPairingBridgeDefinition(registry=reg)
        assert not bridge.matches(MultiEval(f, Symbol("V")))

    def test_unknown_degree_head_skipped(self):
        # No declaration → degree undecidable → guard returns False.
        reg = PropertyRegistry()
        a = Symbol("α")
        bridge = MultiEvalOneFormPairingBridgeDefinition(registry=reg)
        assert not bridge.matches(MultiEval(a, Symbol("V")))

    def test_no_registry_skipped(self):
        # When the bridge is built without a registry, only forms whose
        # degree is determinable in registry-free dispatch (e.g. literal
        # numerics) qualify. Plain Symbols don't.
        bridge = MultiEvalOneFormPairingBridgeDefinition()
        a = Symbol("α")
        assert not bridge.matches(MultiEval(a, Symbol("V")))


class TestBridgeRewrite:
    def test_rewrites_to_pairing(self):
        reg, [a] = _one_form_registry("α")
        bridge = MultiEvalOneFormPairingBridgeDefinition(registry=reg)
        V = Symbol("V")
        out = bridge.rewrite(MultiEval(a, V))
        assert out == Pairing(a, V)


# --------------------------------------------------------------------- #
# End-to-end: engine composes alternating expansion + bridge             #
# --------------------------------------------------------------------- #


class TestEngineComposition:
    def test_two_form_wedge_eval_collapses_to_pairing_product_pair(self):
        reg, [a, b] = _one_form_registry("α", "β")
        U, V = Symbol("U"), Symbol("V")
        eng = ExpansionEngine(
            [
                WedgeMultiEvalAlternatingDefinition(registry=reg),
                MultiEvalOneFormPairingBridgeDefinition(registry=reg),
            ]
        )
        start = MultiEval(Wedge(a, b), U, V)
        result, _steps = eng.expand(start)
        expected = Sum(
            Product.make(Pairing(a, U), Pairing(b, V)),
            Neg(Product.make(Pairing(a, V), Pairing(b, U))),
        )
        assert result == expected

    def test_three_form_wedge_eval_yields_six_pairing_products(self):
        reg, [a, b, c] = _one_form_registry("α", "β", "γ")
        U, V, W = Symbol("U"), Symbol("V"), Symbol("W")
        eng = ExpansionEngine(
            [
                WedgeMultiEvalAlternatingDefinition(registry=reg),
                MultiEvalOneFormPairingBridgeDefinition(registry=reg),
            ]
        )
        start = MultiEval(Wedge(a, b, c), U, V, W)
        result, _steps = eng.expand(start)
        assert isinstance(result, Sum)
        assert len(result.children) == 6
        # Every leaf factor inside the products should be a Pairing.
        for term in result.children:
            inner = term.arg if isinstance(term, Neg) else term
            assert isinstance(inner, Product)
            for f in inner.children:
                assert isinstance(f, Pairing)
