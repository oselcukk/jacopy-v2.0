"""Tests for MultiEval C∞-linearity (Faz 12.B #6)."""

import pytest

from jacopy.calculus.multi_eval_scalar_axioms import (
    MultiEvalScalarPullDefinition,
)
from jacopy.core.expr import Product, Symbol
from jacopy.core.multi_eval import multi_eval
from jacopy.proof.expansion import ExpansionEngine


# --------------------------------------------------------------------- #
# Match logic                                                            #
# --------------------------------------------------------------------- #


class TestMultiEvalScalarPullMatches:
    def test_matches_bivector_first_arg(self):
        rule = MultiEvalScalarPullDefinition()
        f, pi, alpha, beta = (
            Symbol("f"),
            Symbol("π"),
            Symbol("α"),
            Symbol("β"),
        )
        expr = multi_eval(
            pi, Product(f, alpha), beta,
            slot_kind="covector", alternating=False,
        )
        assert rule.matches(expr)

    def test_matches_form_middle_arg(self):
        rule = MultiEvalScalarPullDefinition()
        f, omega, X, Y, Z = (
            Symbol("f"),
            Symbol("ω"),
            Symbol("X"),
            Symbol("Y"),
            Symbol("Z"),
        )
        expr = multi_eval(omega, X, Product(f, Y), Z)
        assert rule.matches(expr)

    def test_no_match_plain_args(self):
        rule = MultiEvalScalarPullDefinition()
        omega, X, Y = Symbol("ω"), Symbol("X"), Symbol("Y")
        assert not rule.matches(multi_eval(omega, X, Y))

    def test_no_match_non_multi_eval(self):
        rule = MultiEvalScalarPullDefinition()
        f, X = Symbol("f"), Symbol("X")
        assert not rule.matches(Product(f, X))


# --------------------------------------------------------------------- #
# Rewrite                                                                #
# --------------------------------------------------------------------- #


class TestMultiEvalScalarPullRewrite:
    def test_pulls_first_arg_factor(self):
        rule = MultiEvalScalarPullDefinition()
        f, pi, alpha, beta = (
            Symbol("f"),
            Symbol("π"),
            Symbol("α"),
            Symbol("β"),
        )
        expr = multi_eval(
            pi, Product(f, alpha), beta,
            slot_kind="covector", alternating=False,
        )
        out = rule.rewrite(expr)
        expected = Product(
            f,
            multi_eval(
                pi, alpha, beta,
                slot_kind="covector", alternating=False,
            ),
        )
        assert out == expected

    def test_pulls_middle_arg_factor(self):
        rule = MultiEvalScalarPullDefinition()
        f, omega, X, Y, Z = (
            Symbol("f"),
            Symbol("ω"),
            Symbol("X"),
            Symbol("Y"),
            Symbol("Z"),
        )
        expr = multi_eval(omega, X, Product(f, Y), Z)
        out = rule.rewrite(expr)
        assert out == Product(f, multi_eval(omega, X, Y, Z))

    def test_three_factor_folds_into_scalar(self):
        rule = MultiEvalScalarPullDefinition()
        g, f, omega, X, Y = (
            Symbol("g"),
            Symbol("f"),
            Symbol("ω"),
            Symbol("X"),
            Symbol("Y"),
        )
        expr = multi_eval(omega, Product(g, f, X), Y)
        out = rule.rewrite(expr)
        assert out == Product(Product(g, f), multi_eval(omega, X, Y))

    def test_preserves_alternating_and_slot_kind(self):
        rule = MultiEvalScalarPullDefinition()
        f, pi, alpha, beta = (
            Symbol("f"),
            Symbol("π"),
            Symbol("α"),
            Symbol("β"),
        )
        expr = multi_eval(
            pi, Product(f, alpha), beta,
            slot_kind="covector", alternating=False,
        )
        inner = rule.rewrite(expr).children[1]
        assert inner.slot_kind == "covector"
        assert inner.alternating is False

    def test_first_slot_wins_when_multiple_products(self):
        # Both slot 0 and slot 1 are Products, slot 0 fires first.
        rule = MultiEvalScalarPullDefinition()
        f, g, pi, alpha, beta = (
            Symbol("f"),
            Symbol("g"),
            Symbol("π"),
            Symbol("α"),
            Symbol("β"),
        )
        expr = multi_eval(
            pi, Product(f, alpha), Product(g, beta),
            slot_kind="covector", alternating=False,
        )
        out = rule.rewrite(expr)
        # Inner still has the second-slot Product to be cleaned next pass.
        assert out == Product(
            f,
            multi_eval(
                pi, alpha, Product(g, beta),
                slot_kind="covector", alternating=False,
            ),
        )


# --------------------------------------------------------------------- #
# Engine integration                                                     #
# --------------------------------------------------------------------- #


class TestMultiEvalScalarPullEngineIntegration:
    def test_engine_clears_all_slots(self):
        engine = ExpansionEngine([MultiEvalScalarPullDefinition()])
        f, g, pi, alpha, beta = (
            Symbol("f"),
            Symbol("g"),
            Symbol("π"),
            Symbol("α"),
            Symbol("β"),
        )
        expr = multi_eval(
            pi, Product(f, alpha), Product(g, beta),
            slot_kind="covector", alternating=False,
        )
        out, steps = engine.expand(expr)
        # Inner MultiEval is now plain π(α, β); the two scalars sit
        # outside as Products. Structurally:
        #   Product(f, Product(g, π(α, β)))
        clean_inner = multi_eval(
            pi, alpha, beta,
            slot_kind="covector", alternating=False,
        )
        assert out == Product(f, Product(g, clean_inner))
        assert len(steps) == 2

    def test_engine_no_op_on_clean(self):
        engine = ExpansionEngine([MultiEvalScalarPullDefinition()])
        omega, X, Y = Symbol("ω"), Symbol("X"), Symbol("Y")
        expr = multi_eval(omega, X, Y)
        out, steps = engine.expand(expr)
        assert out == expr
        assert steps == []
