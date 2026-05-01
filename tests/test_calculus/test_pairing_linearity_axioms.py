"""Tests for the Pairing C∞-linearity axiom (Faz 12.B #12)."""

import pytest

from jacopy.calculus.pairing import Pairing, pairing
from jacopy.calculus.pairing_linearity_axioms import (
    PairingScalarPullDefinition,
)
from jacopy.core.expr import Product, Symbol
from jacopy.proof.expansion import ExpansionEngine


# --------------------------------------------------------------------- #
# Match logic                                                            #
# --------------------------------------------------------------------- #


class TestPairingScalarPullMatches:
    def test_matches_x_slot_product(self):
        rule = PairingScalarPullDefinition()
        f, alpha, X = Symbol("f"), Symbol("α"), Symbol("X")
        assert rule.matches(pairing(alpha, Product(f, X)))

    def test_matches_alpha_slot_product(self):
        rule = PairingScalarPullDefinition()
        f, alpha, X = Symbol("f"), Symbol("α"), Symbol("X")
        assert rule.matches(pairing(Product(f, alpha), X))

    def test_matches_both_slots_product(self):
        rule = PairingScalarPullDefinition()
        f, alpha, X = Symbol("f"), Symbol("α"), Symbol("X")
        assert rule.matches(pairing(Product(f, alpha), Product(f, X)))

    def test_no_match_plain_pairing(self):
        rule = PairingScalarPullDefinition()
        alpha, X = Symbol("α"), Symbol("X")
        assert not rule.matches(pairing(alpha, X))

    def test_no_match_non_pairing(self):
        rule = PairingScalarPullDefinition()
        f, X = Symbol("f"), Symbol("X")
        assert not rule.matches(Product(f, X))

    def test_no_match_one_factor_product(self):
        # Product.make folds a single child away, so only an explicitly
        # constructed Product(x) would have len 1, and even then, the
        # rule requires >= 2 factors, so it shouldn't fire.
        rule = PairingScalarPullDefinition()
        alpha, X = Symbol("α"), Symbol("X")
        single = Product(X)
        assert not rule.matches(pairing(alpha, single))


# --------------------------------------------------------------------- #
# Rewrite                                                                #
# --------------------------------------------------------------------- #


class TestPairingScalarPullRewrite:
    def test_x_slot_pull(self):
        rule = PairingScalarPullDefinition()
        f, alpha, X = Symbol("f"), Symbol("α"), Symbol("X")
        out = rule.rewrite(pairing(alpha, Product(f, X)))
        assert out == Product(f, pairing(alpha, X))

    def test_alpha_slot_pull(self):
        rule = PairingScalarPullDefinition()
        f, alpha, X = Symbol("f"), Symbol("α"), Symbol("X")
        out = rule.rewrite(pairing(Product(f, alpha), X))
        assert out == Product(f, pairing(alpha, X))

    def test_three_factor_folds_leading_into_scalar(self):
        rule = PairingScalarPullDefinition()
        g, f, alpha, X = (
            Symbol("g"),
            Symbol("f"),
            Symbol("α"),
            Symbol("X"),
        )
        out = rule.rewrite(pairing(alpha, Product(g, f, X)))
        assert out == Product(Product(g, f), pairing(alpha, X))

    def test_alpha_first_when_both_slots_product(self):
        # alpha slot pulled first; the engine fires again on the
        # remaining inner Pairing to clear the X slot.
        rule = PairingScalarPullDefinition()
        f, g, alpha, X = (
            Symbol("f"),
            Symbol("g"),
            Symbol("α"),
            Symbol("X"),
        )
        out = rule.rewrite(pairing(Product(f, alpha), Product(g, X)))
        assert out == Product(f, pairing(alpha, Product(g, X)))


# --------------------------------------------------------------------- #
# Engine integration                                                     #
# --------------------------------------------------------------------- #


class TestPairingScalarPullEngineIntegration:
    def test_engine_clears_both_slots(self):
        engine = ExpansionEngine([PairingScalarPullDefinition()])
        f, g, alpha, X = (
            Symbol("f"),
            Symbol("g"),
            Symbol("α"),
            Symbol("X"),
        )
        expr = pairing(Product(f, alpha), Product(g, X))
        out, steps = engine.expand(expr)
        # After two passes both slots are clean, the result is a flat
        # Product(f, g, ⟨α, X⟩) (or with nested f / g that simplify
        # would flatten; structurally we accept either).
        assert pairing(alpha, X) in _atoms(out)
        # Both scalars must appear as factors somewhere in the chain.
        assert _has_scalar(out, f)
        assert _has_scalar(out, g)
        assert len(steps) == 2

    def test_engine_no_op_when_clean(self):
        engine = ExpansionEngine([PairingScalarPullDefinition()])
        alpha, X = Symbol("α"), Symbol("X")
        expr = pairing(alpha, X)
        out, steps = engine.expand(expr)
        assert out == expr
        assert steps == []


def _atoms(e):
    yield e
    for c in getattr(e, "children", ()):
        yield from _atoms(c)


def _has_scalar(e, s):
    return any(a == s for a in _atoms(e))
