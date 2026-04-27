"""Tests for jacopy.algorithms.rewrite."""

import pytest

from jacopy.algorithms.rewrite import (
    Rule,
    apply_bottomup,
    apply_once_at_root,
    apply_topdown,
    normalize,
)
from jacopy.core.expr import Integer, Neg, Product, Sum, Symbol
from jacopy.core.properties import Scalar
from jacopy.core.registry import PropertyRegistry
from jacopy.core.wildcards import SeqWildcard, Wildcard


# --------------------------------------------------------------------- #
# Rule                                                                   #
# --------------------------------------------------------------------- #


class TestRule:
    def test_try_at_match(self):
        x = Symbol("x")
        rule = Rule(lhs=Wildcard("A") + Wildcard("A"),
                    rhs=Integer(2) * Wildcard("A"),
                    name="sum-to-scalar")
        assert rule.try_at(x + x) == Integer(2) * x

    def test_try_at_no_match(self):
        rule = Rule(lhs=Wildcard("A") + Wildcard("A"),
                    rhs=Integer(2) * Wildcard("A"))
        assert rule.try_at(Symbol("x") + Symbol("y")) is None

    def test_guard_rejects(self):
        def only_scalar(bindings, registry):
            if registry is None:
                return False
            return registry.has(bindings["A"], Scalar)

        reg = PropertyRegistry()
        reg.declare(Symbol("f"), Scalar())

        rule = Rule(lhs=Wildcard("A") + Wildcard("A"),
                    rhs=Integer(2) * Wildcard("A"),
                    guard=only_scalar)
        # f is scalar -> fires
        assert rule.try_at(Symbol("f") + Symbol("f"), reg) == Integer(2) * Symbol("f")
        # x is not scalar -> rejected
        assert rule.try_at(Symbol("x") + Symbol("x"), reg) is None

    def test_guard_without_registry(self):
        called = {"n": 0}

        def record(bindings, registry):
            called["n"] += 1
            return True

        rule = Rule(lhs=Wildcard("A"), rhs=Wildcard("A"), guard=record)
        rule.try_at(Symbol("x"))
        assert called["n"] == 1


# --------------------------------------------------------------------- #
# Root-level application                                                 #
# --------------------------------------------------------------------- #


class TestApplyOnceAtRoot:
    def test_first_matching_wins(self):
        x = Symbol("x")
        r1 = Rule(lhs=Wildcard("A") + Wildcard("B"),
                  rhs=Wildcard("B") + Wildcard("A"),
                  name="commute")
        r2 = Rule(lhs=Wildcard("A") + Wildcard("B"),
                  rhs=Integer(0),
                  name="zero")
        # Both would match; first wins.
        assert apply_once_at_root([r1, r2], x + Symbol("y")) == Symbol("y") + x

    def test_no_rule_matches(self):
        r = Rule(lhs=Integer(5), rhs=Integer(6))
        assert apply_once_at_root([r], Symbol("x")) is None


# --------------------------------------------------------------------- #
# Bottom-up                                                              #
# --------------------------------------------------------------------- #


class TestApplyBottomup:
    def test_rewrites_in_children(self):
        # x + x -> 2*x, then the outer expr sees a Sum rewritten.
        r = Rule(lhs=Wildcard("A") + Wildcard("A"),
                 rhs=Integer(2) * Wildcard("A"))
        x, y = Symbol("x"), Symbol("y")
        # Target: (x + x) * y. Bottom-up: inner (x + x) -> 2*x; outer unchanged.
        target = (x + x) * y
        result = apply_bottomup([r], target)
        assert result == (Integer(2) * x) * y

    def test_atom_passes_through(self):
        r = Rule(lhs=Integer(5), rhs=Integer(6))
        assert apply_bottomup([r], Symbol("x")) == Symbol("x")

    def test_cascading_rewrite(self):
        # Nested Sums to keep arity-2 structure (Sum + Sum.make would flatten).
        r = Rule(lhs=Wildcard("A") + Wildcard("A"),
                 rhs=Integer(2) * Wildcard("A"))
        x = Symbol("x")
        inner = Sum(x, x)
        target = Sum(inner, inner)
        # After child rewrites: Sum(2*x, 2*x); outer then matches
        # ?A + ?A with A = 2*x.
        result = apply_bottomup([r], target)
        # wildcards.substitute keeps rhs shape verbatim (no smart-ctor
        # flattening), so the outer rewrite leaves a nested Product.
        assert result == Product(Integer(2), Product(Integer(2), x))

    def test_identity_rule_no_cycle(self):
        """A rule that rewrites to itself produces the same tree."""
        r = Rule(lhs=Wildcard("A"), rhs=Wildcard("A"))
        # apply_bottomup hits every node but result structure is identical.
        x = Symbol("x")
        assert apply_bottomup([r], x + Symbol("y")) == x + Symbol("y")


# --------------------------------------------------------------------- #
# Top-down                                                               #
# --------------------------------------------------------------------- #


class TestApplyTopdown:
    def test_root_rewrite_before_children(self):
        # Rule that consumes the whole thing at the root;
        # children are never examined.
        r = Rule(lhs=Wildcard("A") + Wildcard("B"),
                 rhs=Integer(0))
        x, y = Symbol("x"), Symbol("y")
        target = Sum(Sum(x, y), Sum(x, y))  # explicit nesting
        # Top-down: at root, matches ?A + ?B -> 0. Inner never touched.
        assert apply_topdown([r], target) == Integer(0)

    def test_recurses_after_rewrite(self):
        # If root matches, we rewrite and then recurse into the rewrite.
        r = Rule(lhs=Symbol("x"), rhs=Symbol("y"))
        # Top-down at root=x -> y; then recurse into y (atom).
        assert apply_topdown([r], Symbol("x")) == Symbol("y")


# --------------------------------------------------------------------- #
# normalize                                                              #
# --------------------------------------------------------------------- #


class TestNormalize:
    def test_converges(self):
        # x + x -> 2*x; then no further rewrites.
        r = Rule(lhs=Wildcard("A") + Wildcard("A"),
                 rhs=Integer(2) * Wildcard("A"))
        x = Symbol("x")
        assert normalize([r], x + x) == Integer(2) * x

    def test_idempotent_at_fixed_point(self):
        r = Rule(lhs=Wildcard("A") + Wildcard("A"),
                 rhs=Integer(2) * Wildcard("A"))
        once = normalize([r], Symbol("x") + Symbol("x"))
        twice = normalize([r], once)
        assert once == twice

    def test_non_converging_raises(self):
        # A rule that swaps children: x + y -> y + x, which swaps again…
        r = Rule(lhs=Wildcard("A") + Wildcard("B"),
                 rhs=Wildcard("B") + Wildcard("A"),
                 name="flip")
        with pytest.raises(RuntimeError, match="did not converge"):
            normalize([r], Symbol("x") + Symbol("y"), max_iter=10)

    def test_unknown_strategy_raises(self):
        with pytest.raises(ValueError, match="Unknown strategy"):
            normalize([], Symbol("x"), strategy="sideways")

    def test_sequential_rules_interact(self):
        # d(a+b) -> d(a) + d(b); d(x) -> 0. Normalizing should fold fully.
        A, B = Wildcard("A"), Wildcard("B")
        d = Symbol("d")
        linearity = Rule(
            lhs=d * (A + B),
            rhs=d * A + d * B,
            name="linearity",
        )
        target = d * (Symbol("x") + Symbol("y"))
        result = normalize([linearity], target)
        assert result == (d * Symbol("x")) + (d * Symbol("y"))
