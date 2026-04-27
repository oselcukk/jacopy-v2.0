"""Tests for jacopy.algorithms.base."""

import pytest

from jacopy.algorithms.base import Algorithm, StepResult
from jacopy.core.expr import Expr, Integer, Neg, Symbol


class TestStepResult:
    def test_unchanged(self):
        x = Symbol("x")
        r = StepResult.unchanged(x)
        assert r.before is x
        assert r.after is x
        assert r.changed is False

    def test_rewrite_detects_change(self):
        x, y = Symbol("x"), Symbol("y")
        r = StepResult.rewrite(x, y)
        assert r.changed is True

    def test_rewrite_detects_noop(self):
        x = Symbol("x")
        # Structurally equal before/after => no change.
        r = StepResult.rewrite(x, Symbol("x"))
        assert r.changed is False

    def test_frozen(self):
        r = StepResult.unchanged(Symbol("x"))
        with pytest.raises(Exception):
            r.changed = True  # type: ignore[misc]


class _NegateAlgo(Algorithm):
    """Test algorithm: negates top-level Symbols, passes through everything else."""

    def can_apply(self, expr: Expr) -> bool:
        return isinstance(expr, Symbol)

    def apply(self, expr: Expr) -> Expr:
        return Neg(expr)


class TestAlgorithm:
    def test_run_produces_changed(self):
        algo = _NegateAlgo()
        r = algo.run(Symbol("x"))
        assert r.changed
        assert r.after == Neg(Symbol("x"))

    def test_run_unchanged_when_cannot_apply(self):
        algo = _NegateAlgo()
        expr = Integer(3)
        r = algo.run(expr)
        assert r.changed is False
        assert r.after is expr

    def test_abstract_cannot_instantiate(self):
        with pytest.raises(TypeError):
            Algorithm()  # type: ignore[abstract]
