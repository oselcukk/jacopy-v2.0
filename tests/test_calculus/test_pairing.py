"""Tests for the canonical pairing ``⟨α, X⟩``."""

import pytest

from jacopy.algebra.derivation import degree_of
from jacopy.calculus.pairing import Pairing, pairing
from jacopy.core.expr import Expr, Symbol
from jacopy.core.symbolic_degree import Degree


class TestConstruction:
    def test_factory_matches_class(self):
        alpha, X = Symbol("α"), Symbol("X")
        a = pairing(alpha, X)
        b = Pairing(alpha, X)
        assert a == b
        assert isinstance(a, Pairing)

    def test_accessors(self):
        alpha, X = Symbol("α"), Symbol("X")
        p = pairing(alpha, X)
        assert p.alpha is alpha
        assert p.X is X
        assert p.children == (alpha, X)

    def test_is_not_atom(self):
        p = pairing(Symbol("α"), Symbol("X"))
        assert not p.is_atom

    def test_repr_shape(self):
        p = pairing(Symbol("α"), Symbol("X"))
        assert "⟨" in p._repr_inner() and "⟩" in p._repr_inner()

    def test_alpha_must_be_expr(self):
        with pytest.raises(TypeError):
            Pairing("α", Symbol("X"))  # type: ignore[arg-type]

    def test_X_must_be_expr(self):
        with pytest.raises(TypeError):
            Pairing(Symbol("α"), "X")  # type: ignore[arg-type]


class TestEquality:
    def test_structural_equality(self):
        a1 = pairing(Symbol("α"), Symbol("X"))
        a2 = pairing(Symbol("α"), Symbol("X"))
        assert a1 == a2
        assert hash(a1) == hash(a2)

    def test_slot_order_matters(self):
        """``⟨α, X⟩`` and ``⟨X, α⟩`` are distinct Exprs, the semantic
        convention (first slot: 1-form) is the caller's responsibility."""
        alpha, X = Symbol("α"), Symbol("X")
        assert pairing(alpha, X) != pairing(X, alpha)


class TestDegree:
    def test_pairing_is_scalar(self):
        """The pairing lands in ℝ regardless of its inputs, degree 0."""
        p = pairing(Symbol("α"), Symbol("X"))
        assert degree_of(p) == Degree.const(0)
