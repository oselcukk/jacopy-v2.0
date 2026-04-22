"""Tests for gradalg.core.registry."""

import pytest

from gradalg.core.expr import Symbol, Integer
from gradalg.core.properties import (
    Graded,
    ProofRef,
    Provenance,
    Scalar,
    Symmetric,
)
from gradalg.core.registry import (
    PropertyRegistry,
    default_registry,
    reset_default_registry,
)


@pytest.fixture
def reg():
    return PropertyRegistry()


@pytest.fixture
def strict_reg():
    return PropertyRegistry(strict_axioms_only=True)


class TestDeclare:
    def test_basic(self, reg):
        x = Symbol("x")
        reg.declare(x, Scalar())
        assert reg.has(x, Scalar)

    def test_rejects_non_expr_key(self, reg):
        with pytest.raises(TypeError):
            reg.declare("x", Scalar())  # type: ignore[arg-type]

    def test_rejects_non_property_value(self, reg):
        x = Symbol("x")
        with pytest.raises(TypeError):
            reg.declare(x, "scalar")  # type: ignore[arg-type]

    def test_duplicate_type_raises(self, reg):
        x = Symbol("x")
        reg.declare(x, Scalar())
        with pytest.raises(ValueError):
            reg.declare(x, Scalar())

    def test_different_types_coexist(self, reg):
        x = Symbol("x")
        reg.declare(x, Scalar())
        reg.declare(x, Graded(degree=0))
        assert reg.has(x, Scalar)
        assert reg.has(x, Graded)


class TestGet:
    def test_missing_returns_none(self, reg):
        assert reg.get(Symbol("x"), Scalar) is None

    def test_returns_declared_property(self, reg):
        x = Symbol("x")
        g = Graded(degree=3)
        reg.declare(x, g)
        got = reg.get(x, Graded)
        assert got is g
        assert got.degree == 3

    def test_structural_key_lookup(self, reg):
        """Two structurally equal Symbols find the same entry."""
        reg.declare(Symbol("y"), Scalar())
        assert reg.has(Symbol("y"), Scalar)

    def test_different_types_do_not_collide(self, reg):
        x = Symbol("x")
        reg.declare(x, Scalar())
        assert reg.get(x, Graded) is None


class TestRetract:
    def test_removes_and_returns(self, reg):
        x = Symbol("x")
        s = Scalar()
        reg.declare(x, s)
        out = reg.retract(x, Scalar)
        assert out is s
        assert not reg.has(x, Scalar)

    def test_missing_is_none(self, reg):
        assert reg.retract(Symbol("x"), Scalar) is None

    def test_re_declare_after_retract(self, reg):
        x = Symbol("x")
        reg.declare(x, Scalar())
        reg.retract(x, Scalar)
        # No duplicate error now.
        reg.declare(x, Scalar())


class TestStrictMode:
    def test_hides_derived(self, strict_reg):
        x = Symbol("x")
        strict_reg.declare(
            x,
            Scalar(provenance=Provenance.DERIVED, proof=ProofRef("r")),
        )
        assert strict_reg.get(x, Scalar) is None
        assert not strict_reg.has(x, Scalar)

    def test_keeps_axioms(self, strict_reg):
        x = Symbol("x")
        strict_reg.declare(x, Scalar())
        assert strict_reg.has(x, Scalar)

    def test_toggle_reveals_derived(self):
        reg = PropertyRegistry(strict_axioms_only=True)
        x = Symbol("x")
        reg.declare(
            x,
            Graded(degree=1, provenance=Provenance.DERIVED, proof=ProofRef("r")),
        )
        assert reg.get(x, Graded) is None
        reg.set_strict(False)
        assert reg.has(x, Graded)

    def test_all_for_filters_derived(self, strict_reg):
        x = Symbol("x")
        strict_reg.declare(x, Scalar())  # axiom
        strict_reg.declare(
            x,
            Graded(
                degree=1,
                provenance=Provenance.DERIVED,
                proof=ProofRef("r"),
            ),
        )
        kinds = {type(p) for p in strict_reg.all_for(x)}
        assert kinds == {Scalar}


class TestAllFor:
    def test_yields_every_property(self, reg):
        x = Symbol("x")
        reg.declare(x, Scalar())
        reg.declare(x, Graded(degree=2))
        reg.declare(x, Symmetric())
        kinds = {type(p) for p in reg.all_for(x)}
        assert kinds == {Scalar, Graded, Symmetric}

    def test_empty_for_unknown(self, reg):
        assert list(reg.all_for(Symbol("z"))) == []


class TestDunder:
    def test_len(self, reg):
        x, y = Symbol("x"), Symbol("y")
        assert len(reg) == 0
        reg.declare(x, Scalar())
        reg.declare(x, Graded(degree=1))
        reg.declare(y, Scalar())
        assert len(reg) == 3

    def test_contains(self, reg):
        x = Symbol("x")
        assert x not in reg
        reg.declare(x, Scalar())
        assert x in reg
        assert Symbol("never") not in reg

    def test_contains_non_expr(self, reg):
        assert "x" not in reg


class TestDefaultRegistry:
    def test_singleton(self):
        assert default_registry() is default_registry()

    def test_reset(self):
        r1 = default_registry()
        r1.declare(Symbol("tmp"), Scalar())
        assert len(r1) == 1
        reset_default_registry()
        r2 = default_registry()
        assert r1 is not r2
        assert len(r2) == 0
