"""Tests for jacopy.core.registry."""

import pytest

from jacopy.core.expr import Expr, Integer, Symbol
from jacopy.core.properties import (
    Antisymmetric,
    Graded,
    NonCommuting,
    ProofRef,
    Provenance,
    Scalar,
    Symmetric,
)
from jacopy.core.registry import (
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


# --------------------------------------------------------------------- #
# Class-based declarations                                              #
# --------------------------------------------------------------------- #


class TestDeclareForClass:
    def test_applies_to_every_instance(self, reg):
        reg.declare_for_class(Integer, Scalar())
        assert reg.has(Integer(2), Scalar)
        assert reg.has(Integer(99), Scalar)

    def test_exact_overrides_class(self, reg):
        x = Integer(7)
        reg.declare_for_class(Integer, Graded(degree=0))
        # Exact overrides class for the same type.
        reg.declare(x, Graded(degree=5))
        assert reg.get(x, Graded).degree == 5
        # Other Integers still see the class fallback.
        assert reg.get(Integer(3), Graded).degree == 0

    def test_mro_inheritance(self, reg):
        # Symbol is a subclass of Atom is a subclass of Expr.
        reg.declare_for_class(Expr, NonCommuting())
        assert reg.has(Symbol("x"), NonCommuting)

    def test_rejects_non_expr_class(self, reg):
        with pytest.raises(TypeError):
            reg.declare_for_class(int, Scalar())  # type: ignore[arg-type]

    def test_duplicate_raises(self, reg):
        reg.declare_for_class(Integer, Scalar())
        with pytest.raises(ValueError):
            reg.declare_for_class(Integer, Scalar())

    def test_retract_for_class(self, reg):
        reg.declare_for_class(Integer, Scalar())
        assert reg.has(Integer(2), Scalar)
        reg.retract_for_class(Integer, Scalar)
        assert not reg.has(Integer(2), Scalar)

    def test_retract_exact_does_not_clear_class_fallback(self, reg):
        reg.declare_for_class(Integer, Scalar())
        x = Integer(2)
        # Retracting an exact binding that doesn't exist returns None
        # and leaves the class-based fallback intact.
        assert reg.retract(x, Scalar) is None
        assert reg.has(x, Scalar)


# --------------------------------------------------------------------- #
# Predicate-based declarations                                          #
# --------------------------------------------------------------------- #


class TestDeclareForPredicate:
    def test_matches_by_predicate(self, reg):
        reg.declare_for_predicate(
            lambda e: isinstance(e, Symbol) and e.name.startswith("c_"),
            Scalar(),
        )
        assert reg.has(Symbol("c_1"), Scalar)
        assert not reg.has(Symbol("x"), Scalar)

    def test_registration_order_first_match_wins(self, reg):
        reg.declare_for_predicate(
            lambda e: isinstance(e, Symbol), Graded(degree=1)
        )
        reg.declare_for_predicate(
            lambda e: isinstance(e, Symbol), Graded(degree=2)
        )
        assert reg.get(Symbol("x"), Graded).degree == 1

    def test_rejects_non_callable(self, reg):
        with pytest.raises(TypeError):
            reg.declare_for_predicate("not callable", Scalar())  # type: ignore[arg-type]

    def test_exact_beats_predicate(self, reg):
        reg.declare_for_predicate(lambda e: True, Graded(degree=1))
        x = Symbol("x")
        reg.declare(x, Graded(degree=7))
        assert reg.get(x, Graded).degree == 7


# --------------------------------------------------------------------- #
# scope context manager                                                  #
# --------------------------------------------------------------------- #


class TestScope:
    def test_rolls_back_declarations(self, reg):
        x = Symbol("x")
        with reg.scope():
            reg.declare(x, Scalar())
            assert reg.has(x, Scalar)
        assert not reg.has(x, Scalar)

    def test_rolls_back_on_exception(self, reg):
        x = Symbol("x")
        with pytest.raises(RuntimeError):
            with reg.scope():
                reg.declare(x, Scalar())
                raise RuntimeError("boom")
        assert not reg.has(x, Scalar)

    def test_preserves_prior_state(self, reg):
        x = Symbol("x")
        reg.declare(x, Scalar())
        with reg.scope():
            reg.declare(x, Graded(degree=1))
            reg.retract(x, Scalar)
            assert not reg.has(x, Scalar)
            assert reg.has(x, Graded)
        assert reg.has(x, Scalar)
        assert not reg.has(x, Graded)

    def test_nested(self, reg):
        x = Symbol("x")
        with reg.scope():
            reg.declare(x, Scalar())
            with reg.scope():
                reg.declare(x, Graded(degree=2))
                assert reg.has(x, Graded)
            assert not reg.has(x, Graded)
            assert reg.has(x, Scalar)
        assert not reg.has(x, Scalar)

    def test_rolls_back_class_and_predicate(self, reg):
        with reg.scope():
            reg.declare_for_class(Integer, Scalar())
            reg.declare_for_predicate(lambda e: isinstance(e, Symbol), Antisymmetric())
            assert reg.has(Integer(5), Scalar)
            assert reg.has(Symbol("x"), Antisymmetric)
        assert not reg.has(Integer(5), Scalar)
        assert not reg.has(Symbol("x"), Antisymmetric)


# --------------------------------------------------------------------- #
# axioms context manager                                                 #
# --------------------------------------------------------------------- #


class TestAxioms:
    def test_narrows_visible_types(self, reg):
        x = Symbol("x")
        reg.declare(x, Scalar())
        reg.declare(x, Graded(degree=1))
        with reg.axioms([Scalar]):
            assert reg.has(x, Scalar)
            assert not reg.has(x, Graded)
        # Restored after block.
        assert reg.has(x, Graded)

    def test_empty_whitelist_hides_everything(self, reg):
        x = Symbol("x")
        reg.declare(x, Scalar())
        with reg.axioms([]):
            assert not reg.has(x, Scalar)
        assert reg.has(x, Scalar)

    def test_nests_with_inner_override(self, reg):
        x = Symbol("x")
        reg.declare(x, Scalar())
        reg.declare(x, Graded(degree=1))
        reg.declare(x, Symmetric())
        with reg.axioms([Scalar, Graded]):
            assert reg.has(x, Scalar)
            assert not reg.has(x, Symmetric)
            with reg.axioms([Symmetric]):
                assert reg.has(x, Symmetric)
                assert not reg.has(x, Scalar)
            assert reg.has(x, Scalar)
            assert not reg.has(x, Symmetric)

    def test_rejects_non_property_class(self, reg):
        with pytest.raises(TypeError):
            with reg.axioms([int]):  # type: ignore[list-item]
                pass

    def test_orthogonal_to_strict(self):
        """strict filters by provenance, axioms filters by type, stackable."""
        reg = PropertyRegistry(strict_axioms_only=True)
        x = Symbol("x")
        reg.declare(x, Scalar())  # axiom
        reg.declare(
            x,
            Graded(degree=1, provenance=Provenance.DERIVED, proof=ProofRef("r")),
        )
        with reg.axioms([Scalar, Graded]):
            # strict hides Graded (derived); axioms allows both types.
            assert reg.has(x, Scalar)
            assert not reg.has(x, Graded)

    def test_filters_all_for(self, reg):
        x = Symbol("x")
        reg.declare(x, Scalar())
        reg.declare(x, Graded(degree=0))
        with reg.axioms([Graded]):
            kinds = {type(p) for p in reg.all_for(x)}
            assert kinds == {Graded}
