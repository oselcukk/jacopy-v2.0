"""Tests for :class:`jacopy.core.wedge.Wedge` (Faz 17.F.1.5)."""

import pytest

from jacopy.algebra.derivation import degree_of
from jacopy.core.expr import Expr, Integer, One, Symbol, Zero
from jacopy.core.properties import Graded
from jacopy.core.registry import PropertyRegistry
from jacopy.core.symbolic_degree import Degree
from jacopy.core.wedge import Wedge
from jacopy.display.ascii import to_ascii
from jacopy.display.latex import to_latex


# --------------------------------------------------------------------- #
# Construction                                                          #
# --------------------------------------------------------------------- #


class TestConstruction:
    def test_two_factor_wedge(self):
        a, b = Symbol("α"), Symbol("β")
        w = Wedge(a, b)
        assert w.children == (a, b)
        assert not w.is_atom

    def test_three_factor_wedge(self):
        a, b, c = Symbol("α"), Symbol("β"), Symbol("γ")
        w = Wedge(a, b, c)
        assert w.children == (a, b, c)

    def test_single_factor_rejected(self):
        with pytest.raises(ValueError, match="at least two factors"):
            Wedge(Symbol("α"))

    def test_zero_factor_rejected(self):
        with pytest.raises(ValueError, match="at least two factors"):
            Wedge()

    def test_non_expr_child_rejected(self):
        with pytest.raises(TypeError, match="children must be Expr"):
            Wedge(Symbol("α"), 3)  # type: ignore[arg-type]


# --------------------------------------------------------------------- #
# Smart constructor                                                     #
# --------------------------------------------------------------------- #


class TestMake:
    def test_make_two_factors_returns_wedge(self):
        a, b = Symbol("α"), Symbol("β")
        w = Wedge.make(a, b)
        assert isinstance(w, Wedge)
        assert w.children == (a, b)

    def test_make_flattens_nested_wedge(self):
        a, b, c = Symbol("α"), Symbol("β"), Symbol("γ")
        nested = Wedge.make(Wedge(a, b), c)
        assert isinstance(nested, Wedge)
        assert nested.children == (a, b, c)

    def test_make_flattens_deeply(self):
        a, b, c, d = (Symbol(s) for s in ("α", "β", "γ", "δ"))
        nested = Wedge.make(Wedge(a, b), Wedge(c, d))
        assert nested.children == (a, b, c, d)

    def test_make_zero_absorbs(self):
        a = Symbol("α")
        w = Wedge.make(a, Zero, Symbol("β"))
        assert w is Zero

    def test_make_one_dropped(self):
        a, b = Symbol("α"), Symbol("β")
        w = Wedge.make(a, One, b)
        assert isinstance(w, Wedge)
        assert w.children == (a, b)

    def test_make_single_factor_collapses(self):
        a = Symbol("α")
        w = Wedge.make(a)
        assert w is a

    def test_make_only_ones_collapses_to_one(self):
        w = Wedge.make(One, One)
        assert w is One

    def test_make_no_factors_collapses_to_one(self):
        assert Wedge.make() is One


# --------------------------------------------------------------------- #
# Equality / hashing                                                    #
# --------------------------------------------------------------------- #


class TestEquality:
    def test_structural_equality(self):
        a, b = Symbol("α"), Symbol("β")
        assert Wedge(a, b) == Wedge(a, b)

    def test_order_matters_no_auto_sort(self):
        a, b = Symbol("α"), Symbol("β")
        # Wedge does NOT sort at construction, graded reordering needs
        # degree information and lives elsewhere.
        assert Wedge(a, b) != Wedge(b, a)

    def test_hash_matches_equal(self):
        a, b = Symbol("α"), Symbol("β")
        assert hash(Wedge(a, b)) == hash(Wedge(a, b))


# --------------------------------------------------------------------- #
# Degree law                                                            #
# --------------------------------------------------------------------- #


class TestDegree:
    def test_two_one_forms_have_degree_two(self):
        reg = PropertyRegistry()
        a, b = Symbol("α"), Symbol("β")
        reg.declare(a, Graded(Degree.const(1)))
        reg.declare(b, Graded(Degree.const(1)))
        w = Wedge(a, b)
        assert degree_of(w, reg) == Degree.const(2)

    def test_mixed_degree_sums(self):
        reg = PropertyRegistry()
        a, b = Symbol("α"), Symbol("β")
        reg.declare(a, Graded(Degree.const(1)))
        reg.declare(b, Graded(Degree.const(2)))
        w = Wedge(a, b)
        assert degree_of(w, reg) == Degree.const(3)

    def test_undetermined_factor_propagates(self):
        reg = PropertyRegistry()
        a, b = Symbol("α"), Symbol("β")
        reg.declare(a, Graded(Degree.const(1)))
        # b has no degree declaration
        with pytest.raises(ValueError, match="not determined"):
            degree_of(Wedge(a, b), reg)


# --------------------------------------------------------------------- #
# Display                                                                #
# --------------------------------------------------------------------- #


class TestDisplay:
    def test_ascii_uses_wedge_glyph(self):
        a, b = Symbol("α"), Symbol("β")
        assert to_ascii(Wedge(a, b)) == "α ∧ β"

    def test_latex_uses_wedge_macro(self):
        a, b = Symbol("α"), Symbol("β")
        rendered = to_latex(Wedge(a, b))
        assert "\\wedge" in rendered
