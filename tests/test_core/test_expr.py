"""Tests for jacopy.core.expr, the foundational expression tree."""

from fractions import Fraction

import pytest

from jacopy.core.expr import (
    Expr,
    Integer,
    Neg,
    NegOne,
    One,
    Power,
    Product,
    Rational,
    Sum,
    Symbol,
    Zero,
    _wrap,
)


# --------------------------------------------------------------------- #
# Symbol                                                                #
# --------------------------------------------------------------------- #


class TestSymbol:
    def test_creation_and_name(self):
        x = Symbol("x")
        assert x.name == "x"
        assert x.is_atom
        assert x.children == ()

    def test_equality(self):
        assert Symbol("x") == Symbol("x")
        assert Symbol("x") != Symbol("y")

    def test_hash_consistency(self):
        assert hash(Symbol("x")) == hash(Symbol("x"))

    def test_rejects_non_string(self):
        with pytest.raises(TypeError):
            Symbol(42)

    def test_rejects_empty(self):
        with pytest.raises(ValueError):
            Symbol("")

    def test_repr(self):
        assert repr(Symbol("alpha")) == "alpha"


# --------------------------------------------------------------------- #
# Integer                                                               #
# --------------------------------------------------------------------- #


class TestInteger:
    def test_value(self):
        assert Integer(7).value == 7
        assert Integer(-3).value == -3

    def test_singletons(self):
        assert Integer(0) is Zero
        assert Integer(1) is One
        assert Integer(-1) is NegOne

    def test_non_singleton_values_not_cached(self):
        a = Integer(5)
        b = Integer(5)
        assert a == b
        # Equality yes, identity need not hold for uncached values.

    def test_equality_by_value(self):
        assert Integer(3) == Integer(3)
        assert Integer(3) != Integer(4)

    def test_hash_consistency(self):
        assert hash(Integer(42)) == hash(Integer(42))

    def test_repr(self):
        assert repr(Integer(5)) == "5"
        assert repr(Integer(-2)) == "-2"


# --------------------------------------------------------------------- #
# Rational                                                              #
# --------------------------------------------------------------------- #


class TestRational:
    def test_reduces_to_integer_when_denominator_one(self):
        assert Rational(6, 3) == Integer(2)
        assert isinstance(Rational(6, 3), Integer)

    def test_reduces_to_lowest_terms(self):
        r = Rational(4, 6)
        assert r.p == 2
        assert r.q == 3

    def test_normalizes_sign_to_positive_denominator(self):
        r = Rational(1, -2)
        assert r.p == -1
        assert r.q == 2

    def test_zero_denominator_raises(self):
        with pytest.raises(ZeroDivisionError):
            Rational(1, 0)

    def test_equality(self):
        assert Rational(1, 2) == Rational(2, 4)
        assert Rational(1, 2) != Rational(1, 3)

    def test_repr(self):
        assert repr(Rational(3, 4)) == "3/4"


# --------------------------------------------------------------------- #
# _wrap                                                                  #
# --------------------------------------------------------------------- #


class TestWrap:
    def test_passes_through_expr(self):
        x = Symbol("x")
        assert _wrap(x) is x

    def test_wraps_int(self):
        assert _wrap(5) == Integer(5)

    def test_rejects_bool(self):
        with pytest.raises(TypeError):
            _wrap(True)

    def test_wraps_fraction(self):
        r = _wrap(Fraction(1, 3))
        assert r == Rational(1, 3)

    def test_rejects_float(self):
        with pytest.raises(TypeError):
            _wrap(1.5)


# --------------------------------------------------------------------- #
# Neg                                                                   #
# --------------------------------------------------------------------- #


class TestNeg:
    def test_first_class_node(self):
        x = Symbol("x")
        n = Neg(x)
        assert n.arg is x
        assert n.children == (x,)

    def test_equality(self):
        x = Symbol("x")
        assert Neg(x) == Neg(x)
        assert Neg(x) != Neg(Symbol("y"))

    def test_rejects_non_expr(self):
        with pytest.raises(TypeError):
            Neg(5)

    def test_unary_minus_builds_neg(self):
        x = Symbol("x")
        assert -x == Neg(x)


# --------------------------------------------------------------------- #
# Sum                                                                   #
# --------------------------------------------------------------------- #


class TestSum:
    def test_make_flattens_nested(self):
        x, y, z = Symbol("x"), Symbol("y"), Symbol("z")
        s = Sum.make(Sum.make(x, y), z)
        assert s.children == (x, y, z)

    def test_make_drops_zeros(self):
        x, y = Symbol("x"), Symbol("y")
        s = Sum.make(x, Zero, y, Zero)
        assert s.children == (x, y)

    def test_make_empty_yields_zero(self):
        assert Sum.make() is Zero
        assert Sum.make(Zero, Zero) is Zero

    def test_make_singleton_collapses(self):
        x = Symbol("x")
        assert Sum.make(x, Zero) is x

    def test_operator_overload(self):
        x, y = Symbol("x"), Symbol("y")
        s = x + y
        assert isinstance(s, Sum)
        assert s.children == (x, y)

    def test_radd_with_int(self):
        x = Symbol("x")
        s = 3 + x
        assert isinstance(s, Sum)
        assert s.children == (Integer(3), x)

    def test_subtraction(self):
        x, y = Symbol("x"), Symbol("y")
        s = x - y
        assert isinstance(s, Sum)
        assert s.children == (x, Neg(y))

    def test_rejects_non_expr_children(self):
        with pytest.raises(TypeError):
            Sum(Symbol("x"), 5)

    def test_equality(self):
        x, y = Symbol("x"), Symbol("y")
        assert Sum(x, y) == Sum(x, y)
        assert Sum(x, y) != Sum(y, x)  # order preserved


# --------------------------------------------------------------------- #
# Product                                                               #
# --------------------------------------------------------------------- #


class TestProduct:
    def test_make_flattens_nested(self):
        x, y, z = Symbol("x"), Symbol("y"), Symbol("z")
        p = Product.make(Product.make(x, y), z)
        assert p.children == (x, y, z)

    def test_make_absorbs_zero(self):
        x, y = Symbol("x"), Symbol("y")
        assert Product.make(x, Zero, y) is Zero

    def test_make_drops_ones(self):
        x, y = Symbol("x"), Symbol("y")
        p = Product.make(x, One, y, One)
        assert p.children == (x, y)

    def test_make_empty_yields_one(self):
        assert Product.make() is One
        assert Product.make(One, One) is One

    def test_make_singleton_collapses(self):
        x = Symbol("x")
        assert Product.make(x, One) is x

    def test_operator_overload(self):
        x, y = Symbol("x"), Symbol("y")
        p = x * y
        assert isinstance(p, Product)
        assert p.children == (x, y)

    def test_non_commutative_by_default(self):
        x, y = Symbol("x"), Symbol("y")
        # Structural inequality: order matters.
        assert x * y != y * x

    def test_rmul_with_int(self):
        x = Symbol("x")
        p = 2 * x
        assert isinstance(p, Product)
        assert p.children == (Integer(2), x)


# --------------------------------------------------------------------- #
# Power                                                                 #
# --------------------------------------------------------------------- #


class TestPower:
    def test_no_auto_eval(self):
        x = Symbol("x")
        p = x ** 2
        assert isinstance(p, Power)
        assert p.base is x
        assert p.exp == Integer(2)

    def test_even_trivial_powers_preserved(self):
        p = Power(One, Symbol("x"))
        assert p.base is One

    def test_rejects_non_expr(self):
        with pytest.raises(TypeError):
            Power(5, Symbol("x"))


# --------------------------------------------------------------------- #
# walk / find                                                           #
# --------------------------------------------------------------------- #


class TestWalk:
    def test_atom_yields_self(self):
        x = Symbol("x")
        assert list(x.walk()) == [x]

    def test_compound_preorder(self):
        x, y = Symbol("x"), Symbol("y")
        s = x + y
        assert list(s.walk()) == [s, x, y]

    def test_find_predicate(self):
        x, y = Symbol("x"), Symbol("y")
        tree = (x + y) * x
        symbols = list(tree.find(lambda n: isinstance(n, Symbol)))
        # x appears twice, once in sum, once as top-level factor.
        assert symbols.count(x) == 2
        assert y in symbols


# --------------------------------------------------------------------- #
# Equality / hashing cross-type                                         #
# --------------------------------------------------------------------- #


class TestEqualityHashing:
    def test_different_types_not_equal(self):
        x = Symbol("x")
        # Symbol("x") and Integer(0) share nothing structural.
        assert x != Zero

    def test_hashable_in_set(self):
        x, y = Symbol("x"), Symbol("y")
        s = {x, y, Symbol("x")}
        assert len(s) == 2

    def test_eq_with_non_expr_returns_notimplemented(self):
        # Python falls back to identity comparison -> False.
        assert (Symbol("x") == "x") is False


# --------------------------------------------------------------------- #
# replace_at                                                            #
# --------------------------------------------------------------------- #


class TestReplaceAt:
    def test_empty_path_replaces_root(self):
        x, y = Symbol("x"), Symbol("y")
        assert x.replace_at((), y) is y

    def test_atom_non_empty_path_raises(self):
        x = Symbol("x")
        with pytest.raises(IndexError):
            x.replace_at((0,), Symbol("y"))

    def test_replace_shallow_child(self):
        x, y, z = Symbol("x"), Symbol("y"), Symbol("z")
        tree = Sum(x, y)
        out = tree.replace_at((1,), z)
        assert out == Sum(x, z)

    def test_replace_deep(self):
        x, y, z, w = Symbol("x"), Symbol("y"), Symbol("z"), Symbol("w")
        tree = Sum(x, Product(y, z))
        out = tree.replace_at((1, 0), w)
        assert out == Sum(x, Product(w, z))

    def test_preserves_raw_structure_no_smart_ctor(self):
        # replace_at must use the raw constructor: a Sum-of-Sum
        # parent should NOT get flattened by Sum.make after splicing.
        x, y, z = Symbol("x"), Symbol("y"), Symbol("z")
        outer = Sum(Sum(x, y), z)
        out = outer.replace_at((0, 1), z)
        # First child should remain a Sum(x, z), not flattened with z.
        assert isinstance(out.children[0], Sum)
        assert out.children[0].children == (x, z)

    def test_index_out_of_range_raises(self):
        x, y = Symbol("x"), Symbol("y")
        tree = Sum(x, y)
        with pytest.raises(IndexError):
            tree.replace_at((5,), Symbol("z"))

    def test_non_expr_new_raises(self):
        x = Symbol("x")
        with pytest.raises(TypeError):
            Sum(x, x).replace_at((0,), 5)


# --------------------------------------------------------------------- #
# clone                                                                 #
# --------------------------------------------------------------------- #


class TestClone:
    def test_atom_returns_self(self):
        x = Symbol("x")
        assert x.clone() is x

    def test_compound_returns_self(self):
        x, y = Symbol("x"), Symbol("y")
        tree = Sum(x, y)
        assert tree.clone() is tree
