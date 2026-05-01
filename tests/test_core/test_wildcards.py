"""Tests for jacopy.core.wildcards."""

import pytest

from jacopy.core.expr import Integer, Neg, Product, Sum, Symbol, Zero
from jacopy.core.properties import Graded, Scalar
from jacopy.core.registry import PropertyRegistry
from jacopy.core.wildcards import (
    SeqWildcard,
    Wildcard,
    match,
    substitute,
)


# --------------------------------------------------------------------- #
# Construction                                                          #
# --------------------------------------------------------------------- #


class TestWildcardConstruction:
    def test_name_only(self):
        w = Wildcard("A")
        assert w.name == "A"
        assert w.type_filter is None
        assert w.is_atom

    def test_with_type_filter(self):
        w = Wildcard("X", type_filter=Scalar)
        assert w.type_filter is Scalar

    def test_rejects_empty_name(self):
        with pytest.raises(ValueError):
            Wildcard("")

    def test_rejects_non_string_name(self):
        with pytest.raises(TypeError):
            Wildcard(42)  # type: ignore[arg-type]

    def test_rejects_bad_type_filter(self):
        with pytest.raises(TypeError):
            Wildcard("A", type_filter=int)  # type: ignore[arg-type]

    def test_with_expr_type_single(self):
        w = Wildcard("S", expr_type=Symbol)
        assert w.expr_type == (Symbol,)

    def test_with_expr_type_tuple(self):
        w = Wildcard("N", expr_type=(Integer, Sum))
        assert w.expr_type == (Integer, Sum)

    def test_rejects_non_expr_type(self):
        with pytest.raises(TypeError):
            Wildcard("A", expr_type=int)  # type: ignore[arg-type]

    def test_rejects_tuple_with_non_expr(self):
        with pytest.raises(TypeError):
            Wildcard("A", expr_type=(Symbol, int))  # type: ignore[arg-type]

    def test_repr(self):
        assert repr(Wildcard("A")) == "?A"
        assert repr(Wildcard("X", type_filter=Scalar)) == "?X:Scalar"
        assert repr(Wildcard("S", expr_type=Symbol)) == "?S<Symbol>"
        assert (
            repr(Wildcard("N", expr_type=(Integer, Sum)))
            == "?N<Integer|Sum>"
        )

    def test_equality(self):
        assert Wildcard("A") == Wildcard("A")
        assert Wildcard("A") != Wildcard("B")
        assert Wildcard("A") != Wildcard("A", type_filter=Scalar)
        assert Wildcard("A") != Wildcard("A", expr_type=Symbol)

    def test_usable_in_expression_builders(self):
        a, b = Wildcard("A"), Wildcard("B")
        s = a + b
        assert isinstance(s, Sum)
        assert s.children == (a, b)


class TestSeqWildcardConstruction:
    def test_basic(self):
        s = SeqWildcard("rest")
        assert s.name == "rest"
        assert s.children == ()

    def test_repr(self):
        assert repr(SeqWildcard("rest")) == "?*rest"

    def test_rejects_empty(self):
        with pytest.raises(ValueError):
            SeqWildcard("")


# --------------------------------------------------------------------- #
# Basic matching                                                        #
# --------------------------------------------------------------------- #


class TestMatchAtoms:
    def test_wildcard_matches_anything(self):
        w = Wildcard("A")
        x = Symbol("x")
        assert match(w, x) == {"A": x}

    def test_wildcard_matches_compound(self):
        w = Wildcard("A")
        target = Symbol("x") + Symbol("y")
        assert match(w, target) == {"A": target}

    def test_concrete_match(self):
        x = Symbol("x")
        assert match(x, x) == {}

    def test_concrete_mismatch(self):
        assert match(Symbol("x"), Symbol("y")) is None

    def test_type_mismatch(self):
        assert match(Symbol("x"), Integer(1)) is None


class TestMatchCompound:
    def test_sum_pairwise(self):
        pattern = Wildcard("A") + Wildcard("B")
        target = Symbol("x") + Symbol("y")
        b = match(pattern, target)
        assert b == {"A": Symbol("x"), "B": Symbol("y")}

    def test_sum_arity_mismatch(self):
        pattern = Wildcard("A") + Wildcard("B")
        target = Sum(Symbol("x"), Symbol("y"), Symbol("z"))
        assert match(pattern, target) is None

    def test_product_pairwise(self):
        pattern = Wildcard("A") * Wildcard("B")
        target = Symbol("x") * Symbol("y")
        b = match(pattern, target)
        assert b == {"A": Symbol("x"), "B": Symbol("y")}

    def test_order_sensitive(self):
        """Structural, order-preserving matching, non-commutative."""
        pattern = Symbol("x") * Wildcard("A")
        # Target starts with y, not x.
        target = Symbol("y") * Symbol("z")
        assert match(pattern, target) is None

    def test_nested(self):
        x = Symbol("x")
        pattern = Wildcard("A") + (Wildcard("B") * Wildcard("C"))
        target = x + (Symbol("y") * Symbol("z"))
        b = match(pattern, target)
        assert b == {"A": x, "B": Symbol("y"), "C": Symbol("z")}


class TestRepeatedWildcard:
    def test_same_binding_succeeds(self):
        x = Symbol("x")
        pattern = Wildcard("A") + Wildcard("A")
        target = x + x
        assert match(pattern, target) == {"A": x}

    def test_inconsistent_binding_fails(self):
        pattern = Wildcard("A") + Wildcard("A")
        target = Symbol("x") + Symbol("y")
        assert match(pattern, target) is None


# --------------------------------------------------------------------- #
# Type-filtered matching                                                #
# --------------------------------------------------------------------- #


class TestTypeFilter:
    @pytest.fixture
    def reg(self):
        r = PropertyRegistry()
        r.declare(Symbol("f"), Scalar())
        r.declare(Symbol("g"), Scalar())
        r.declare(Symbol("X"), Graded(degree=1))
        return r

    def test_filter_pass(self, reg):
        w = Wildcard("F", type_filter=Scalar)
        assert match(w, Symbol("f"), reg) == {"F": Symbol("f")}

    def test_filter_fail(self, reg):
        w = Wildcard("F", type_filter=Scalar)
        # X is Graded, not Scalar
        assert match(w, Symbol("X"), reg) is None

    def test_filter_without_registry_fails(self):
        w = Wildcard("F", type_filter=Scalar)
        assert match(w, Symbol("f")) is None


# --------------------------------------------------------------------- #
# Expr-type filter                                                      #
# --------------------------------------------------------------------- #


class TestExprTypeFilter:
    def test_matches_symbol_only(self):
        w = Wildcard("S", expr_type=Symbol)
        x = Symbol("x")
        assert match(w, x) == {"S": x}
        # Sum should not match.
        assert match(w, Sum(x, Symbol("y"))) is None

    def test_matches_integer_only(self):
        w = Wildcard("N", expr_type=Integer)
        assert match(w, Integer(3)) == {"N": Integer(3)}
        assert match(w, Symbol("x")) is None

    def test_tuple_accepts_any(self):
        w = Wildcard("A", expr_type=(Symbol, Integer))
        assert match(w, Symbol("x")) == {"A": Symbol("x")}
        assert match(w, Integer(7)) == {"A": Integer(7)}
        assert match(w, Sum(Symbol("x"), Symbol("y"))) is None

    def test_combines_with_type_filter(self):
        reg = PropertyRegistry()
        reg.declare(Symbol("f"), Scalar())
        # Must be a Symbol AND registered as Scalar.
        w = Wildcard("F", type_filter=Scalar, expr_type=Symbol)
        assert match(w, Symbol("f"), reg) == {"F": Symbol("f")}
        # Integer(3) fails expr_type even though numerics are "scalar-ish".
        assert match(w, Integer(3), reg) is None


# --------------------------------------------------------------------- #
# Sequence wildcards                                                    #
# --------------------------------------------------------------------- #


class TestSeqWildcard:
    def test_captures_middle(self):
        x, y, z, w = Symbol("x"), Symbol("y"), Symbol("z"), Symbol("w")
        pattern = Sum(Wildcard("A"), SeqWildcard("mid"), Wildcard("B"))
        target = Sum(x, y, z, w)
        b = match(pattern, target)
        assert b == {"A": x, "mid": (y, z), "B": w}

    def test_captures_empty(self):
        x, y = Symbol("x"), Symbol("y")
        pattern = Sum(Wildcard("A"), SeqWildcard("mid"), Wildcard("B"))
        target = Sum(x, y)
        b = match(pattern, target)
        assert b == {"A": x, "mid": (), "B": y}

    def test_leading_seq(self):
        x, y, z = Symbol("x"), Symbol("y"), Symbol("z")
        pattern = Product(SeqWildcard("head"), Wildcard("last"))
        target = Product(x, y, z)
        b = match(pattern, target)
        assert b == {"head": (x, y), "last": z}

    def test_trailing_seq(self):
        x, y, z = Symbol("x"), Symbol("y"), Symbol("z")
        pattern = Product(Wildcard("first"), SeqWildcard("tail"))
        target = Product(x, y, z)
        b = match(pattern, target)
        assert b == {"first": x, "tail": (y, z)}

    def test_too_few_children(self):
        pattern = Sum(Wildcard("A"), SeqWildcard("mid"), Wildcard("B"))
        # Only one child, can't match two non-seq wildcards even with empty mid.
        target = Sum(Symbol("x"))
        # Wait, Sum(x) collapses via make but we used constructor directly.
        # It has one child here.
        assert match(pattern, target) is None

    def test_two_seq_wildcards_split(self):
        """Multiple SeqWildcards per level are allowed, backtracks over splits.

        Leftmost-shortest-first: the first SeqWildcard consumes 0 elements,
        the middle non-seq takes one, the second SeqWildcard takes the rest.
        """
        x, y, z = Symbol("x"), Symbol("y"), Symbol("z")
        pattern = Sum(
            SeqWildcard("a"),
            Wildcard("mid"),
            SeqWildcard("b"),
        )
        b = match(pattern, Sum(x, y, z))
        assert b == {"a": (), "mid": x, "b": (y, z)}

    def test_two_seq_wildcards_pinned_by_non_seq(self):
        """The non-seq wildcard inside pins one element in place."""
        x, y, z, w = (
            Symbol("x"), Symbol("y"), Symbol("z"), Symbol("w"),
        )
        pattern = Product(
            SeqWildcard("pre"),
            Symbol("z"),
            SeqWildcard("post"),
        )
        # z sits at index 2; pre takes (x, y), post takes (w,).
        b = match(pattern, Product(x, y, z, w))
        assert b == {"pre": (x, y), "post": (w,)}

    def test_seq_backtracks_for_binding_consistency(self):
        """Repeat wildcard forces backtracking over seq splits."""
        y, z = Symbol("y"), Symbol("z")
        pattern = Product(
            SeqWildcard("pre"),
            Wildcard("A"),
            SeqWildcard("mid"),
            Wildcard("A"),
            SeqWildcard("post"),
        )
        # Target: y z y. Only consistent binding is A=y. The matcher
        # first tries pre=(), A=y, mid=(), A=z which fails (y != z),
        # then pre=(), A=y, mid=(z,), A=y which succeeds.
        target = Product(y, z, y)
        b = match(pattern, target)
        assert b is not None
        assert b["A"] == y
        assert b["pre"] == ()
        assert b["mid"] == (z,)
        assert b["post"] == ()


# --------------------------------------------------------------------- #
# Substitute                                                            #
# --------------------------------------------------------------------- #


class TestSubstitute:
    def test_replaces_wildcards(self):
        pattern = Wildcard("A") + Wildcard("B")
        bindings = {"A": Symbol("x"), "B": Symbol("y")}
        result = substitute(pattern, bindings)
        assert result == Sum(Symbol("x"), Symbol("y"))

    def test_unbound_wildcard_kept(self):
        pattern = Wildcard("A") + Wildcard("B")
        bindings = {"A": Symbol("x")}
        result = substitute(pattern, bindings)
        assert result == Sum(Symbol("x"), Wildcard("B"))

    def test_seq_wildcard_expands(self):
        x, y, z = Symbol("x"), Symbol("y"), Symbol("z")
        pattern = Sum(Wildcard("A"), SeqWildcard("rest"))
        bindings = {"A": x, "rest": (y, z)}
        result = substitute(pattern, bindings)
        assert result == Sum(x, y, z)

    def test_seq_wildcard_empty(self):
        x = Symbol("x")
        pattern = Sum(Wildcard("A"), SeqWildcard("rest"))
        bindings = {"A": x, "rest": ()}
        result = substitute(pattern, bindings)
        assert result == Sum(x)

    def test_round_trip_match_then_substitute(self):
        """match + substitute on an unchanged pattern recovers the target."""
        pattern = Wildcard("A") + Wildcard("B") * Wildcard("C")
        target = Symbol("x") + Symbol("y") * Symbol("z")
        b = match(pattern, target)
        assert b is not None
        assert substitute(pattern, b) == target

    def test_nested_substitute(self):
        """Substitute into a right-hand side that shares wildcards."""
        # Rewrite rule: ?A + ?B -> ?A + ?A + ?B (silly but tests logic).
        lhs = Wildcard("A") + Wildcard("B")
        rhs = Wildcard("A") + Wildcard("A") + Wildcard("B")
        target = Symbol("x") + Symbol("y")
        b = match(lhs, target)
        assert b is not None
        result = substitute(rhs, b)
        assert result == Sum(Symbol("x"), Symbol("x"), Symbol("y"))


# --------------------------------------------------------------------- #
# Walk & integration with Expr                                          #
# --------------------------------------------------------------------- #


class TestIntegration:
    def test_wildcard_in_walk(self):
        w = Wildcard("A")
        tree = w + Symbol("x")
        nodes = list(tree.walk())
        assert w in nodes

    def test_wildcard_equality_in_set(self):
        s = {Wildcard("A"), Wildcard("A"), Wildcard("B")}
        assert len(s) == 2

    def test_non_matching_neg(self):
        pattern = Neg(Wildcard("A"))
        target = -Symbol("x")
        b = match(pattern, target)
        assert b == {"A": Symbol("x")}
