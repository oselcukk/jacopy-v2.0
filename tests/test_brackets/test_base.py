"""Tests for gradalg.brackets.base."""

import pytest

from gradalg.brackets.base import BracketApply, GradedBracket, expand_bracket
from gradalg.brackets.lie import LieBracket
from gradalg.core.expr import Integer, Neg, Product, Sum, Symbol
from gradalg.core.properties import Graded
from gradalg.core.registry import PropertyRegistry
from gradalg.core.symbolic_degree import Degree


# --------------------------------------------------------------------- #
# Fixtures                                                               #
# --------------------------------------------------------------------- #


@pytest.fixture
def reg():
    r = PropertyRegistry()
    r.declare(Symbol("X"), Graded(degree=1))
    r.declare(Symbol("Y"), Graded(degree=1))
    r.declare(Symbol("Z"), Graded(degree=1))
    r.declare(Symbol("a"), Graded(degree=1))  # odd
    r.declare(Symbol("b"), Graded(degree=2))  # even
    r.declare(Symbol("c"), Graded(degree=2))  # even
    return r


# --------------------------------------------------------------------- #
# GradedBracket construction                                             #
# --------------------------------------------------------------------- #


class TestConstruction:
    def test_defaults(self):
        lie = LieBracket()
        assert lie.name == "[·,·]"
        assert lie.degree == Degree.const(0)
        assert lie.is_graded_antisymmetric
        assert lie.satisfies_leibniz
        assert lie.satisfies_graded_jacobi is True

    def test_rejects_empty_name(self):
        with pytest.raises(ValueError):
            LieBracket(name="")

    def test_equality(self):
        # LieBracket takes only a name; same name → equal.
        assert LieBracket() == LieBracket()
        assert LieBracket("A") != LieBracket("B")

    def test_hashable(self):
        s = {LieBracket(), LieBracket()}
        assert len(s) == 1

    def test_repr(self):
        assert "LieBracket" in repr(LieBracket())


# --------------------------------------------------------------------- #
# BracketApply node                                                      #
# --------------------------------------------------------------------- #


class TestBracketApply:
    def test_children_exclude_bracket(self):
        lie = LieBracket()
        X, Y = Symbol("X"), Symbol("Y")
        node = lie(X, Y)
        assert isinstance(node, BracketApply)
        assert node.children == (X, Y)  # bracket ref is NOT a child
        assert node.bracket is lie

    def test_callable_shortcut_equals_constructor(self):
        lie = LieBracket()
        X, Y = Symbol("X"), Symbol("Y")
        assert lie(X, Y) == BracketApply(lie, X, Y)

    def test_equality_structural(self):
        lie = LieBracket()
        X, Y = Symbol("X"), Symbol("Y")
        assert lie(X, Y) == lie(X, Y)
        assert lie(X, Y) != lie(Y, X)

    def test_distinct_brackets_distinct_nodes(self):
        A = LieBracket("A")
        B = LieBracket("B")
        X, Y = Symbol("X"), Symbol("Y")
        assert A(X, Y) != B(X, Y)

    def test_rejects_non_bracket(self):
        with pytest.raises(TypeError):
            BracketApply("not-a-bracket", Symbol("a"), Symbol("b"))  # type: ignore[arg-type]

    def test_rejects_non_expr_operand(self):
        lie = LieBracket()
        with pytest.raises(TypeError):
            BracketApply(lie, "a", Symbol("b"))  # type: ignore[arg-type]

    def test_repr_uses_bracket_name(self):
        lie = LieBracket(name="L")
        assert repr(lie(Symbol("X"), Symbol("Y"))) == "L(X, Y)"

    def test_walk_descends_into_operands(self):
        lie = LieBracket()
        X, Y = Symbol("X"), Symbol("Y")
        node = lie(X, Y)
        nodes = list(node.walk())
        assert node in nodes and X in nodes and Y in nodes

    def test_expand_delegates_to_bracket(self, reg):
        lie = LieBracket()
        X, Y = Symbol("X"), Symbol("Y")
        node = lie(X, Y)
        assert node.expand(reg) == Sum(Product(X, Y), Neg(Product(Y, X)))

    def test_standalone_expand_bracket(self, reg):
        lie = LieBracket()
        X, Y = Symbol("X"), Symbol("Y")
        node = lie(X, Y)
        assert expand_bracket(node, reg) == node.expand(reg)


# --------------------------------------------------------------------- #
# Obstruction helpers                                                    #
# --------------------------------------------------------------------- #


class TestObstructions:
    def test_antisymmetry_even_parity(self, reg):
        """|a||b| even → [a,b] + [b,a] as the claimed-zero expression."""
        lie = LieBracket()
        # b has degree 2, c has degree 2 → product parity 0.
        b, c = Symbol("b"), Symbol("c")
        obs = lie.graded_antisymmetry_obstruction(b, c, reg)
        assert obs == Sum(lie(b, c), lie(c, b))

    def test_antisymmetry_odd_parity(self, reg):
        """|a||b| odd → [a,b] − [b,a]."""
        lie = LieBracket()
        X, Y = Symbol("X"), Symbol("Y")  # both odd → product odd
        obs = lie.graded_antisymmetry_obstruction(X, Y, reg)
        assert obs == Sum(lie(X, Y), Neg(lie(Y, X)))

    def test_antisymmetry_symbolic_parity_raises(self):
        """Symbolic degree → parity undecidable, axiom check punts."""
        reg = PropertyRegistry()
        a = Symbol("a")
        b = Symbol("b")
        reg.declare(a, Graded(degree=Degree.var("|a|")))
        reg.declare(b, Graded(degree=Degree.var("|b|")))
        lie = LieBracket()
        with pytest.raises(ValueError, match="symbolic"):
            lie.graded_antisymmetry_obstruction(a, b, reg)

    def test_jacobi_three_odd_operands(self, reg):
        """X, Y, Z all odd → every parity product is 1 (odd).

        Each cyclic term gets a Neg.
        """
        lie = LieBracket()
        X, Y, Z = Symbol("X"), Symbol("Y"), Symbol("Z")
        obs = lie.graded_jacobi_obstruction(X, Y, Z, reg)
        assert obs == Sum(
            Neg(lie(X, lie(Y, Z))),
            Neg(lie(Y, lie(Z, X))),
            Neg(lie(Z, lie(X, Y))),
        )

    def test_jacobi_even_operand(self, reg):
        """Mix even b (degree 2) with odd X, Y (degree 1).

        Parities: |X||b|=0 even, |Y||X|=1 odd, |b||Y|=0 even.
        Order in obstruction is cyclic: (X, Y, b), (Y, b, X), (b, X, Y).
        Signs: even→+, odd→−, even→+.
        """
        lie = LieBracket()
        X, Y, b = Symbol("X"), Symbol("Y"), Symbol("b")
        # (X, Y, b): |X||b| = 1*2 = 2 even → +
        # (Y, b, X): |Y||X| = 1*1 = 1 odd → −
        # (b, X, Y): |b||Y| = 2*1 = 2 even → +
        obs = lie.graded_jacobi_obstruction(X, Y, b, reg)
        assert obs == Sum(
            lie(X, lie(Y, b)),
            Neg(lie(Y, lie(b, X))),
            lie(b, lie(X, Y)),
        )

    def test_leibniz_even_parity(self, reg):
        """|a||b| even → [a, b*c] − [a,b]*c − b*[a,c]."""
        lie = LieBracket()
        b, c = Symbol("b"), Symbol("c")  # both degree 2
        # Use a third even operand for the left slot.
        a = Symbol("a")
        reg.declare(Symbol("t"), Graded(degree=2))
        t = Symbol("t")
        obs = lie.leibniz_obstruction(t, b, c, reg)
        assert obs == Sum(
            lie(t, Product(b, c)),
            Neg(Product(lie(t, b), c)),
            Neg(Product(b, lie(t, c))),
        )

    def test_leibniz_odd_parity(self, reg):
        """|a||b| odd → [a, b*c] − [a,b]*c + b*[a,c]. The +sign on the
        third term comes from folding (−1)^{|a||b|} = −1 into the Neg."""
        lie = LieBracket()
        X, Y = Symbol("X"), Symbol("Y")  # both degree 1
        Z = Symbol("Z")  # degree 1
        obs = lie.leibniz_obstruction(X, Y, Z, reg)
        assert obs == Sum(
            lie(X, Product(Y, Z)),
            Neg(Product(lie(X, Y), Z)),
            # −(−(Y * [X,Z])) → +(Y * [X,Z]) because signed_t2 = Neg(t2),
            # and the Sum includes Neg(signed_t2).
            Product(Y, lie(X, Z)),
        )


# --------------------------------------------------------------------- #
# ABC enforcement                                                        #
# --------------------------------------------------------------------- #


class TestABC:
    def test_cannot_instantiate_bracketabstract(self):
        """GradedBracket without expand() is abstract."""
        with pytest.raises(TypeError):
            GradedBracket("B")  # type: ignore[abstract]
