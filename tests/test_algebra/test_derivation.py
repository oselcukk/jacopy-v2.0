"""Tests for jacopy.algebra.derivation."""

import pytest

from jacopy.algebra.derivation import Act, Derivation, compose, degree_of
from jacopy.core.expr import Integer, Neg, Product, Rational, Sum, Symbol
from jacopy.core.properties import Graded, Scalar
from jacopy.core.registry import PropertyRegistry
from jacopy.core.symbolic_degree import Degree


# --------------------------------------------------------------------- #
# Derivation construction                                                #
# --------------------------------------------------------------------- #


class TestDerivationConstruction:
    def test_basic(self):
        d = Derivation("d", degree=1)
        assert d.name == "d"
        assert d.degree == Degree.const(1)
        assert d.is_atom

    def test_int_degree_coerced(self):
        d = Derivation("d", degree=2)
        assert d.degree == Degree.const(2)

    def test_symbolic_degree(self):
        d = Derivation("D", degree=Degree.var("|D|"))
        assert d.degree == Degree.var("|D|")

    def test_default_degree_zero(self):
        d = Derivation("X")
        assert d.degree == Degree.const(0)

    def test_rejects_empty_name(self):
        with pytest.raises(ValueError):
            Derivation("", degree=1)

    def test_rejects_non_string_name(self):
        with pytest.raises(TypeError):
            Derivation(42, degree=1)  # type: ignore[arg-type]

    def test_equality_structural(self):
        assert Derivation("d", degree=1) == Derivation("d", degree=1)
        assert Derivation("d", degree=1) != Derivation("d", degree=2)
        assert Derivation("d", degree=1) != Derivation("e", degree=1)

    def test_hashable(self):
        s = {Derivation("d", degree=1), Derivation("d", degree=1)}
        assert len(s) == 1

    def test_repr(self):
        assert repr(Derivation("d", degree=1)) == "d"


# --------------------------------------------------------------------- #
# Act construction                                                       #
# --------------------------------------------------------------------- #


class TestAct:
    def test_basic_children(self):
        d = Derivation("d", degree=1)
        x = Symbol("x")
        a = Act(d, x)
        assert a.op is d
        assert a.arg is x
        assert a.children == (d, x)

    def test_callable_shortcut(self):
        d = Derivation("d", degree=1)
        x = Symbol("x")
        assert d(x) == Act(d, x)

    def test_rejects_non_expr_op(self):
        with pytest.raises(TypeError):
            Act("d", Symbol("x"))  # type: ignore[arg-type]

    def test_rejects_non_expr_arg(self):
        with pytest.raises(TypeError):
            Act(Derivation("d", 1), "x")  # type: ignore[arg-type]

    def test_equality_structural(self):
        d = Derivation("d", degree=1)
        x = Symbol("x")
        assert Act(d, x) == Act(d, x)
        assert Act(d, x) != Act(d, Symbol("y"))

    def test_repr(self):
        d = Derivation("d", degree=1)
        assert repr(Act(d, Symbol("x"))) == "d(x)"

    def test_walk_includes_op_and_arg(self):
        d = Derivation("d", degree=1)
        x = Symbol("x")
        a = Act(d, x)
        nodes = list(a.walk())
        assert d in nodes
        assert x in nodes
        assert a in nodes


# --------------------------------------------------------------------- #
# degree_of                                                              #
# --------------------------------------------------------------------- #


class TestDegreeOf:
    def test_derivation_self_describes(self):
        d = Derivation("d", degree=1)
        assert degree_of(d) == Degree.const(1)

    def test_integer_is_zero(self):
        assert degree_of(Integer(5)) == Degree.const(0)

    def test_rational_is_zero(self):
        assert degree_of(Rational(1, 3)) == Degree.const(0)

    def test_scalar_in_registry(self):
        reg = PropertyRegistry()
        f = Symbol("f")
        reg.declare(f, Scalar())
        assert degree_of(f, reg) == Degree.const(0)

    def test_graded_in_registry(self):
        reg = PropertyRegistry()
        x = Symbol("x")
        reg.declare(x, Graded(degree=2))
        assert degree_of(x, reg) == Degree.const(2)

    def test_graded_symbolic(self):
        reg = PropertyRegistry()
        a = Symbol("a")
        reg.declare(a, Graded(degree=Degree.var("|a|")))
        assert degree_of(a, reg) == Degree.var("|a|")

    def test_unregistered_raises(self):
        x = Symbol("x")
        with pytest.raises(ValueError, match="not determined"):
            degree_of(x)

    def test_without_registry_plain_symbol_raises(self):
        x = Symbol("x")
        with pytest.raises(ValueError):
            degree_of(x, None)

    def test_sum_raises(self):
        # Sum has a degree only when every term's degree agrees, a
        # policy we leave to the caller. degree_of refuses to walk it.
        s = Sum(Symbol("x"), Symbol("y"))
        with pytest.raises(ValueError):
            degree_of(s)

    def test_product_sums_child_degrees(self):
        """|a*b| = |a| + |b|. Covers both graded tensor products and
        operator composition, which share the same Product node."""
        D1 = Derivation("D1", degree=1)
        D2 = Derivation("D2", degree=2)
        assert degree_of(Product(D1, D2)) == Degree.const(3)

    def test_product_mixes_registry_and_derivation(self):
        reg = PropertyRegistry()
        a = Symbol("a")
        reg.declare(a, Graded(degree=3))
        d = Derivation("d", degree=1)
        # Product of a registry-graded atom with a derivation.
        assert degree_of(Product(a, d), reg) == Degree.const(4)

    def test_product_propagates_symbolic(self):
        D = Derivation("D", degree=Degree.var("|D|"))
        E = Derivation("E", degree=2)
        assert degree_of(Product(D, E)) == Degree.var("|D|") + Degree.const(2)

    def test_neg_passthrough(self):
        """Negation is a scalar sign, doesn't shift degree."""
        d = Derivation("d", degree=3)
        assert degree_of(Neg(d)) == Degree.const(3)

    def test_act_adds_op_and_arg_degrees(self):
        """|D(x)| = |D| + |x|."""
        reg = PropertyRegistry()
        x = Symbol("x")
        reg.declare(x, Graded(degree=2))
        d = Derivation("d", degree=1)
        assert degree_of(Act(d, x), reg) == Degree.const(3)

    def test_act_nested(self):
        """|D(E(x))| = |D| + |E| + |x|."""
        reg = PropertyRegistry()
        x = Symbol("x")
        reg.declare(x, Graded(degree=1))
        d = Derivation("d", degree=1)
        e = Derivation("e", degree=2)
        assert degree_of(Act(d, Act(e, x)), reg) == Degree.const(4)

    def test_bracketapply_adds_bracket_degree(self):
        """|[a, b]| = |a| + |b| + bracket.degree."""
        # Bracket of degree −1 (e.g. derived bracket from a deg-1
        # generator: |Q| − 2 = −1) over two degree-1 operands.
        from jacopy.brackets.custom import CustomBracket
        reg = PropertyRegistry()
        a = Symbol("a")
        b = Symbol("b")
        reg.declare(a, Graded(degree=1))
        reg.declare(b, Graded(degree=1))
        B = CustomBracket(
            "B",
            lambda x, y, r: x,  # expansion is irrelevant for degree
            degree=-1,
        )
        assert degree_of(B(a, b), reg) == Degree.const(1)

    def test_bracketapply_symbolic_degree(self):
        """Bracket degree propagates symbolically."""
        from jacopy.brackets.custom import CustomBracket
        reg = PropertyRegistry()
        a, b = Symbol("a"), Symbol("b")
        reg.declare(a, Graded(degree=Degree.var("|a|")))
        reg.declare(b, Graded(degree=Degree.var("|b|")))
        B = CustomBracket("B", lambda x, y, r: x, degree=0)
        assert degree_of(B(a, b), reg) == (
            Degree.var("|a|") + Degree.var("|b|")
        )


# --------------------------------------------------------------------- #
# compose                                                                #
# --------------------------------------------------------------------- #


class TestCompose:
    def test_two_operators_build_product(self):
        D1 = Derivation("D1", degree=1)
        D2 = Derivation("D2", degree=2)
        assert compose(D1, D2) == Product(D1, D2)

    def test_three_operators_preserve_order(self):
        D1 = Derivation("D1", degree=1)
        D2 = Derivation("D2", degree=2)
        D3 = Derivation("D3", degree=3)
        assert compose(D1, D2, D3) == Product(D1, D2, D3)

    def test_single_operator_collapses(self):
        D = Derivation("D", degree=1)
        # No wrapping Product for a lone operator, compose(D) is D.
        assert compose(D) is D

    def test_empty_rejected(self):
        with pytest.raises(ValueError):
            compose()

    def test_non_expr_rejected(self):
        with pytest.raises(TypeError):
            compose(Derivation("D", 1), "not-an-expr")  # type: ignore[arg-type]

    def test_composition_degree_is_sum(self):
        """|D1 ∘ D2| = |D1| + |D2|, falls out of the Product rule."""
        D1 = Derivation("D1", degree=1)
        D2 = Derivation("D2", degree=2)
        assert degree_of(compose(D1, D2)) == Degree.const(3)
