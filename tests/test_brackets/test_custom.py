"""Tests for jacopy.brackets.custom."""

import pytest

from jacopy.brackets.base import BracketApply
from jacopy.brackets.custom import CustomBracket
from jacopy.core.expr import Neg, Product, Sum, Symbol
from jacopy.core.symbolic_degree import Degree


def _anticommutator(a, b, registry):
    """A B + B A, not antisymmetric, used as a custom rule."""
    return Sum(Product(a, b), Product(b, a))


# --------------------------------------------------------------------- #
# Construction                                                           #
# --------------------------------------------------------------------- #


class TestConstruction:
    def test_basic(self):
        B = CustomBracket(
            "anti",
            _anticommutator,
            degree=0,
            is_graded_antisymmetric=False,
            satisfies_leibniz=True,
            satisfies_graded_jacobi=False,
        )
        assert B.name == "anti"
        assert B.degree == Degree.const(0)
        assert not B.is_graded_antisymmetric
        assert B.satisfies_leibniz
        assert B.satisfies_graded_jacobi is False

    def test_rejects_non_callable_expand(self):
        with pytest.raises(TypeError):
            CustomBracket("bad", "not-callable")  # type: ignore[arg-type]

    def test_equality_accounts_for_expand_fn(self):
        """Two CustomBrackets with the same axioms but different
        expansion rules must compare unequal."""
        rule_a = lambda a, b, reg: Sum(Product(a, b), Neg(Product(b, a)))
        rule_b = lambda a, b, reg: Sum(Product(a, b), Product(b, a))
        A = CustomBracket("B", rule_a)
        B = CustomBracket("B", rule_b)
        assert A != B

    def test_equality_same_rule_matches(self):
        rule = lambda a, b, reg: Sum(Product(a, b), Product(b, a))
        A = CustomBracket("B", rule)
        B = CustomBracket("B", rule)
        assert A == B


# --------------------------------------------------------------------- #
# Application + expansion                                                #
# --------------------------------------------------------------------- #


class TestBehavior:
    def test_application_produces_bracketapply(self):
        B = CustomBracket("anti", _anticommutator)
        X, Y = Symbol("X"), Symbol("Y")
        node = B(X, Y)
        assert isinstance(node, BracketApply)
        assert node.bracket is B
        assert node.children == (X, Y)

    def test_expansion_calls_user_rule(self):
        B = CustomBracket("anti", _anticommutator)
        X, Y = Symbol("X"), Symbol("Y")
        assert B(X, Y).expand() == Sum(Product(X, Y), Product(Y, X))

    def test_registry_forwarded_to_rule(self):
        captured = {}

        def rule(a, b, registry):
            captured["registry"] = registry
            return Product(a, b)

        B = CustomBracket("capture", rule)
        sentinel = object()
        B.expand(Symbol("a"), Symbol("b"), sentinel)  # type: ignore[arg-type]
        assert captured["registry"] is sentinel
