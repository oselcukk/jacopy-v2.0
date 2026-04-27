"""Tests for the Dorfman bracket on TM ⊕ T*M section pairs."""

import pytest

from jacopy.algebra.derivation import Act
from jacopy.brackets.dorfman import DorfmanBracket, SectionPair
from jacopy.brackets.lie import LieBracket
from jacopy.calculus.exterior_d import d
from jacopy.core.expr import Neg, Sum, Symbol


class TestSectionPair:
    def test_stores_vector_and_form(self):
        p = SectionPair(Symbol("X"), Symbol("α"))
        assert p.vector == Symbol("X")
        assert p.form == Symbol("α")

    def test_children_are_vector_and_form(self):
        p = SectionPair(Symbol("X"), Symbol("α"))
        assert p.children == (Symbol("X"), Symbol("α"))

    def test_equality_is_structural(self):
        p1 = SectionPair(Symbol("X"), Symbol("α"))
        p2 = SectionPair(Symbol("X"), Symbol("α"))
        assert p1 == p2
        assert hash(p1) == hash(p2)

    def test_rejects_non_expr(self):
        with pytest.raises(TypeError):
            SectionPair("X", Symbol("α"))  # type: ignore[arg-type]


class TestDorfmanBracketAxioms:
    def test_degree_zero(self):
        from jacopy.core.symbolic_degree import Degree
        db = DorfmanBracket()
        assert db.degree == Degree.const(0)

    def test_not_antisymmetric(self):
        db = DorfmanBracket()
        assert db.is_graded_antisymmetric is False

    def test_satisfies_leibniz_and_jacobi(self):
        db = DorfmanBracket()
        assert db.satisfies_leibniz is True
        assert db.satisfies_graded_jacobi is True


class TestDorfmanBracketExpansion:
    def test_vector_part_is_lie_bracket(self):
        db = DorfmanBracket()
        X, Y = Symbol("X"), Symbol("Y")
        alpha, beta = Symbol("α"), Symbol("β")
        result = db.expand(SectionPair(X, alpha), SectionPair(Y, beta))
        assert isinstance(result, SectionPair)
        assert result.vector == LieBracket().expand(X, Y)

    def test_form_part_is_L_X_beta_minus_iota_Y_d_alpha(self):
        db = DorfmanBracket()
        X, Y = Symbol("X"), Symbol("Y")
        alpha, beta = Symbol("α"), Symbol("β")
        result = db.expand(SectionPair(X, alpha), SectionPair(Y, beta))
        form = result.form
        assert isinstance(form, Sum)
        # The form component must mention both β (via L_X) and dα (via ι_Y∘d).
        descendants = list(form.walk())
        assert any(node == beta for node in descendants)
        assert any(
            isinstance(node, Act) and node.op == d and node.arg == alpha
            for node in descendants
        )
        # The ι_Y branch must be subtracted, not added.
        assert any(isinstance(node, Neg) for node in form.children)

    def test_rejects_raw_vector_operand(self):
        db = DorfmanBracket()
        with pytest.raises(TypeError, match="SectionPair"):
            db.expand(Symbol("X"), SectionPair(Symbol("Y"), Symbol("β")))

    def test_rejects_raw_form_operand(self):
        db = DorfmanBracket()
        with pytest.raises(TypeError, match="SectionPair"):
            db.expand(SectionPair(Symbol("X"), Symbol("α")), Symbol("β"))


class TestDorfmanBracketIdentity:
    def test_two_defaults_are_equal(self):
        db1 = DorfmanBracket()
        db2 = DorfmanBracket()
        assert db1 == db2
        assert hash(db1) == hash(db2)

    def test_custom_name_preserved(self):
        db = DorfmanBracket(name="[·,·]_Dorf")
        assert db.name == "[·,·]_Dorf"
