"""Tests for the Courant bracket on TM ⊕ T*M section pairs."""

import pytest

from jacopy.algebra.derivation import Act
from jacopy.brackets.courant import CourantBracket
from jacopy.brackets.derived import VanishingCondition
from jacopy.brackets.dorfman import SectionPair
from jacopy.brackets.lie import LieBracket
from jacopy.calculus.exterior_d import d
from jacopy.calculus.interior import interior
from jacopy.calculus.lie_derivative import lie_derivative
from jacopy.core.expr import Integer, Neg, Product, Rational, Sum, Symbol, Zero
from jacopy.core.symbolic_degree import Degree


# --------------------------------------------------------------------- #
# Construction + axiom flags                                             #
# --------------------------------------------------------------------- #


class TestConstruction:
    def test_default_name(self):
        C = CourantBracket()
        assert C.name == "[·,·]_C"

    def test_custom_name(self):
        C = CourantBracket(name="[·,·]_{C,alt}")
        assert C.name == "[·,·]_{C,alt}"

    def test_default_vector_bracket_is_lie(self):
        C = CourantBracket()
        assert C.vector_bracket == LieBracket()

    def test_custom_vector_bracket_honoured(self):
        custom = LieBracket(name="[·,·]_custom")
        C = CourantBracket(vector_bracket=custom)
        assert C.vector_bracket is custom

    def test_background_H_defaults_to_none(self):
        C = CourantBracket()
        assert C.background_H is None
        assert C.is_twisted is False

    def test_background_H_stored(self):
        H = Symbol("H")
        C = CourantBracket(background_H=H)
        assert C.background_H is H
        assert C.is_twisted is True

    def test_rejects_non_expr_background_H(self):
        with pytest.raises(TypeError):
            CourantBracket(background_H="H")  # type: ignore[arg-type]


class TestAxiomFlags:
    def test_degree_zero(self):
        C = CourantBracket()
        assert C.degree == Degree.const(0)

    def test_graded_antisymmetric(self):
        """Courant is skew, this is the defining difference from Dorfman."""
        C = CourantBracket()
        assert C.is_graded_antisymmetric is True

    def test_does_not_satisfy_leibniz(self):
        """The Dorfman/Courant dichotomy: Courant loses Leibniz."""
        C = CourantBracket()
        assert C.satisfies_leibniz is False

    def test_jacobi_is_conditional(self):
        """Jacobi reported as None (conditional), the H-twisted case
        needs dH = 0 and the untwisted case is cleaner at the proof
        layer than as an unconditional flag."""
        C = CourantBracket()
        assert C.satisfies_graded_jacobi is None


# --------------------------------------------------------------------- #
# Untwisted expansion                                                    #
# --------------------------------------------------------------------- #


class TestUntwistedExpansion:
    def test_result_is_section_pair(self):
        C = CourantBracket()
        X, Y = Symbol("X"), Symbol("Y")
        alpha, beta = Symbol("α"), Symbol("β")
        out = C.expand(SectionPair(X, alpha), SectionPair(Y, beta))
        assert isinstance(out, SectionPair)

    def test_vector_part_is_lie_bracket(self):
        C = CourantBracket()
        X, Y = Symbol("X"), Symbol("Y")
        alpha, beta = Symbol("α"), Symbol("β")
        out = C.expand(SectionPair(X, alpha), SectionPair(Y, beta))
        assert out.vector == LieBracket().expand(X, Y)

    def test_form_part_has_three_terms(self):
        """L_X β − L_Y α − ½ d(ι_X β − ι_Y α), three terms in the Sum."""
        C = CourantBracket()
        X, Y = Symbol("X"), Symbol("Y")
        alpha, beta = Symbol("α"), Symbol("β")
        out = C.expand(SectionPair(X, alpha), SectionPair(Y, beta))
        assert isinstance(out.form, Sum)
        assert len(out.form.children) == 3

    def test_form_part_first_term_is_L_X_beta(self):
        C = CourantBracket()
        X, Y = Symbol("X"), Symbol("Y")
        alpha, beta = Symbol("α"), Symbol("β")
        out = C.expand(SectionPair(X, alpha), SectionPair(Y, beta))
        assert out.form.children[0] == Act(lie_derivative(X), beta)

    def test_form_part_second_term_is_neg_L_Y_alpha(self):
        C = CourantBracket()
        X, Y = Symbol("X"), Symbol("Y")
        alpha, beta = Symbol("α"), Symbol("β")
        out = C.expand(SectionPair(X, alpha), SectionPair(Y, beta))
        assert out.form.children[1] == Neg(Act(lie_derivative(Y), alpha))

    def test_form_part_third_term_is_neg_half_d_contraction_diff(self):
        """−½ d(ι_X β − ι_Y α): the exact correction that distinguishes
        Courant from Dorfman."""
        C = CourantBracket()
        X, Y = Symbol("X"), Symbol("Y")
        alpha, beta = Symbol("α"), Symbol("β")
        out = C.expand(SectionPair(X, alpha), SectionPair(Y, beta))
        expected_inner = Sum(Act(interior(X), beta), Neg(Act(interior(Y), alpha)))
        expected = Neg(Product(Rational(1, 2), Act(d, expected_inner)))
        assert out.form.children[2] == expected

    def test_form_part_uses_half_rational(self):
        """The coefficient on the Cartan correction is a Rational(1, 2),
        not an Integer, not a float."""
        C = CourantBracket()
        out = C.expand(
            SectionPair(Symbol("X"), Symbol("α")),
            SectionPair(Symbol("Y"), Symbol("β")),
        )
        third = out.form.children[2]  # Neg(Product(½, d(...)))
        product = third.arg
        assert product.children[0] == Rational(1, 2)


class TestOperandTypes:
    def test_rejects_non_section_pair_left(self):
        C = CourantBracket()
        with pytest.raises(TypeError, match="left operand"):
            C.expand(Symbol("X"), SectionPair(Symbol("Y"), Symbol("β")))

    def test_rejects_non_section_pair_right(self):
        C = CourantBracket()
        with pytest.raises(TypeError, match="right operand"):
            C.expand(SectionPair(Symbol("X"), Symbol("α")), Symbol("Y"))


# --------------------------------------------------------------------- #
# H-twisted expansion                                                    #
# --------------------------------------------------------------------- #


class TestTwistedExpansion:
    def test_twisted_form_has_four_terms(self):
        """Plain three + the ι_Y ι_X H correction."""
        H = Symbol("H")
        C = CourantBracket(background_H=H)
        X, Y = Symbol("X"), Symbol("Y")
        alpha, beta = Symbol("α"), Symbol("β")
        out = C.expand(SectionPair(X, alpha), SectionPair(Y, beta))
        assert len(out.form.children) == 4

    def test_twist_term_is_iota_Y_iota_X_H(self):
        H = Symbol("H")
        C = CourantBracket(background_H=H)
        X, Y = Symbol("X"), Symbol("Y")
        alpha, beta = Symbol("α"), Symbol("β")
        out = C.expand(SectionPair(X, alpha), SectionPair(Y, beta))
        expected = Act(interior(Y), Act(interior(X), H))
        assert out.form.children[3] == expected

    def test_twist_term_has_no_sign(self):
        """Classical H-twist convention: the twist is additive, no Neg."""
        H = Symbol("H")
        C = CourantBracket(background_H=H)
        out = C.expand(
            SectionPair(Symbol("X"), Symbol("α")),
            SectionPair(Symbol("Y"), Symbol("β")),
        )
        assert not isinstance(out.form.children[3], Neg)

    def test_vector_part_unaffected_by_twist(self):
        """The twist only touches the form half; the vector half is
        still the Lie bracket."""
        X, Y = Symbol("X"), Symbol("Y")
        alpha, beta = Symbol("α"), Symbol("β")
        C_plain = CourantBracket()
        C_twist = CourantBracket(background_H=Symbol("H"))
        plain = C_plain.expand(SectionPair(X, alpha), SectionPair(Y, beta))
        twisted = C_twist.expand(SectionPair(X, alpha), SectionPair(Y, beta))
        assert plain.vector == twisted.vector

    def test_untwisted_minus_twisted_is_only_twist_term(self):
        """First three form terms coincide between plain and H-twisted."""
        X, Y = Symbol("X"), Symbol("Y")
        alpha, beta = Symbol("α"), Symbol("β")
        C_plain = CourantBracket()
        C_twist = CourantBracket(background_H=Symbol("H"))
        plain = C_plain.expand(SectionPair(X, alpha), SectionPair(Y, beta))
        twisted = C_twist.expand(SectionPair(X, alpha), SectionPair(Y, beta))
        assert plain.form.children == twisted.form.children[:3]


# --------------------------------------------------------------------- #
# Jacobi condition                                                       #
# --------------------------------------------------------------------- #


class TestJacobiCondition:
    def test_untwisted_condition_is_vacuous(self):
        """Plain Courant Jacobi holds on (TM ⊕ T*M); the obstruction is
        the literal zero."""
        C = CourantBracket()
        cond = C.jacobi_condition()
        assert isinstance(cond, VanishingCondition)
        assert cond.obstruction == Zero
        assert cond.holds() is True

    def test_twisted_condition_obstruction_is_dH(self):
        """H-twisted Courant Jacobi ≡ dH = 0."""
        H = Symbol("H")
        C = CourantBracket(background_H=H)
        cond = C.jacobi_condition()
        assert cond.obstruction == Act(d, H)

    def test_twisted_condition_name_references_twist(self):
        H = Symbol("H")
        C = CourantBracket(background_H=H)
        cond = C.jacobi_condition()
        assert "H-twisted" in cond.name

    def test_untwisted_condition_name_marks_vacuous(self):
        C = CourantBracket()
        cond = C.jacobi_condition()
        assert "vacuous" in cond.name


# --------------------------------------------------------------------- #
# Identity / equality                                                    #
# --------------------------------------------------------------------- #


class TestIdentity:
    def test_two_default_brackets_are_equal(self):
        assert CourantBracket() == CourantBracket()

    def test_brackets_with_different_twist_not_equal(self):
        assert CourantBracket() != CourantBracket(background_H=Symbol("H"))

    def test_brackets_with_different_H_not_equal(self):
        c1 = CourantBracket(background_H=Symbol("H1"))
        c2 = CourantBracket(background_H=Symbol("H2"))
        assert c1 != c2

    def test_hash_consistent_with_equality(self):
        assert hash(CourantBracket()) == hash(CourantBracket())

    def test_repr_includes_name(self):
        C = CourantBracket(name="[·,·]_C")
        assert "[·,·]_C" in repr(C)
