"""Tests for ``jacopy.display.ascii``."""

from __future__ import annotations

import pytest

from jacopy.algebra.commutator import Commutator
from jacopy.algebra.derivation import Act, Derivation
from jacopy.brackets.dorfman import SectionPair
from jacopy.brackets.lie import LieBracket
from jacopy.calculus.pairing import Pairing
from jacopy.core.expr import (
    Integer,
    Neg,
    Power,
    Product,
    Rational,
    Sum,
    Symbol,
)
from jacopy.display.ascii import (
    chain_to_ascii,
    step_to_ascii,
    to_ascii,
)
from jacopy.proof.chain import ProofChain
from jacopy.proof.step import ProofStep


# --------------------------------------------------------------------- #
# Core atoms                                                            #
# --------------------------------------------------------------------- #


class TestAtoms:
    def test_symbol_is_its_name(self):
        assert to_ascii(Symbol("X")) == "X"

    def test_symbol_with_greek_passes_through(self):
        # ASCII renderer does not sanitise Unicode; that is LaTeX's job.
        assert to_ascii(Symbol("ω")) == "ω"

    def test_positive_integer(self):
        assert to_ascii(Integer(3)) == "3"

    def test_zero(self):
        assert to_ascii(Integer(0)) == "0"

    def test_negative_integer_does_not_wrap_at_top_level(self):
        # ctx_precedence=0 means no wrapping, even for negative.
        assert to_ascii(Integer(-4)) == "-4"

    def test_negative_integer_wraps_when_below_context(self):
        # Inside a Product(-4 * X), the factor is rendered at PRODUCT+1,
        # which is above NEG, so it parenthesises.
        x = Symbol("X")
        assert to_ascii(Product(Integer(-4), x)) == "(-4) * X"

    def test_positive_rational(self):
        assert to_ascii(Rational(3, 5)) == "3/5"

    def test_negative_rational_wraps_in_product(self):
        assert "(-" in to_ascii(Product(Rational(-1, 2), Symbol("X")))

    def test_positive_rational_bare(self):
        assert to_ascii(Rational(3, 5)) == "3/5"


# --------------------------------------------------------------------- #
# Compound expressions                                                  #
# --------------------------------------------------------------------- #


class TestNegSum:
    def test_neg_symbol(self):
        assert to_ascii(Neg(Symbol("X"))) == "-X"

    def test_sum_two_positives(self):
        assert to_ascii(Sum(Symbol("X"), Symbol("Y"))) == "X + Y"

    def test_sum_with_neg_folds_to_minus(self):
        # Sum(X, Neg(Y)) → "X - Y", not "X + -Y" or "X + (-Y)".
        s = Sum(Symbol("X"), Neg(Symbol("Y")))
        assert to_ascii(s) == "X - Y"

    def test_sum_leading_neg(self):
        # Leading Neg child: "-X + Y".
        s = Sum(Neg(Symbol("X")), Symbol("Y"))
        assert to_ascii(s) == "-X + Y"

    def test_empty_sum_renders_as_zero(self):
        # Sum with no children; constructed directly (not via make).
        assert to_ascii(Sum()) == "0"

    def test_three_term_sum_with_minus(self):
        s = Sum(Symbol("X"), Neg(Symbol("Y")), Symbol("Z"))
        assert to_ascii(s) == "X - Y + Z"


class TestProductPower:
    def test_product(self):
        assert to_ascii(Product(Symbol("X"), Symbol("Y"))) == "X * Y"

    def test_product_wraps_sum_factor(self):
        # Sum's precedence is below Product's, parens required.
        s = Sum(Symbol("A"), Symbol("B"))
        out = to_ascii(Product(s, Symbol("C")))
        assert out == "(A + B) * C"

    def test_sum_does_not_wrap_product_child(self):
        p = Product(Symbol("A"), Symbol("B"))
        out = to_ascii(Sum(p, Symbol("C")))
        assert out == "A * B + C"

    def test_power(self):
        assert to_ascii(Power(Symbol("X"), Integer(2))) == "X**2"

    def test_power_of_sum_wraps_base(self):
        s = Sum(Symbol("A"), Symbol("B"))
        assert to_ascii(Power(s, Integer(2))) == "(A + B)**2"


# --------------------------------------------------------------------- #
# Algebra                                                               #
# --------------------------------------------------------------------- #


class TestAlgebra:
    def test_derivation_is_its_name(self):
        assert to_ascii(Derivation("d", 1)) == "d"

    def test_act_simple(self):
        d = Derivation("d", 1)
        assert to_ascii(Act(d, Symbol("X"))) == "d(X)"

    def test_act_with_composite_operator_wraps(self):
        # Sum operator: (d + L)(X), parens around the sum.
        op = Sum(Derivation("d", 1), Derivation("L", 0))
        out = to_ascii(Act(op, Symbol("X")))
        assert out == "(d + L)(X)"

    def test_commutator(self):
        out = to_ascii(Commutator(Symbol("X"), Symbol("Y")))
        assert out == "[X, Y]"


# --------------------------------------------------------------------- #
# Brackets / sections / pairing                                         #
# --------------------------------------------------------------------- #


class TestBracketNodes:
    def test_bracket_apply(self):
        lie = LieBracket("[·,·]_L")
        expr = lie(Symbol("X"), Symbol("Y"))
        out = to_ascii(expr)
        assert out.startswith("[X, Y]_{")
        assert "[·,·]_L" in out

    def test_section_pair(self):
        sp = SectionPair(Symbol("X"), Symbol("α"))
        assert to_ascii(sp) == "(X, α)"

    def test_pairing(self):
        p = Pairing(Symbol("α"), Symbol("X"))
        assert to_ascii(p) == "<α, X>"


# --------------------------------------------------------------------- #
# Error handling                                                        #
# --------------------------------------------------------------------- #


class TestErrors:
    def test_non_expr_raises(self):
        with pytest.raises(TypeError):
            to_ascii("not an expr")  # type: ignore[arg-type]


# --------------------------------------------------------------------- #
# ProofStep / ProofChain                                                #
# --------------------------------------------------------------------- #


class TestProofTranscript:
    def test_step_basic(self):
        step = ProofStep(Symbol("X"), Symbol("Y"), rule="demo")
        out = step_to_ascii(step)
        assert "[demo]" in out
        assert "X -> Y" in out

    def test_step_with_provenance_tag(self):
        step = ProofStep(
            Symbol("X"),
            Symbol("Y"),
            rule="axiom-fire",
            provenance_tag="axiom",
        )
        out = step_to_ascii(step)
        assert "(axiom)" in out

    def test_step_with_justification(self):
        step = ProofStep(
            Symbol("X"),
            Symbol("Y"),
            rule="demo",
            justification="because reasons",
        )
        out = step_to_ascii(step)
        assert "-- because reasons" in out

    def test_step_nests_children(self):
        child = ProofStep(Symbol("A"), Symbol("B"), rule="sub")
        parent = ProofStep(
            Symbol("X"),
            Symbol("Y"),
            rule="outer",
            children=[child],
        )
        out = step_to_ascii(parent)
        lines = out.splitlines()
        assert len(lines) == 2
        assert lines[0].startswith("[outer]")
        assert lines[1].startswith("  [sub]")

    def test_step_max_depth_zero_hides_children(self):
        child = ProofStep(Symbol("A"), Symbol("B"), rule="sub")
        parent = ProofStep(
            Symbol("X"),
            Symbol("Y"),
            rule="outer",
            children=[child],
        )
        out = step_to_ascii(parent, max_depth=0)
        assert "sub" not in out

    def test_chain_empty(self):
        assert chain_to_ascii(ProofChain()) == "(empty proof chain)"

    def test_chain_multiple_steps(self):
        s1 = ProofStep(Symbol("X"), Symbol("Y"), rule="r1")
        s2 = ProofStep(Symbol("Y"), Symbol("Z"), rule="r2")
        out = chain_to_ascii(ProofChain([s1, s2]))
        assert "[r1] X -> Y" in out
        assert "[r2] Y -> Z" in out

    def test_step_type_error(self):
        with pytest.raises(TypeError):
            step_to_ascii("not a step")  # type: ignore[arg-type]

    def test_chain_type_error(self):
        with pytest.raises(TypeError):
            chain_to_ascii("not a chain")  # type: ignore[arg-type]


# --------------------------------------------------------------------- #
# Verbosity modes                                                       #
# --------------------------------------------------------------------- #


class TestVerbosity:
    def _demo_step(self):
        return ProofStep(
            Symbol("X"),
            Symbol("Y"),
            rule="demo",
            justification="because of Cartan",
            provenance_tag="axiom",
        )

    def _demo_chain_with_children(self):
        child = ProofStep(Symbol("A"), Symbol("B"), rule="sub")
        parent = ProofStep(
            Symbol("X"),
            Symbol("Y"),
            rule="outer",
            justification="outer-j",
            children=[child],
        )
        return ProofChain([parent])

    def test_full_is_default_and_includes_justification(self):
        out = step_to_ascii(self._demo_step())
        assert "X -> Y" in out
        assert "because of Cartan" in out
        # Default must equal explicit "full".
        assert out == step_to_ascii(self._demo_step(), verbosity="full")

    def test_summary_drops_justification_but_keeps_arrow(self):
        out = step_to_ascii(self._demo_step(), verbosity="summary")
        assert "X -> Y" in out
        assert "because of Cartan" not in out

    def test_compact_drops_arrow_and_justification(self):
        out = step_to_ascii(self._demo_step(), verbosity="compact")
        assert "X -> Y" not in out
        assert "because of Cartan" not in out
        # Rule + tag only.
        assert out == "[demo] (axiom)"

    def test_summary_keeps_children(self):
        chain = self._demo_chain_with_children()
        out = chain_to_ascii(chain, verbosity="summary")
        assert "[outer]" in out
        assert "[sub]" in out
        assert "outer-j" not in out

    def test_compact_suppresses_children(self):
        chain = self._demo_chain_with_children()
        out = chain_to_ascii(chain, verbosity="compact")
        assert "[outer]" in out
        assert "[sub]" not in out

    def test_invalid_verbosity_on_step_raises(self):
        with pytest.raises(ValueError, match="verbosity must be one of"):
            step_to_ascii(self._demo_step(), verbosity="loud")

    def test_invalid_verbosity_on_chain_raises(self):
        with pytest.raises(ValueError, match="verbosity must be one of"):
            chain_to_ascii(
                self._demo_chain_with_children(), verbosity="loud"
            )

    def test_empty_chain_is_placeholder_in_every_mode(self):
        for mode in ("full", "summary", "compact"):
            assert (
                chain_to_ascii(ProofChain(), verbosity=mode)
                == "(empty proof chain)"
            )


# --------------------------------------------------------------------- #
# Sign normalisation integration                                        #
# --------------------------------------------------------------------- #


class TestSignNormalisation:
    def test_nested_minus_in_product_of_sum(self):
        # (X - Y) * Z
        inner = Sum(Symbol("X"), Neg(Symbol("Y")))
        out = to_ascii(Product(inner, Symbol("Z")))
        assert out == "(X - Y) * Z"

    def test_derivation_applied_to_signed_sum(self):
        d = Derivation("d", 1)
        expr = Act(d, Sum(Symbol("X"), Neg(Symbol("Y"))))
        assert to_ascii(expr) == "d(X - Y)"
