"""Tests for ``jacopy.display.latex``."""

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
from jacopy.display.latex import (
    chain_to_latex,
    chain_to_latex_document,
    chain_to_tikz,
    chain_to_tikz_document,
    latex_name,
    step_to_latex,
    to_latex,
)
from jacopy.proof.chain import ProofChain
from jacopy.proof.step import ProofStep


# --------------------------------------------------------------------- #
# Name sanitising                                                       #
# --------------------------------------------------------------------- #


class TestLatexName:
    def test_ascii_passthrough(self):
        assert latex_name("X") == "X"

    def test_lowercase_greek(self):
        assert latex_name("ω") == r"\omega"
        assert latex_name("α") == r"\alpha"
        assert latex_name("ι") == r"\iota"

    def test_uppercase_greek(self):
        assert latex_name("Θ") == r"\Theta"
        assert latex_name("Σ") == r"\Sigma"

    def test_musical_and_algebraic(self):
        assert latex_name("♭") == r"\flat"
        assert latex_name("♯") == r"\sharp"
        assert latex_name("∧") == r"\wedge"

    def test_single_char_subscript_not_braced(self):
        assert latex_name("X_f") == "X_f"

    def test_multi_char_subscript_is_braced(self):
        assert latex_name("X_ab") == "X_{ab}"

    def test_greek_with_subscript(self):
        # ι_X, single-char subscript stays bare.
        assert latex_name("ι_X") == r"\iota_X"

    def test_greek_with_multichar_subscript(self):
        assert latex_name("ω_xy") == r"\omega_{xy}"

    def test_non_str_raises(self):
        with pytest.raises(TypeError):
            latex_name(42)  # type: ignore[arg-type]


# --------------------------------------------------------------------- #
# Core atoms                                                            #
# --------------------------------------------------------------------- #


class TestAtoms:
    def test_symbol(self):
        assert to_latex(Symbol("X")) == "X"

    def test_symbol_with_greek(self):
        assert to_latex(Symbol("ω")) == r"\omega"

    def test_positive_integer(self):
        assert to_latex(Integer(7)) == "7"

    def test_zero(self):
        assert to_latex(Integer(0)) == "0"

    def test_negative_integer_bare(self):
        assert to_latex(Integer(-4)) == "-4"

    def test_positive_rational_as_frac(self):
        assert to_latex(Rational(3, 5)) == r"\frac{3}{5}"

    def test_negative_rational_has_minus_outside(self):
        out = to_latex(Rational(-3, 5))
        assert out == r"-\frac{3}{5}"


# --------------------------------------------------------------------- #
# Compound expressions                                                  #
# --------------------------------------------------------------------- #


class TestNegSum:
    def test_neg_symbol(self):
        assert to_latex(Neg(Symbol("X"))) == "-X"

    def test_sum_two_positives(self):
        assert to_latex(Sum(Symbol("X"), Symbol("Y"))) == "X + Y"

    def test_sum_with_neg_folds(self):
        s = Sum(Symbol("X"), Neg(Symbol("Y")))
        assert to_latex(s) == "X - Y"

    def test_sum_leading_neg(self):
        s = Sum(Neg(Symbol("X")), Symbol("Y"))
        assert to_latex(s) == "-X + Y"


class TestProductPower:
    def test_product_uses_thin_space(self):
        out = to_latex(Product(Symbol("X"), Symbol("Y")))
        assert out == r"X \, Y"

    def test_product_wraps_sum_factor(self):
        s = Sum(Symbol("A"), Symbol("B"))
        out = to_latex(Product(s, Symbol("C")))
        assert out == r"\left(A + B\right) \, C"

    def test_power(self):
        out = to_latex(Power(Symbol("X"), Integer(2)))
        assert out == "{X}^{2}"


# --------------------------------------------------------------------- #
# Algebra                                                               #
# --------------------------------------------------------------------- #


class TestAlgebra:
    def test_derivation_name_sanitised(self):
        d = Derivation("ι_X", -1)
        assert to_latex(d) == r"\iota_X"

    def test_act_uses_left_right_parens(self):
        d = Derivation("d", 1)
        out = to_latex(Act(d, Symbol("X")))
        assert out == r"d\!\left(X\right)"

    def test_commutator(self):
        out = to_latex(Commutator(Symbol("X"), Symbol("Y")))
        assert out == r"\left[X,\, Y\right]"


# --------------------------------------------------------------------- #
# Bracket / section / pairing                                           #
# --------------------------------------------------------------------- #


class TestBracketNodes:
    def test_bracket_apply_tag_is_sanitised(self):
        # Bracket name containing Greek should come out with LaTeX commands.
        lie = LieBracket("[·,·]_ω")
        expr = lie(Symbol("X"), Symbol("Y"))
        out = to_latex(expr)
        assert out.startswith(r"\left[X,\, Y\right]_{")
        assert r"\omega" in out

    def test_section_pair(self):
        sp = SectionPair(Symbol("X"), Symbol("α"))
        out = to_latex(sp)
        assert out == r"\left(X,\, \alpha\right)"

    def test_pairing(self):
        p = Pairing(Symbol("α"), Symbol("X"))
        out = to_latex(p)
        assert out == r"\langle \alpha,\, X \rangle"


# --------------------------------------------------------------------- #
# Error handling                                                        #
# --------------------------------------------------------------------- #


class TestErrors:
    def test_non_expr_raises(self):
        with pytest.raises(TypeError):
            to_latex("not an expr")  # type: ignore[arg-type]


# --------------------------------------------------------------------- #
# ProofStep / ProofChain                                                #
# --------------------------------------------------------------------- #


class TestProofTranscript:
    def test_step_basic_shape(self):
        step = ProofStep(Symbol("X"), Symbol("Y"), rule="demo")
        out = step_to_latex(step)
        assert r"X \to Y" in out
        assert r"\text{[demo]}" in out

    def test_step_with_provenance_tag(self):
        step = ProofStep(
            Symbol("X"),
            Symbol("Y"),
            rule="r",
            provenance_tag="theorem",
        )
        out = step_to_latex(step)
        assert "(theorem)" in out

    def test_step_with_justification(self):
        step = ProofStep(
            Symbol("X"),
            Symbol("Y"),
            rule="demo",
            justification="because of Cartan",
        )
        out = step_to_latex(step)
        assert r"\text{--- because of Cartan}" in out

    def test_step_escapes_underscores_in_rule(self):
        step = ProofStep(Symbol("X"), Symbol("Y"), rule="rule_name")
        out = step_to_latex(step)
        assert r"rule\_name" in out

    def test_step_translates_unicode_in_rule_via_ensuremath(self):
        """Regression: rule names carry math glyphs (ι, ∘, ω). Text-mode
        \\text{…} can't host raw Unicode, pdfLaTeX errors with
        'Unicode character not set up for use with LaTeX'. Each glyph
        must land inside \\ensuremath{…} so math mode is entered
        locally."""
        step = ProofStep(Symbol("X"), Symbol("Y"), rule="L_X := d∘ι_X")
        out = step_to_latex(step)
        # Raw glyphs must NOT survive into the output.
        assert "∘" not in out
        assert "ι" not in out
        # Each translated glyph must be wrapped in \ensuremath{…}.
        assert r"\ensuremath{\circ}" in out
        assert r"\ensuremath{\iota}" in out

    def test_step_translates_unicode_in_justification(self):
        step = ProofStep(
            Symbol("X"),
            Symbol("Y"),
            rule="r",
            justification="uses ω and ∂",
        )
        out = step_to_latex(step)
        assert "ω" not in out
        assert "∂" not in out
        assert r"\ensuremath{\omega}" in out
        assert r"\ensuremath{\partial}" in out

    def test_chain_empty_produces_placeholder(self):
        out = chain_to_latex(ProofChain())
        assert r"\begin{gather*}" in out
        assert r"\end{gather*}" in out
        assert "empty proof chain" in out

    def test_chain_multiple_steps_joined_with_double_backslash(self):
        s1 = ProofStep(Symbol("X"), Symbol("Y"), rule="r1")
        s2 = ProofStep(Symbol("Y"), Symbol("Z"), rule="r2")
        out = chain_to_latex(ProofChain([s1, s2]))
        # ``gather*`` (no shared alignment column) so chain rows with
        # very wide intermediate expressions don't shove every other
        # row off the right margin.
        assert r"\begin{gather*}" in out
        assert r"\end{gather*}" in out
        # \allowdisplaybreaks + \scriptsize wrapper lets long chains
        # page-break and fit on the page width.
        assert r"\allowdisplaybreaks" in out
        assert r"\scriptsize" in out
        # Two rendered lines separated by "\\".
        assert r" \\" in out

    def test_step_type_error(self):
        with pytest.raises(TypeError):
            step_to_latex("not a step")  # type: ignore[arg-type]

    def test_chain_type_error(self):
        with pytest.raises(TypeError):
            chain_to_latex("not a chain")  # type: ignore[arg-type]


# --------------------------------------------------------------------- #
# Sign normalisation integration                                        #
# --------------------------------------------------------------------- #


class TestSignNormalisation:
    def test_product_of_sum_with_neg_child(self):
        inner = Sum(Symbol("X"), Neg(Symbol("Y")))
        out = to_latex(Product(inner, Symbol("Z")))
        assert out == r"\left(X - Y\right) \, Z"

    def test_derivation_act_on_signed_sum(self):
        d = Derivation("d", 1)
        expr = Act(d, Sum(Symbol("X"), Neg(Symbol("Y"))))
        out = to_latex(expr)
        assert out == r"d\!\left(X - Y\right)"


# --------------------------------------------------------------------- #
# Standalone document export                                             #
# --------------------------------------------------------------------- #


class TestChainToLatexDocument:
    @pytest.fixture
    def chain(self):
        s1 = ProofStep(Symbol("X"), Symbol("Y"), rule="r1")
        s2 = ProofStep(Symbol("Y"), Symbol("Z"), rule="r2")
        return ProofChain([s1, s2])

    def test_wraps_align_in_documentclass(self, chain):
        out = chain_to_latex_document(chain)
        assert out.startswith(r"\documentclass{article}")
        assert r"\usepackage{amsmath}" in out
        assert r"\usepackage{amssymb}" in out
        assert r"\begin{document}" in out
        assert r"\end{document}" in out
        # Body: the gather* block from chain_to_latex should be present.
        assert r"\begin{gather*}" in out
        assert r"\end{gather*}" in out

    def test_no_maketitle_without_title_or_author(self, chain):
        out = chain_to_latex_document(chain)
        assert r"\maketitle" not in out

    def test_title_and_author_trigger_maketitle(self, chain):
        out = chain_to_latex_document(
            chain, title="Jacobi for Poisson", author="Test"
        )
        assert r"\title{Jacobi for Poisson}" in out
        assert r"\author{Test}" in out
        assert r"\maketitle" in out

    def test_title_alone_triggers_maketitle(self, chain):
        out = chain_to_latex_document(chain, title="X")
        assert r"\title{X}" in out
        assert r"\maketitle" in out

    def test_preamble_extras_inserted(self, chain):
        extras = r"\newcommand{\foo}{bar}"
        out = chain_to_latex_document(chain, preamble_extras=extras)
        # Extras must appear after default packages, before \begin{document}.
        extras_pos = out.find(extras)
        doc_pos = out.find(r"\begin{document}")
        assert extras_pos > 0
        assert extras_pos < doc_pos

    def test_title_escapes_special_chars(self, chain):
        out = chain_to_latex_document(chain, title="X_1 & Y")
        assert r"X\_1 \& Y" in out

    def test_trailing_newline(self, chain):
        out = chain_to_latex_document(chain)
        assert out.endswith("\n")

    def test_type_error_on_non_chain(self):
        with pytest.raises(TypeError):
            chain_to_latex_document("not a chain")  # type: ignore[arg-type]

    def test_empty_chain_still_renders_document(self):
        out = chain_to_latex_document(ProofChain())
        assert r"\documentclass" in out
        assert "empty proof chain" in out


# --------------------------------------------------------------------- #
# TikZ export                                                            #
# --------------------------------------------------------------------- #


class TestChainToTikz:
    @pytest.fixture
    def chain(self):
        s1 = ProofStep(Symbol("X"), Symbol("Y"), rule="r1")
        s2 = ProofStep(Symbol("Y"), Symbol("Z"), rule="r2")
        return ProofChain([s1, s2])

    def test_emits_tikzpicture(self, chain):
        out = chain_to_tikz(chain)
        assert out.startswith(r"\begin{tikzpicture}")
        assert out.endswith(r"\end{tikzpicture}")

    def test_n_plus_one_nodes_for_n_steps(self, chain):
        out = chain_to_tikz(chain)
        # 2 steps → 3 nodes (e0, e1, e2).
        for i in range(3):
            assert f"(e{i})" in out
        assert "(e3)" not in out

    def test_arrows_carry_rule_labels(self, chain):
        out = chain_to_tikz(chain)
        assert r"\draw[->] (e0)" in out
        assert r"\draw[->] (e1)" in out
        assert "r1" in out
        assert "r2" in out

    def test_provenance_tag_in_label(self):
        step = ProofStep(
            Symbol("X"), Symbol("Y"), rule="rule", provenance_tag="axiom"
        )
        out = chain_to_tikz(ProofChain([step]))
        assert "(axiom)" in out

    def test_custom_node_distance(self, chain):
        out = chain_to_tikz(chain, node_distance="2.5cm")
        assert "node distance=2.5cm" in out

    def test_empty_chain_placeholder(self):
        out = chain_to_tikz(ProofChain())
        assert r"\begin{tikzpicture}" in out
        assert "empty proof chain" in out

    def test_type_error_on_non_chain(self):
        with pytest.raises(TypeError):
            chain_to_tikz("not a chain")  # type: ignore[arg-type]


class TestChainToTikzDocument:
    @pytest.fixture
    def chain(self):
        s1 = ProofStep(Symbol("X"), Symbol("Y"), rule="r1")
        return ProofChain([s1])

    def test_loads_tikz_and_positioning(self, chain):
        out = chain_to_tikz_document(chain)
        assert r"\usepackage{tikz}" in out
        assert r"\usetikzlibrary{positioning}" in out

    def test_body_wraps_tikzpicture(self, chain):
        out = chain_to_tikz_document(chain)
        assert r"\begin{tikzpicture}" in out
        assert r"\end{tikzpicture}" in out
        assert r"\begin{center}" in out

    def test_title_triggers_maketitle(self, chain):
        out = chain_to_tikz_document(chain, title="Demo")
        assert r"\title{Demo}" in out
        assert r"\maketitle" in out

    def test_type_error_on_non_chain(self):
        with pytest.raises(TypeError):
            chain_to_tikz_document("not a chain")  # type: ignore[arg-type]
