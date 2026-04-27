"""Tests for ``jacopy.display.jupyter``."""

from __future__ import annotations

import pytest

from jacopy.algebra.derivation import Derivation
from jacopy.core.expr import Symbol
from jacopy.display.jupyter import (
    HtmlProofDisplay,
    LatexDisplay,
    display_chain,
    display_chain_collapsible,
    display_expr,
    display_proof,
    display_step,
    display_step_collapsible,
)
from jacopy.proof.chain import ProofChain
from jacopy.proof.step import ProofStep


# --------------------------------------------------------------------- #
# LatexDisplay wrapper                                                  #
# --------------------------------------------------------------------- #


class TestLatexDisplay:
    def test_stores_latex(self):
        ld = LatexDisplay("X + Y")
        assert ld.latex == "X + Y"
        assert ld.environment is False

    def test_inline_wraps_in_dollar(self):
        ld = LatexDisplay("X + Y")
        assert ld._repr_latex_() == "$X + Y$"

    def test_environment_passthrough(self):
        body = r"\begin{align*}X &\to Y\end{align*}"
        ld = LatexDisplay(body, environment=True)
        assert ld._repr_latex_() == body

    def test_html_uses_mathjax_delimiters(self):
        ld = LatexDisplay("X + Y")
        html = ld._repr_html_()
        assert "jacopy-latex" in html
        assert "\\(X + Y\\)" in html

    def test_html_environment_passes_raw(self):
        body = r"\begin{align*}X &\to Y\end{align*}"
        ld = LatexDisplay(body, environment=True)
        html = ld._repr_html_()
        assert r"\begin{align*}" in html
        # Environment form is NOT wrapped in \( \).
        assert "\\(" not in html

    def test_mimebundle_has_three_mimetypes(self):
        ld = LatexDisplay("X")
        bundle = ld._repr_mimebundle_()
        assert set(bundle.keys()) == {"text/latex", "text/html", "text/plain"}
        assert bundle["text/latex"] == "$X$"
        assert bundle["text/plain"] == "X"

    def test_mimebundle_include_filter(self):
        ld = LatexDisplay("X")
        bundle = ld._repr_mimebundle_(include={"text/latex"})
        assert list(bundle.keys()) == ["text/latex"]

    def test_mimebundle_exclude_filter(self):
        ld = LatexDisplay("X")
        bundle = ld._repr_mimebundle_(exclude={"text/html"})
        assert "text/html" not in bundle
        assert "text/latex" in bundle

    def test_str_returns_raw_latex(self):
        ld = LatexDisplay("\\alpha")
        assert str(ld) == "\\alpha"

    def test_repr_returns_raw_latex(self):
        ld = LatexDisplay("\\alpha")
        assert repr(ld) == "\\alpha"

    def test_equality(self):
        assert LatexDisplay("X") == LatexDisplay("X")
        assert LatexDisplay("X") != LatexDisplay("Y")
        assert LatexDisplay("X") != LatexDisplay("X", environment=True)

    def test_hashable(self):
        assert hash(LatexDisplay("X")) == hash(LatexDisplay("X"))
        # Usable in a set.
        assert len({LatexDisplay("X"), LatexDisplay("X"), LatexDisplay("Y")}) == 2

    def test_non_str_payload_raises(self):
        with pytest.raises(TypeError):
            LatexDisplay(42)  # type: ignore[arg-type]


# --------------------------------------------------------------------- #
# display_expr                                                          #
# --------------------------------------------------------------------- #


class TestDisplayExpr:
    def test_wraps_in_inline_math(self):
        out = display_expr(Symbol("X"))
        assert isinstance(out, LatexDisplay)
        assert out._repr_latex_() == "$X$"

    def test_sanitises_greek(self):
        out = display_expr(Symbol("ω"))
        assert "\\omega" in out.latex
        assert out._repr_latex_() == "$\\omega$"

    def test_rejects_non_expr(self):
        with pytest.raises(TypeError):
            display_expr("X")  # type: ignore[arg-type]


# --------------------------------------------------------------------- #
# display_step                                                          #
# --------------------------------------------------------------------- #


class TestDisplayStep:
    def test_wraps_in_gather_environment(self):
        step = ProofStep(Symbol("X"), Symbol("Y"), rule="demo")
        out = display_step(step)
        assert isinstance(out, LatexDisplay)
        assert out.environment is True
        assert out.latex.startswith(r"\begin{gather*}")
        assert out.latex.endswith(r"\end{gather*}")
        assert r"X \to Y" in out.latex

    def test_repr_latex_is_raw_environment(self):
        step = ProofStep(Symbol("X"), Symbol("Y"), rule="demo")
        out = display_step(step)
        # environment=True => no $ wrapping.
        assert out._repr_latex_() == out.latex
        assert not out._repr_latex_().startswith("$")

    def test_rejects_non_step(self):
        with pytest.raises(TypeError):
            display_step("not a step")  # type: ignore[arg-type]


# --------------------------------------------------------------------- #
# display_chain / display_proof                                         #
# --------------------------------------------------------------------- #


class TestDisplayChain:
    def test_empty_chain(self):
        out = display_chain(ProofChain())
        assert isinstance(out, LatexDisplay)
        assert out.environment is True
        assert "empty proof chain" in out.latex

    def test_multiple_steps(self):
        s1 = ProofStep(Symbol("X"), Symbol("Y"), rule="r1")
        s2 = ProofStep(Symbol("Y"), Symbol("Z"), rule="r2")
        out = display_chain(ProofChain([s1, s2]))
        assert r"\begin{gather*}" in out.latex
        assert r"X \to Y" in out.latex
        assert r"Y \to Z" in out.latex

    def test_display_proof_is_alias(self):
        assert display_proof is display_chain

    def test_display_proof_returns_displayable(self):
        chain = ProofChain(
            [ProofStep(Symbol("X"), Symbol("Y"), rule="r")]
        )
        out = display_proof(chain)
        assert isinstance(out, LatexDisplay)
        assert out.environment is True

    def test_rejects_non_chain(self):
        with pytest.raises(TypeError):
            display_chain("not a chain")  # type: ignore[arg-type]


# --------------------------------------------------------------------- #
# Integration: Greek + proof rendering                                  #
# --------------------------------------------------------------------- #


class TestIntegration:
    def test_derivation_with_greek_in_chain(self):
        iota = Derivation("ι_X", -1)
        step = ProofStep(iota, iota, rule="identity")
        out = display_step(step)
        assert r"\iota_X" in out.latex

    def test_chain_round_trips_through_mimebundle(self):
        s = ProofStep(Symbol("X"), Symbol("Y"), rule="r")
        out = display_chain(ProofChain([s]))
        bundle = out._repr_mimebundle_()
        assert r"\begin{gather*}" in bundle["text/latex"]
        assert r"\begin{gather*}" in bundle["text/html"]
        assert r"\begin{gather*}" in bundle["text/plain"]


# --------------------------------------------------------------------- #
# HtmlProofDisplay wrapper                                              #
# --------------------------------------------------------------------- #


class TestHtmlProofDisplay:
    def test_stores_html(self):
        d = HtmlProofDisplay("<div>X</div>")
        assert d.html == "<div>X</div>"

    def test_repr_html_returns_raw(self):
        d = HtmlProofDisplay("<div>X</div>")
        assert d._repr_html_() == "<div>X</div>"

    def test_mimebundle_has_html_and_plain(self):
        d = HtmlProofDisplay("<div>X</div>")
        bundle = d._repr_mimebundle_()
        assert set(bundle) == {"text/html", "text/plain"}
        assert bundle["text/html"] == "<div>X</div>"

    def test_str_and_repr_are_raw(self):
        d = HtmlProofDisplay("<p>Y</p>")
        assert str(d) == "<p>Y</p>"
        assert repr(d) == "<p>Y</p>"

    def test_equality_and_hash(self):
        a = HtmlProofDisplay("<p>X</p>")
        b = HtmlProofDisplay("<p>X</p>")
        c = HtmlProofDisplay("<p>Y</p>")
        assert a == b
        assert a != c
        assert hash(a) == hash(b)

    def test_non_str_payload_raises(self):
        with pytest.raises(TypeError):
            HtmlProofDisplay(42)  # type: ignore[arg-type]


# --------------------------------------------------------------------- #
# display_step_collapsible                                              #
# --------------------------------------------------------------------- #


class TestDisplayStepCollapsible:
    def test_leaf_wraps_in_div_not_details(self):
        step = ProofStep(Symbol("X"), Symbol("Y"), rule="demo")
        out = display_step_collapsible(step)
        assert isinstance(out, HtmlProofDisplay)
        html = out.html
        assert '<div class="jacopy-step">' in html
        assert "<details" not in html
        assert "[demo]" in html

    def test_with_children_uses_details_open(self):
        child = ProofStep(Symbol("A"), Symbol("B"), rule="sub")
        parent = ProofStep(
            Symbol("X"), Symbol("Y"), rule="outer", children=[child]
        )
        html = display_step_collapsible(parent).html
        assert '<details open class="jacopy-step">' in html
        assert "<summary>" in html
        assert "[outer]" in html
        assert "[sub]" in html

    def test_math_uses_mathjax_delimiters(self):
        step = ProofStep(Symbol("X"), Symbol("Y"), rule="demo")
        html = display_step_collapsible(step).html
        # `\to` inside `\(...\)` so MathJax picks it up.
        assert r"\(X \to Y\)" in html

    def test_rule_is_html_escaped(self):
        step = ProofStep(Symbol("X"), Symbol("Y"), rule="a<b&c")
        html = display_step_collapsible(step).html
        assert "a&lt;b&amp;c" in html
        assert "a<b&c" not in html

    def test_compact_suppresses_children_and_math(self):
        child = ProofStep(Symbol("A"), Symbol("B"), rule="sub")
        parent = ProofStep(
            Symbol("X"), Symbol("Y"), rule="outer", children=[child]
        )
        html = display_step_collapsible(parent, verbosity="compact").html
        assert "[outer]" in html
        assert "[sub]" not in html
        assert "\\to" not in html

    def test_summary_drops_justification(self):
        step = ProofStep(
            Symbol("X"),
            Symbol("Y"),
            rule="demo",
            justification="because-j",
        )
        html = display_step_collapsible(step, verbosity="summary").html
        assert r"\to" in html
        assert "because-j" not in html

    def test_max_depth_zero_suppresses_children(self):
        child = ProofStep(Symbol("A"), Symbol("B"), rule="sub")
        parent = ProofStep(
            Symbol("X"), Symbol("Y"), rule="outer", children=[child]
        )
        html = display_step_collapsible(parent, max_depth=0).html
        assert "[outer]" in html
        assert "[sub]" not in html
        # With no visible children, falls back to the leaf div form.
        assert "<details" not in html

    def test_rejects_non_step(self):
        with pytest.raises(TypeError):
            display_step_collapsible("not a step")  # type: ignore[arg-type]

    def test_rejects_unknown_verbosity(self):
        step = ProofStep(Symbol("X"), Symbol("Y"), rule="demo")
        with pytest.raises(ValueError, match="verbosity must be one of"):
            display_step_collapsible(step, verbosity="loud")


# --------------------------------------------------------------------- #
# display_chain_collapsible                                             #
# --------------------------------------------------------------------- #


class TestDisplayChainCollapsible:
    def test_empty_chain(self):
        out = display_chain_collapsible(ProofChain())
        assert isinstance(out, HtmlProofDisplay)
        assert "empty proof chain" in out.html

    def test_header_includes_step_count(self):
        s1 = ProofStep(Symbol("X"), Symbol("Y"), rule="r1")
        s2 = ProofStep(Symbol("Y"), Symbol("Z"), rule="r2")
        html = display_chain_collapsible(ProofChain([s1, s2])).html
        assert 'class="jacopy-proof-header"' in html
        assert "Proof (2 steps)" in html
        assert "[r1]" in html
        assert "[r2]" in html

    def test_title_false_suppresses_header(self):
        s1 = ProofStep(Symbol("X"), Symbol("Y"), rule="r1")
        html = display_chain_collapsible(
            ProofChain([s1]), title=False
        ).html
        assert "jacopy-proof-header" not in html
        assert "[r1]" in html

    def test_nested_children_are_expandable(self):
        child = ProofStep(Symbol("A"), Symbol("B"), rule="sub")
        parent = ProofStep(
            Symbol("X"), Symbol("Y"), rule="outer", children=[child]
        )
        html = display_chain_collapsible(ProofChain([parent])).html
        assert '<details open class="jacopy-step">' in html
        assert '<div class="jacopy-children">' in html
        assert "[outer]" in html
        assert "[sub]" in html

    def test_compact_flattens_tree(self):
        child = ProofStep(Symbol("A"), Symbol("B"), rule="sub")
        parent = ProofStep(
            Symbol("X"), Symbol("Y"), rule="outer", children=[child]
        )
        html = display_chain_collapsible(
            ProofChain([parent]), verbosity="compact"
        ).html
        assert "[outer]" in html
        assert "[sub]" not in html
        assert "<details" not in html

    def test_rejects_non_chain(self):
        with pytest.raises(TypeError):
            display_chain_collapsible("not a chain")  # type: ignore[arg-type]

    def test_rejects_unknown_verbosity(self):
        chain = ProofChain([ProofStep(Symbol("X"), Symbol("Y"), rule="r")])
        with pytest.raises(ValueError, match="verbosity must be one of"):
            display_chain_collapsible(chain, verbosity="loud")

    def test_mimebundle_is_html_only(self):
        s = ProofStep(Symbol("X"), Symbol("Y"), rule="r")
        out = display_chain_collapsible(ProofChain([s]))
        bundle = out._repr_mimebundle_()
        assert "text/html" in bundle
        assert "text/latex" not in bundle
