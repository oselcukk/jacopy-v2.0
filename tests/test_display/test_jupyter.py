"""Tests for ``gradalg.display.jupyter``."""

from __future__ import annotations

import pytest

from gradalg.algebra.derivation import Derivation
from gradalg.core.expr import Symbol
from gradalg.display.jupyter import (
    LatexDisplay,
    display_chain,
    display_expr,
    display_proof,
    display_step,
)
from gradalg.proof.chain import ProofChain
from gradalg.proof.step import ProofStep


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
        assert "gradalg-latex" in html
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
    def test_wraps_in_align_environment(self):
        step = ProofStep(Symbol("X"), Symbol("Y"), rule="demo")
        out = display_step(step)
        assert isinstance(out, LatexDisplay)
        assert out.environment is True
        assert out.latex.startswith(r"\begin{align*}")
        assert out.latex.endswith(r"\end{align*}")
        assert r"X &\to Y" in out.latex

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
        assert out.latex.startswith(r"\begin{align*}")
        assert r"X &\to Y" in out.latex
        assert r"Y &\to Z" in out.latex

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
        assert r"\begin{align*}" in bundle["text/latex"]
        assert r"\begin{align*}" in bundle["text/html"]
        assert r"\begin{align*}" in bundle["text/plain"]
