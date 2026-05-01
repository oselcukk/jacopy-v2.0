"""Tests for ``jacopy.display.terminal``."""

from __future__ import annotations

import pytest

from jacopy.core.expr import Symbol
from jacopy.display import terminal as term_mod
from jacopy.display.ascii import chain_to_ascii, step_to_ascii, to_ascii
from jacopy.display.terminal import (
    HAS_RICH,
    print_chain,
    print_expr,
    print_step,
    render_chain,
    render_expr,
    render_step,
)
from jacopy.proof.chain import ProofChain
from jacopy.proof.step import ProofStep


# --------------------------------------------------------------------- #
# Module-level flag                                                     #
# --------------------------------------------------------------------- #


class TestHasRichFlag:
    def test_is_bool(self):
        assert isinstance(HAS_RICH, bool)


# --------------------------------------------------------------------- #
# Input validation                                                      #
# --------------------------------------------------------------------- #


class TestTypeErrors:
    def test_render_expr_rejects_non_expr(self):
        with pytest.raises(TypeError):
            render_expr("X")  # type: ignore[arg-type]

    def test_render_step_rejects_non_step(self):
        with pytest.raises(TypeError):
            render_step("not a step")  # type: ignore[arg-type]

    def test_render_chain_rejects_non_chain(self):
        with pytest.raises(TypeError):
            render_chain("not a chain")  # type: ignore[arg-type]

    def test_print_expr_rejects_non_expr(self):
        with pytest.raises(TypeError):
            print_expr("X")  # type: ignore[arg-type]

    def test_print_step_rejects_non_step(self):
        with pytest.raises(TypeError):
            print_step("not a step")  # type: ignore[arg-type]

    def test_print_chain_rejects_non_chain(self):
        with pytest.raises(TypeError):
            print_chain("not a chain")  # type: ignore[arg-type]


# --------------------------------------------------------------------- #
# render_expr                                                           #
# --------------------------------------------------------------------- #


class TestRenderExpr:
    def test_symbol(self):
        # render_expr is symmetric, always returns the ASCII form.
        assert render_expr(Symbol("X")) == to_ascii(Symbol("X"))

    def test_greek_passthrough(self):
        # ASCII renderer leaves Unicode glyphs alone.
        assert render_expr(Symbol("ω")) == "ω"


# --------------------------------------------------------------------- #
# Fallback path (HAS_RICH == False)                                     #
#                                                                       #
# These tests force the fallback branch via monkeypatch so they pass    #
# regardless of whether rich is installed in the test environment.     #
# --------------------------------------------------------------------- #


class TestFallbackRender:
    def test_render_step_falls_back_to_ascii(self, monkeypatch):
        monkeypatch.setattr(term_mod, "HAS_RICH", False)
        step = ProofStep(Symbol("X"), Symbol("Y"), rule="demo")
        assert render_step(step) == step_to_ascii(step)

    def test_render_step_respects_max_depth_in_fallback(self, monkeypatch):
        monkeypatch.setattr(term_mod, "HAS_RICH", False)
        child = ProofStep(Symbol("A"), Symbol("B"), rule="sub")
        parent = ProofStep(
            Symbol("X"), Symbol("Y"), rule="outer", children=[child]
        )
        assert render_step(parent, max_depth=0) == step_to_ascii(
            parent, max_depth=0
        )

    def test_render_chain_falls_back_to_ascii(self, monkeypatch):
        monkeypatch.setattr(term_mod, "HAS_RICH", False)
        s1 = ProofStep(Symbol("X"), Symbol("Y"), rule="r1")
        s2 = ProofStep(Symbol("Y"), Symbol("Z"), rule="r2")
        chain = ProofChain([s1, s2])
        assert render_chain(chain) == chain_to_ascii(chain)

    def test_render_chain_empty_in_fallback(self, monkeypatch):
        monkeypatch.setattr(term_mod, "HAS_RICH", False)
        assert render_chain(ProofChain()) == "(empty proof chain)"

    def test_print_expr_fallback_writes_to_stdout(self, monkeypatch, capsys):
        monkeypatch.setattr(term_mod, "HAS_RICH", False)
        print_expr(Symbol("X"))
        captured = capsys.readouterr()
        assert captured.out.strip() == "X"

    def test_print_step_fallback_writes_to_stdout(self, monkeypatch, capsys):
        monkeypatch.setattr(term_mod, "HAS_RICH", False)
        step = ProofStep(Symbol("X"), Symbol("Y"), rule="demo")
        print_step(step)
        captured = capsys.readouterr()
        assert "[demo]" in captured.out
        assert "X -> Y" in captured.out

    def test_print_chain_fallback_writes_to_stdout(self, monkeypatch, capsys):
        monkeypatch.setattr(term_mod, "HAS_RICH", False)
        s1 = ProofStep(Symbol("X"), Symbol("Y"), rule="r1")
        print_chain(ProofChain([s1]))
        captured = capsys.readouterr()
        assert "[r1]" in captured.out

    def test_print_chain_fallback_empty(self, monkeypatch, capsys):
        monkeypatch.setattr(term_mod, "HAS_RICH", False)
        print_chain(ProofChain())
        captured = capsys.readouterr()
        assert "(empty proof chain)" in captured.out


# --------------------------------------------------------------------- #
# Rich path (skipped when rich is absent)                               #
# --------------------------------------------------------------------- #


_requires_rich = pytest.mark.skipif(
    not HAS_RICH, reason="rich not installed"
)


@_requires_rich
class TestRichPath:
    def test_render_step_contains_rule_and_expressions(self):
        step = ProofStep(Symbol("X"), Symbol("Y"), rule="demo")
        out = render_step(step)
        assert "[demo]" in out
        assert "X" in out
        assert "Y" in out

    def test_render_chain_has_title(self):
        s1 = ProofStep(Symbol("X"), Symbol("Y"), rule="r1")
        s2 = ProofStep(Symbol("Y"), Symbol("Z"), rule="r2")
        out = render_chain(ProofChain([s1, s2]))
        assert "Proof" in out
        assert "2 steps" in out
        assert "[r1]" in out
        assert "[r2]" in out

    def test_render_chain_title_false_suppresses_header(self):
        s1 = ProofStep(Symbol("X"), Symbol("Y"), rule="r1")
        out = render_chain(ProofChain([s1]), title=False)
        assert "Proof (" not in out

    def test_render_chain_empty(self):
        assert render_chain(ProofChain()) == "(empty proof chain)"

    def test_render_step_nests_children(self):
        child = ProofStep(Symbol("A"), Symbol("B"), rule="sub")
        parent = ProofStep(
            Symbol("X"), Symbol("Y"), rule="outer", children=[child]
        )
        out = render_step(parent)
        assert "[outer]" in out
        assert "[sub]" in out

    def test_render_step_respects_max_depth(self):
        child = ProofStep(Symbol("A"), Symbol("B"), rule="sub")
        parent = ProofStep(
            Symbol("X"), Symbol("Y"), rule="outer", children=[child]
        )
        out = render_step(parent, max_depth=0)
        assert "[outer]" in out
        assert "[sub]" not in out

    def test_render_chain_does_not_leak_to_stdout(self, capsys):
        """Regression: render_chain must be pure, no stray stdout write.

        Earlier the recording Console had no explicit ``file`` arg and
        rich writes to stdout by default even when recording, so
        callers who printed the returned string saw the tree twice.
        """
        s1 = ProofStep(Symbol("X"), Symbol("Y"), rule="r1")
        _ = render_chain(ProofChain([s1]))
        assert capsys.readouterr().out == ""

    def test_render_step_does_not_leak_to_stdout(self, capsys):
        step = ProofStep(Symbol("X"), Symbol("Y"), rule="demo")
        _ = render_step(step)
        assert capsys.readouterr().out == ""


# --------------------------------------------------------------------- #
# Verbosity, applies to both the rich and fallback paths               #
# --------------------------------------------------------------------- #


class TestVerbosityInputValidation:
    def test_render_step_rejects_unknown_mode(self):
        step = ProofStep(Symbol("X"), Symbol("Y"), rule="demo")
        with pytest.raises(ValueError, match="verbosity must be one of"):
            render_step(step, verbosity="loud")

    def test_render_chain_rejects_unknown_mode(self):
        chain = ProofChain([ProofStep(Symbol("X"), Symbol("Y"), rule="r")])
        with pytest.raises(ValueError, match="verbosity must be one of"):
            render_chain(chain, verbosity="loud")

    def test_print_step_rejects_unknown_mode(self):
        step = ProofStep(Symbol("X"), Symbol("Y"), rule="demo")
        with pytest.raises(ValueError, match="verbosity must be one of"):
            print_step(step, verbosity="loud")

    def test_print_chain_rejects_unknown_mode(self):
        chain = ProofChain([ProofStep(Symbol("X"), Symbol("Y"), rule="r")])
        with pytest.raises(ValueError, match="verbosity must be one of"):
            print_chain(chain, verbosity="loud")


class TestVerbosityFallback:
    """Fallback path delegates to ascii.step_to_ascii / chain_to_ascii."""

    def _demo_step(self):
        return ProofStep(
            Symbol("X"),
            Symbol("Y"),
            rule="demo",
            justification="because-j",
            provenance_tag="axiom",
        )

    def _demo_chain_with_children(self):
        child = ProofStep(Symbol("A"), Symbol("B"), rule="sub")
        return ProofChain(
            [
                ProofStep(
                    Symbol("X"),
                    Symbol("Y"),
                    rule="outer",
                    justification="outer-j",
                    children=[child],
                )
            ]
        )

    def test_summary_drops_justification_in_fallback(self, monkeypatch):
        monkeypatch.setattr(term_mod, "HAS_RICH", False)
        out = render_step(self._demo_step(), verbosity="summary")
        assert "X -> Y" in out
        assert "because-j" not in out

    def test_compact_drops_arrow_in_fallback(self, monkeypatch):
        monkeypatch.setattr(term_mod, "HAS_RICH", False)
        out = render_step(self._demo_step(), verbosity="compact")
        assert "X -> Y" not in out
        assert out.strip() == "[demo] (axiom)"

    def test_compact_suppresses_children_in_fallback(self, monkeypatch):
        monkeypatch.setattr(term_mod, "HAS_RICH", False)
        out = render_chain(
            self._demo_chain_with_children(), verbosity="compact"
        )
        assert "[outer]" in out
        assert "[sub]" not in out


@_requires_rich
class TestVerbosityRichPath:
    def _demo_step(self):
        return ProofStep(
            Symbol("X"),
            Symbol("Y"),
            rule="demo",
            justification="because-j",
        )

    def _demo_chain_with_children(self):
        child = ProofStep(Symbol("A"), Symbol("B"), rule="sub")
        return ProofChain(
            [
                ProofStep(
                    Symbol("X"),
                    Symbol("Y"),
                    rule="outer",
                    justification="outer-j",
                    children=[child],
                )
            ]
        )

    def test_summary_omits_justification(self):
        out = render_step(self._demo_step(), verbosity="summary")
        assert "[demo]" in out
        assert "X" in out and "Y" in out
        assert "because-j" not in out

    def test_compact_shows_only_rule_and_tag(self):
        out = render_step(self._demo_step(), verbosity="compact")
        assert "[demo]" in out
        # No arrow and no before/after glyph.
        assert "→" not in out
        assert "because-j" not in out

    def test_compact_suppresses_children_in_tree(self):
        out = render_chain(
            self._demo_chain_with_children(), verbosity="compact"
        )
        assert "[outer]" in out
        assert "[sub]" not in out
