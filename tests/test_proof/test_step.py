"""Tests for jacopy.proof.step."""

import pytest

from jacopy.core.expr import Integer, Symbol
from jacopy.proof.step import ProofStep


class TestConstruction:
    def test_basic_fields(self):
        a, b = Symbol("a"), Symbol("b")
        step = ProofStep(a, b, rule="rename", justification="a is b")
        assert step.before is a
        assert step.after is b
        assert step.rule == "rename"
        assert step.justification == "a is b"
        assert step.children == ()

    def test_justification_defaults_to_empty(self):
        a = Symbol("a")
        step = ProofStep(a, a, rule="identity")
        assert step.justification == ""

    def test_rejects_non_expr_before(self):
        with pytest.raises(TypeError):
            ProofStep("a", Symbol("b"), rule="r")  # type: ignore[arg-type]

    def test_rejects_non_expr_after(self):
        with pytest.raises(TypeError):
            ProofStep(Symbol("a"), "b", rule="r")  # type: ignore[arg-type]

    def test_rejects_non_str_rule(self):
        a = Symbol("a")
        with pytest.raises(TypeError):
            ProofStep(a, a, rule=42)  # type: ignore[arg-type]

    def test_children_seed_accepted(self):
        a, b, c = Symbol("a"), Symbol("b"), Symbol("c")
        inner = ProofStep(a, b, rule="inner")
        outer = ProofStep(a, c, rule="outer", children=[inner])
        assert outer.children == (inner,)


class TestChildren:
    def test_add_child(self):
        a, b, c = Symbol("a"), Symbol("b"), Symbol("c")
        outer = ProofStep(a, c, rule="outer")
        inner = ProofStep(a, b, rule="inner")
        outer.add_child(inner)
        assert outer.children == (inner,)

    def test_add_child_rejects_non_step(self):
        a = Symbol("a")
        step = ProofStep(a, a, rule="r")
        with pytest.raises(TypeError):
            step.add_child("not a step")  # type: ignore[arg-type]


class TestFormat:
    def test_single_step_renders(self):
        a, b = Symbol("a"), Symbol("b")
        step = ProofStep(a, b, rule="r", justification="why")
        text = step.format()
        assert "[r]" in text
        assert "a" in text
        assert "b" in text
        assert "why" in text

    def test_children_indented(self):
        a, b, c = Symbol("a"), Symbol("b"), Symbol("c")
        inner = ProofStep(b, c, rule="inner-rule")
        outer = ProofStep(a, c, rule="outer-rule")
        outer.add_child(inner)
        text = outer.format()
        lines = text.splitlines()
        assert len(lines) == 2
        assert lines[0].startswith("[outer-rule]")
        assert lines[1].startswith("  [inner-rule]")

    def test_max_depth_prunes(self):
        a, b = Symbol("a"), Symbol("b")
        inner = ProofStep(a, b, rule="inner")
        outer = ProofStep(a, b, rule="outer")
        outer.add_child(inner)
        text = outer.format(max_depth=0)
        assert "inner" not in text
        assert "outer" in text

    def test_integer_zero_renders(self):
        a = Symbol("a")
        step = ProofStep(a, Integer(0), rule="cancel")
        text = step.format()
        assert "0" in text


class TestProvenanceTag:
    def test_default_is_none(self):
        a = Symbol("a")
        step = ProofStep(a, a, rule="r")
        assert step.provenance_tag is None

    def test_axiom_tag_accepted(self):
        a = Symbol("a")
        step = ProofStep(a, a, rule="r", provenance_tag="axiom")
        assert step.provenance_tag == "axiom"

    def test_theorem_tag_accepted(self):
        a = Symbol("a")
        step = ProofStep(a, a, rule="r", provenance_tag="theorem")
        assert step.provenance_tag == "theorem"

    def test_rejects_unknown_tag(self):
        a = Symbol("a")
        with pytest.raises(ValueError, match="provenance_tag"):
            ProofStep(a, a, rule="r", provenance_tag="lemma")

    def test_tag_rendered_in_format(self):
        a = Symbol("a")
        step = ProofStep(a, a, rule="r", provenance_tag="axiom")
        assert "(axiom)" in step.format()

    def test_no_tag_means_no_tag_text(self):
        a = Symbol("a")
        step = ProofStep(a, a, rule="r")
        assert "(axiom)" not in step.format()
        assert "(theorem)" not in step.format()
