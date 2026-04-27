"""Tests for jacopy.proof.chain."""

import pytest

from jacopy.core.expr import Symbol
from jacopy.proof.chain import ProofChain
from jacopy.proof.step import ProofStep


def _step(before_name, after_name, rule="r"):
    return ProofStep(Symbol(before_name), Symbol(after_name), rule=rule)


class TestConstruction:
    def test_empty_chain(self):
        chain = ProofChain()
        assert len(chain) == 0
        assert not chain

    def test_seed_steps(self):
        s = _step("a", "b")
        chain = ProofChain([s])
        assert len(chain) == 1
        assert chain.steps == (s,)

    def test_rejects_non_step_in_seed(self):
        with pytest.raises(TypeError):
            ProofChain(["not a step"])  # type: ignore[list-item]


class TestMutation:
    def test_append(self):
        chain = ProofChain()
        s = _step("a", "b")
        chain.append(s)
        assert len(chain) == 1
        assert chain.steps[0] is s

    def test_append_rejects_non_step(self):
        chain = ProofChain()
        with pytest.raises(TypeError):
            chain.append("not a step")  # type: ignore[arg-type]

    def test_extend(self):
        chain = ProofChain()
        s1, s2 = _step("a", "b"), _step("b", "c")
        chain.extend([s1, s2])
        assert chain.steps == (s1, s2)


class TestBoundaries:
    def test_initial_and_final(self):
        chain = ProofChain([_step("a", "b"), _step("b", "c")])
        assert chain.initial == Symbol("a")
        assert chain.final == Symbol("c")

    def test_initial_raises_on_empty(self):
        chain = ProofChain()
        with pytest.raises(ValueError):
            _ = chain.initial

    def test_final_raises_on_empty(self):
        chain = ProofChain()
        with pytest.raises(ValueError):
            _ = chain.final


class TestIteration:
    def test_iter(self):
        s1, s2 = _step("a", "b"), _step("b", "c")
        chain = ProofChain([s1, s2])
        assert list(chain) == [s1, s2]


class TestFormat:
    def test_compact(self):
        chain = ProofChain([_step("a", "b", "r1"), _step("b", "c", "r2")])
        text = chain.format(verbosity="compact")
        lines = text.splitlines()
        assert len(lines) == 2
        assert "[r1]" in lines[0]
        assert "[r2]" in lines[1]

    def test_full(self):
        chain = ProofChain([_step("a", "b", "r1")])
        text = chain.format(verbosity="full")
        assert "[r1]" in text

    def test_unknown_verbosity_raises(self):
        chain = ProofChain()
        with pytest.raises(ValueError):
            chain.format(verbosity="fancy")

    def test_compact_hides_children(self):
        inner = _step("x", "y", "inner-rule")
        outer = _step("a", "b", "outer-rule")
        outer.add_child(inner)
        chain = ProofChain([outer])
        assert "inner-rule" not in chain.format(verbosity="compact")
        assert "inner-rule" in chain.format(verbosity="full")
