"""Tests for Faz 13.D, SnBivectorFormulaDefinition."""

import pytest

from jacopy.algebra.derivation import Act
from jacopy.algebra.lie_bracket_vf import LieBracketVF
from jacopy.brackets.base import BracketApply
from jacopy.brackets.schouten import sn as default_sn
from jacopy.calculus.lie_derivative import lie_derivative
from jacopy.calculus.musical import sharp
from jacopy.calculus.sn_axiom import SnBivectorFormulaDefinition
from jacopy.core.expr import Integer, Neg, Sum, Symbol
from jacopy.proof.expansion import ExpansionEngine


def _vf_term(sh, a, b, c):
    """Build ``Act(L_{[π^♯a, π^♯b]_VF}, c)``."""
    return Act(
        lie_derivative(LieBracketVF(Act(sh, a), Act(sh, b))), c,
    )


# --------------------------------------------------------------------- #
# Construction / validation                                              #
# --------------------------------------------------------------------- #


class TestSnBivectorFormulaConstruction:
    def test_requires_sharp(self):
        with pytest.raises(TypeError):
            SnBivectorFormulaDefinition("not a sharp")  # type: ignore[arg-type]

    def test_default_sn_bracket(self):
        pi = Symbol("π")
        rule = SnBivectorFormulaDefinition(sharp(pi))
        assert rule._sn is default_sn

    def test_custom_sn_bracket(self):
        from jacopy.brackets.schouten import SchoutenBracket

        pi = Symbol("π")
        custom = SchoutenBracket(name="[·,·]_SN'")
        rule = SnBivectorFormulaDefinition(sharp(pi), sn_bracket=custom)
        assert rule._sn is custom

    def test_name(self):
        pi = Symbol("π")
        rule = SnBivectorFormulaDefinition(sharp(pi))
        assert rule.name == "[π,π]_SN bivector formula"


# --------------------------------------------------------------------- #
# matches()                                                              #
# --------------------------------------------------------------------- #


class TestSnBivectorFormulaMatches:
    def test_matches_canonical_cyclic_triple(self):
        pi = Symbol("π")
        a, b, c = Symbol("α"), Symbol("β"), Symbol("γ")
        sh = sharp(pi)
        rule = SnBivectorFormulaDefinition(sh)
        s = Sum(
            _vf_term(sh, a, b, c),
            _vf_term(sh, b, c, a),
            _vf_term(sh, c, a, b),
        )
        assert rule.matches(s)

    def test_matches_reversed_cycle(self):
        # The reverse cyclic class { (α,γ,β), (γ,β,α), (β,α,γ) } is
        # the post-commutator residue shape from the 2f-deep probe.
        pi = Symbol("π")
        a, b, c = Symbol("α"), Symbol("β"), Symbol("γ")
        sh = sharp(pi)
        rule = SnBivectorFormulaDefinition(sh)
        s = Sum(
            _vf_term(sh, a, c, b),
            _vf_term(sh, c, b, a),
            _vf_term(sh, b, a, c),
        )
        assert rule.matches(s)

    def test_no_match_on_two_term_subset(self):
        pi = Symbol("π")
        a, b, c = Symbol("α"), Symbol("β"), Symbol("γ")
        sh = sharp(pi)
        rule = SnBivectorFormulaDefinition(sh)
        s = Sum(
            _vf_term(sh, a, b, c),
            _vf_term(sh, b, c, a),
        )
        assert not rule.matches(s)

    def test_no_match_on_non_cyclic_triple(self):
        pi = Symbol("π")
        a, b, c = Symbol("α"), Symbol("β"), Symbol("γ")
        sh = sharp(pi)
        rule = SnBivectorFormulaDefinition(sh)
        # Three terms, but not a cyclic permutation of any anchor.
        s = Sum(
            _vf_term(sh, a, b, c),
            _vf_term(sh, a, b, c),
            _vf_term(sh, b, c, a),
        )
        assert not rule.matches(s)

    def test_no_match_with_different_sharp(self):
        pi1, pi2 = Symbol("π1"), Symbol("π2")
        a, b, c = Symbol("α"), Symbol("β"), Symbol("γ")
        sh1, sh2 = sharp(pi1), sharp(pi2)
        rule = SnBivectorFormulaDefinition(sh1)
        # Mixing sharps, the rule is set to π1, so a triple under π2
        # is not its formula.
        s = Sum(
            _vf_term(sh2, a, b, c),
            _vf_term(sh2, b, c, a),
            _vf_term(sh2, c, a, b),
        )
        assert not rule.matches(s)

    def test_no_match_when_inner_not_sharp_act(self):
        # LieBracketVF(X, Y) where X is bare Symbol, not Act(Sharp, _).
        pi = Symbol("π")
        a, b, c = Symbol("α"), Symbol("β"), Symbol("γ")
        sh = sharp(pi)
        rule = SnBivectorFormulaDefinition(sh)
        # Bare-symbol vector fields under the LieBracketVF.
        bad = Act(
            lie_derivative(LieBracketVF(a, Act(sh, b))), c,
        )
        # Build a triple where one term has the wrong inner shape.
        s = Sum(
            bad,
            _vf_term(sh, b, c, a),
            _vf_term(sh, c, a, b),
        )
        assert not rule.matches(s)

    def test_no_match_on_negated_term(self):
        # The cyclic SN formula is the *positive* triple, Neg-wrapped
        # terms decline. (Mirrors LieVfJacobiDefinition's policy.)
        pi = Symbol("π")
        a, b, c = Symbol("α"), Symbol("β"), Symbol("γ")
        sh = sharp(pi)
        rule = SnBivectorFormulaDefinition(sh)
        s = Sum(
            Neg(_vf_term(sh, a, b, c)),
            _vf_term(sh, b, c, a),
            _vf_term(sh, c, a, b),
        )
        assert not rule.matches(s)

    def test_no_match_atomic_term(self):
        pi = Symbol("π")
        rule = SnBivectorFormulaDefinition(sharp(pi))
        assert not rule.matches(Symbol("ω"))


# --------------------------------------------------------------------- #
# rewrite()                                                              #
# --------------------------------------------------------------------- #


class TestSnBivectorFormulaRewrite:
    def test_collapses_to_sn_bracket_apply_when_only_triple(self):
        pi = Symbol("π")
        a, b, c = Symbol("α"), Symbol("β"), Symbol("γ")
        sh = sharp(pi)
        rule = SnBivectorFormulaDefinition(sh)
        s = Sum(
            _vf_term(sh, a, b, c),
            _vf_term(sh, b, c, a),
            _vf_term(sh, c, a, b),
        )
        out = rule.rewrite(s)
        assert out == BracketApply(default_sn, pi, pi)

    def test_keeps_residue_terms(self):
        pi = Symbol("π")
        a, b, c = Symbol("α"), Symbol("β"), Symbol("γ")
        sh = sharp(pi)
        residue = Symbol("residue")
        rule = SnBivectorFormulaDefinition(sh)
        s = Sum(
            _vf_term(sh, a, b, c),
            _vf_term(sh, b, c, a),
            _vf_term(sh, c, a, b),
            residue,
        )
        out = rule.rewrite(s)
        assert out == Sum(BracketApply(default_sn, pi, pi), residue)

    def test_uses_supplied_sn_bracket(self):
        from jacopy.brackets.schouten import SchoutenBracket

        pi = Symbol("π")
        a, b, c = Symbol("α"), Symbol("β"), Symbol("γ")
        sh = sharp(pi)
        custom = SchoutenBracket(name="[·,·]_SN'")
        rule = SnBivectorFormulaDefinition(sh, sn_bracket=custom)
        s = Sum(
            _vf_term(sh, a, b, c),
            _vf_term(sh, b, c, a),
            _vf_term(sh, c, a, b),
        )
        out = rule.rewrite(s)
        assert out == BracketApply(custom, pi, pi)


# --------------------------------------------------------------------- #
# Engine integration                                                     #
# --------------------------------------------------------------------- #


class TestSnBivectorFormulaEngine:
    def test_engine_fires(self):
        pi = Symbol("π")
        a, b, c = Symbol("α"), Symbol("β"), Symbol("γ")
        sh = sharp(pi)
        engine = ExpansionEngine([SnBivectorFormulaDefinition(sh)])
        s = Sum(
            _vf_term(sh, a, b, c),
            _vf_term(sh, b, c, a),
            _vf_term(sh, c, a, b),
        )
        result, steps = engine.expand(s)
        assert result == BracketApply(default_sn, pi, pi)
        assert any(
            "[π,π]_SN bivector formula" in step.rule for step in steps
        )

    def test_engine_preserves_other_terms(self):
        pi = Symbol("π")
        a, b, c = Symbol("α"), Symbol("β"), Symbol("γ")
        sh = sharp(pi)
        residue = Symbol("residue")
        engine = ExpansionEngine([SnBivectorFormulaDefinition(sh)])
        s = Sum(
            _vf_term(sh, a, b, c),
            _vf_term(sh, b, c, a),
            _vf_term(sh, c, a, b),
            residue,
        )
        result, steps = engine.expand(s)
        assert result == Sum(BracketApply(default_sn, pi, pi), residue)
