"""Tests for Faz 13.A, Sharp R-linearity and Sharp-on-exact rewrites."""

import pytest

from jacopy.algebra.derivation import Act
from jacopy.calculus.exterior_d import ExteriorDerivative, d
from jacopy.calculus.hamiltonian_vf import HamiltonianVectorField
from jacopy.calculus.musical import Sharp, sharp
from jacopy.calculus.sharp_axioms import (
    SharpLinearityDefinition,
    SharpOnExactDefinition,
)
from jacopy.core.expr import Integer, Sum, Symbol
from jacopy.core.properties import Graded
from jacopy.core.registry import PropertyRegistry
from jacopy.proof.expansion import ExpansionEngine


# --------------------------------------------------------------------- #
# SharpLinearityDefinition (axiom 1)                                     #
# --------------------------------------------------------------------- #


class TestSharpLinearityMatches:
    def test_matches_sum_argument(self):
        pi = Symbol("π")
        sh = sharp(pi)
        a, b = Symbol("a"), Symbol("b")
        rule = SharpLinearityDefinition(sh)
        assert rule.matches(Act(sh, Sum(a, b)))

    def test_no_match_on_atomic_argument(self):
        pi = Symbol("π")
        sh = sharp(pi)
        a = Symbol("a")
        rule = SharpLinearityDefinition(sh)
        assert not rule.matches(Act(sh, a))

    def test_no_match_on_other_sharp(self):
        pi1 = Symbol("π1")
        pi2 = Symbol("π2")
        sh1, sh2 = sharp(pi1), sharp(pi2)
        a, b = Symbol("a"), Symbol("b")
        rule = SharpLinearityDefinition(sh1)
        assert not rule.matches(Act(sh2, Sum(a, b)))

    def test_no_match_on_non_act(self):
        pi = Symbol("π")
        sh = sharp(pi)
        rule = SharpLinearityDefinition(sh)
        assert not rule.matches(Sum(Symbol("a"), Symbol("b")))


class TestSharpLinearityRewrite:
    def test_distributes_two_term_sum(self):
        pi = Symbol("π")
        sh = sharp(pi)
        a, b = Symbol("a"), Symbol("b")
        rule = SharpLinearityDefinition(sh)
        out = rule.rewrite(Act(sh, Sum(a, b)))
        assert out == Sum(Act(sh, a), Act(sh, b))

    def test_distributes_three_term_sum(self):
        pi = Symbol("π")
        sh = sharp(pi)
        a, b, c = Symbol("a"), Symbol("b"), Symbol("c")
        rule = SharpLinearityDefinition(sh)
        out = rule.rewrite(Act(sh, Sum(a, b, c)))
        assert out == Sum(Act(sh, a), Act(sh, b), Act(sh, c))

    def test_engine_fires_on_act_over_sum(self):
        pi = Symbol("π")
        sh = sharp(pi)
        a, b = Symbol("a"), Symbol("b")
        engine = ExpansionEngine([SharpLinearityDefinition(sh)])
        result, steps = engine.expand(Act(sh, Sum(a, b)))
        assert result == Sum(Act(sh, a), Act(sh, b))
        assert any("R-linearity" in s.rule for s in steps)


class TestSharpLinearityValidation:
    def test_rejects_non_sharp(self):
        with pytest.raises(TypeError):
            SharpLinearityDefinition(Symbol("not a sharp"))


# --------------------------------------------------------------------- #
# SharpOnExactDefinition (axiom 2)                                       #
# --------------------------------------------------------------------- #


def _registry_with_function(name: str) -> tuple[Symbol, PropertyRegistry]:
    f = Symbol(name)
    reg = PropertyRegistry()
    reg.declare(f, Graded(degree=0))
    return f, reg


class TestSharpOnExactMatches:
    def test_matches_sharp_of_df(self):
        pi = Symbol("π")
        sh = sharp(pi)
        f, reg = _registry_with_function("f")
        rule = SharpOnExactDefinition(sh, registry=reg)
        assert rule.matches(Act(sh, Act(d, f)))

    def test_no_match_on_non_zero_form_argument(self):
        pi = Symbol("π")
        sh = sharp(pi)
        # alpha is a 1-form (not a 0-form), so d(alpha) is a 2-form,
        # but the *argument* of d (alpha) is what gets degree-checked
        # against zero. This rule only fires on 0-forms.
        alpha = Symbol("α")
        reg = PropertyRegistry()
        reg.declare(alpha, Graded(degree=1))
        rule = SharpOnExactDefinition(sh, registry=reg)
        assert not rule.matches(Act(sh, Act(d, alpha)))

    def test_no_match_when_inner_op_is_not_d(self):
        pi = Symbol("π")
        sh = sharp(pi)
        f, reg = _registry_with_function("f")
        d_other = ExteriorDerivative("d_E")
        rule = SharpOnExactDefinition(sh, registry=reg)
        assert not rule.matches(Act(sh, Act(d_other, f)))

    def test_targets_specific_d(self):
        pi = Symbol("π")
        sh = sharp(pi)
        f, reg = _registry_with_function("f")
        d_E = ExteriorDerivative("d_E")
        rule = SharpOnExactDefinition(sh, d=d_E, registry=reg)
        assert rule.matches(Act(sh, Act(d_E, f)))
        assert not rule.matches(Act(sh, Act(d, f)))

    def test_no_match_when_outer_is_not_target_sharp(self):
        pi1 = Symbol("π1")
        pi2 = Symbol("π2")
        sh1, sh2 = sharp(pi1), sharp(pi2)
        f, reg = _registry_with_function("f")
        rule = SharpOnExactDefinition(sh1, registry=reg)
        assert not rule.matches(Act(sh2, Act(d, f)))

    def test_no_match_without_registry(self):
        pi = Symbol("π")
        sh = sharp(pi)
        f = Symbol("f")
        rule = SharpOnExactDefinition(sh)
        # f's degree is undeterminable without a registry, so the rule
        # leaves the shape untouched rather than firing on undefined
        # input.
        assert not rule.matches(Act(sh, Act(d, f)))


class TestSharpOnExactRewrite:
    def test_produces_hamiltonian_vector_field(self):
        pi = Symbol("π")
        sh = sharp(pi)
        f, reg = _registry_with_function("f")
        rule = SharpOnExactDefinition(sh, registry=reg)
        out = rule.rewrite(Act(sh, Act(d, f)))
        assert isinstance(out, HamiltonianVectorField)
        assert out.function is f
        assert out.bivector is pi

    def test_engine_fires(self):
        pi = Symbol("π")
        sh = sharp(pi)
        f, reg = _registry_with_function("f")
        engine = ExpansionEngine([SharpOnExactDefinition(sh, registry=reg)])
        result, steps = engine.expand(Act(sh, Act(d, f)))
        assert isinstance(result, HamiltonianVectorField)
        assert any("X_f" in s.rule for s in steps)

    def test_combined_with_linearity(self):
        # Sharp(π)(d(f) + d(g)), linearity first, then SharpOnExact
        # twice, gives X_f + X_g.
        pi = Symbol("π")
        sh = sharp(pi)
        f, reg = _registry_with_function("f")
        g = Symbol("g")
        reg.declare(g, Graded(degree=0))
        engine = ExpansionEngine(
            [
                SharpLinearityDefinition(sh),
                SharpOnExactDefinition(sh, registry=reg),
            ]
        )
        expr = Act(sh, Sum(Act(d, f), Act(d, g)))
        result, _ = engine.expand(expr)
        Xf = HamiltonianVectorField(f, bivector=pi)
        Xg = HamiltonianVectorField(g, bivector=pi)
        assert result == Sum(Xf, Xg)


class TestSharpOnExactValidation:
    def test_rejects_non_sharp(self):
        with pytest.raises(TypeError):
            SharpOnExactDefinition(Symbol("not a sharp"))


# --------------------------------------------------------------------- #
# Display / introspection                                                #
# --------------------------------------------------------------------- #


class TestSharpAxiomNames:
    def test_linearity_name_includes_bivector(self):
        pi = Symbol("π")
        rule = SharpLinearityDefinition(sharp(pi))
        assert "π" in rule.name

    def test_on_exact_name_includes_bivector(self):
        pi = Symbol("π")
        rule = SharpOnExactDefinition(sharp(pi))
        assert "π" in rule.name
        assert "X_f" in rule.name
