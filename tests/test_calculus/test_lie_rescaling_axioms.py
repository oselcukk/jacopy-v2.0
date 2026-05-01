"""Tests for the Lie-derivative rescaling axiom (Faz 12.B #10)."""

import pytest

from jacopy.algebra.derivation import Act
from jacopy.calculus.exterior_d import ExteriorDerivative, d as default_d
from jacopy.calculus.interior import InteriorProduct, interior
from jacopy.calculus.lie_derivative import LieDerivative, lie_derivative
from jacopy.calculus.lie_rescaling_axioms import LieRescalingDefinition
from jacopy.core.expr import Product, Sum, Symbol
from jacopy.proof.expansion import ExpansionEngine


# --------------------------------------------------------------------- #
# Match logic                                                            #
# --------------------------------------------------------------------- #


class TestLieRescalingMatches:
    def test_matches_two_factor_product(self):
        rule = LieRescalingDefinition()
        f, X, omega = Symbol("f"), Symbol("X"), Symbol("ω")
        expr = Act(lie_derivative(Product(f, X)), omega)
        assert rule.matches(expr)

    def test_matches_three_factor_product(self):
        # L_{g·f·X}(ω), leading two folded into f.
        rule = LieRescalingDefinition()
        g, f, X, omega = Symbol("g"), Symbol("f"), Symbol("X"), Symbol("ω")
        expr = Act(lie_derivative(Product(g, f, X)), omega)
        assert rule.matches(expr)

    def test_no_match_plain_vector_field(self):
        rule = LieRescalingDefinition()
        X, omega = Symbol("X"), Symbol("ω")
        expr = Act(lie_derivative(X), omega)
        assert not rule.matches(expr)

    def test_no_match_non_lie_derivative_op(self):
        rule = LieRescalingDefinition()
        f, X, omega = Symbol("f"), Symbol("X"), Symbol("ω")
        # ι_{f·X}(ω), also not handled here.
        expr = Act(interior(Product(f, X)), omega)
        assert not rule.matches(expr)

    def test_no_match_non_act(self):
        rule = LieRescalingDefinition()
        f, X = Symbol("f"), Symbol("X")
        # The vector field expression itself, not an Act.
        assert not rule.matches(Product(f, X))


# --------------------------------------------------------------------- #
# Rewrite                                                                #
# --------------------------------------------------------------------- #


class TestLieRescalingRewrite:
    def test_two_factor_rewrite(self):
        rule = LieRescalingDefinition()
        f, X, omega = Symbol("f"), Symbol("X"), Symbol("ω")
        expr = Act(lie_derivative(Product(f, X)), omega)
        out = rule.rewrite(expr)
        expected = Sum(
            Product(f, Act(lie_derivative(X), omega)),
            Product(Act(default_d, f), Act(interior(X), omega)),
        )
        assert out == expected

    def test_three_factor_folds_leading_into_scalar(self):
        rule = LieRescalingDefinition()
        g, f, X, omega = Symbol("g"), Symbol("f"), Symbol("X"), Symbol("ω")
        expr = Act(lie_derivative(Product(g, f, X)), omega)
        out = rule.rewrite(expr)
        expected = Sum(
            Product(Product(g, f), Act(lie_derivative(X), omega)),
            Product(
                Act(default_d, Product(g, f)),
                Act(interior(X), omega),
            ),
        )
        assert out == expected

    def test_preserves_definition_mode(self):
        rule = LieRescalingDefinition()
        f, X, omega = Symbol("f"), Symbol("X"), Symbol("ω")
        L = lie_derivative(Product(f, X), definition="flow")
        out = rule.rewrite(Act(L, omega))
        # Walk the inner L_X to confirm it's flow-mode too.
        inner_L = out.children[0].children[1].op
        assert isinstance(inner_L, LieDerivative)
        assert inner_L.definition == "flow"

    def test_uses_bundle_d_override(self):
        # Faux algebroid d.
        rule = LieRescalingDefinition()
        f, X, omega = Symbol("f"), Symbol("X"), Symbol("ω")
        d_E = ExteriorDerivative(name="d_E")
        L = lie_derivative(Product(f, X), d=d_E)
        out = rule.rewrite(Act(L, omega))
        # The df term must use d_E, not default_d.
        df_term = out.children[1]
        assert df_term.children[0] == Act(d_E, f)

    def test_uses_iota_factory(self):
        rule = LieRescalingDefinition()
        f, X, omega = Symbol("f"), Symbol("X"), Symbol("ω")
        iota_calls: list = []

        def factory(x):
            iota_calls.append(x)
            return InteriorProduct(x, name=f"ι_E_{x._repr_inner()}")

        L = lie_derivative(Product(f, X), iota_factory=factory)
        out = rule.rewrite(Act(L, omega))
        assert iota_calls == [X]
        # Confirm the rebuilt L_X also carries the same factory.
        inner_L = out.children[0].children[1].op
        assert inner_L.iota_factory is factory


# --------------------------------------------------------------------- #
# Engine integration                                                     #
# --------------------------------------------------------------------- #


class TestLieRescalingEngineIntegration:
    def test_engine_rewrites(self):
        engine = ExpansionEngine([LieRescalingDefinition()])
        f, X, omega = Symbol("f"), Symbol("X"), Symbol("ω")
        expr = Act(lie_derivative(Product(f, X)), omega)
        out, steps = engine.expand(expr)
        assert out == Sum(
            Product(f, Act(lie_derivative(X), omega)),
            Product(Act(default_d, f), Act(interior(X), omega)),
        )
        assert len(steps) == 1
        assert steps[0].rule == "L_{fX}: L_{f·X}(ω) = f·L_X(ω) + df∧ι_X(ω)"

    def test_engine_no_op_on_plain(self):
        engine = ExpansionEngine([LieRescalingDefinition()])
        X, omega = Symbol("X"), Symbol("ω")
        expr = Act(lie_derivative(X), omega)
        out, steps = engine.expand(expr)
        assert out == expr
        assert steps == []
