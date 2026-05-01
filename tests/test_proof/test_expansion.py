"""Tests for jacopy.proof.expansion."""

import pytest

from jacopy.algebra.derivation import Act, Derivation, compose
from jacopy.calculus.exterior_d import ExteriorDerivative, d
from jacopy.calculus.interior import interior
from jacopy.calculus.lie_derivative import lie_derivative
from jacopy.core.expr import Expr, Integer, Product, Sum, Symbol
from jacopy.core.properties import Graded
from jacopy.core.registry import PropertyRegistry
from jacopy.proof.chain import ProofChain
from jacopy.proof.expansion import (
    MODES,
    ActOverSumOpDefinition,
    DSquaredZeroDefinition,
    Definition,
    ExpansionEngine,
    IotaOnExactOneFormDefinition,
    IotaOnZeroFormDefinition,
    IotaSquaredZeroDefinition,
    LieDerivativeCartanDefinition,
    LieDerivativeCommutesWithDDefinition,
    LieDerivativeOnZeroFormDefinition,
    default_engine,
)
from jacopy.proof.step import ProofStep


# --------------------------------------------------------------------- #
# LieDerivativeCartanDefinition                                          #
# --------------------------------------------------------------------- #


class TestLieDerivativeCartanDefinition:
    def test_matches_cartan_mode(self):
        X = Symbol("X")
        L = lie_derivative(X, definition="cartan")
        omega = Symbol("omega")
        defn = LieDerivativeCartanDefinition()
        assert defn.matches(Act(L, omega))

    def test_does_not_match_flow_mode(self):
        X = Symbol("X")
        L = lie_derivative(X, definition="flow")
        omega = Symbol("omega")
        defn = LieDerivativeCartanDefinition()
        assert not defn.matches(Act(L, omega))

    def test_does_not_match_non_act(self):
        X = Symbol("X")
        L = lie_derivative(X, definition="cartan")
        defn = LieDerivativeCartanDefinition()
        assert not defn.matches(L)

    def test_rewrite_produces_distributed_sum(self):
        X = Symbol("X")
        L = lie_derivative(X, definition="cartan")
        omega = Symbol("omega")
        defn = LieDerivativeCartanDefinition()
        result = defn.rewrite(Act(L, omega))
        iota_X = interior(X)
        expected = Sum(
            Act(compose(d, iota_X), omega),
            Act(compose(iota_X, d), omega),
        )
        assert result == expected

    def test_rewrite_uses_bundle_specific_d(self):
        """L.d overrides the default exterior derivative in the rewrite."""
        X = Symbol("X")
        d_E = ExteriorDerivative("d_E")
        L = lie_derivative(X, definition="cartan", d=d_E)
        omega = Symbol("omega")
        defn = LieDerivativeCartanDefinition()
        result = defn.rewrite(Act(L, omega))
        iota_X = interior(X)
        expected = Sum(
            Act(compose(d_E, iota_X), omega),
            Act(compose(iota_X, d_E), omega),
        )
        assert result == expected

    def test_rewrite_uses_bundle_specific_iota_factory(self):
        """L.iota_factory decides which ι_X instance is produced."""
        X = Symbol("X")

        def factory(Y: Expr):
            return interior(Y, name=f"ι_E,{Y._repr_inner()}")

        L = lie_derivative(X, definition="cartan", iota_factory=factory)
        omega = Symbol("omega")
        defn = LieDerivativeCartanDefinition()
        result = defn.rewrite(Act(L, omega))
        iota_EX = factory(X)
        expected = Sum(
            Act(compose(d, iota_EX), omega),
            Act(compose(iota_EX, d), omega),
        )
        assert result == expected

    def test_rewrite_uses_both_bundle_slots(self):
        """Algebroid case, L carries both d_E and the ι_E factory."""
        X = Symbol("X")
        d_E = ExteriorDerivative("d_E")

        def factory(Y: Expr):
            return interior(Y, name=f"ι_E,{Y._repr_inner()}")

        L = lie_derivative(
            X, definition="cartan", d=d_E, iota_factory=factory
        )
        omega = Symbol("omega")
        defn = LieDerivativeCartanDefinition()
        result = defn.rewrite(Act(L, omega))
        iota_EX = factory(X)
        expected = Sum(
            Act(compose(d_E, iota_EX), omega),
            Act(compose(iota_EX, d_E), omega),
        )
        assert result == expected


# --------------------------------------------------------------------- #
# LieDerivativeOnZeroFormDefinition                                      #
# --------------------------------------------------------------------- #


class TestLieDerivativeOnZeroFormDefinition:
    def test_matches_flow_on_zero_form(self):
        reg = PropertyRegistry()
        f = Symbol("f")
        reg.declare(f, Graded(degree=0))
        X = Derivation("X", degree=0)
        L = lie_derivative(X, definition="flow")
        defn = LieDerivativeOnZeroFormDefinition(registry=reg)
        assert defn.matches(Act(L, f))

    def test_does_not_match_cartan_mode(self):
        reg = PropertyRegistry()
        f = Symbol("f")
        reg.declare(f, Graded(degree=0))
        X = Derivation("X", degree=0)
        L = lie_derivative(X, definition="cartan")
        defn = LieDerivativeOnZeroFormDefinition(registry=reg)
        assert not defn.matches(Act(L, f))

    def test_does_not_match_one_form(self):
        reg = PropertyRegistry()
        alpha = Symbol("alpha")
        reg.declare(alpha, Graded(degree=1))
        X = Derivation("X", degree=0)
        L = lie_derivative(X, definition="flow")
        defn = LieDerivativeOnZeroFormDefinition(registry=reg)
        assert not defn.matches(Act(L, alpha))

    def test_does_not_match_non_derivation_field(self):
        """X must be a Derivation or composite, bare Symbol shouldn't fire."""
        reg = PropertyRegistry()
        f = Symbol("f")
        reg.declare(f, Graded(degree=0))
        X = Symbol("X")
        L = lie_derivative(X, definition="flow")
        defn = LieDerivativeOnZeroFormDefinition(registry=reg)
        assert not defn.matches(Act(L, f))

    def test_rewrite_produces_X_of_f(self):
        reg = PropertyRegistry()
        f = Symbol("f")
        reg.declare(f, Graded(degree=0))
        X = Derivation("X", degree=0)
        L = lie_derivative(X, definition="flow")
        defn = LieDerivativeOnZeroFormDefinition(registry=reg)
        assert defn.rewrite(Act(L, f)) == Act(X, f)


# --------------------------------------------------------------------- #
# LieDerivativeCommutesWithDDefinition                                   #
# --------------------------------------------------------------------- #


class TestLieDerivativeCommutesWithDDefinition:
    def test_matches_flow_on_exact_form(self):
        X = Derivation("X", degree=0)
        L = lie_derivative(X, definition="flow")
        omega = Symbol("omega")
        defn = LieDerivativeCommutesWithDDefinition()
        assert defn.matches(Act(L, Act(d, omega)))

    def test_does_not_match_cartan_mode(self):
        X = Derivation("X", degree=0)
        L = lie_derivative(X, definition="cartan")
        omega = Symbol("omega")
        defn = LieDerivativeCommutesWithDDefinition()
        assert not defn.matches(Act(L, Act(d, omega)))

    def test_does_not_match_non_d_inner(self):
        X = Derivation("X", degree=0)
        L = lie_derivative(X, definition="flow")
        omega = Symbol("omega")
        defn = LieDerivativeCommutesWithDDefinition()
        assert not defn.matches(Act(L, omega))

    def test_pins_to_bundle_specific_d(self):
        """If L carries d_E, default d doesn't match."""
        X = Derivation("X", degree=0)
        d_E = ExteriorDerivative("d_E")
        L = lie_derivative(X, definition="flow", d=d_E)
        omega = Symbol("omega")
        defn = LieDerivativeCommutesWithDDefinition()
        assert not defn.matches(Act(L, Act(d, omega)))
        assert defn.matches(Act(L, Act(d_E, omega)))

    def test_rewrite_moves_d_outside(self):
        X = Derivation("X", degree=0)
        L = lie_derivative(X, definition="flow")
        omega = Symbol("omega")
        defn = LieDerivativeCommutesWithDDefinition()
        result = defn.rewrite(Act(L, Act(d, omega)))
        assert result == Act(d, Act(L, omega))


# --------------------------------------------------------------------- #
# ExpansionEngine                                                        #
# --------------------------------------------------------------------- #


class TestExpansionEngine:
    def test_expand_once_on_matching_expr(self):
        X = Symbol("X")
        L = lie_derivative(X, definition="cartan")
        omega = Symbol("omega")
        engine = default_engine()
        after, step = engine.expand_once(Act(L, omega))
        assert step is not None
        assert isinstance(step, ProofStep)
        assert step.before == Act(L, omega)
        assert after == step.after

    def test_expand_once_no_match(self):
        engine = default_engine()
        expr = Symbol("x")
        after, step = engine.expand_once(expr)
        assert step is None
        assert after is expr

    def test_expand_once_descends_into_children(self):
        X = Symbol("X")
        L = lie_derivative(X, definition="cartan")
        omega = Symbol("omega")
        # Bury L(ω) inside a Sum.
        outer = Sum(Symbol("a"), Act(L, omega))
        engine = default_engine()
        after, step = engine.expand_once(outer)
        assert step is not None
        assert step.before == Act(L, omega)
        # Result has outer structure preserved; only the matching subtree changed.
        assert isinstance(after, Sum)
        assert after.children[0] == Symbol("a")

    def test_expand_reaches_fixed_point(self):
        X = Symbol("X")
        L = lie_derivative(X, definition="cartan")
        omega = Symbol("omega")
        engine = default_engine()
        after, steps = engine.expand(Act(L, omega))
        assert len(steps) == 1
        # After one rewrite, no further Cartan-mode L_X remains.
        after2, more = engine.expand(after)
        assert more == []

    def test_expand_empty_when_nothing_matches(self):
        engine = default_engine()
        expr = Sum(Symbol("a"), Symbol("b"))
        after, steps = engine.expand(expr)
        assert steps == []
        assert after == expr

    def test_flow_mode_is_not_expanded(self):
        X = Symbol("X")
        L = lie_derivative(X, definition="flow")
        omega = Symbol("omega")
        engine = default_engine()
        after, steps = engine.expand(Act(L, omega))
        assert steps == []
        assert after == Act(L, omega)


# --------------------------------------------------------------------- #
# Registration & custom definitions                                      #
# --------------------------------------------------------------------- #


class _SymbolRenameDefinition(Definition):
    """Toy definition: rename Symbol("a") to Symbol("b")."""

    name = "rename a → b"

    def matches(self, expr: Expr) -> bool:
        return isinstance(expr, Symbol) and expr.name == "a"

    def rewrite(self, expr: Expr) -> Expr:
        return Symbol("b")


class TestRegistration:
    def test_register_adds_definition(self):
        engine = ExpansionEngine()
        assert engine.definitions == ()
        defn = _SymbolRenameDefinition()
        engine.register(defn)
        assert engine.definitions == (defn,)

    def test_register_rejects_non_definition(self):
        engine = ExpansionEngine()
        with pytest.raises(TypeError):
            engine.register("not a definition")  # type: ignore[arg-type]

    def test_custom_definition_fires(self):
        engine = ExpansionEngine([_SymbolRenameDefinition()])
        after, steps = engine.expand(Symbol("a"))
        assert after == Symbol("b")
        assert len(steps) == 1

    def test_custom_definition_fires_inside_sum(self):
        engine = ExpansionEngine([_SymbolRenameDefinition()])
        expr = Sum(Symbol("a"), Symbol("c"))
        after, steps = engine.expand(expr)
        assert after == Sum(Symbol("b"), Symbol("c"))
        assert len(steps) == 1

    def test_divergent_rule_raises(self):
        class _Bad(Definition):
            name = "cycle"

            def matches(self, expr):
                return isinstance(expr, Symbol) and expr.name == "x"

            def rewrite(self, expr):
                # Stays "x" → engine loops forever.
                return Symbol("x_rewritten_to_x") if False else Symbol("x")

        # Actually this rewrite would be a fixpoint immediately; force
        # divergence by always producing a fresh equivalent expression.
        class _Div(Definition):
            name = "grow"

            def __init__(self):
                self._n = [0]

            def matches(self, expr):
                return isinstance(expr, Symbol) and expr.name.startswith("g")

            def rewrite(self, expr):
                self._n[0] += 1
                return Symbol(f"g{self._n[0]}")

        engine = ExpansionEngine([_Div()])
        with pytest.raises(RuntimeError):
            engine.expand(Symbol("g"), max_steps=8)


# --------------------------------------------------------------------- #
# ActOverSumOpDefinition                                                 #
# --------------------------------------------------------------------- #


class TestActOverSumOpDefinition:
    def test_matches_sum_operator(self):
        A, B = Symbol("A"), Symbol("B")
        x = Symbol("x")
        assert ActOverSumOpDefinition().matches(Act(Sum(A, B), x))

    def test_does_not_match_product_operator(self):
        A, B = Symbol("A"), Symbol("B")
        x = Symbol("x")
        assert not ActOverSumOpDefinition().matches(Act(Product(A, B), x))

    def test_does_not_match_plain_symbol_operator(self):
        X = Symbol("X")
        x = Symbol("x")
        assert not ActOverSumOpDefinition().matches(Act(X, x))

    def test_rewrite_distributes(self):
        A, B = Symbol("A"), Symbol("B")
        x = Symbol("x")
        result = ActOverSumOpDefinition().rewrite(Act(Sum(A, B), x))
        assert result == Sum(Act(A, x), Act(B, x))


# --------------------------------------------------------------------- #
# DSquaredZeroDefinition                                                 #
# --------------------------------------------------------------------- #


class TestDSquaredZeroDefinition:
    def test_matches_nested_d_d(self):
        f = Symbol("f")
        assert DSquaredZeroDefinition().matches(Act(d, Act(d, f)))

    def test_does_not_match_single_d(self):
        f = Symbol("f")
        assert not DSquaredZeroDefinition().matches(Act(d, f))

    def test_rewrite_produces_zero(self):
        f = Symbol("f")
        assert DSquaredZeroDefinition().rewrite(Act(d, Act(d, f))) == Integer(0)

    def test_targeted_pins_specific_instance(self):
        d_E = ExteriorDerivative("d_E")
        defn = DSquaredZeroDefinition(target=d_E)
        f = Symbol("f")
        # Standard d fails the pin even though its structure matches.
        assert not defn.matches(Act(d, Act(d, f)))
        assert defn.matches(Act(d_E, Act(d_E, f)))

    def test_does_not_match_mixed_d_and_d_E(self):
        d_E = ExteriorDerivative("d_E")
        f = Symbol("f")
        # Outer d, inner d_E, not a d² occurrence.
        assert not DSquaredZeroDefinition().matches(Act(d, Act(d_E, f)))


# --------------------------------------------------------------------- #
# DSquaredZeroDefinition, classification (axiom vs theorem)             #
# --------------------------------------------------------------------- #


class TestDSquaredZeroClassification:
    def test_default_classification_is_axiom(self):
        defn = DSquaredZeroDefinition()
        assert defn.classification == "axiom"
        assert defn.is_theorem is False
        assert defn.theorem_proof_builder() is None

    def test_theorem_classification_flips_is_theorem(self):
        defn = DSquaredZeroDefinition(classification="theorem")
        assert defn.classification == "theorem"
        assert defn.is_theorem is True
        assert defn.theorem_proof_builder() is not None

    def test_rewrite_unchanged_by_classification(self):
        """Both classifications rewrite to zero, only bookkeeping differs."""
        f = Symbol("f")
        dd_f = Act(d, Act(d, f))
        assert DSquaredZeroDefinition().rewrite(dd_f) == Integer(0)
        assert (
            DSquaredZeroDefinition(classification="theorem").rewrite(dd_f)
            == Integer(0)
        )

    def test_matches_unchanged_by_classification(self):
        f = Symbol("f")
        dd_f = Act(d, Act(d, f))
        assert DSquaredZeroDefinition().matches(dd_f)
        assert DSquaredZeroDefinition(classification="theorem").matches(dd_f)

    def test_rejects_unknown_classification(self):
        with pytest.raises(ValueError, match="classification"):
            DSquaredZeroDefinition(classification="lemma")

    def test_theorem_builder_cites_generator_axiom(self):
        """Sub-proof bottoms out at d(df) = 0, the foundational primitive."""
        f = Symbol("f")
        dd_f = Act(d, Act(d, f))
        defn = DSquaredZeroDefinition(classification="theorem")
        builder = defn.theorem_proof_builder()
        assert builder is not None
        sub_chain = builder(dd_f)
        assert isinstance(sub_chain, ProofChain)
        assert len(sub_chain) == 1
        step = sub_chain.steps[0]
        assert step.before == dd_f
        assert step.after == Integer(0)
        assert step.provenance_tag == "axiom"
        assert "generator" in step.rule


class TestDSquaredZeroEngineClassification:
    def test_efficient_mode_theorem_classification_tags_theorem(self):
        """Theorem classification: step tagged 'theorem', no sub-proof in efficient mode."""
        f = Symbol("f")
        engine = ExpansionEngine(
            [DSquaredZeroDefinition(classification="theorem")],
            mode="efficient",
        )
        _, step = engine.expand_once(Act(d, Act(d, f)))
        assert step is not None
        assert step.provenance_tag == "theorem"
        assert step.children == ()

    def test_foundational_mode_theorem_attaches_sub_proof(self):
        f = Symbol("f")
        engine = ExpansionEngine(
            [DSquaredZeroDefinition(classification="theorem")],
            mode="foundational",
        )
        _, step = engine.expand_once(Act(d, Act(d, f)))
        assert step is not None
        assert step.provenance_tag == "theorem"
        assert len(step.children) == 1
        sub = step.children[0]
        assert sub.provenance_tag == "axiom"
        assert "generator" in sub.rule

    def test_axiom_classification_stays_axiom_under_foundational(self):
        """Axiom classification does NOT attach sub-proofs, even in foundational mode."""
        f = Symbol("f")
        engine = ExpansionEngine(
            [DSquaredZeroDefinition()],  # default = axiom
            mode="foundational",
        )
        _, step = engine.expand_once(Act(d, Act(d, f)))
        assert step is not None
        assert step.provenance_tag == "axiom"
        assert step.children == ()


class TestDefaultEngineDSquaredMode:
    def test_default_d_squared_mode_is_axiom(self):
        """Backward-compat: default_engine with no kwarg keeps d²=0 as axiom."""
        engine = default_engine()
        f = Symbol("f")
        _, step = engine.expand_once(Act(d, Act(d, f)))
        assert step is not None
        assert step.provenance_tag == "axiom"

    def test_theorem_mode_registers_theorem_classified_rule(self):
        engine = default_engine(d_squared_mode="theorem")
        f = Symbol("f")
        _, step = engine.expand_once(Act(d, Act(d, f)))
        assert step is not None
        assert step.provenance_tag == "theorem"

    def test_foundational_plus_theorem_mode_unrolls(self):
        engine = default_engine(mode="foundational", d_squared_mode="theorem")
        f = Symbol("f")
        _, step = engine.expand_once(Act(d, Act(d, f)))
        assert step is not None
        assert step.provenance_tag == "theorem"
        assert len(step.children) == 1

    def test_unknown_d_squared_mode_raises(self):
        with pytest.raises(ValueError, match="classification"):
            default_engine(d_squared_mode="lemma")


# --------------------------------------------------------------------- #
# IotaSquaredZeroDefinition                                              #
# --------------------------------------------------------------------- #


class TestIotaSquaredZeroDefinition:
    def test_matches_same_iota(self):
        iota_X = interior(Symbol("X"))
        assert IotaSquaredZeroDefinition().matches(
            Act(iota_X, Act(iota_X, Symbol("a")))
        )

    def test_does_not_match_different_iotas(self):
        iota_X = interior(Symbol("X"))
        iota_Y = interior(Symbol("Y"))
        assert not IotaSquaredZeroDefinition().matches(
            Act(iota_X, Act(iota_Y, Symbol("a")))
        )

    def test_rewrite_produces_zero(self):
        iota_X = interior(Symbol("X"))
        result = IotaSquaredZeroDefinition().rewrite(
            Act(iota_X, Act(iota_X, Symbol("a")))
        )
        assert result == Integer(0)


# --------------------------------------------------------------------- #
# IotaOnZeroFormDefinition                                               #
# --------------------------------------------------------------------- #


class TestIotaOnZeroFormDefinition:
    def test_matches_declared_zero_form(self):
        reg = PropertyRegistry()
        f = Symbol("f")
        reg.declare(f, Graded(degree=0))
        iota_X = interior(Symbol("X"))
        assert IotaOnZeroFormDefinition(registry=reg).matches(Act(iota_X, f))

    def test_does_not_match_one_form(self):
        reg = PropertyRegistry()
        alpha = Symbol("alpha")
        reg.declare(alpha, Graded(degree=1))
        iota_X = interior(Symbol("X"))
        assert not IotaOnZeroFormDefinition(registry=reg).matches(
            Act(iota_X, alpha)
        )

    def test_does_not_match_undeclared(self):
        iota_X = interior(Symbol("X"))
        # No registry → undecidable degree → conservative no-match.
        assert not IotaOnZeroFormDefinition(registry=None).matches(
            Act(iota_X, Symbol("f"))
        )

    def test_rewrite_produces_zero(self):
        iota_X = interior(Symbol("X"))
        result = IotaOnZeroFormDefinition().rewrite(Act(iota_X, Symbol("f")))
        assert result == Integer(0)


# --------------------------------------------------------------------- #
# IotaOnExactOneFormDefinition                                           #
# --------------------------------------------------------------------- #


class TestIotaOnExactOneFormDefinition:
    def test_matches_iota_d_f(self):
        reg = PropertyRegistry()
        f = Symbol("f")
        reg.declare(f, Graded(degree=0))
        X_deriv = Derivation("X", degree=0)
        iota_X = interior(X_deriv)
        assert IotaOnExactOneFormDefinition(registry=reg).matches(
            Act(iota_X, Act(d, f))
        )

    def test_requires_X_to_be_derivation(self):
        reg = PropertyRegistry()
        f = Symbol("f")
        reg.declare(f, Graded(degree=0))
        iota_X = interior(Symbol("X"))  # Symbol, not Derivation
        assert not IotaOnExactOneFormDefinition(registry=reg).matches(
            Act(iota_X, Act(d, f))
        )

    def test_pinned_to_specific_d(self):
        reg = PropertyRegistry()
        f = Symbol("f")
        reg.declare(f, Graded(degree=0))
        d_E = ExteriorDerivative("d_E")
        X_deriv = Derivation("X", degree=0)
        iota_X = interior(X_deriv)
        defn = IotaOnExactOneFormDefinition(d=d_E, registry=reg)
        # Standard d doesn't match when pinned to d_E.
        assert not defn.matches(Act(iota_X, Act(d, f)))
        assert defn.matches(Act(iota_X, Act(d_E, f)))

    def test_rewrite_produces_X_of_f(self):
        reg = PropertyRegistry()
        f = Symbol("f")
        reg.declare(f, Graded(degree=0))
        X_deriv = Derivation("X", degree=0)
        iota_X = interior(X_deriv)
        defn = IotaOnExactOneFormDefinition(registry=reg)
        result = defn.rewrite(Act(iota_X, Act(d, f)))
        assert result == Act(X_deriv, f)


# --------------------------------------------------------------------- #
# default_engine with registry                                           #
# --------------------------------------------------------------------- #


class TestDefaultEngineWithRegistry:
    def test_iota_on_zero_form_fires(self):
        reg = PropertyRegistry()
        f = Symbol("f")
        reg.declare(f, Graded(degree=0))
        iota_X = interior(Symbol("X"))
        engine = default_engine(registry=reg)
        after, steps = engine.expand(Act(iota_X, f))
        assert after == Integer(0)
        assert len(steps) == 1

    def test_iota_on_exact_one_form_fires(self):
        reg = PropertyRegistry()
        f = Symbol("f")
        reg.declare(f, Graded(degree=0))
        X_deriv = Derivation("X", degree=0)
        iota_X = interior(X_deriv)
        engine = default_engine(registry=reg)
        after, steps = engine.expand(Act(iota_X, Act(d, f)))
        assert after == Act(X_deriv, f)

    def test_d_squared_fires(self):
        f = Symbol("f")
        engine = default_engine()
        after, steps = engine.expand(Act(d, Act(d, f)))
        assert after == Integer(0)

    def test_iota_squared_fires(self):
        iota_X = interior(Symbol("X"))
        engine = default_engine()
        after, steps = engine.expand(Act(iota_X, Act(iota_X, Symbol("a"))))
        assert after == Integer(0)

    def test_act_over_sum_fires(self):
        A, B = Symbol("A"), Symbol("B")
        x = Symbol("x")
        engine = default_engine()
        after, steps = engine.expand(Act(Sum(A, B), x))
        assert after == Sum(Act(A, x), Act(B, x))


# --------------------------------------------------------------------- #
# Provenance and mode                                                    #
# --------------------------------------------------------------------- #


class _StubAxiom(Definition):
    """Minimal axiom definition that rewrites Symbol('a') → Integer(0)."""

    name = "stub-axiom"

    def matches(self, expr):
        return isinstance(expr, Symbol) and expr.name == "a"

    def rewrite(self, expr):
        return Integer(0)


class _StubTheorem(Definition):
    """Minimal theorem definition: rewrites Symbol('t') → Integer(0) with sub-proof."""

    name = "stub-theorem"

    def matches(self, expr):
        return isinstance(expr, Symbol) and expr.name == "t"

    def rewrite(self, expr):
        return Integer(0)

    def theorem_proof_builder(self):
        def _build(matched):
            chain = ProofChain()
            chain.append(
                ProofStep(
                    matched,
                    Integer(0),
                    rule="stub-sub-proof",
                    justification="stand-in sub-proof",
                    provenance_tag="axiom",
                )
            )
            return chain
        return _build


class TestDefinitionTheoremFlag:
    def test_default_is_axiom(self):
        assert _StubAxiom().is_theorem is False
        assert _StubAxiom().theorem_proof_builder() is None

    def test_theorem_flag_true_when_builder_present(self):
        assert _StubTheorem().is_theorem is True
        assert _StubTheorem().theorem_proof_builder() is not None


class TestEngineModes:
    def test_default_mode_efficient(self):
        eng = ExpansionEngine([])
        assert eng.mode == "efficient"

    def test_constructor_accepts_foundational(self):
        eng = ExpansionEngine([], mode="foundational")
        assert eng.mode == "foundational"

    def test_constructor_rejects_unknown_mode(self):
        with pytest.raises(ValueError, match="mode"):
            ExpansionEngine([], mode="sloppy")

    def test_modes_constant_lists_known_modes(self):
        assert set(MODES) == {"efficient", "foundational"}

    def test_with_mode_returns_new_engine(self):
        eng = ExpansionEngine([_StubAxiom()], mode="efficient")
        eng2 = eng.with_mode("foundational")
        assert eng2.mode == "foundational"
        assert eng.mode == "efficient"
        # Definitions are preserved.
        assert len(eng2.definitions) == 1

    def test_default_engine_accepts_mode(self):
        eng = default_engine(mode="foundational")
        assert eng.mode == "foundational"


class TestEngineProvenanceTagging:
    def test_axiom_step_tagged_axiom(self):
        eng = ExpansionEngine([_StubAxiom()])
        _, step = eng.expand_once(Symbol("a"))
        assert step is not None
        assert step.provenance_tag == "axiom"
        assert step.children == ()

    def test_theorem_step_tagged_theorem_efficient_no_children(self):
        eng = ExpansionEngine([_StubTheorem()], mode="efficient")
        _, step = eng.expand_once(Symbol("t"))
        assert step is not None
        assert step.provenance_tag == "theorem"
        # Efficient mode does not unroll sub-proofs.
        assert step.children == ()

    def test_theorem_step_attaches_sub_proof_foundational(self):
        eng = ExpansionEngine([_StubTheorem()], mode="foundational")
        _, step = eng.expand_once(Symbol("t"))
        assert step is not None
        assert step.provenance_tag == "theorem"
        # Foundational mode attaches the theorem's sub-proof as children.
        assert len(step.children) == 1
        assert step.children[0].rule == "stub-sub-proof"

    def test_axiom_step_unchanged_under_foundational(self):
        eng = ExpansionEngine([_StubAxiom()], mode="foundational")
        _, step = eng.expand_once(Symbol("a"))
        assert step is not None
        assert step.provenance_tag == "axiom"
        assert step.children == ()
