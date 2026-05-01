"""Tests for musical isomorphisms ``♭``, ``♯`` and their compatibility."""

import pytest

from jacopy.algebra.derivation import Act, Derivation
from jacopy.calculus.interior import interior
from jacopy.calculus.musical import (
    ArgNegLinearityDefinition,
    Flat,
    IotaFlatDefinition,
    MusicalCompatibility,
    MusicalCompatibilityBilinearDefinition,
    MusicalCompatibilityDefinition,
    Sharp,
    flat,
    sharp,
)
from jacopy.core.expr import Integer, Neg, Symbol
from jacopy.core.multi_eval import multi_eval
from jacopy.core.symbolic_degree import Degree
from jacopy.proof.expansion import ExpansionEngine


# --------------------------------------------------------------------- #
# Flat / Sharp                                                           #
# --------------------------------------------------------------------- #


class TestFlat:
    def test_is_degree_zero_derivation(self):
        omega = Symbol("ω")
        fl = flat(omega)
        assert isinstance(fl, Derivation)
        assert fl.degree == Degree.const(0)

    def test_default_name(self):
        omega = Symbol("ω")
        assert flat(omega).name == "ω♭"

    def test_custom_name(self):
        omega = Symbol("ω")
        assert flat(omega, name="ω_E♭").name == "ω_E♭"

    def test_carries_form(self):
        omega = Symbol("ω")
        assert flat(omega).form is omega

    def test_equality_on_form(self):
        omega = Symbol("ω")
        assert flat(omega) == flat(omega)

    def test_distinct_forms_give_distinct_flats(self):
        assert flat(Symbol("ω1")) != flat(Symbol("ω2"))

    def test_requires_expr(self):
        with pytest.raises(TypeError):
            Flat("ω")  # type: ignore[arg-type]


class TestSharp:
    def test_is_degree_zero_derivation(self):
        pi = Symbol("π")
        sh = sharp(pi)
        assert isinstance(sh, Derivation)
        assert sh.degree == Degree.const(0)

    def test_default_name(self):
        pi = Symbol("π")
        assert sharp(pi).name == "π♯"

    def test_carries_bivector(self):
        pi = Symbol("π")
        assert sharp(pi).bivector is pi

    def test_equality_on_bivector(self):
        pi = Symbol("π")
        assert sharp(pi) == sharp(pi)


# --------------------------------------------------------------------- #
# MusicalCompatibility                                                   #
# --------------------------------------------------------------------- #


class TestMusicalCompatibility:
    def test_between_builds_flat_and_sharp(self):
        omega, pi = Symbol("ω"), Symbol("π")
        compat = MusicalCompatibility.between(omega, pi)
        assert compat.omega is omega
        assert compat.pi is pi
        assert isinstance(compat.flat, Flat)
        assert isinstance(compat.sharp, Sharp)
        assert compat.flat.form is omega
        assert compat.sharp.bivector is pi

    def test_custom_flat_and_sharp_honoured(self):
        omega, pi = Symbol("ω"), Symbol("π")
        fl = flat(omega, name="ω̃♭")
        sh = sharp(pi, name="π̃♯")
        compat = MusicalCompatibility.between(
            omega, pi, flat_instance=fl, sharp_instance=sh
        )
        assert compat.flat is fl
        assert compat.sharp is sh

    def test_default_name_mentions_operators(self):
        compat = MusicalCompatibility.between(Symbol("ω"), Symbol("π"))
        assert "ω" in compat.name and "π" in compat.name

    def test_frozen(self):
        """Dataclass is frozen, mutating attributes must fail."""
        compat = MusicalCompatibility.between(Symbol("ω"), Symbol("π"))
        with pytest.raises(Exception):
            compat.omega = Symbol("ω2")  # type: ignore[misc]

    def test_musical_definitions_quartet(self):
        compat = MusicalCompatibility.between(Symbol("ω"), Symbol("π"))
        defs = compat.musical_definitions()
        assert len(defs) == 4
        kinds = {type(d).__name__ for d in defs}
        assert kinds == {
            "IotaFlatDefinition",
            "ArgNegLinearityDefinition",
            "MusicalCompatibilityDefinition",
            "MusicalCompatibilityBilinearDefinition",
        }

    def test_as_definition_returns_singleton_rule(self):
        compat = MusicalCompatibility.between(Symbol("ω"), Symbol("π"))
        single = compat.as_definition()
        assert isinstance(single, MusicalCompatibilityDefinition)
        assert single.compatibility is compat


# --------------------------------------------------------------------- #
# Rewrite rules                                                          #
# --------------------------------------------------------------------- #


class TestMusicalCompatibilityDefinition:
    def test_forward_composition_rewrites(self):
        omega, pi, alpha = Symbol("ω"), Symbol("π"), Symbol("α")
        compat = MusicalCompatibility.between(omega, pi)
        rule = compat.as_definition()
        expr = Act(compat.flat, Act(compat.sharp, alpha))
        assert rule.matches(expr)
        assert rule.rewrite(expr) == alpha

    def test_reverse_composition_rewrites(self):
        """``π^♯ ∘ ω^♭ = id``, the dual direction fires on the same rule."""
        omega, pi, X = Symbol("ω"), Symbol("π"), Symbol("X")
        compat = MusicalCompatibility.between(omega, pi)
        rule = compat.as_definition()
        expr = Act(compat.sharp, Act(compat.flat, X))
        assert rule.matches(expr)
        assert rule.rewrite(expr) == X

    def test_mismatched_pair_does_not_match(self):
        """Compat of (ω1, π1) must not fire on (ω2, π2)."""
        compat = MusicalCompatibility.between(Symbol("ω1"), Symbol("π1"))
        other_flat = flat(Symbol("ω2"))
        other_sharp = sharp(Symbol("π2"))
        rule = compat.as_definition()
        expr = Act(other_flat, Act(other_sharp, Symbol("α")))
        assert not rule.matches(expr)

    def test_rejects_non_compatibility(self):
        with pytest.raises(TypeError):
            MusicalCompatibilityDefinition("not-a-compat")  # type: ignore[arg-type]


class TestIotaFlatDefinition:
    def test_rewrites_iota_to_flat(self):
        """``ι_X ω → ω^♭(X)`` when ω matches the compatibility's form."""
        omega, pi, X = Symbol("ω"), Symbol("π"), Symbol("X")
        compat = MusicalCompatibility.between(omega, pi)
        rule = IotaFlatDefinition(compat)
        iota_X = interior(X)
        expr = Act(iota_X, omega)
        assert rule.matches(expr)
        out = rule.rewrite(expr)
        assert out == Act(compat.flat, X)

    def test_ignores_other_forms(self):
        omega, pi = Symbol("ω"), Symbol("π")
        compat = MusicalCompatibility.between(omega, pi)
        rule = IotaFlatDefinition(compat)
        other = Symbol("ω_other")
        expr = Act(interior(Symbol("X")), other)
        assert not rule.matches(expr)

    def test_ignores_non_iota_op(self):
        omega, pi = Symbol("ω"), Symbol("π")
        compat = MusicalCompatibility.between(omega, pi)
        rule = IotaFlatDefinition(compat)
        # Generic Derivation, not an InteriorProduct.
        D = Derivation("D", degree=-1)
        expr = Act(D, omega)
        assert not rule.matches(expr)


class TestArgNegLinearityDefinition:
    def test_pulls_neg_outward(self):
        rule = ArgNegLinearityDefinition()
        D = Derivation("D", degree=0)
        x = Symbol("x")
        expr = Act(D, Neg(x))
        assert rule.matches(expr)
        assert rule.rewrite(expr) == Neg(Act(D, x))

    def test_ignores_non_neg_arg(self):
        rule = ArgNegLinearityDefinition()
        D = Derivation("D", degree=0)
        expr = Act(D, Symbol("x"))
        assert not rule.matches(expr)

    def test_ignores_non_derivation_op(self):
        """Safety: rule only fires when op is a Derivation, arbitrary
        Exprs in the op slot aren't guaranteed to be linear."""
        rule = ArgNegLinearityDefinition()
        expr = Act(Symbol("op"), Neg(Symbol("x")))
        assert not rule.matches(expr)


# --------------------------------------------------------------------- #
# Musical bilinear, ω(π♯α, π♯β) = π(α, β)                                #
# --------------------------------------------------------------------- #


class TestMusicalCompatibilityBilinear:
    def _compat(self):
        omega = Symbol("ω")
        pi = Symbol("π")
        return MusicalCompatibility.between(omega, pi)

    def test_matches_paired_sharp_args(self):
        compat = self._compat()
        rule = MusicalCompatibilityBilinearDefinition(compat)
        alpha, beta = Symbol("α"), Symbol("β")
        expr = multi_eval(
            compat.omega,
            Act(compat.sharp, alpha),
            Act(compat.sharp, beta),
        )
        assert rule.matches(expr)

    def test_no_match_wrong_outer_form(self):
        compat = self._compat()
        rule = MusicalCompatibilityBilinearDefinition(compat)
        other = Symbol("ω₂")
        alpha, beta = Symbol("α"), Symbol("β")
        expr = multi_eval(
            other,
            Act(compat.sharp, alpha),
            Act(compat.sharp, beta),
        )
        assert not rule.matches(expr)

    def test_no_match_wrong_sharp_op(self):
        compat = self._compat()
        rule = MusicalCompatibilityBilinearDefinition(compat)
        other_sharp = Sharp(Symbol("π'"))
        alpha, beta = Symbol("α"), Symbol("β")
        expr = multi_eval(
            compat.omega,
            Act(compat.sharp, alpha),
            Act(other_sharp, beta),
        )
        assert not rule.matches(expr)

    def test_no_match_arity_three(self):
        compat = self._compat()
        rule = MusicalCompatibilityBilinearDefinition(compat)
        alpha, beta, gamma = Symbol("α"), Symbol("β"), Symbol("γ")
        expr = multi_eval(
            compat.omega,
            Act(compat.sharp, alpha),
            Act(compat.sharp, beta),
            Act(compat.sharp, gamma),
        )
        assert not rule.matches(expr)

    def test_no_match_when_arg_is_not_act(self):
        compat = self._compat()
        rule = MusicalCompatibilityBilinearDefinition(compat)
        alpha = Symbol("α")
        X = Symbol("X")  # bare VF, not Act(sharp, _)
        expr = multi_eval(compat.omega, Act(compat.sharp, alpha), X)
        assert not rule.matches(expr)

    def test_no_match_covector_slot_kind(self):
        # The bilinear identity is a vector-slot statement; covector-
        # slot arguments would mean a different evaluation contract.
        compat = self._compat()
        rule = MusicalCompatibilityBilinearDefinition(compat)
        alpha, beta = Symbol("α"), Symbol("β")
        expr = multi_eval(
            compat.omega,
            Act(compat.sharp, alpha),
            Act(compat.sharp, beta),
            slot_kind="covector",
        )
        assert not rule.matches(expr)

    def test_rewrite_to_pi_on_covectors(self):
        compat = self._compat()
        rule = MusicalCompatibilityBilinearDefinition(compat)
        alpha, beta = Symbol("α"), Symbol("β")
        expr = multi_eval(
            compat.omega,
            Act(compat.sharp, alpha),
            Act(compat.sharp, beta),
        )
        out = rule.rewrite(expr)
        assert out == multi_eval(
            compat.pi, alpha, beta, slot_kind="covector"
        )

    def test_rewrite_preserves_alternating_flag(self):
        compat = self._compat()
        rule = MusicalCompatibilityBilinearDefinition(compat)
        alpha, beta = Symbol("α"), Symbol("β")
        expr = multi_eval(
            compat.omega,
            Act(compat.sharp, alpha),
            Act(compat.sharp, beta),
            alternating=False,
        )
        out = rule.rewrite(expr)
        assert out.alternating is False
        assert out.slot_kind == "covector"

    def test_engine_rewrites(self):
        compat = self._compat()
        engine = ExpansionEngine(
            [MusicalCompatibilityBilinearDefinition(compat)]
        )
        alpha, beta = Symbol("α"), Symbol("β")
        expr = multi_eval(
            compat.omega,
            Act(compat.sharp, alpha),
            Act(compat.sharp, beta),
        )
        out, steps = engine.expand(expr)
        assert out == multi_eval(
            compat.pi, alpha, beta, slot_kind="covector"
        )
        assert len(steps) == 1

    def test_compatibility_bundles_bilinear_in_definitions(self):
        compat = self._compat()
        defs = compat.musical_definitions()
        assert any(
            isinstance(d, MusicalCompatibilityBilinearDefinition) for d in defs
        )
