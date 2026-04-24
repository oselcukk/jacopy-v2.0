"""Tests for musical isomorphisms ``♭``, ``♯`` and their compatibility."""

import pytest

from gradalg.algebra.derivation import Act, Derivation
from gradalg.calculus.interior import interior
from gradalg.calculus.musical import (
    ArgNegLinearityDefinition,
    Flat,
    IotaFlatDefinition,
    MusicalCompatibility,
    MusicalCompatibilityDefinition,
    Sharp,
    flat,
    sharp,
)
from gradalg.core.expr import Integer, Neg, Symbol
from gradalg.core.symbolic_degree import Degree


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
        """Dataclass is frozen — mutating attributes must fail."""
        compat = MusicalCompatibility.between(Symbol("ω"), Symbol("π"))
        with pytest.raises(Exception):
            compat.omega = Symbol("ω2")  # type: ignore[misc]

    def test_musical_definitions_triplet(self):
        compat = MusicalCompatibility.between(Symbol("ω"), Symbol("π"))
        defs = compat.musical_definitions()
        assert len(defs) == 3
        kinds = {type(d).__name__ for d in defs}
        assert kinds == {
            "IotaFlatDefinition",
            "ArgNegLinearityDefinition",
            "MusicalCompatibilityDefinition",
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
        """``π^♯ ∘ ω^♭ = id`` — the dual direction fires on the same rule."""
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
        """Safety: rule only fires when op is a Derivation — arbitrary
        Exprs in the op slot aren't guaranteed to be linear."""
        rule = ArgNegLinearityDefinition()
        expr = Act(Symbol("op"), Neg(Symbol("x")))
        assert not rule.matches(expr)
