"""Tests for ``jacopy.library.koszul_problem.KoszulProblem``."""

from __future__ import annotations

import pytest

from jacopy.algebra.derivation import Act, Derivation
from jacopy.brackets.base import BracketApply
from jacopy.brackets.koszul import KoszulBracket
from jacopy.brackets.schouten import sn
from jacopy.calculus.antisym_axioms import RegistryAntiSymCanonicalDefinition
from jacopy.calculus.hamiltonian_vf import HamiltonianVectorField
from jacopy.calculus.interior import InteriorProduct
from jacopy.calculus.musical import Sharp
from jacopy.calculus.sharp_axioms import (
    SharpLinearityDefinition,
    SharpOnExactDefinition,
)
from jacopy.calculus.tilde import (
    TildeDOfFunctionDefinition,
    TildeDSquaredPoissonDefinition,
    TildeExteriorDLichnerowiczDefinition,
    TildeExteriorDerivative,
    TildeInteriorProduct,
    TildeIotaOnZeroVectorDefinition,
    TildeIotaSquaredZeroDefinition,
    TildeIotaSwapDefinition,
    TildeLieDerivative,
    TildeLieMagicDefinition,
    TildeLieOnZeroVectorDefinition,
)
from jacopy.core.expr import Neg, Symbol
from jacopy.core.properties import Antisymmetric, Graded, Poisson
from jacopy.core.registry import PropertyRegistry
from jacopy.library.koszul_problem import (
    KoszulBracketExpansionDefinition,
    KoszulProblem,
)


# --------------------------------------------------------------------- #
# Fixtures                                                              #
# --------------------------------------------------------------------- #


@pytest.fixture
def pi():
    return Symbol("π")


@pytest.fixture
def omega():
    return Symbol("ω")


@pytest.fixture
def eta():
    return Symbol("η")


@pytest.fixture
def reg(pi, omega, eta):
    r = PropertyRegistry()
    r.declare(pi, Graded(degree=1))
    r.declare(omega, Graded(degree=2))
    r.declare(eta, Graded(degree=1))
    return r


# --------------------------------------------------------------------- #
# Construction                                                          #
# --------------------------------------------------------------------- #


class TestConstruction:
    def test_basic(self, pi, omega, eta, reg):
        kp = KoszulProblem(pi, (omega, eta), registry=reg)
        assert kp.pi is pi
        assert kp.forms == (omega, eta)
        assert isinstance(kp.sharp, Sharp)
        assert kp.sharp.bivector is pi
        assert isinstance(kp.koszul_bracket, KoszulBracket)

    def test_auto_declares_antisymmetric(self, pi, omega, reg):
        assert not reg.has(pi, Antisymmetric)
        KoszulProblem(pi, (omega,), registry=reg)
        assert reg.has(pi, Antisymmetric)

    def test_does_not_overwrite_existing_antisym(self, pi, omega, reg):
        prior = Antisymmetric()
        reg.declare(pi, prior)
        KoszulProblem(pi, (omega,), registry=reg)
        assert reg.get(pi, Antisymmetric) is prior

    def test_default_name(self, pi, omega, eta, reg):
        kp = KoszulProblem(pi, (omega, eta), registry=reg)
        assert "π" in kp.name

    def test_custom_name(self, pi, omega, reg):
        kp = KoszulProblem(pi, (omega,), registry=reg, name="MyKP")
        assert kp.name == "MyKP"

    def test_repr_includes_pi_and_forms(self, pi, omega, eta, reg):
        kp = KoszulProblem(pi, (omega, eta), registry=reg)
        s = repr(kp)
        assert "π" in s and "ω" in s and "η" in s

    # ---- validation --------------------------------------------- #

    def test_rejects_non_expr_pi(self, omega, reg):
        with pytest.raises(TypeError, match="pi"):
            KoszulProblem("π", (omega,), registry=reg)  # type: ignore[arg-type]

    def test_rejects_non_expr_form(self, pi, reg):
        with pytest.raises(TypeError, match="forms"):
            KoszulProblem(pi, ("ω",), registry=reg)  # type: ignore[arg-type]

    def test_rejects_empty_forms(self, pi, reg):
        with pytest.raises(ValueError, match="form"):
            KoszulProblem(pi, (), registry=reg)

    def test_rejects_missing_registry(self, pi, omega):
        with pytest.raises(TypeError, match="PropertyRegistry"):
            KoszulProblem(pi, (omega,), registry=None)  # type: ignore[arg-type]


# --------------------------------------------------------------------- #
# Engine pre-loaded with the right axioms                               #
# --------------------------------------------------------------------- #


class TestEngineAxioms:
    def _kinds(self, kp):
        return [type(d) for d in kp.engine.definitions]

    def test_engine_carries_sharp_linearity(self, pi, omega, reg):
        kp = KoszulProblem(pi, (omega,), registry=reg)
        assert SharpLinearityDefinition in self._kinds(kp)

    def test_engine_carries_sharp_on_exact(self, pi, omega, reg):
        kp = KoszulProblem(pi, (omega,), registry=reg)
        assert SharpOnExactDefinition in self._kinds(kp)

    def test_engine_carries_registry_antisym(self, pi, omega, reg):
        kp = KoszulProblem(pi, (omega,), registry=reg)
        assert RegistryAntiSymCanonicalDefinition in self._kinds(kp)

    def test_engine_carries_bracket_expansion_rule(self, pi, omega, reg):
        kp = KoszulProblem(pi, (omega,), registry=reg)
        assert KoszulBracketExpansionDefinition in self._kinds(kp)


# --------------------------------------------------------------------- #
# anchor / bracket / bracket_expansion                                  #
# --------------------------------------------------------------------- #


class TestHelpers:
    def test_anchor_returns_act_of_sharp(self, pi, omega, reg):
        kp = KoszulProblem(pi, (omega,), registry=reg)
        out = kp.anchor(omega)
        assert isinstance(out, Act)
        assert out.op is kp.sharp
        assert out.arg is omega

    def test_bracket_is_inert_bracket_apply(self, pi, omega, eta, reg):
        kp = KoszulProblem(pi, (omega, eta), registry=reg)
        out = kp.bracket(omega, eta)
        assert isinstance(out, BracketApply)
        assert out.bracket is kp.koszul_bracket
        assert out.a is omega and out.b is eta

    def test_bracket_expansion_is_three_term_sum(self, pi, omega, eta, reg):
        from jacopy.core.expr import Sum

        kp = KoszulProblem(pi, (omega, eta), registry=reg)
        out = kp.bracket_expansion(omega, eta)
        assert isinstance(out, Sum)
        assert len(out.children) == 3

    def test_anchor_rejects_non_expr(self, pi, omega, reg):
        kp = KoszulProblem(pi, (omega,), registry=reg)
        with pytest.raises(TypeError):
            kp.anchor("ω")  # type: ignore[arg-type]

    def test_bracket_rejects_non_expr(self, pi, omega, reg):
        kp = KoszulProblem(pi, (omega,), registry=reg)
        with pytest.raises(TypeError):
            kp.bracket("ω", omega)  # type: ignore[arg-type]
        with pytest.raises(TypeError):
            kp.bracket(omega, "η")  # type: ignore[arg-type]


# --------------------------------------------------------------------- #
# KoszulBracketExpansionDefinition rule                                 #
# --------------------------------------------------------------------- #


class TestKoszulBracketExpansionDefinition:
    def test_matches_own_bracket(self, pi, omega, eta, reg):
        kp = KoszulProblem(pi, (omega, eta), registry=reg)
        node = kp.bracket(omega, eta)
        assert kp.bracket_expansion_rule.matches(node)

    def test_skips_other_bracket(self, pi, omega, eta, reg):
        kp = KoszulProblem(pi, (omega, eta), registry=reg)
        other_sharp = Sharp(Symbol("π2"))
        other = KoszulBracket(other_sharp)
        unrelated = BracketApply(other, omega, eta)
        assert not kp.bracket_expansion_rule.matches(unrelated)

    def test_rewrites_to_cartan_expansion(self, pi, omega, eta, reg):
        kp = KoszulProblem(pi, (omega, eta), registry=reg)
        node = kp.bracket(omega, eta)
        out = kp.bracket_expansion_rule.rewrite(node)
        assert out == kp.bracket_expansion(omega, eta)

    def test_constructor_rejects_non_koszul(self):
        with pytest.raises(TypeError, match="KoszulBracket"):
            KoszulBracketExpansionDefinition("not-a-bracket")  # type: ignore[arg-type]


# --------------------------------------------------------------------- #
# End-to-end: engine fires bracket-expansion rule                       #
# --------------------------------------------------------------------- #


class TestEndToEnd:
    def test_engine_expands_bracket_node(self, pi, omega, eta, reg):
        """Pipe a [ω, η]_K node through the engine and watch it expand."""
        kp = KoszulProblem(pi, (omega, eta), registry=reg)
        node = kp.bracket(omega, eta)
        result, steps = kp.engine.expand(node)
        # At minimum the bracket-expansion rule fires.
        assert any(
            isinstance(s.rule, str)
            and "L_ρα" in s.rule or "[α, β]_K" in s.rule
            for s in steps
        ) or result != node


# --------------------------------------------------------------------- #
# Faz 14.C, multivectors auto-declare                                  #
# --------------------------------------------------------------------- #


class TestMultivectorsParam:
    def test_default_is_empty_tuple(self, pi, omega, reg):
        kp = KoszulProblem(pi, (omega,), registry=reg)
        assert kp.multivectors == ()

    def test_auto_declares_graded(self, pi, omega, reg):
        f = Symbol("f")
        X = Symbol("X")
        assert not reg.has(f, Graded)
        assert not reg.has(X, Graded)
        kp = KoszulProblem(
            pi, (omega,), registry=reg, multivectors=((f, 0), (X, 1))
        )
        assert reg.get(f, Graded).degree == 0
        assert reg.get(X, Graded).degree == 1
        assert kp.multivectors == ((f, 0), (X, 1))

    def test_does_not_overwrite_existing_graded(self, pi, omega, reg):
        f = Symbol("f")
        prior = Graded(degree=0)
        reg.declare(f, prior)
        KoszulProblem(pi, (omega,), registry=reg, multivectors=((f, 5),))
        # Pre-existing Graded survives the auto-declare attempt.
        assert reg.get(f, Graded) is prior

    def test_rejects_non_pair_entry(self, pi, omega, reg):
        with pytest.raises(TypeError, match="multivectors"):
            KoszulProblem(
                pi, (omega,), registry=reg, multivectors=(Symbol("f"),)  # type: ignore[arg-type]
            )

    def test_rejects_non_expr_operand(self, pi, omega, reg):
        with pytest.raises(TypeError, match="multivectors"):
            KoszulProblem(
                pi, (omega,), registry=reg,
                multivectors=(("f", 0),),  # type: ignore[arg-type]
            )

    def test_rejects_non_int_degree(self, pi, omega, reg):
        with pytest.raises(TypeError, match="multivectors"):
            KoszulProblem(
                pi, (omega,), registry=reg,
                multivectors=((Symbol("f"), "0"),),  # type: ignore[arg-type]
            )

    def test_rejects_negative_degree(self, pi, omega, reg):
        with pytest.raises(TypeError, match="non-negative"):
            KoszulProblem(
                pi, (omega,), registry=reg,
                multivectors=((Symbol("f"), -1),),
            )


# --------------------------------------------------------------------- #
# Faz 14.C, tilde axiom registration                                   #
# --------------------------------------------------------------------- #


class TestTildeAxiomRegistration:
    def _kinds(self, kp):
        return [type(d) for d in kp.engine.definitions]

    def test_engine_carries_tilde_swap(self, pi, omega, reg):
        kp = KoszulProblem(pi, (omega,), registry=reg)
        assert TildeIotaSwapDefinition in self._kinds(kp)

    def test_engine_carries_tilde_d_lichnerowicz(self, pi, omega, reg):
        kp = KoszulProblem(pi, (omega,), registry=reg)
        assert TildeExteriorDLichnerowiczDefinition in self._kinds(kp)

    def test_engine_carries_tilde_lie_magic(self, pi, omega, reg):
        kp = KoszulProblem(pi, (omega,), registry=reg)
        assert TildeLieMagicDefinition in self._kinds(kp)

    def test_tilde_d_rule_is_pi_scoped(self, pi, omega, reg):
        kp = KoszulProblem(pi, (omega,), registry=reg)
        assert kp.tilde_d_rule.pi is pi

    def test_tilde_lie_rule_is_pi_scoped(self, pi, omega, reg):
        kp = KoszulProblem(pi, (omega,), registry=reg)
        assert kp.tilde_lie_rule.pi is pi


# --------------------------------------------------------------------- #
# Faz 14.C, tilde factory methods                                      #
# --------------------------------------------------------------------- #


class TestTildeFactories:
    def test_tilde_d_returns_pi_bound_head(self, pi, omega, reg):
        kp = KoszulProblem(pi, (omega,), registry=reg)
        td = kp.tilde_d()
        assert isinstance(td, TildeExteriorDerivative)
        assert td.bivector is pi
        assert td == TildeExteriorDerivative(pi)

    def test_tilde_interior_returns_form_indexed_head(self, pi, omega, reg):
        kp = KoszulProblem(pi, (omega,), registry=reg)
        ti = kp.tilde_interior(omega)
        assert isinstance(ti, TildeInteriorProduct)
        assert ti.form is omega
        assert ti == TildeInteriorProduct(omega)

    def test_tilde_lie_returns_form_and_pi_bound_head(self, pi, omega, reg):
        kp = KoszulProblem(pi, (omega,), registry=reg)
        tl = kp.tilde_lie(omega)
        assert isinstance(tl, TildeLieDerivative)
        assert tl.form is omega
        assert tl.bivector is pi
        assert tl == TildeLieDerivative(omega, pi)

    def test_tilde_interior_rejects_non_expr(self, pi, omega, reg):
        kp = KoszulProblem(pi, (omega,), registry=reg)
        with pytest.raises(TypeError):
            kp.tilde_interior("ω")  # type: ignore[arg-type]

    def test_tilde_lie_rejects_non_expr(self, pi, omega, reg):
        kp = KoszulProblem(pi, (omega,), registry=reg)
        with pytest.raises(TypeError):
            kp.tilde_lie("ω")  # type: ignore[arg-type]


# --------------------------------------------------------------------- #
# Faz 14.C, assume_poisson                                             #
# --------------------------------------------------------------------- #


class TestAssumePoisson:
    def test_declares_poisson_property(self, pi, omega, reg):
        kp = KoszulProblem(pi, (omega,), registry=reg)
        assert not reg.has(pi, Poisson)
        kp.assume_poisson()
        assert reg.has(pi, Poisson)

    def test_idempotent(self, pi, omega, reg):
        kp = KoszulProblem(pi, (omega,), registry=reg)
        kp.assume_poisson()
        prior = reg.get(pi, Poisson)
        kp.assume_poisson()  # second call is a no-op
        assert reg.get(pi, Poisson) is prior

    def test_does_not_overwrite_existing_poisson(self, pi, omega, reg):
        prior = Poisson()
        reg.declare(pi, prior)
        kp = KoszulProblem(pi, (omega,), registry=reg)
        kp.assume_poisson()
        assert reg.get(pi, Poisson) is prior


# --------------------------------------------------------------------- #
# Faz 14.C, engine fires tilde rules end-to-end                        #
# --------------------------------------------------------------------- #


class TestTildeEndToEnd:
    def test_engine_expands_tilde_lie_act(self, pi, omega, reg):
        """L̃_ω X → magic → swap + Lichnerowicz, all inside KoszulProblem."""
        X = Symbol("X")
        kp = KoszulProblem(
            pi, (omega,), registry=reg, multivectors=((X, 1),)
        )
        node = Act(kp.tilde_lie(omega), X)
        out, steps = kp.engine.expand(node)
        # Magic fires once on the outer head.
        assert any("L̃_ω V" in s.rule for s in steps)
        # Both branches of the magic Sum are unfolded.
        assert any("ι̃_ω V" in s.rule for s in steps)
        assert any("d̃ V" in s.rule for s in steps)
        # Final form has neither tilde-d nor tilde-iota at the surface.
        assert "ι̃" not in str(out)
        assert "d̃" not in str(out)

    def test_engine_expands_tilde_d_act(self, pi, omega, reg):
        X = Symbol("X")
        kp = KoszulProblem(
            pi, (omega,), registry=reg, multivectors=((X, 1),)
        )
        node = Act(kp.tilde_d(), X)
        out, _ = kp.engine.expand(node)
        assert out == BracketApply(sn, pi, X)

    def test_engine_expands_tilde_interior_act(self, pi, omega, reg):
        X = Symbol("X")
        kp = KoszulProblem(
            pi, (omega,), registry=reg, multivectors=((X, 1),)
        )
        node = Act(kp.tilde_interior(omega), X)
        out, _ = kp.engine.expand(node)
        assert out == Act(InteriorProduct(X), omega)

    def test_two_problems_with_distinct_pi_do_not_alias(
        self, pi, omega, reg
    ):
        """Two KoszulProblems on distinct π's keep their tilde-d rules apart."""
        pi2 = Symbol("π2")
        reg.declare(pi2, Graded(degree=1))
        X = Symbol("X")
        reg.declare(X, Graded(degree=1))

        kp1 = KoszulProblem(pi, (omega,), registry=reg)
        kp2 = KoszulProblem(pi2, (omega,), registry=reg)

        # kp1's engine fires only its own tilde-d on Act(d̃_π, X).
        out1, _ = kp1.engine.expand(Act(kp1.tilde_d(), X))
        out2, _ = kp2.engine.expand(Act(kp2.tilde_d(), X))
        assert out1 == BracketApply(sn, pi, X)
        assert out2 == BracketApply(sn, pi2, X)
        # kp1's rule does not match the other π's head.
        assert not kp1.tilde_d_rule.matches(Act(kp2.tilde_d(), X))


# --------------------------------------------------------------------- #
# Faz 14.D, KoszulProblem auto-registers all 5 auxiliary axioms        #
# --------------------------------------------------------------------- #


class TestTildeAuxRegistration:
    def _kinds(self, kp):
        return [type(d) for d in kp.engine.definitions]

    def test_engine_carries_iota0(self, pi, omega, reg):
        kp = KoszulProblem(pi, (omega,), registry=reg)
        assert TildeIotaOnZeroVectorDefinition in self._kinds(kp)

    def test_engine_carries_iota_squared(self, pi, omega, reg):
        kp = KoszulProblem(pi, (omega,), registry=reg)
        assert TildeIotaSquaredZeroDefinition in self._kinds(kp)

    def test_engine_carries_lie0(self, pi, omega, reg):
        kp = KoszulProblem(pi, (omega,), registry=reg)
        assert TildeLieOnZeroVectorDefinition in self._kinds(kp)

    def test_engine_carries_d_of_function(self, pi, omega, reg):
        kp = KoszulProblem(pi, (omega,), registry=reg)
        assert TildeDOfFunctionDefinition in self._kinds(kp)

    def test_engine_carries_d_squared(self, pi, omega, reg):
        kp = KoszulProblem(pi, (omega,), registry=reg)
        assert TildeDSquaredPoissonDefinition in self._kinds(kp)

    def test_aux_rules_property_returns_five(self, pi, omega, reg):
        kp = KoszulProblem(pi, (omega,), registry=reg)
        assert len(kp.tilde_aux_rules) == 5
        kinds = [type(r) for r in kp.tilde_aux_rules]
        assert kinds == [
            TildeIotaOnZeroVectorDefinition,
            TildeIotaSquaredZeroDefinition,
            TildeLieOnZeroVectorDefinition,
            TildeDOfFunctionDefinition,
            TildeDSquaredPoissonDefinition,
        ]

    def test_aux_registered_before_defining(self, pi, omega, reg):
        """Aux rules must register before their fallback defining axioms.

        Engine matches in registration order; the auxiliaries are
        specificity shortcuts, so the first matcher of each pair must
        be the auxiliary.
        """
        kp = KoszulProblem(pi, (omega,), registry=reg)
        kinds = self._kinds(kp)
        # ι̃-on-0 before swap.
        assert kinds.index(TildeIotaOnZeroVectorDefinition) < kinds.index(
            TildeIotaSwapDefinition
        )
        # L̃-on-0 before magic.
        assert kinds.index(TildeLieOnZeroVectorDefinition) < kinds.index(
            TildeLieMagicDefinition
        )
        # d̃ f shortcut before Lichnerowicz.
        assert kinds.index(TildeDOfFunctionDefinition) < kinds.index(
            TildeExteriorDLichnerowiczDefinition
        )


# --------------------------------------------------------------------- #
# Faz 14.D, engine fires the right shortcut on the right shape          #
# --------------------------------------------------------------------- #


class TestTildeAuxEndToEnd:
    def test_iota0_shortcut_in_engine(self, pi, omega, reg):
        """ι̃_ω f → 0 in 1 step (Aux-1 wins over swap)."""
        f = Symbol("f")
        kp = KoszulProblem(
            pi, (omega,), registry=reg, multivectors=((f, 0),)
        )
        from jacopy.core.expr import Zero
        out, steps = kp.engine.expand(Act(kp.tilde_interior(omega), f))
        assert out is Zero
        assert len(steps) == 1
        assert "ι̃_ω f = 0" in steps[0].rule

    def test_lie0_shortcut_in_engine(self, pi, omega, reg):
        """L̃_ω f → π^♯(ω)(f) in 1 step (Aux-3 wins over magic)."""
        f = Symbol("f")
        kp = KoszulProblem(
            pi, (omega,), registry=reg, multivectors=((f, 0),)
        )
        out, steps = kp.engine.expand(Act(kp.tilde_lie(omega), f))
        assert out == Act(Act(Sharp(pi), omega), f)
        assert len(steps) == 1
        assert "L̃_ω f" in steps[0].rule

    def test_d_func_shortcut_in_engine(self, pi, omega, reg):
        """d̃ f → -X_f via Aux-4 + SharpOnExact (2 steps total).

        The Aux-4 short-circuit produces ``-π^♯(df)`` in one step;
        the bundled :class:`SharpOnExactDefinition` then names that
        composition as a Hamiltonian vector field, so the final
        transcript is two short-circuit steps instead of the multi-step
        SN expansion the bare Lichnerowicz definition would emit.
        """
        f = Symbol("f")
        kp = KoszulProblem(
            pi, (omega,), registry=reg, multivectors=((f, 0),)
        )
        out, steps = kp.engine.expand(Act(kp.tilde_d(), f))
        assert out == Neg(HamiltonianVectorField(f, bivector=pi))
        assert len(steps) == 2
        rules = [s.rule for s in steps]
        assert any("d̃ f" in r for r in rules)

    def test_assume_poisson_does_not_change_d_squared_engine_path(
        self, pi, omega, reg
    ):
        """The engine's children-first traversal beats Aux-5.

        Aux-5 matches the strict shape ``Act(d̃, Act(d̃, V))``, but the
        engine rewrites the inner ``Act(d̃, V)`` first via Lichnerowicz,
        so by the time the parent is checked the shape is already
        ``Act(d̃, [π,V]_SN)`` and Aux-5's pattern is gone. Stage E will
        either invoke Aux-5 directly or extend the engine; for now the
        engine pipeline normalises to ``[π, [π, V]_SN]_SN`` regardless
        of the Poisson flag, this test pins the current behaviour.
        """
        V = Symbol("V")
        kp = KoszulProblem(
            pi, (omega,), registry=reg, multivectors=((V, 1),)
        )
        target = Act(kp.tilde_d(), Act(kp.tilde_d(), V))
        out_unpoisson, _ = kp.engine.expand(target)
        kp.assume_poisson()
        out_poisson, _ = kp.engine.expand(target)
        assert out_unpoisson == out_poisson
