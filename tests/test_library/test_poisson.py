"""Tests for ``jacopy.library.poisson``."""

from __future__ import annotations

import pytest

from jacopy.algebra.derivation import Act
from jacopy.brackets.base import BracketApply
from jacopy.brackets.derived import DerivedBracket, VanishingCondition
from jacopy.brackets.koszul import KoszulBracket
from jacopy.brackets.schouten import sn
from jacopy.calculus.hamiltonian_vf import HamiltonianVectorField
from jacopy.calculus.musical import Sharp
from jacopy.core.expr import Sum, Symbol
from jacopy.core.properties import Graded
from jacopy.core.registry import PropertyRegistry
from jacopy.core.symbolic_degree import Degree
from jacopy.library import theorem_book
from jacopy.library.poisson import (
    THEOREM_POISSON_JACOBI,
    THEOREM_POISSON_KOSZUL_EQUIVALENCE,
    THEOREM_POISSON_KOSZUL_JACOBI,
    PoissonBracket,
    poisson_bracket,
)
from jacopy.proof.chain import ProofChain


# --------------------------------------------------------------------- #
# Fixtures                                                               #
# --------------------------------------------------------------------- #


@pytest.fixture
def registry():
    r = PropertyRegistry()
    pi = Symbol("π")
    f = Symbol("f")
    g = Symbol("g")
    h = Symbol("h")
    r.declare(pi, Graded(degree=1))
    r.declare(f, Graded(degree=-1))
    r.declare(g, Graded(degree=-1))
    r.declare(h, Graded(degree=-1))
    return r


# --------------------------------------------------------------------- #
# Construction                                                           #
# --------------------------------------------------------------------- #


class TestConstruction:
    def test_basic(self):
        pi = Symbol("π")
        P = PoissonBracket(pi)
        assert P.bivector is pi
        assert isinstance(P.derived, DerivedBracket)
        assert P.derived.base is sn
        assert P.derived.Q is pi

    def test_from_bivector_classmethod(self):
        pi = Symbol("π")
        P = PoissonBracket.from_bivector(pi)
        assert P.bivector is pi

    def test_factory(self):
        pi = Symbol("π")
        P = poisson_bracket(pi)
        assert isinstance(P, PoissonBracket)
        assert P.bivector is pi

    def test_default_name(self):
        pi = Symbol("π")
        P = PoissonBracket(pi)
        assert "π" in P.name

    def test_custom_name(self):
        pi = Symbol("π")
        P = PoissonBracket(pi, name="{·,·}_myPoisson")
        assert P.name == "{·,·}_myPoisson"

    def test_rejects_non_expr(self):
        with pytest.raises(TypeError, match="Expr"):
            PoissonBracket("π")  # type: ignore[arg-type]

    def test_derived_bracket_degree_is_minus_one(self):
        """``|{·,·}_π| = |π| − 2 = 1 − 2 = −1`` in the SN grading."""
        pi = Symbol("π")
        P = PoissonBracket(pi)
        assert P.derived.degree == Degree.const(-1)


# --------------------------------------------------------------------- #
# Three equivalent views                                                #
# --------------------------------------------------------------------- #


class TestViews:
    def test_expand_delegates_to_derived(self, registry):
        pi = Symbol("π")
        f, g = Symbol("f"), Symbol("g")
        P = PoissonBracket(pi)
        direct = P.expand(f, g, registry)
        via_db = P.derived.expand(f, g, registry)
        assert direct == via_db

    def test_via_hamiltonian_shape(self):
        """``{f, g}_π = X_f(g)``, an :class:`Act` of ``X_f`` on ``g``."""
        pi = Symbol("π")
        f, g = Symbol("f"), Symbol("g")
        P = PoissonBracket(pi)
        out = P.via_hamiltonian(f, g)
        assert isinstance(out, Act)
        assert isinstance(out.op, HamiltonianVectorField)
        assert out.op.function is f
        assert out.op.bivector is pi
        assert out.arg is g

    def test_hamiltonian_vf(self):
        pi = Symbol("π")
        f = Symbol("f")
        P = PoissonBracket(pi)
        Xf = P.hamiltonian_vf(f)
        assert isinstance(Xf, HamiltonianVectorField)
        assert Xf.function is f
        assert Xf.bivector is pi

    def test_via_hamiltonian_rejects_non_expr(self):
        P = PoissonBracket(Symbol("π"))
        with pytest.raises(TypeError, match="Expr"):
            P.via_hamiltonian("f", Symbol("g"))  # type: ignore[arg-type]

    def test_bivector_eval_shape(self):
        """``{f, g}_π = π(df, dg)``, alternating covector MultiEval."""
        from jacopy.calculus.exterior_d import d as default_d
        from jacopy.core.multi_eval import MultiEval

        pi = Symbol("π")
        f, g = Symbol("f"), Symbol("g")
        P = PoissonBracket(pi)
        out = P.bivector_eval(f, g)
        assert isinstance(out, MultiEval)
        assert out.head is pi
        assert out.alternating is True
        assert out.slot_kind == "covector"
        assert out.args == (Act(default_d, f), Act(default_d, g))

    def test_bivector_eval_repeat_collapses_under_engine(self):
        """``π(df, df) → 0`` via the alternating repeat-arg rule."""
        from jacopy.calculus.multi_eval_axioms import (
            MultiEvalRepeatArgZeroDefinition,
        )
        from jacopy.core.expr import Integer
        from jacopy.proof.expansion import ExpansionEngine

        pi = Symbol("π")
        f = Symbol("f")
        P = PoissonBracket(pi)
        engine = ExpansionEngine([MultiEvalRepeatArgZeroDefinition()])
        out, _ = engine.expand(P.bivector_eval(f, f))
        assert out == Integer(0)

    def test_bivector_eval_swap_introduces_sign(self):
        """``π(dg, df) → -π(df, dg)`` via the alternating canonicaliser."""
        from jacopy.calculus.multi_eval_axioms import (
            MultiEvalAlternatingNormalDefinition,
        )
        from jacopy.core.expr import Neg
        from jacopy.proof.expansion import ExpansionEngine

        pi = Symbol("π")
        f, g = Symbol("f"), Symbol("g")
        P = PoissonBracket(pi)
        engine = ExpansionEngine([MultiEvalAlternatingNormalDefinition()])
        # repr("d(g)") > repr("d(f)") so {g, f} is out-of-order and
        # the rule swaps it.
        out, _ = engine.expand(P.bivector_eval(g, f))
        assert out == Neg(P.bivector_eval(f, g))

    def test_bivector_eval_uses_d_override(self):
        from jacopy.calculus.exterior_d import ExteriorDerivative
        from jacopy.core.multi_eval import MultiEval

        pi = Symbol("π")
        f, g = Symbol("f"), Symbol("g")
        d_E = ExteriorDerivative(name="d_E")
        P = PoissonBracket(pi)
        out = P.bivector_eval(f, g, d=d_E)
        assert isinstance(out, MultiEval)
        assert out.args == (Act(d_E, f), Act(d_E, g))

    def test_bivector_eval_rejects_non_expr(self):
        P = PoissonBracket(Symbol("π"))
        with pytest.raises(TypeError, match="Expr"):
            P.bivector_eval("f", Symbol("g"))  # type: ignore[arg-type]


# --------------------------------------------------------------------- #
# Jacobi                                                                 #
# --------------------------------------------------------------------- #


class TestJacobi:
    def test_jacobi_obstruction_matches_sn_self_bracket(self, registry):
        """The Poisson Jacobi obstruction is ``[π, π]_SN``, for atomic
        ``π`` it stays opaque."""
        pi = Symbol("π")
        P = PoissonBracket(pi)
        assert P.jacobi_obstruction(registry) == sn.self_bracket(pi, registry)

    def test_jacobi_condition_is_vanishing_condition(self, registry):
        pi = Symbol("π")
        P = PoissonBracket(pi)
        cond = P.jacobi_condition(registry)
        assert isinstance(cond, VanishingCondition)
        assert cond.obstruction == sn.self_bracket(pi, registry)
        assert "Poisson" in cond.name

    def test_prove_jacobi_reduction_produces_theorem_step(self, registry):
        """The reduction chain's first step cites the Derived Bracket
        Theorem and rewrites the triple Jacobi sum to the raw
        obstruction ``[π, π]_SN``."""
        pi = Symbol("π")
        f, g, h = Symbol("f"), Symbol("g"), Symbol("h")
        P = PoissonBracket(pi)
        chain = P.prove_jacobi_reduction(f, g, h, registry=registry)
        assert isinstance(chain, ProofChain)
        assert len(chain) >= 1
        first = chain.steps[0]
        assert first.rule == "DerivedBracketTheorem"
        assert first.provenance_tag == "theorem"
        assert isinstance(first.after, BracketApply)
        assert first.after.bracket is sn

    def test_prove_jacobi_reduction_final_is_obstruction(self, registry):
        """Chain ends on ``[π, π]_SN``, not on zero, discharging the
        obstruction is the caller's job (the Poisson hypothesis)."""
        pi = Symbol("π")
        f, g, h = Symbol("f"), Symbol("g"), Symbol("h")
        P = PoissonBracket(pi)
        chain = P.prove_jacobi_reduction(f, g, h, registry=registry)
        assert chain.final == P.jacobi_obstruction(registry)


# --------------------------------------------------------------------- #
# Seeded theorem                                                         #
# --------------------------------------------------------------------- #


class TestSeededTheorem:
    def test_theorem_registered(self):
        """Importing :mod:`jacopy.library.poisson` seeds
        ``poisson_jacobi`` into the package-wide theorem book."""
        assert "poisson_jacobi" in theorem_book
        assert theorem_book.get("poisson_jacobi") is THEOREM_POISSON_JACOBI

    def test_theorem_from_axioms(self):
        thm = THEOREM_POISSON_JACOBI
        assert thm.name == "poisson_jacobi"
        assert "Derived Bracket Theorem" in thm.from_axioms
        assert any("[π, π]_SN" in ax for ax in thm.from_axioms)

    def test_theorem_proof_is_proofchain(self):
        thm = THEOREM_POISSON_JACOBI
        assert isinstance(thm.proof, ProofChain)
        assert len(thm.proof) >= 1
        assert thm.proof.steps[0].rule == "DerivedBracketTheorem"


# --------------------------------------------------------------------- #
# Stage B.2, form-level (Koszul) view                                  #
# --------------------------------------------------------------------- #


class TestKoszulViews:
    def test_sharp_built_from_bivector(self):
        pi = Symbol("π")
        P = PoissonBracket(pi)
        assert isinstance(P.sharp, Sharp)
        assert P.sharp.bivector is pi

    def test_koszul_derived_uses_sharp_as_anchor(self):
        pi = Symbol("π")
        P = PoissonBracket(pi)
        D = P.koszul_derived
        assert isinstance(D, DerivedBracket)
        assert D.acting_on is P.sharp
        assert D.base is sn
        assert D.Q is pi

    def test_koszul_classical_uses_sharp_as_anchor(self):
        """Classical :class:`KoszulBracket` built with the manifold's
        ``Sharp(π)``, the relaxed anchor type (``Derivation`` rather
        than strictly ``Anchor``) is what unlocks this path."""
        pi = Symbol("π")
        P = PoissonBracket(pi)
        K = P.koszul_classical
        assert isinstance(K, KoszulBracket)
        assert K.anchor is P.sharp

    def test_koszul_expand_shape_is_three_term_sum(self, registry):
        """Expansion should be a :class:`Sum` of three Koszul pieces,
        ``L_{π^♯(α)} β``, ``−L_{π^♯(β)} α``, ``−d⟨π^♯(α), β⟩``. We
        don't probe the exact operator identities beyond counting the
        top-level summands; the structural agreement with
        :attr:`koszul_classical` is covered below."""
        pi = Symbol("π")
        alpha, beta = Symbol("α"), Symbol("β")
        registry.declare(alpha, Graded(degree=1))
        registry.declare(beta, Graded(degree=1))
        P = PoissonBracket(pi)
        out = P.koszul_expand(alpha, beta, registry)
        assert isinstance(out, Sum)
        assert len(out.children) == 3

    def test_koszul_expand_matches_classical_structurally(self, registry):
        """The two views share the same ``Sharp(π)`` anchor by
        construction, so their Exprs are *structurally* equal, this
        is what makes :meth:`prove_koszul_equivalence` close in one
        reflexive step."""
        pi = Symbol("π")
        alpha, beta = Symbol("α"), Symbol("β")
        registry.declare(alpha, Graded(degree=1))
        registry.declare(beta, Graded(degree=1))
        P = PoissonBracket(pi)
        derived_out = P.koszul_expand(alpha, beta, registry)
        classical_out = P.koszul_classical.expand(alpha, beta, registry)
        assert derived_out == classical_out


class TestProveKoszulEquivalence:
    def test_returns_proof_chain(self, registry):
        pi = Symbol("π")
        alpha, beta = Symbol("α"), Symbol("β")
        registry.declare(alpha, Graded(degree=1))
        registry.declare(beta, Graded(degree=1))
        P = PoissonBracket(pi)
        chain = P.prove_koszul_equivalence(alpha, beta, registry=registry)
        assert isinstance(chain, ProofChain)
        assert len(chain) >= 1

    def test_closes_in_one_reflexive_step(self, registry):
        """Both Koszul views emit the same Expr, no rewrites needed,
        :func:`prove_equivalence` reports a single ``reflexive`` step."""
        pi = Symbol("π")
        alpha, beta = Symbol("α"), Symbol("β")
        registry.declare(alpha, Graded(degree=1))
        registry.declare(beta, Graded(degree=1))
        P = PoissonBracket(pi)
        chain = P.prove_koszul_equivalence(alpha, beta, registry=registry)
        assert len(chain) == 1
        assert chain.steps[0].rule == "reflexive"


class TestSeededKoszulTheorem:
    def test_theorem_registered(self):
        assert "poisson_koszul_equivalence" in theorem_book
        assert (
            theorem_book.get("poisson_koszul_equivalence")
            is THEOREM_POISSON_KOSZUL_EQUIVALENCE
        )

    def test_theorem_proof_closes_reflexively(self):
        thm = THEOREM_POISSON_KOSZUL_EQUIVALENCE
        assert isinstance(thm.proof, ProofChain)
        assert len(thm.proof) == 1
        assert thm.proof.steps[0].rule == "reflexive"

    def test_theorem_from_axioms_names_sharp_as_anchor(self):
        thm = THEOREM_POISSON_KOSZUL_EQUIVALENCE
        assert any("Sharp" in ax or "π^♯" in ax for ax in thm.from_axioms)


# --------------------------------------------------------------------- #
# Stage B.3, form-level (Koszul) Jacobi reduction ProofChain            #
# --------------------------------------------------------------------- #


@pytest.fixture
def koszul_registry():
    r = PropertyRegistry()
    pi = Symbol("π")
    alpha = Symbol("α")
    beta = Symbol("β")
    gamma = Symbol("γ")
    r.declare(pi, Graded(degree=1))
    r.declare(alpha, Graded(degree=1))
    r.declare(beta, Graded(degree=1))
    r.declare(gamma, Graded(degree=1))
    return r


class TestKoszulJacobiCondition:
    def test_returns_vanishing_condition(self, koszul_registry):
        pi = Symbol("π")
        P = PoissonBracket(pi)
        cond = P.koszul_jacobi_condition(koszul_registry)
        assert isinstance(cond, VanishingCondition)

    def test_obstruction_matches_function_level(self, koszul_registry):
        """Form-level and function-level Jacobi share the same
        ``[π, π]_SN`` universal obstruction, ``acting_on`` doesn't
        alter ``[Q, Q]_base`` on a DerivedBracket."""
        pi = Symbol("π")
        P = PoissonBracket(pi)
        assert (
            P.koszul_jacobi_condition(koszul_registry).obstruction
            == P.jacobi_condition(koszul_registry).obstruction
        )

    def test_name_mentions_koszul(self, koszul_registry):
        pi = Symbol("π")
        P = PoissonBracket(pi)
        cond = P.koszul_jacobi_condition(koszul_registry)
        assert "Koszul" in cond.name


class TestProveKoszulJacobiReduction:
    def test_returns_proof_chain(self, koszul_registry):
        pi = Symbol("π")
        alpha, beta, gamma = Symbol("α"), Symbol("β"), Symbol("γ")
        P = PoissonBracket(pi)
        chain = P.prove_koszul_jacobi_reduction(
            alpha, beta, gamma, registry=koszul_registry
        )
        assert isinstance(chain, ProofChain)
        assert len(chain) >= 1

    def test_first_step_cites_derived_bracket_theorem(self, koszul_registry):
        pi = Symbol("π")
        alpha, beta, gamma = Symbol("α"), Symbol("β"), Symbol("γ")
        P = PoissonBracket(pi)
        chain = P.prove_koszul_jacobi_reduction(
            alpha, beta, gamma, registry=koszul_registry
        )
        first = chain.steps[0]
        assert first.rule == "DerivedBracketTheorem"
        assert first.provenance_tag == "theorem"
        assert isinstance(first.after, BracketApply)
        assert first.after.bracket is sn

    def test_final_is_shared_obstruction(self, koszul_registry):
        """The chain terminates at the same ``[π, π]_SN`` the
        function-level reduction lands on, one hypothesis discharges
        both views."""
        pi = Symbol("π")
        alpha, beta, gamma = Symbol("α"), Symbol("β"), Symbol("γ")
        f, g, h = Symbol("f"), Symbol("g"), Symbol("h")
        reg = koszul_registry
        reg.declare(f, Graded(degree=-1))
        reg.declare(g, Graded(degree=-1))
        reg.declare(h, Graded(degree=-1))
        P = PoissonBracket(pi)
        koszul_chain = P.prove_koszul_jacobi_reduction(
            alpha, beta, gamma, registry=reg
        )
        fn_chain = P.prove_jacobi_reduction(f, g, h, registry=reg)
        assert koszul_chain.final == fn_chain.final

    def test_first_step_starts_on_cyclic_sum(self, koszul_registry):
        """The reduction begins on the cyclic Jacobi Sum built from the
        form-level derived bracket, not the function-level one, so the
        inner BracketApply nodes carry ``koszul_derived``'s name."""
        pi = Symbol("π")
        alpha, beta, gamma = Symbol("α"), Symbol("β"), Symbol("γ")
        P = PoissonBracket(pi)
        chain = P.prove_koszul_jacobi_reduction(
            alpha, beta, gamma, registry=koszul_registry
        )
        expected_start = P.koszul_derived.graded_jacobi_obstruction(
            alpha, beta, gamma, koszul_registry
        )
        assert chain.steps[0].before == expected_start


class TestSeededKoszulJacobiTheorem:
    def test_theorem_registered(self):
        assert "poisson_koszul_jacobi" in theorem_book
        assert (
            theorem_book.get("poisson_koszul_jacobi")
            is THEOREM_POISSON_KOSZUL_JACOBI
        )

    def test_theorem_from_axioms_includes_poisson_hypothesis(self):
        thm = THEOREM_POISSON_KOSZUL_JACOBI
        assert any("[π, π]_SN" in ax for ax in thm.from_axioms)
        assert "Derived Bracket Theorem" in thm.from_axioms

    def test_theorem_proof_has_derived_bracket_step(self):
        thm = THEOREM_POISSON_KOSZUL_JACOBI
        assert isinstance(thm.proof, ProofChain)
        assert thm.proof.steps[0].rule == "DerivedBracketTheorem"
        assert thm.proof.steps[0].provenance_tag == "theorem"
