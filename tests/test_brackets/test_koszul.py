"""Tests for the classical Koszul bracket on 1-forms."""

import pytest

from jacopy.algebra.derivation import Act
from jacopy.brackets.derived import DerivedBracket, VanishingCondition
from jacopy.brackets.koszul import KoszulBracket
from jacopy.brackets.schouten import sn
from jacopy.calculus.anchor import Anchor
from jacopy.calculus.exterior_d import d
from jacopy.calculus.lie_derivative import lie_derivative
from jacopy.calculus.pairing import Pairing, pairing
from jacopy.core.expr import Neg, Sum, Symbol
from jacopy.core.properties import Graded
from jacopy.core.registry import PropertyRegistry
from jacopy.core.symbolic_degree import Degree
from jacopy.proof.verifier import prove_equivalence


class TestConstruction:
    def test_requires_derivation(self):
        """Non-``Derivation`` anchors rejected at construction."""
        with pytest.raises(TypeError):
            KoszulBracket("ρ")  # type: ignore[arg-type]

    def test_accepts_sharp_as_anchor(self):
        """Any :class:`Derivation` is a valid anchor, the relaxed check
        lets the musical map ``π^♯`` stand in as the anchor on a
        Poisson manifold, which is what
        :class:`jacopy.library.poisson.PoissonBracket` relies on."""
        from jacopy.calculus.musical import Sharp
        pi = Symbol("π")
        sh = Sharp(pi)
        K = KoszulBracket(sh)
        assert K.anchor is sh

    def test_default_name(self):
        K = KoszulBracket(Anchor("ρ"))
        assert K.name == "[·,·]_K"

    def test_custom_name(self):
        K = KoszulBracket(Anchor("ρ"), name="[·,·]_{Pois}")
        assert K.name == "[·,·]_{Pois}"

    def test_stores_anchor(self):
        rho = Anchor("ρ")
        K = KoszulBracket(rho)
        assert K.anchor is rho

    def test_wires_default_d_and_lie_derivative(self):
        """No Cartan operators supplied → the smooth-manifold defaults."""
        K = KoszulBracket(Anchor("ρ"))
        # Indirect check: expansion uses the default d singleton and
        # default lie_derivative factory. We probe by comparing keys.
        other = KoszulBracket(Anchor("ρ"))
        assert K == other


class TestAxiomFlags:
    def test_degree_zero(self):
        K = KoszulBracket(Anchor("ρ"))
        assert K.degree == Degree.const(0)

    def test_graded_antisymmetric(self):
        K = KoszulBracket(Anchor("ρ"))
        assert K.is_graded_antisymmetric is True

    def test_satisfies_leibniz(self):
        K = KoszulBracket(Anchor("ρ"))
        assert K.satisfies_leibniz is True

    def test_jacobi_is_conditional(self):
        """Koszul Jacobi is conditional on anchor-compat + [Q,Q]=0."""
        K = KoszulBracket(Anchor("ρ"))
        assert K.satisfies_graded_jacobi is None


class TestExpand:
    def test_shape_is_sum_of_three_terms(self):
        K = KoszulBracket(Anchor("ρ"))
        alpha, beta = Symbol("α"), Symbol("β")
        out = K.expand(alpha, beta)
        assert isinstance(out, Sum)
        assert len(out.children) == 3

    def test_first_term_is_L_rho_alpha_beta(self):
        rho = Anchor("ρ")
        K = KoszulBracket(rho)
        alpha, beta = Symbol("α"), Symbol("β")
        out = K.expand(alpha, beta)
        expected_first = Act(lie_derivative(Act(rho, alpha)), beta)
        assert out.children[0] == expected_first

    def test_second_term_is_neg_L_rho_beta_alpha(self):
        rho = Anchor("ρ")
        K = KoszulBracket(rho)
        alpha, beta = Symbol("α"), Symbol("β")
        out = K.expand(alpha, beta)
        expected_second = Neg(Act(lie_derivative(Act(rho, beta)), alpha))
        assert out.children[1] == expected_second

    def test_third_term_is_neg_d_pairing(self):
        rho = Anchor("ρ")
        K = KoszulBracket(rho)
        alpha, beta = Symbol("α"), Symbol("β")
        out = K.expand(alpha, beta)
        expected_third = Neg(Act(d, pairing(Act(rho, alpha), beta)))
        assert out.children[2] == expected_third

    def test_pairing_slot_order_is_rho_alpha_then_beta(self):
        """The convention ⟨ρα, β⟩, 1-form slot first, vector slot second."""
        rho = Anchor("ρ")
        K = KoszulBracket(rho)
        alpha, beta = Symbol("α"), Symbol("β")
        out = K.expand(alpha, beta)
        inner = out.children[2].arg.arg  # Neg(Act(d, Pairing))
        assert isinstance(inner, Pairing)
        assert inner.alpha == Act(rho, alpha)
        assert inner.X == beta

    def test_rejects_non_expr_first_operand(self):
        K = KoszulBracket(Anchor("ρ"))
        with pytest.raises(TypeError):
            K.expand("α", Symbol("β"))  # type: ignore[arg-type]

    def test_rejects_non_expr_second_operand(self):
        K = KoszulBracket(Anchor("ρ"))
        with pytest.raises(TypeError):
            K.expand(Symbol("α"), "β")  # type: ignore[arg-type]


class TestAnchorEffect:
    def test_different_anchor_gives_different_expansion(self):
        """ρ₁α and ρ₂α differ structurally, so expansions differ."""
        alpha, beta = Symbol("α"), Symbol("β")
        out1 = KoszulBracket(Anchor("ρ1")).expand(alpha, beta)
        out2 = KoszulBracket(Anchor("ρ2")).expand(alpha, beta)
        assert out1 != out2

    def test_same_anchor_gives_equal_expansion(self):
        rho = Anchor("ρ")
        alpha, beta = Symbol("α"), Symbol("β")
        out1 = KoszulBracket(rho).expand(alpha, beta)
        out2 = KoszulBracket(rho).expand(alpha, beta)
        assert out1 == out2


class TestIdentity:
    def test_two_brackets_with_same_anchor_are_equal(self):
        rho = Anchor("ρ")
        assert KoszulBracket(rho) == KoszulBracket(rho)

    def test_brackets_with_different_anchors_not_equal(self):
        assert KoszulBracket(Anchor("ρ1")) != KoszulBracket(Anchor("ρ2"))

    def test_hash_consistent_with_equality(self):
        rho = Anchor("ρ")
        a, b = KoszulBracket(rho), KoszulBracket(rho)
        assert hash(a) == hash(b)

    def test_repr_mentions_name(self):
        K = KoszulBracket(Anchor("ρ"), name="[·,·]_K")
        assert "[·,·]_K" in repr(K)


class TestAxiomObstructionApi:
    def test_antisymmetry_obstruction_builds(self):
        """The inherited obstruction helper runs on Koszul operands."""
        K = KoszulBracket(Anchor("ρ"))
        alpha, beta = Symbol("α"), Symbol("β")
        reg = PropertyRegistry()
        reg.declare(alpha, Graded(degree=1))  # 1-forms
        reg.declare(beta, Graded(degree=1))
        obs = K.graded_antisymmetry_obstruction(alpha, beta, reg)
        assert isinstance(obs, Sum)


# --------------------------------------------------------------------- #
# Classical ↔ derived equivalence                                        #
# --------------------------------------------------------------------- #


class TestKoszulDerivedEquivalence:
    def test_equivalence_chain_closes(self):
        """`KoszulBracket(ρ)` and `DerivedBracket(sn, π, acting_on=ρ)`
        produce identical Exprs on the same operands, the
        classical-vs-derived theorem reduces to reflexive closure."""
        rho = Anchor("ρ")
        pi = Symbol("π")
        alpha, beta = Symbol("α"), Symbol("β")
        K = KoszulBracket(rho)
        D = DerivedBracket(sn, pi, degree_Q=1, acting_on=rho)
        chain = prove_equivalence(K, D, alpha, beta)
        assert len(chain) == 1
        # Reflexive step, justification wording is engine-provided.
        assert chain.steps[0].rule == "reflexive"

    def test_reversed_direction_also_closes(self):
        rho = Anchor("ρ")
        pi = Symbol("π")
        alpha, beta = Symbol("α"), Symbol("β")
        K = KoszulBracket(rho)
        D = DerivedBracket(sn, pi, degree_Q=1, acting_on=rho)
        chain = prove_equivalence(D, K, alpha, beta)
        assert len(chain) == 1

    def test_mismatched_anchor_does_not_close_trivially(self):
        """Two brackets with distinct anchors emit distinct expansions;
        reflexive closure is not enough, so the chain's single step is
        non-reflexive (or the proof fails outright)."""
        from jacopy.proof.verifier import ProofFailure
        rho1, rho2 = Anchor("ρ1"), Anchor("ρ2")
        pi = Symbol("π")
        alpha, beta = Symbol("α"), Symbol("β")
        K = KoszulBracket(rho1)
        D = DerivedBracket(sn, pi, degree_Q=1, acting_on=rho2)
        # The two expansions differ structurally, no common rewrite
        # rule bridges them, so ExpandAndSimplify leaves a residual.
        with pytest.raises(ProofFailure):
            prove_equivalence(K, D, alpha, beta)


class TestJacobiCondition:
    def test_returns_vanishing_condition(self):
        K = KoszulBracket(Anchor("π♯"))
        pi = Symbol("π")
        cond = K.jacobi_condition(pi)
        assert isinstance(cond, VanishingCondition)

    def test_obstruction_is_sn_self_bracket(self):
        """Koszul Jacobi ≡ `[π, π]_SN = 0`. Matches the
        DerivedBracket(sn, π) Jacobi obstruction exactly."""
        K = KoszulBracket(Anchor("π♯"))
        pi = Symbol("π")
        cond = K.jacobi_condition(pi)
        expected = sn.expand(pi, pi)
        assert cond.obstruction == expected

    def test_matches_derived_bracket_jacobi_obstruction(self):
        """The whole point: classical Koszul Jacobi and derived-bracket
        Jacobi on the SN side share the same condition."""
        pi = Symbol("π")
        K = KoszulBracket(Anchor("π♯"))
        D = DerivedBracket(sn, pi, degree_Q=1)
        assert K.jacobi_condition(pi).obstruction == D.jacobi_obstruction()

    def test_name_mentions_koszul(self):
        K = KoszulBracket(Anchor("π♯"))
        cond = K.jacobi_condition(Symbol("π"))
        assert "Koszul" in cond.name

    def test_rejects_non_expr_bivector(self):
        K = KoszulBracket(Anchor("π♯"))
        with pytest.raises(TypeError):
            K.jacobi_condition("π")  # type: ignore[arg-type]

    def test_is_anchor_agnostic(self):
        """Two Koszul brackets with different anchors return the same
        condition on the same generator, the condition lives on π,
        not ρ."""
        pi = Symbol("π")
        c1 = KoszulBracket(Anchor("ρ1")).jacobi_condition(pi)
        c2 = KoszulBracket(Anchor("ρ2")).jacobi_condition(pi)
        assert c1.obstruction == c2.obstruction


class TestProveJacobiReduction:
    def _setup(self):
        from jacopy.calculus.musical import Sharp
        reg = PropertyRegistry()
        pi = Symbol("π"); reg.declare(pi, Graded(degree=1))
        alpha = Symbol("α"); reg.declare(alpha, Graded(degree=1))
        beta = Symbol("β"); reg.declare(beta, Graded(degree=1))
        gamma = Symbol("γ"); reg.declare(gamma, Graded(degree=1))
        K = KoszulBracket(Sharp(pi))
        return K, pi, alpha, beta, gamma, reg

    def test_returns_proof_chain(self):
        from jacopy.proof.chain import ProofChain
        K, pi, a, b, g, reg = self._setup()
        chain = K.prove_jacobi_reduction(a, b, g, bivector=pi, registry=reg)
        assert isinstance(chain, ProofChain)

    def test_chain_is_non_empty(self):
        K, pi, a, b, g, reg = self._setup()
        chain = K.prove_jacobi_reduction(a, b, g, bivector=pi, registry=reg)
        assert len(chain) >= 1

    def test_first_step_cites_derived_bracket_theorem(self):
        K, pi, a, b, g, reg = self._setup()
        chain = K.prove_jacobi_reduction(a, b, g, bivector=pi, registry=reg)
        first = chain.steps[0]
        assert first.rule == "DerivedBracketTheorem"
        assert first.provenance_tag == "theorem"

    def test_terminal_obstruction_is_sn_self_bracket(self):
        """Final expression in the chain is ``[π, π]_SN``, the same
        universal obstruction the typed condition wraps."""
        K, pi, a, b, g, reg = self._setup()
        chain = K.prove_jacobi_reduction(a, b, g, bivector=pi, registry=reg)
        expected = K.jacobi_condition(pi, registry=reg).obstruction
        assert chain.final == expected

    def test_matches_poisson_bracket_view(self):
        """KoszulBracket.prove_jacobi_reduction with anchor=Sharp(π)
        produces the same terminal obstruction as PoissonBracket's
        Koszul-view reduction, one Poisson hypothesis closes both."""
        from jacopy.library.poisson import PoissonBracket
        K, pi, a, b, g, reg = self._setup()
        chain_K = K.prove_jacobi_reduction(a, b, g, bivector=pi, registry=reg)
        chain_P = PoissonBracket.from_bivector(pi).prove_koszul_jacobi_reduction(
            a, b, g, registry=reg,
        )
        assert chain_K.final == chain_P.final

    def test_rejects_non_expr_operands(self):
        K, pi, _, b, g, reg = self._setup()
        with pytest.raises(TypeError):
            K.prove_jacobi_reduction("α", b, g, bivector=pi, registry=reg)  # type: ignore[arg-type]

    def test_rejects_non_expr_bivector(self):
        K, _, a, b, g, reg = self._setup()
        with pytest.raises(TypeError):
            K.prove_jacobi_reduction(a, b, g, bivector="π", registry=reg)  # type: ignore[arg-type]
