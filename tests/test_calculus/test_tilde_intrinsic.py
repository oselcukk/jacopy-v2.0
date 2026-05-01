"""Tests for tilde-calculus intrinsic axioms + engine (Faz 14.E)."""

import pytest

from jacopy.algebra.derivation import Act
from jacopy.brackets.base import BracketApply
from jacopy.brackets.koszul import KoszulBracket
from jacopy.calculus.musical import Sharp
from jacopy.calculus.tilde import (
    TildeDIntrinsicDefinition,
    TildeExteriorDerivative,
    TildeInteriorProduct,
    TildeIntrinsicFormulaMatch,
    TildeIntrinsicFormulaRecognizer,
    TildeIotaIntrinsicDefinition,
    TildeLieDerivative,
    TildeLieIntrinsicDefinition,
    prove_tilde_cartan_relation,
    tilde_intrinsic_engine,
)
from jacopy.core.expr import Integer, Neg, Symbol
from jacopy.core.multi_eval import multi_eval
from jacopy.core.properties import Graded
from jacopy.core.registry import PropertyRegistry
from jacopy.library.koszul_problem import KoszulProblem
from jacopy.proof.chain import ProofChain
from jacopy.proof.expansion import Definition, ExpansionEngine
from jacopy.proof.strategies import ProofFailure


# --------------------------------------------------------------------- #
# Helpers                                                               #
# --------------------------------------------------------------------- #


def _setup():
    """Return a fresh (registry, π, ω, η, V, sharp, koszul) bundle."""
    reg = PropertyRegistry()
    pi = Symbol("π")
    omega = Symbol("ω")
    eta = Symbol("η")
    V = Symbol("V")
    reg.declare(omega, Graded(degree=1))
    reg.declare(eta, Graded(degree=1))
    sharp = Sharp(pi)
    koszul = KoszulBracket(sharp, name="[·,·]_K[π]")
    return reg, pi, omega, eta, V, sharp, koszul


# --------------------------------------------------------------------- #
# TildeIotaIntrinsicDefinition                                          #
# --------------------------------------------------------------------- #


class TestTildeIotaIntrinsic:
    def test_is_definition(self):
        rule = TildeIotaIntrinsicDefinition()
        assert isinstance(rule, Definition)

    def test_matches_covector_slot_eval(self):
        _, _, omega, eta, V, _, _ = _setup()
        rule = TildeIotaIntrinsicDefinition()
        head = Act(TildeInteriorProduct(omega), V)
        expr = multi_eval(head, eta, slot_kind="covector")
        assert rule.matches(expr)

    def test_no_match_on_vector_slot(self):
        _, _, omega, eta, V, _, _ = _setup()
        rule = TildeIotaIntrinsicDefinition()
        head = Act(TildeInteriorProduct(omega), V)
        # Wrong slot kind, standard-side recognisers would handle this.
        expr = multi_eval(head, eta, slot_kind="vector")
        assert not rule.matches(expr)

    def test_no_match_on_plain_head(self):
        _, _, _, eta, V, _, _ = _setup()
        rule = TildeIotaIntrinsicDefinition()
        expr = multi_eval(V, eta, slot_kind="covector")
        assert not rule.matches(expr)

    def test_rewrite_absorbs_form_into_first_slot(self):
        _, _, omega, eta, V, _, _ = _setup()
        rule = TildeIotaIntrinsicDefinition()
        head = Act(TildeInteriorProduct(omega), V)
        out = rule.rewrite(multi_eval(head, eta, slot_kind="covector"))
        expected = multi_eval(V, omega, eta, slot_kind="covector")
        assert out == expected


# --------------------------------------------------------------------- #
# TildeLieIntrinsicDefinition                                           #
# --------------------------------------------------------------------- #


class TestTildeLieIntrinsic:
    def test_is_definition(self):
        _, pi, _, _, _, _, koszul = _setup()
        rule = TildeLieIntrinsicDefinition(pi, koszul)
        assert isinstance(rule, Definition)

    def test_pi_scoped_match(self):
        _, pi, omega, eta, V, _, koszul = _setup()
        rule = TildeLieIntrinsicDefinition(pi, koszul)
        head = Act(TildeLieDerivative(omega, pi), V)
        expr = multi_eval(head, eta, slot_kind="covector")
        assert rule.matches(expr)

    def test_no_match_on_other_pi(self):
        _, pi, omega, eta, V, _, koszul = _setup()
        rule = TildeLieIntrinsicDefinition(pi, koszul)
        other_pi = Symbol("π'")
        head = Act(TildeLieDerivative(omega, other_pi), V)
        expr = multi_eval(head, eta, slot_kind="covector")
        assert not rule.matches(expr)

    def test_rewrite_emits_anchor_and_bracket_terms(self):
        _, pi, omega, eta, V, sharp, koszul = _setup()
        rule = TildeLieIntrinsicDefinition(pi, koszul, sharp=sharp)
        head = Act(TildeLieDerivative(omega, pi), V)
        out = rule.rewrite(multi_eval(head, eta, slot_kind="covector"))
        # Expected: π^♯(ω)·V(η) − V([ω, η]_K)
        first = Act(
            Act(sharp, omega),
            multi_eval(V, eta, slot_kind="covector"),
        )
        bracket_term = Neg(
            multi_eval(
                V,
                BracketApply(koszul, omega, eta),
                slot_kind="covector",
            )
        )
        from jacopy.core.expr import Sum
        expected = Sum.make(first, bracket_term)
        assert out == expected


# --------------------------------------------------------------------- #
# TildeDIntrinsicDefinition                                             #
# --------------------------------------------------------------------- #


class TestTildeDIntrinsic:
    def test_is_definition(self):
        _, pi, _, _, _, _, koszul = _setup()
        rule = TildeDIntrinsicDefinition(pi, koszul)
        assert isinstance(rule, Definition)

    def test_pi_scoped_match(self):
        _, pi, _, eta, V, _, koszul = _setup()
        rule = TildeDIntrinsicDefinition(pi, koszul)
        head = Act(TildeExteriorDerivative(pi), V)
        expr = multi_eval(head, eta, slot_kind="covector")
        assert rule.matches(expr)

    def test_no_match_on_other_pi(self):
        _, pi, _, eta, V, _, koszul = _setup()
        rule = TildeDIntrinsicDefinition(pi, koszul)
        other_pi = Symbol("π'")
        head = Act(TildeExteriorDerivative(other_pi), V)
        expr = multi_eval(head, eta, slot_kind="covector")
        assert not rule.matches(expr)

    def test_arity_one_collapses_to_anchor_action(self):
        # (d̃ f)(η_0) = π^♯(η_0)·f, single term, no inner MultiEval, no bracket sum.
        _, pi, _, eta, _, sharp, koszul = _setup()
        f = Symbol("f")
        rule = TildeDIntrinsicDefinition(pi, koszul, sharp=sharp)
        head = Act(TildeExteriorDerivative(pi), f)
        out = rule.rewrite(multi_eval(head, eta, slot_kind="covector"))
        from jacopy.core.expr import Sum
        expected = Sum.make(Act(Act(sharp, eta), f))
        assert out == expected

    def test_arity_two_emits_two_anchor_terms_plus_one_bracket(self):
        # (d̃ V)(η_0, η_1) = π^♯(η_0)·V(η_1) − π^♯(η_1)·V(η_0) + V([η_0, η_1]_K)
        _, pi, _, _, V, sharp, koszul = _setup()
        eta0 = Symbol("η_0")
        eta1 = Symbol("η_1")
        rule = TildeDIntrinsicDefinition(pi, koszul, sharp=sharp)
        head = Act(TildeExteriorDerivative(pi), V)
        out = rule.rewrite(
            multi_eval(head, eta0, eta1, slot_kind="covector")
        )
        from jacopy.core.expr import Sum
        term0 = Act(Act(sharp, eta0), multi_eval(V, eta1, slot_kind="covector"))
        term1 = Neg(
            Act(Act(sharp, eta1), multi_eval(V, eta0, slot_kind="covector"))
        )
        # i=0, j=1 → (i+j) % 2 == 1 → Neg
        bracket_term = Neg(
            multi_eval(
                V,
                BracketApply(koszul, eta0, eta1),
                slot_kind="covector",
            )
        )
        expected = Sum.make(term0, term1, bracket_term)
        assert out == expected


# --------------------------------------------------------------------- #
# tilde_intrinsic_engine factory                                        #
# --------------------------------------------------------------------- #


class TestTildeIntrinsicEngineFactory:
    def test_returns_expansion_engine(self):
        _, pi, _, _, _, _, koszul = _setup()
        eng = tilde_intrinsic_engine(pi, koszul)
        assert isinstance(eng, ExpansionEngine)

    def test_bundles_twentysix_rules(self):
        _, pi, _, _, _, _, koszul = _setup()
        eng = tilde_intrinsic_engine(pi, koszul)
        # 3 tilde intrinsic + 1 tilde aux + 5 multi-eval + 2 sharp +
        # 3 closure + 3 tilde-closure (slot-Lie commutator + L-of-anchor-bracket
        # + bare anchor LH) + 4 tilde-rel-4/6 closure
        # (L_X-d-commute, Pairing linearity, Pairing-Lie Leibniz,
        # L_πα(π^♯β) anchor) + 1 Hamiltonian-anchor-pairing antisymmetry
        # + 1 Wrapped-pairing-anchor-antisymmetry
        # + 1 Tilde-SN-Jacobi-residue
        # + 1 Pairing-with-exact-form + 1 Act-over-Sum-op = 26
        assert len(eng.definitions) == 26

    def test_intrinsic_rules_first(self):
        _, pi, _, _, _, _, koszul = _setup()
        eng = tilde_intrinsic_engine(pi, koszul)
        names = [type(d).__name__ for d in eng.definitions]
        assert names.index("TildeIotaIntrinsicDefinition") < names.index(
            "MultiEvalHeadLinearityDefinition"
        )
        assert names.index("TildeLieIntrinsicDefinition") < names.index(
            "MultiEvalHeadLinearityDefinition"
        )
        assert names.index("TildeDIntrinsicDefinition") < names.index(
            "MultiEvalHeadLinearityDefinition"
        )

    def test_each_call_returns_fresh_engine(self):
        _, pi, _, _, _, _, koszul = _setup()
        eng1 = tilde_intrinsic_engine(pi, koszul)
        eng2 = tilde_intrinsic_engine(pi, koszul)
        assert eng1 is not eng2

    def test_engine_expands_iota_one_form(self):
        # ι̃_ω V evaluated on η: should rewrite to V(ω, η). The alternating
        # canonicaliser may reorder the slots, compare structurally
        # against the alternating-MultiEval normal form by re-running the
        # alternating rule on the expected RHS.
        _, pi, omega, eta, V, _, koszul = _setup()
        eng = tilde_intrinsic_engine(pi, koszul)
        head = Act(TildeInteriorProduct(omega), V)
        expr = multi_eval(head, eta, slot_kind="covector")
        out, steps = eng.expand(expr)
        rhs = multi_eval(V, omega, eta, slot_kind="covector")
        rhs_canonical, _ = eng.expand(rhs)
        assert out == rhs_canonical
        assert len(steps) >= 1


# --------------------------------------------------------------------- #
# TildeIntrinsicFormulaRecognizer                                       #
# --------------------------------------------------------------------- #


class TestTildeRecognizer:
    def test_recognizes_tilde_iota_head(self):
        _, _, omega, eta, V, _, _ = _setup()
        rec = TildeIntrinsicFormulaRecognizer()
        m = rec.recognize(
            multi_eval(
                Act(TildeInteriorProduct(omega), V),
                eta,
                slot_kind="covector",
            )
        )
        assert m is not None
        assert m.operator == "tilde_interior"
        assert m.form == omega
        assert m.multivector == V
        assert m.args == (eta,)

    def test_recognizes_tilde_lie_head(self):
        _, pi, omega, eta, V, _, _ = _setup()
        rec = TildeIntrinsicFormulaRecognizer()
        m = rec.recognize(
            multi_eval(
                Act(TildeLieDerivative(omega, pi), V),
                eta,
                slot_kind="covector",
            )
        )
        assert m is not None
        assert m.operator == "tilde_lie"
        assert m.form == omega
        assert m.bivector == pi
        assert m.multivector == V

    def test_recognizes_tilde_d_head(self):
        _, pi, _, eta, V, _, _ = _setup()
        rec = TildeIntrinsicFormulaRecognizer()
        m = rec.recognize(
            multi_eval(
                Act(TildeExteriorDerivative(pi), V),
                eta,
                slot_kind="covector",
            )
        )
        assert m is not None
        assert m.operator == "tilde_exterior_d"
        assert m.form is None
        assert m.bivector == pi

    def test_classify(self):
        _, _, omega, eta, V, _, _ = _setup()
        rec = TildeIntrinsicFormulaRecognizer()
        expr = multi_eval(
            Act(TildeInteriorProduct(omega), V),
            eta,
            slot_kind="covector",
        )
        assert rec.classify(expr) == "tilde_interior"

    def test_vector_slot_returns_none(self):
        # A vector-slot evaluation belongs to the standard-side recogniser.
        _, _, omega, eta, V, _, _ = _setup()
        rec = TildeIntrinsicFormulaRecognizer()
        expr = multi_eval(
            Act(TildeInteriorProduct(omega), V),
            eta,
            slot_kind="vector",
        )
        assert rec.recognize(expr) is None
        assert rec.classify(expr) is None

    def test_non_multieval_returns_none(self):
        rec = TildeIntrinsicFormulaRecognizer()
        assert rec.recognize(Symbol("V")) is None

    def test_match_is_frozen_dataclass(self):
        m = TildeIntrinsicFormulaMatch(
            operator="tilde_interior",
            form=Symbol("ω"),
            bivector=Symbol("π"),
            multivector=Symbol("V"),
            args=(),
            alternating=True,
            slot_kind="covector",
        )
        with pytest.raises(Exception):
            m.operator = "tilde_lie"


# --------------------------------------------------------------------- #
# prove_tilde_cartan_relation                                           #
# --------------------------------------------------------------------- #


class TestProveTildeCartanRelation:
    def test_reflexive_returns_proof_chain(self):
        # Driver should accept reflexive equalities, single-step chain.
        _, pi, omega, eta, V, _, koszul = _setup()
        eng = tilde_intrinsic_engine(pi, koszul)
        head = Act(TildeInteriorProduct(omega), V)
        chain = prove_tilde_cartan_relation(
            head, head, etas=(eta,), engine=eng
        )
        assert isinstance(chain, ProofChain)
        assert chain.steps

    def test_rejects_non_tuple_etas(self):
        _, pi, omega, _, V, _, koszul = _setup()
        eng = tilde_intrinsic_engine(pi, koszul)
        head = Act(TildeInteriorProduct(omega), V)
        with pytest.raises(TypeError):
            prove_tilde_cartan_relation(
                head, head, etas=[Symbol("η")], engine=eng
            )

    def test_rejects_empty_etas(self):
        _, pi, omega, _, V, _, koszul = _setup()
        eng = tilde_intrinsic_engine(pi, koszul)
        head = Act(TildeInteriorProduct(omega), V)
        with pytest.raises(TypeError):
            prove_tilde_cartan_relation(head, head, etas=(), engine=eng)

    def test_rejects_non_engine(self):
        _, _, omega, _, V, _, _ = _setup()
        head = Act(TildeInteriorProduct(omega), V)
        with pytest.raises(TypeError):
            prove_tilde_cartan_relation(
                head, head, etas=(Symbol("η"),), engine="not-an-engine"
            )


# --------------------------------------------------------------------- #
# KoszulProblem.tilde_intrinsic_engine + prove_tilde_cartan             #
# --------------------------------------------------------------------- #


class TestKoszulProblemTildeAccessors:
    def _build_problem(self):
        reg = PropertyRegistry()
        pi = Symbol("π")
        omega = Symbol("ω")
        eta = Symbol("η")
        reg.declare(omega, Graded(degree=1))
        reg.declare(eta, Graded(degree=1))
        return KoszulProblem(pi, (omega, eta), registry=reg), omega, eta

    def test_engine_is_fresh_each_call(self):
        prob, _, _ = self._build_problem()
        eng1 = prob.tilde_intrinsic_engine()
        eng2 = prob.tilde_intrinsic_engine()
        assert eng1 is not eng2

    def test_engine_includes_bracket_expansion_rule(self):
        prob, _, _ = self._build_problem()
        eng = prob.tilde_intrinsic_engine()
        # Factory bundles 26 rules, KoszulProblem wires bracket-expansion
        # + the Poisson-gated d̃² aux rule (Aux-5) on top → 28 total.
        assert len(eng.definitions) == 28
        from jacopy.library.koszul_problem import KoszulBracketExpansionDefinition
        assert any(
            isinstance(d, KoszulBracketExpansionDefinition)
            for d in eng.definitions
        )

    def test_engine_is_pi_scoped(self):
        prob, _, _ = self._build_problem()
        eng = prob.tilde_intrinsic_engine()
        # The π-scoped d̃ rule on this engine should not match a head
        # built around a different π.
        d_rule = next(
            d for d in eng.definitions
            if isinstance(d, TildeDIntrinsicDefinition)
        )
        other_pi = Symbol("π'")
        V = Symbol("V")
        eta = Symbol("η")
        head = Act(TildeExteriorDerivative(other_pi), V)
        expr = multi_eval(head, eta, slot_kind="covector")
        assert not d_rule.matches(expr)

    def test_prove_tilde_cartan_reflexive(self):
        # Driver accepts a reflexive equality wrapped against an η-tuple.
        prob, omega, eta = self._build_problem()
        V = Symbol("V")
        head = Act(TildeInteriorProduct(omega), V)
        chain = prob.prove_tilde_cartan(head, head, etas=(eta,))
        assert isinstance(chain, ProofChain)
        assert chain.steps

    def test_prove_tilde_cartan_rejects_bad_lhs_type(self):
        prob, omega, eta = self._build_problem()
        with pytest.raises(TypeError):
            prob.prove_tilde_cartan(
                "not-an-expr", omega, etas=(eta,)
            )


# --------------------------------------------------------------------- #
# Faz 14.G, WrappedPairingAnchorAntisymmetryDefinition                  #
# --------------------------------------------------------------------- #


class TestWrappedPairingAnchorAntisymmetry:
    """Sum-level cancellation of π-antisymmetric pairings under common wrap."""

    def _setup(self):
        from jacopy.calculus.tilde.closure_axioms import (
            WrappedPairingAnchorAntisymmetryDefinition,
        )
        from jacopy.core.properties import Poisson
        from jacopy.calculus.pairing import Pairing

        reg = PropertyRegistry()
        pi = Symbol("π")
        a = Symbol("a")
        b = Symbol("b")
        for f in (a, b):
            reg.declare(f, Graded(degree=1))
        reg.declare(pi, Poisson())
        sharp = Sharp(pi)
        rule = WrappedPairingAnchorAntisymmetryDefinition(
            pi, registry=reg, sharp=sharp
        )
        return rule, sharp, pi, a, b, Pairing

    def test_cancels_bare_pairing_pair(self):
        from jacopy.core.expr import Sum

        rule, sharp, _, a, b, Pairing = self._setup()
        # ⟨π^♯a, b⟩ + ⟨π^♯b, a⟩ → 0
        p1 = Pairing(Act(sharp, a), b)
        p2 = Pairing(Act(sharp, b), a)
        s = Sum.make(p1, p2)
        assert rule.matches(s)
        out = rule.rewrite(s)
        assert out == Integer(0)

    def test_cancels_through_wrap(self):
        from jacopy.core.expr import Sum

        rule, sharp, _, a, b, Pairing = self._setup()
        V = Symbol("V")
        d = Symbol("d")
        # V(d(⟨π^♯a, b⟩)) + V(d(⟨π^♯b, a⟩)) → 0 (same wrap on both sides).
        wrapped = lambda p: Act(V, Act(d, p))
        s = Sum.make(wrapped(Pairing(Act(sharp, a), b)),
                     wrapped(Pairing(Act(sharp, b), a)))
        assert rule.matches(s)
        out = rule.rewrite(s)
        assert out == Integer(0)

    def test_declines_when_signs_differ(self):
        from jacopy.core.expr import Sum

        rule, sharp, _, a, b, Pairing = self._setup()
        # ⟨π^♯a, b⟩ − ⟨π^♯b, a⟩, opposite signs, doesn't cancel.
        s = Sum.make(
            Pairing(Act(sharp, a), b),
            Neg(Pairing(Act(sharp, b), a)),
        )
        assert not rule.matches(s)

    def test_poisson_gated(self):
        from jacopy.core.expr import Sum

        rule, sharp, pi, a, b, Pairing = self._setup()
        # Build a fresh registry without Poisson.
        reg2 = PropertyRegistry()
        reg2.declare(a, Graded(degree=1))
        reg2.declare(b, Graded(degree=1))
        from jacopy.calculus.tilde.closure_axioms import (
            WrappedPairingAnchorAntisymmetryDefinition,
        )
        rule2 = WrappedPairingAnchorAntisymmetryDefinition(
            pi, registry=reg2, sharp=sharp
        )
        s = Sum.make(
            Pairing(Act(sharp, a), b),
            Pairing(Act(sharp, b), a),
        )
        assert not rule2.matches(s)


# --------------------------------------------------------------------- #
# Faz 14.G, TildeSnJacobiResidueDefinition                              #
# --------------------------------------------------------------------- #


class TestTildeSnJacobiResidue:
    """5-term SN-Jacobi residue zeroing under Poisson."""

    def _setup(self):
        from jacopy.calculus.tilde.closure_axioms import (
            TildeSnJacobiResidueDefinition,
        )
        from jacopy.core.properties import Poisson
        from jacopy.calculus.pairing import Pairing
        from jacopy.calculus.hamiltonian_vf import HamiltonianVectorField
        from jacopy.calculus.lie_derivative import lie_derivative

        reg = PropertyRegistry()
        pi = Symbol("π")
        a = Symbol("a")
        b = Symbol("b")
        c = Symbol("c")
        for f in (a, b, c):
            reg.declare(f, Graded(degree=1))
        reg.declare(pi, Poisson())
        sharp = Sharp(pi)
        rule = TildeSnJacobiResidueDefinition(pi, registry=reg, sharp=sharp)
        return (rule, sharp, pi, a, b, c, Pairing, HamiltonianVectorField,
                lie_derivative)

    def _build_residue(self, sharp, pi, a, b, c, Pairing, HVF, lie_der):
        from jacopy.core.expr import Sum

        # P1: +⟨X_{⟨π^♯a, b⟩}, c⟩
        p1 = Pairing(HVF(Pairing(Act(sharp, a), b), bivector=pi), c)
        # P2: -⟨X_{⟨π^♯c, a⟩}, b⟩
        p2 = Neg(Pairing(HVF(Pairing(Act(sharp, c), a), bivector=pi), b))
        # P3: -⟨π^♯(L_{π^♯a} b), c⟩
        p3 = Neg(Pairing(
            Act(sharp, Act(lie_der(Act(sharp, a)), b)),
            c,
        ))
        # P4: +⟨π^♯(L_{π^♯b} a), c⟩
        p4 = Pairing(
            Act(sharp, Act(lie_der(Act(sharp, b)), a)),
            c,
        )
        # P5: +⟨π^♯a, L_{π^♯b} c⟩
        p5 = Pairing(Act(sharp, a), Act(lie_der(Act(sharp, b)), c))
        return Sum.make(p1, p2, p3, p4, p5)

    def test_cancels_bare_residue(self):
        rule, sharp, pi, a, b, c, P, HVF, ld = self._setup()
        s = self._build_residue(sharp, pi, a, b, c, P, HVF, ld)
        assert rule.matches(s)
        out = rule.rewrite(s)
        assert out == Integer(0)

    def test_declines_without_poisson(self):
        from jacopy.calculus.tilde.closure_axioms import (
            TildeSnJacobiResidueDefinition,
        )
        from jacopy.calculus.pairing import Pairing
        from jacopy.calculus.hamiltonian_vf import HamiltonianVectorField
        from jacopy.calculus.lie_derivative import lie_derivative

        reg = PropertyRegistry()
        pi = Symbol("π")
        a, b, c = Symbol("a"), Symbol("b"), Symbol("c")
        for f in (a, b, c):
            reg.declare(f, Graded(degree=1))
        sharp = Sharp(pi)
        # Note: no reg.declare(pi, Poisson()).
        rule = TildeSnJacobiResidueDefinition(pi, registry=reg, sharp=sharp)
        s = self._build_residue(
            sharp, pi, a, b, c, Pairing, HamiltonianVectorField, lie_derivative
        )
        assert not rule.matches(s)

    def test_declines_on_partial_residue(self):
        from jacopy.core.expr import Sum

        rule, sharp, pi, a, b, c, P, HVF, ld = self._setup()
        full = self._build_residue(sharp, pi, a, b, c, P, HVF, ld)
        # Drop the mixed-shape term; the recognizer should decline.
        partial = Sum.make(*full.children[:-1])
        assert not rule.matches(partial)

    def _build_flipped_residue(self, sharp, pi, a, b, c, Pairing, HVF, lie_der):
        """Same SN-Jacobi 5-term identity, every sign flipped, the form
        Faz 14.G+H residues take when wrapped under ``MultiEval(V, d(·),
        γ)`` (T2/T3 derived identities)."""
        from jacopy.core.expr import Sum

        # F1: -⟨X_{⟨π^♯a, b⟩}, c⟩
        p1 = Neg(Pairing(HVF(Pairing(Act(sharp, a), b), bivector=pi), c))
        # F2: +⟨X_{⟨π^♯c, a⟩}, b⟩
        p2 = Pairing(HVF(Pairing(Act(sharp, c), a), bivector=pi), b)
        # F3: +⟨π^♯(L_{π^♯a} b), c⟩
        p3 = Pairing(
            Act(sharp, Act(lie_der(Act(sharp, a)), b)),
            c,
        )
        # F4: -⟨π^♯(L_{π^♯b} a), c⟩
        p4 = Neg(Pairing(
            Act(sharp, Act(lie_der(Act(sharp, b)), a)),
            c,
        ))
        # F5: -⟨π^♯a, L_{π^♯b} c⟩
        p5 = Neg(Pairing(Act(sharp, a), Act(lie_der(Act(sharp, b)), c)))
        return Sum.make(p1, p2, p3, p4, p5)

    def test_cancels_flipped_residue(self):
        rule, sharp, pi, a, b, c, P, HVF, ld = self._setup()
        s = self._build_flipped_residue(sharp, pi, a, b, c, P, HVF, ld)
        assert rule.matches(s)
        out = rule.rewrite(s)
        assert out == Integer(0)
