"""
Koszul-bracket problem wrapper.

A :class:`KoszulProblem` is the Poisson-side counterpart to
:class:`SymplecticProblem`: it bundles a Poisson bivector ``π``, a
form inventory, a registry, and a pre-configured
:class:`ExpansionEngine` carrying every axiom a Koszul-bracket
problem typically cites:

* :class:`SharpLinearityDefinition`, ``π^♯(A + B) → π^♯(A) + π^♯(B)``,
* :class:`SharpOnExactDefinition`, ``π^♯(df) → X_f``,
* :class:`RegistryAntiSymCanonicalDefinition`, registry-driven
  canonicalization of ``π(α, β)``,
* :class:`KoszulBracketExpansionDefinition`, Cartan-style expansion
  of ``[α, β]_K`` for any pair of forms (the engine will fire on a
  :class:`BracketApply` whose bracket is this problem's Koszul
  bracket),
* the registry-free C∞-linearity rules
  (:class:`MultiEvalScalarPullDefinition`,
  :class:`PairingScalarPullDefinition`,
  :class:`LieRescalingDefinition`).

Where 2d's notebook hand-rolled four bracket / linearity / antisym
inline definitions, ``KoszulProblem(π, (ω, η, f, fη))`` puts the
same setup in one constructor call.
"""

from __future__ import annotations

from typing import Iterable, Optional, Tuple

from jacopy.algebra.derivation import Act
from jacopy.brackets.base import BracketApply
from jacopy.brackets.koszul import KoszulBracket
from jacopy.calculus.antisym_axioms import RegistryAntiSymCanonicalDefinition
from jacopy.calculus.closure_axioms import (
    IotaActAsScalarDefinition,
    LieBracketVfJacobiDefinition,
)
from jacopy.calculus.cartan_remainder_axioms import (
    CartanRemainderDefinition,
    TildeCartanRemainderDefinition,
)
from jacopy.calculus.derivator import prove_derivator_identity
from jacopy.calculus.derivator_index_pass import (
    AtomSlotLiftDefinition,
    BareAtomSlotLiftDefinition,
    canonicalize_indices,
)
from jacopy.calculus.intrinsic_axioms import (
    ExteriorDIntrinsicDefinition,
    InteriorProductIntrinsicDefinition,
    LieDerivativeIntrinsicDefinition,
)
from jacopy.calculus.intrinsic_engine import intrinsic_engine_with_closure
from jacopy.calculus.exterior_d import ExteriorDerivative, d as default_d
from jacopy.calculus.lie_derivative import lie_derivative as default_lie_derivative
from jacopy.calculus.lie_rescaling_axioms import LieRescalingDefinition
from jacopy.calculus.multi_eval_scalar_axioms import (
    MultiEvalScalarPullDefinition,
)
from jacopy.calculus.musical import Sharp
from jacopy.calculus.pairing_linearity_axioms import (
    PairingScalarPullDefinition,
)
from jacopy.calculus.sharp_axioms import (
    SharpLinearityDefinition,
    SharpNegLinearityDefinition,
    SharpOnExactDefinition,
)
from jacopy.calculus.sn_function_axiom import (
    HamiltonianPairingAntisymmetryDefinition,
    LieBracketVfAntisymmetryDefinition,
    LieBracketVfNegLinearityDefinition,
    MultiEvalBivectorAntisymmetryDefinition,
    MultiEvalCovectorDArityOneDefinition,
    MultiEvalCovectorPairingFlipDefinition,
    MultiEvalOnHamiltonianDefinition,
    PairingToMultiEvalBridgeDefinition,
    PermissiveIotaActAsScalarDefinition,
    PermissiveVfActCommutatorDefinition,
    PoissonCommutatorOnInteriorDefinition,
    HamiltonianVfInnerIotaToMultiEvalDefinition,
    SnBracketNegLinearityDefinition,
    SnBracketOfFunctionDefinition,
    SnBracketOfOneVectorsToLieBracketVfDefinition,
    SnBracketSumLinearityDefinition,
    LieBracketVfSumLinearityDefinition,
    TildeDOnScalarToHamiltonianVfDefinition,
    TildeIotaOnLieBracketVfActAsScalarDefinition,
    TildeLieOnOneVectorLichnerowiczDefinition,
    VfActOnExteriorDOfScalarDefinition,
    VfActOnLieOfScalarDefinition,
    VfActOnPairingLeibnizDefinition,
)
from jacopy.calculus.tilde.aux_axioms import (
    TildeDOfFunctionDefinition,
    TildeDSquaredPoissonDefinition,
    TildeIotaOnZeroVectorDefinition,
    TildeIotaSquaredZeroDefinition,
    TildeLieOnZeroVectorDefinition,
)
from jacopy.calculus.tilde.axioms import (
    TildeExteriorDLichnerowiczDefinition,
    TildeIotaSwapDefinition,
    TildeLieMagicDefinition,
)
from jacopy.calculus.tilde.intrinsic_engine import (
    prove_tilde_cartan_relation,
    tilde_intrinsic_engine as _tilde_intrinsic_engine_factory,
)
from jacopy.calculus.tilde.operators import (
    TildeExteriorDerivative,
    TildeInteriorProduct,
    TildeLieDerivative,
)
from jacopy.core.expr import Expr
from jacopy.core.properties import Antisymmetric, Graded, Poisson
from jacopy.core.registry import PropertyRegistry
from jacopy.proof.expansion import Definition, ExpansionEngine, default_engine


# --------------------------------------------------------------------- #
# Engine rule, [α, β]_K → Cartan expansion                             #
# --------------------------------------------------------------------- #


class KoszulBracketExpansionDefinition(Definition):
    """Rewrite ``BracketApply(K, α, β) → L_{ρα}β − L_{ρβ}α − d⟨ρα, β⟩``.

    Scoped to a specific :class:`KoszulBracket` instance so that
    bracket nodes from unrelated brackets in the same proof are left
    untouched.
    """

    def __init__(self, koszul: KoszulBracket) -> None:
        if not isinstance(koszul, KoszulBracket):
            raise TypeError(
                "KoszulBracketExpansionDefinition expects a KoszulBracket"
            )
        self._koszul = koszul
        self.name = f"[α, β]_K = L_ρα(β) − L_ρβ(α) − d⟨ρα, β⟩ [{koszul.name}]"

    def matches(self, expr: Expr) -> bool:
        return isinstance(expr, BracketApply) and expr.bracket is self._koszul

    def rewrite(self, expr: Expr) -> Expr:
        return self._koszul.expand(expr.a, expr.b)


# --------------------------------------------------------------------- #
# KoszulProblem                                                         #
# --------------------------------------------------------------------- #


class KoszulProblem:
    """``(π, ρ = π^♯, K = [·,·]_K, {α_i})``, Koszul-bracket problem bundle.

    Parameters
    ----------
    pi
        Poisson bivector ``π``. Required.
    forms
        Iterable of form operands designated for this problem (e.g.
        ``(ω, η, f, fη)``). Each must be an :class:`Expr`.
    registry
        :class:`PropertyRegistry`. The wrapper auto-declares
        :class:`Antisymmetric` on ``π`` if not already declared.
        Grading declarations stay the caller's responsibility.
    base_engine
        Optional starting engine; defaults to
        :func:`~jacopy.proof.expansion.default_engine`.
    d
        Optional :class:`ExteriorDerivative` instance; defaults to
        the module-level singleton.
    lie_derivative
        Optional Lie-derivative factory; defaults to the module-level
        :func:`~jacopy.calculus.lie_derivative.lie_derivative`.
    name
        Display name; defaults to ``f"KoszulProblem(π; α_1, …, α_n)"``.
    """

    __slots__ = (
        "_pi",
        "_forms",
        "_multivectors",
        "_registry",
        "_engine",
        "_sharp",
        "_koszul",
        "_d",
        "_lie",
        "_name",
        "_bracket_rule",
        "_tilde_swap_rule",
        "_tilde_d_rule",
        "_tilde_lie_rule",
        "_tilde_aux_rules",
        "_tilde_d_sq_rule",
    )

    def __init__(
        self,
        pi: Expr,
        forms: Iterable[Expr],
        *,
        registry: PropertyRegistry,
        multivectors: Optional[Iterable[Tuple[Expr, int]]] = None,
        base_engine: Optional[ExpansionEngine] = None,
        d: Optional[ExteriorDerivative] = None,
        lie_derivative=None,
        name: Optional[str] = None,
    ) -> None:
        if not isinstance(pi, Expr):
            raise TypeError("KoszulProblem pi must be an Expr")
        if not isinstance(registry, PropertyRegistry):
            raise TypeError("KoszulProblem requires a PropertyRegistry")
        forms_tuple: Tuple[Expr, ...] = tuple(forms)
        if not forms_tuple:
            raise ValueError(
                "KoszulProblem requires at least one form in 'forms'"
            )
        for alpha in forms_tuple:
            if not isinstance(alpha, Expr):
                raise TypeError(
                    "KoszulProblem 'forms' must contain Expr instances only"
                )

        # ``multivectors`` is an optional iterable of ``(operand, k)`` pairs
        # giving each multivector its SN-degree (0 for a scalar function,
        # 1 for a vector field, 2 for a bivector, ...). The wrapper auto-
        # declares ``Graded(degree=k)`` if the operand has no Graded
        # property yet, symmetric to ``Antisymmetric`` on ``π`` but for
        # the tilde operands.
        mv_tuple: Tuple[Tuple[Expr, int], ...] = (
            () if multivectors is None else tuple(multivectors)
        )
        for entry in mv_tuple:
            if not (isinstance(entry, tuple) and len(entry) == 2):
                raise TypeError(
                    "KoszulProblem 'multivectors' entries must be "
                    "(Expr, int) pairs"
                )
            operand, k = entry
            if not isinstance(operand, Expr):
                raise TypeError(
                    "KoszulProblem 'multivectors' operand must be an Expr"
                )
            if not isinstance(k, int) or k < 0:
                raise TypeError(
                    "KoszulProblem 'multivectors' degree must be a "
                    "non-negative int"
                )

        if not registry.has(pi, Antisymmetric):
            registry.declare(pi, Antisymmetric())

        for operand, k in mv_tuple:
            if not registry.has(operand, Graded):
                registry.declare(operand, Graded(degree=k))

        d_op = default_d if d is None else d
        lie = default_lie_derivative if lie_derivative is None else lie_derivative

        sharp = Sharp(pi)
        koszul = KoszulBracket(
            sharp, name=f"[·,·]_K[{pi._repr_inner()}]",
            d=d_op, lie_derivative=lie,
        )

        if base_engine is None:
            engine = default_engine(registry=registry, d=d_op)
        else:
            engine = ExpansionEngine(
                list(base_engine.definitions),
                mode=base_engine.mode,
            )
        # Sharp axioms.
        engine.register(SharpLinearityDefinition(sharp))
        engine.register(SharpOnExactDefinition(sharp, registry=registry, d=d_op))
        # π anti-sym (#11) + registry-free C∞-linearity (#6/#10/#12).
        engine.register(RegistryAntiSymCanonicalDefinition(registry=registry))
        engine.register(MultiEvalScalarPullDefinition())
        engine.register(PairingScalarPullDefinition())
        engine.register(LieRescalingDefinition())
        # Koszul-bracket expansion axiom.
        bracket_rule = KoszulBracketExpansionDefinition(koszul)
        engine.register(bracket_rule)
        # Tilde calculus auxiliary axioms (Faz 14.D) register *before*
        # the defining axioms (Faz 14.B). The auxiliaries are
        # specificity shortcuts, each one's match condition is a strict
        # subset of a defining axiom (Aux-1/2 narrow swap; Aux-3 narrows
        # magic; Aux-4/5 narrow Lichnerowicz). Engine ordering picks the
        # first matching rule, so registering the specific ones first
        # makes them fire when applicable, falling through to the
        # general defining axiom otherwise. Aux-5 reads ``Poisson(π)``
        # off the registry at match time, so it stays inert until the
        # caller invokes ``assume_poisson()``.
        tilde_iota0_rule = TildeIotaOnZeroVectorDefinition(registry=registry)
        tilde_iota_sq_rule = TildeIotaSquaredZeroDefinition()
        tilde_lie0_rule = TildeLieOnZeroVectorDefinition(pi, registry=registry)
        tilde_d_func_rule = TildeDOfFunctionDefinition(
            pi, d=d_op, registry=registry
        )
        tilde_d_sq_rule = TildeDSquaredPoissonDefinition(pi, registry=registry)
        engine.register(tilde_iota0_rule)
        engine.register(tilde_iota_sq_rule)
        engine.register(tilde_lie0_rule)
        engine.register(tilde_d_func_rule)
        engine.register(tilde_d_sq_rule)
        # Tilde calculus defining axioms (Faz 14.B). The Lichnerowicz and
        # magic rules are π-scoped, so two KoszulProblems with distinct
        # π's never cross-fire; the swap rule is registry-free and
        # idempotent across instances.
        tilde_swap_rule = TildeIotaSwapDefinition()
        tilde_d_rule = TildeExteriorDLichnerowiczDefinition(pi)
        tilde_lie_rule = TildeLieMagicDefinition(pi)
        engine.register(tilde_swap_rule)
        engine.register(tilde_d_rule)
        engine.register(tilde_lie_rule)

        self._pi = pi
        self._forms = forms_tuple
        self._multivectors = mv_tuple
        self._registry = registry
        self._engine = engine
        self._sharp = sharp
        self._koszul = koszul
        self._d = d_op
        self._lie = lie
        self._bracket_rule = bracket_rule
        self._tilde_swap_rule = tilde_swap_rule
        self._tilde_d_rule = tilde_d_rule
        self._tilde_lie_rule = tilde_lie_rule
        self._tilde_aux_rules = (
            tilde_iota0_rule,
            tilde_iota_sq_rule,
            tilde_lie0_rule,
            tilde_d_func_rule,
            tilde_d_sq_rule,
        )
        self._tilde_d_sq_rule = tilde_d_sq_rule
        if name is not None:
            self._name = name
        else:
            forms_repr = ", ".join(a._repr_inner() for a in forms_tuple)
            self._name = f"KoszulProblem({pi._repr_inner()}; {forms_repr})"

    # ---- accessors ------------------------------------------------- #

    @property
    def pi(self) -> Expr:
        return self._pi

    @property
    def forms(self) -> Tuple[Expr, ...]:
        return self._forms

    @property
    def registry(self) -> PropertyRegistry:
        return self._registry

    @property
    def engine(self) -> ExpansionEngine:
        return self._engine

    @property
    def sharp(self) -> Sharp:
        return self._sharp

    @property
    def koszul_bracket(self) -> KoszulBracket:
        return self._koszul

    @property
    def name(self) -> str:
        return self._name

    @property
    def bracket_expansion_rule(self) -> KoszulBracketExpansionDefinition:
        return self._bracket_rule

    @property
    def multivectors(self) -> Tuple[Tuple[Expr, int], ...]:
        return self._multivectors

    @property
    def tilde_swap_rule(self) -> TildeIotaSwapDefinition:
        return self._tilde_swap_rule

    @property
    def tilde_d_rule(self) -> TildeExteriorDLichnerowiczDefinition:
        return self._tilde_d_rule

    @property
    def tilde_lie_rule(self) -> TildeLieMagicDefinition:
        return self._tilde_lie_rule

    @property
    def tilde_aux_rules(self) -> Tuple[Definition, ...]:
        r"""The five Faz 14.D auxiliary rules in registration order.

        Order: ``(iota0, iota_sq, lie0, d_func, d_sq)``,
        :class:`TildeIotaOnZeroVectorDefinition`,
        :class:`TildeIotaSquaredZeroDefinition`,
        :class:`TildeLieOnZeroVectorDefinition`,
        :class:`TildeDOfFunctionDefinition`,
        :class:`TildeDSquaredPoissonDefinition`.
        """
        return self._tilde_aux_rules

    # ---- tilde intrinsic engine (Faz 14.E) ------------------------- #

    def tilde_intrinsic_engine(self) -> ExpansionEngine:
        r"""Return a fresh tilde-intrinsic :class:`ExpansionEngine`.

        Bundle (in match order):

        * the three Faz 14.E.1 intrinsic rules (ι̃, L̃, d̃), each one
          unfolds an operator-on-multivector head wrapped in a
          ``slot_kind="covector"`` :class:`MultiEval`,
        * the four MultiEval helpers (Faz 12.A.0/12.A.4),
        * :class:`SharpLinearityDefinition` and
          :class:`SharpOnExactDefinition` (Faz 13.A) plumbed against
          this problem's :class:`Sharp` and registry,
        * this problem's :class:`KoszulBracketExpansionDefinition` so
          ``[α, β]_K`` residues from the L̃/d̃ intrinsic rules unfold
          to their Lichnerowicz form.

        Each call returns a fresh :class:`ExpansionEngine`; mutate it
        freely (e.g. register extra rules for a particular proof) without
        contaminating the problem's main engine.
        """
        engine = _tilde_intrinsic_engine_factory(
            self._pi,
            self._koszul,
            sharp=self._sharp,
            registry=self._registry,
            d_op=self._d,
        )
        engine.register(self._bracket_rule)
        # Wire the Faz 14.D Poisson-gated d̃² aux rule onto the proof
        # engine so relation 2 can close. The other aux rules
        # (TildeIotaOnZero / TildeIotaSquaredZero / TildeLieOnZero /
        # TildeDOfFunction) are *not* registered here: each one
        # competes with the intrinsic-formula rules at the operator
        # level and races with the d̃/L̃ intrinsic expansion when the
        # operand is a MultiEval-derived scalar, registering them on
        # the proof engine breaks the Cartan-magic closure (Aux-4
        # rewrites the post-Aux-6 ``d̃(V(ω))`` into ``-X_{V(ω)}``,
        # blocking syntactic equality with the d̃-intrinsic-derived
        # ``π^♯(η)·V`` shape).
        engine.register(self._tilde_d_sq_rule)
        return engine

    def prove_tilde_cartan(
        self,
        lhs: Expr,
        rhs: Expr,
        *,
        etas: Tuple[Expr, ...],
        alternating: bool = True,
    ):
        r"""Prove ``lhs == rhs`` by evaluating both sides on ``(η_1, …, η_p)``.

        Thin wrapper around
        :func:`~jacopy.calculus.tilde.intrinsic_engine.prove_tilde_cartan_relation`
        that supplies this problem's tilde intrinsic engine and
        registry. ``etas`` must be a non-empty tuple of 1-forms.
        """
        return prove_tilde_cartan_relation(
            lhs,
            rhs,
            etas=etas,
            engine=self.tilde_intrinsic_engine(),
            registry=self._registry,
            alternating=alternating,
        )

    # ---- derivator engines (Faz 15.B) ------------------------------ #

    def derivator_form_engine(self) -> ExpansionEngine:
        r"""Form-side engine for Section 3.1.5 derivator identities.

        Bundles, in match order:

        * standard intrinsic + closure axioms (Faz 12.A.5/12.A.6)
        * :class:`KoszulBracketExpansionDefinition` for this problem
        * :class:`SharpLinearityDefinition` and
          :class:`SharpOnExactDefinition`
        * :class:`CartanRemainderDefinition`,
          ``K_V ω → −L_V ω + d ι_V ω``
        * :class:`TildeCartanRemainderDefinition`,
          ``K̃_η V → −L̃_η V + d̃ ι̃_η V``
        * the three tilde defining axioms (swap / Lichnerowicz / magic)
          plus the Poisson-gated d̃² aux rule, so a ``K̃_η V`` chain
          reaches the standard side.

        Returns a fresh engine; mutate freely. Used via
        :func:`~jacopy.calculus.derivator.prove_derivator_identity`
        with ``slot_kind="vector"`` for identities (1)/(2)/(3) of
        Section 3.1.5.
        """
        engine = intrinsic_engine_with_closure()
        engine.register(self._bracket_rule)
        engine.register(SharpLinearityDefinition(self._sharp))
        engine.register(
            SharpOnExactDefinition(
                self._sharp, registry=self._registry, d=self._d
            )
        )
        engine.register(CartanRemainderDefinition(d=self._d, lie_derivative=self._lie))
        engine.register(TildeCartanRemainderDefinition())
        engine.register(self._tilde_swap_rule)
        engine.register(self._tilde_d_rule)
        engine.register(self._tilde_lie_rule)
        engine.register(self._tilde_d_sq_rule)
        sn_func_rule = SnBracketOfFunctionDefinition(
            self._sharp, d=self._d, registry=self._registry
        )
        engine.register(sn_func_rule)
        ham_pairing_rule = HamiltonianPairingAntisymmetryDefinition(
            self._sharp, registry=self._registry
        )
        engine.register(ham_pairing_rule)
        me_ham_rule = MultiEvalOnHamiltonianDefinition(
            self._sharp, registry=self._registry
        )
        engine.register(me_ham_rule)
        # Faz 15.C, §3.1.5 form-side identity (3) closure rules.
        # Order: vf-Leibniz on Pairing first (opens scalar pairings into
        # standard-shape sums), then Poisson commutator (opens
        # ι_[π,W]_SN(ω) shapes), then Pairing→MultiEval bridge (unifies
        # canonical scalar form), then bivector antisymmetry on MultiEval
        # (lines up sibling cancellations).
        vf_pairing_leibniz_rule = VfActOnPairingLeibnizDefinition(
            lie_derivative=self._lie
        )
        engine.register(vf_pairing_leibniz_rule)
        poisson_commutator_rule = PoissonCommutatorOnInteriorDefinition(
            self._sharp,
            registry=self._registry,
            lie_derivative=self._lie,
        )
        engine.register(poisson_commutator_rule)
        pairing_to_me_rule = PairingToMultiEvalBridgeDefinition(
            registry=self._registry
        )
        engine.register(pairing_to_me_rule)
        me_bivector_antisym_rule = MultiEvalBivectorAntisymmetryDefinition(
            self._sharp, registry=self._registry
        )
        engine.register(me_bivector_antisym_rule)
        # Faz 15.C, identity (1) closure: permissive VF-commutator pair
        # finder + LieBracketVF antisymmetry. The pair finder folds bare
        # ``Symbol`` vfs (which the original closure rule rejects) into
        # ``LieBracketVF`` atoms; the antisymmetry rule canonicalizes
        # arg order so two paths emitting ``[U, π^♯η]`` and ``[π^♯η, U]``
        # cancel via collect_terms.
        permissive_commutator_rule = PermissiveVfActCommutatorDefinition()
        engine.register(permissive_commutator_rule)
        lbvf_neg_linearity_rule = LieBracketVfNegLinearityDefinition()
        engine.register(lbvf_neg_linearity_rule)
        lbvf_antisym_rule = LieBracketVfAntisymmetryDefinition()
        engine.register(lbvf_antisym_rule)
        # After the permissive commutator + antisymmetry rules surface
        # nested ``[A,[B,C]]`` triples wrapped in ``η(·)``/``μ(·)``
        # MultiEval, the Jacobi rule collapses each cyclic triple to 0.
        lbvf_jacobi_rule = LieBracketVfJacobiDefinition()
        engine.register(lbvf_jacobi_rule)
        # AtomSlotLift, opens opaque atom slots (LieDerivative.vector_field,
        # LieBracketVF.X/.Y, etc.) so the SharpOnExact / SnBracketOfFunction
        # rewrites can reach inside. The inner engine carries only the
        # canonicalization rules to keep the lift terminating; the main
        # engine does the rest after the slots are opened.
        inner_engine = ExpansionEngine([
            SharpOnExactDefinition(
                self._sharp, registry=self._registry, d=self._d
            ),
            sn_func_rule,
            ham_pairing_rule,
            me_ham_rule,
            vf_pairing_leibniz_rule,
            poisson_commutator_rule,
            pairing_to_me_rule,
            me_bivector_antisym_rule,
            permissive_commutator_rule,
            lbvf_neg_linearity_rule,
            lbvf_antisym_rule,
            lbvf_jacobi_rule,
            CartanRemainderDefinition(d=self._d, lie_derivative=self._lie),
            TildeCartanRemainderDefinition(),
            self._tilde_swap_rule,
        ])
        engine.register(AtomSlotLiftDefinition(inner_engine))
        return engine

    def derivator_multivector_engine(self) -> ExpansionEngine:
        r"""Multivector-side engine for Section 3.1.5 derivator identities.

        Extends :meth:`tilde_intrinsic_engine` with both Cartan-remainder
        rules and the full family of Faz 15.C closure axioms, the dual
        identities (1')/(2')/(3') need the same Pairing-Leibniz +
        Poisson-commutator + commutator-pair-finder + Jacobi machinery
        as the form side, just routed through the tilde-intrinsic
        operator-on-multivector base.

        Returns a fresh engine; mutate freely. Used via
        :func:`~jacopy.calculus.derivator.prove_derivator_identity`
        with ``slot_kind="covector"``.
        """
        engine = self.tilde_intrinsic_engine()
        engine.register(SharpLinearityDefinition(self._sharp))
        engine.register(
            SharpOnExactDefinition(
                self._sharp, registry=self._registry, d=self._d
            )
        )
        # Form-side Cartan intrinsics, the dual identities reduce
        # ``ι̃_α(V)`` shapes to scalar pairings then flip arity-1
        # covector evaluations to vector-slot ones, where the
        # form-side ``L_X`` magic and ``d``-arity-1 rules close them.
        engine.register(InteriorProductIntrinsicDefinition())
        engine.register(LieDerivativeIntrinsicDefinition())
        engine.register(ExteriorDIntrinsicDefinition())
        engine.register(IotaActAsScalarDefinition())
        engine.register(CartanRemainderDefinition(d=self._d, lie_derivative=self._lie))
        engine.register(TildeCartanRemainderDefinition())
        sn_func_rule = SnBracketOfFunctionDefinition(
            self._sharp, d=self._d, registry=self._registry
        )
        engine.register(sn_func_rule)
        ham_pairing_rule = HamiltonianPairingAntisymmetryDefinition(
            self._sharp, registry=self._registry
        )
        engine.register(ham_pairing_rule)
        me_ham_rule = MultiEvalOnHamiltonianDefinition(
            self._sharp, registry=self._registry
        )
        engine.register(me_ham_rule)
        vf_pairing_leibniz_rule = VfActOnPairingLeibnizDefinition(
            lie_derivative=self._lie
        )
        engine.register(vf_pairing_leibniz_rule)
        poisson_commutator_rule = PoissonCommutatorOnInteriorDefinition(
            self._sharp,
            registry=self._registry,
            lie_derivative=self._lie,
        )
        engine.register(poisson_commutator_rule)
        pairing_to_me_rule = PairingToMultiEvalBridgeDefinition(
            registry=self._registry
        )
        engine.register(pairing_to_me_rule)
        me_bivector_antisym_rule = MultiEvalBivectorAntisymmetryDefinition(
            self._sharp, registry=self._registry
        )
        engine.register(me_bivector_antisym_rule)
        permissive_commutator_rule = PermissiveVfActCommutatorDefinition()
        engine.register(permissive_commutator_rule)
        lbvf_neg_linearity_rule = LieBracketVfNegLinearityDefinition()
        engine.register(lbvf_neg_linearity_rule)
        lbvf_antisym_rule = LieBracketVfAntisymmetryDefinition()
        engine.register(lbvf_antisym_rule)
        lbvf_jacobi_rule = LieBracketVfJacobiDefinition()
        engine.register(lbvf_jacobi_rule)
        # Faz 15.C, multivector-side dual closures: SN→VF for 1-vfs,
        # ι̃-act-as-scalar bridge for LieBracketVF, arity-1 covector
        # pairing flip. Together these route the (3') residue through
        # the form-side intrinsic Cartan-magic / d-arity-1 rules.
        sn_to_lbvf_rule = SnBracketOfOneVectorsToLieBracketVfDefinition(
            registry=self._registry
        )
        engine.register(sn_to_lbvf_rule)
        tilde_iota_lbvf_rule = TildeIotaOnLieBracketVfActAsScalarDefinition()
        engine.register(tilde_iota_lbvf_rule)
        me_pairing_flip_rule = MultiEvalCovectorPairingFlipDefinition(
            registry=self._registry
        )
        engine.register(me_pairing_flip_rule)
        permissive_iota_rule = PermissiveIotaActAsScalarDefinition()
        engine.register(permissive_iota_rule)
        vf_act_on_df_rule = VfActOnExteriorDOfScalarDefinition(
            registry=self._registry
        )
        engine.register(vf_act_on_df_rule)
        me_cov_d_arity1_rule = MultiEvalCovectorDArityOneDefinition(
            registry=self._registry
        )
        engine.register(me_cov_d_arity1_rule)
        vf_act_on_lf_rule = VfActOnLieOfScalarDefinition()
        engine.register(vf_act_on_lf_rule)
        tilde_d_to_hvf_rule = TildeDOnScalarToHamiltonianVfDefinition(self._pi)
        engine.register(tilde_d_to_hvf_rule)
        hvf_inner_normalize_rule = HamiltonianVfInnerIotaToMultiEvalDefinition()
        engine.register(hvf_inner_normalize_rule)
        sn_neg_rule = SnBracketNegLinearityDefinition()
        engine.register(sn_neg_rule)
        # Faz 15.C, (1') closure: tilde-Lie Lichnerowicz on a 1-vector
        # plus SN-bracket Sum-distribution. The Lichnerowicz rewrite
        # emits a three-term Sum into a SN-bracket slot; the Sum
        # linearity then peels each summand into its own bracket so the
        # SN→LBVF coercion fires per term.
        tilde_lie_lich_rule = TildeLieOnOneVectorLichnerowiczDefinition(
            self._sharp,
            registry=self._registry,
            lie_derivative=self._lie,
        )
        engine.register(tilde_lie_lich_rule)
        sn_sum_rule = SnBracketSumLinearityDefinition()
        engine.register(sn_sum_rule)
        lbvf_sum_rule = LieBracketVfSumLinearityDefinition()
        engine.register(lbvf_sum_rule)
        sharp_neg_rule = SharpNegLinearityDefinition(self._sharp)
        engine.register(sharp_neg_rule)
        sharp_lin_rule = SharpLinearityDefinition(self._sharp)
        inner_engine = ExpansionEngine([
            sharp_lin_rule,
            sharp_neg_rule,
            SharpOnExactDefinition(
                self._sharp, registry=self._registry, d=self._d
            ),
            sn_func_rule,
            ham_pairing_rule,
            me_ham_rule,
            vf_pairing_leibniz_rule,
            poisson_commutator_rule,
            pairing_to_me_rule,
            me_bivector_antisym_rule,
            permissive_commutator_rule,
            lbvf_neg_linearity_rule,
            lbvf_antisym_rule,
            lbvf_jacobi_rule,
            sn_to_lbvf_rule,
            tilde_iota_lbvf_rule,
            me_pairing_flip_rule,
            permissive_iota_rule,
            vf_act_on_df_rule,
            me_cov_d_arity1_rule,
            vf_act_on_lf_rule,
            tilde_d_to_hvf_rule,
            hvf_inner_normalize_rule,
            sn_neg_rule,
            tilde_lie_lich_rule,
            sn_sum_rule,
            lbvf_sum_rule,
            CartanRemainderDefinition(d=self._d, lie_derivative=self._lie),
            TildeCartanRemainderDefinition(),
            self._tilde_swap_rule,
        ])
        engine.register(AtomSlotLiftDefinition(inner_engine))
        engine.register(BareAtomSlotLiftDefinition(inner_engine))
        return engine

    def prove_derivator(
        self,
        lhs: Expr,
        rhs: Expr,
        *,
        eval_args: Tuple[Expr, ...],
        side: str = "form",
        alternating: bool = True,
    ):
        r"""Prove a Section 3.1.5 derivator identity by ``p``-tuple evaluation.

        Wraps both sides under ``MultiEval`` and delegates to
        :func:`~jacopy.calculus.derivator.prove_derivator_identity`.

        Parameters
        ----------
        lhs, rhs
            Operator-valued expressions whose equality is to be shown.
        eval_args
            Tuple of expressions to evaluate against, vector fields
            for ``side="form"``, 1-forms for ``side="multivector"``.
        side
            ``"form"`` (default) routes through
            :meth:`derivator_form_engine` with
            ``slot_kind="vector"``; ``"multivector"`` routes through
            :meth:`derivator_multivector_engine` with
            ``slot_kind="covector"``.
        alternating
            Whether the :class:`MultiEval` wrap is graded-antisymmetric
            in its argument slots; defaults to ``True``.
        """
        if side == "form":
            engine = self.derivator_form_engine()
            slot_kind = "vector"
        elif side == "multivector":
            engine = self.derivator_multivector_engine()
            slot_kind = "covector"
        else:
            raise ValueError(
                "KoszulProblem.prove_derivator side must be "
                "'form' or 'multivector'"
            )
        # Pre-pass, canonicalize operator-atom index slots before the
        # MultiEval wrap. The engine's bottom-up walk treats those
        # slots as opaque (see operator_atom_index_opacity memo); the
        # pass expands K̃/K-remainder atoms there and distributes any
        # Sum/Neg index outward so the post-MultiEval intrinsic rules
        # see fully-flat operator heads.
        lhs = canonicalize_indices(lhs, engine, self._registry)
        rhs = canonicalize_indices(rhs, engine, self._registry)
        return prove_derivator_identity(
            lhs,
            rhs,
            engine=engine,
            eval_args=eval_args,
            slot_kind=slot_kind,
            alternating=alternating,
            registry=self._registry,
        )

    # ---- helpers --------------------------------------------------- #

    def anchor(self, alpha: Expr) -> Expr:
        r"""``ρ(α) = π^♯(α)`` as an :class:`Act` node."""
        if not isinstance(alpha, Expr):
            raise TypeError("KoszulProblem.anchor requires an Expr operand")
        return Act(self._sharp, alpha)

    def bracket(self, alpha: Expr, beta: Expr) -> Expr:
        r"""Inert ``[α, β]_K`` node (a :class:`BracketApply`).

        Use the engine to expand it: the bundled
        :class:`KoszulBracketExpansionDefinition` will fire on this
        node and rewrite to the Cartan formula
        ``L_{ρα}β − L_{ρβ}α − d⟨ρα, β⟩``.
        """
        if not isinstance(alpha, Expr):
            raise TypeError("KoszulProblem.bracket alpha must be an Expr")
        if not isinstance(beta, Expr):
            raise TypeError("KoszulProblem.bracket beta must be an Expr")
        return BracketApply(self._koszul, alpha, beta)

    def tilde_d(self) -> TildeExteriorDerivative:
        r"""``d̃ = [π, ·]_SN``, the Lichnerowicz differential bound to ``π``.

        Returns a fresh :class:`TildeExteriorDerivative` instance; structural
        equality on the class makes it compare equal to any other
        ``TildeExteriorDerivative(π)`` built elsewhere, so engine rules
        registered on this problem will fire on the returned head.
        """
        return TildeExteriorDerivative(self._pi)

    def tilde_interior(self, omega: Expr) -> TildeInteriorProduct:
        r"""``ι̃_ω``, form-indexed contraction on multivectors.

        ``omega`` is a 1-form; the engine's :class:`TildeIotaSwapDefinition`
        will rewrite ``Act(ι̃_ω, V) → Act(ι_V, ω)``.
        """
        if not isinstance(omega, Expr):
            raise TypeError(
                "KoszulProblem.tilde_interior omega must be an Expr"
            )
        return TildeInteriorProduct(omega)

    def tilde_lie(self, omega: Expr) -> TildeLieDerivative:
        r"""``L̃_ω = d̃∘ι̃_ω + ι̃_ω∘d̃``, tilde Lie derivative bound to ``π``.

        The form parameter ``omega`` is the indexing 1-form; the bivector
        is the problem's ``π``. The engine's :class:`TildeLieMagicDefinition`
        will rewrite ``Act(L̃_ω, V)`` into the Cartan magic Sum on ``π``.
        """
        if not isinstance(omega, Expr):
            raise TypeError("KoszulProblem.tilde_lie omega must be an Expr")
        return TildeLieDerivative(omega, self._pi)

    def assume_poisson(self) -> None:
        r"""Declare ``[π, π]_SN = 0`` on the registry, Poisson condition.

        Idempotent: calling this on a problem whose ``π`` already carries
        :class:`Poisson` is a no-op, so callers can invoke it freely from
        proof scripts without bookkeeping. Auxiliary engine rules added in
        Faz 14.D consume this flag (e.g. ``d̃² V → 0``).
        """
        if not self._registry.has(self._pi, Poisson):
            self._registry.declare(self._pi, Poisson())

    def bracket_expansion(self, alpha: Expr, beta: Expr) -> Expr:
        r"""Cartan expansion of ``[α, β]_K`` as a :class:`Sum` node.

        Skips the inert :class:`BracketApply` shape, equivalent to
        feeding :meth:`bracket` through the engine, but cheaper when
        the caller wants the expanded form directly (e.g. as the RHS
        of a manually authored proof step).
        """
        if not isinstance(alpha, Expr):
            raise TypeError("KoszulProblem.bracket_expansion alpha must be an Expr")
        if not isinstance(beta, Expr):
            raise TypeError("KoszulProblem.bracket_expansion beta must be an Expr")
        return self._koszul.expand(alpha, beta, self._registry)

    # ---- dunder --------------------------------------------------- #

    def __repr__(self) -> str:
        forms_repr = ", ".join(a._repr_inner() for a in self._forms)
        return f"KoszulProblem({self._pi._repr_inner()}; forms={{{forms_repr}}})"
