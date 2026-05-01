r"""
Tilde-side closure axioms, Faz 14.G.

Two engine rewrite rules that close the residues left by the Faz 14.E
intrinsic formulas on the harder Cartan relations
(``[L̃_α, L̃_β] = L̃_{[α,β]_K}`` rel-4 and ``[L̃_α, d̃] = 0`` rel-6):

* :class:`MultiEvalLieCommutatorSlotDefinition`, Sum-level rule that
  combines two ``MultiEval`` children differing in exactly one slot,
  where the differing slots are
  ``Act(LieDerivative(X), Act(LieDerivative(Y), ω))`` (positive child)
  and ``Act(LieDerivative(Y), Act(LieDerivative(X), ω))`` (negative
  child) for the same ``X, Y, ω``. The pair collapses into a single
  ``MultiEval`` whose differing slot is
  ``Act(LieDerivative(LieBracketVF(X, Y)), ω)``, the operator-level
  Lie-commutator-equals-bracket-VF identity, lifted from the bare
  ``Act`` level (Faz 13.C ``OpCommutatorVfDefinition``) up into a
  ``MultiEval`` slot.

* :class:`AnchorLieHomomorphismDefinition`, fires on a
  ``LieBracketVF(Act(Sharp(π), α), Act(Sharp(π), β))`` (anywhere in the
  expression tree) and rewrites to
  ``Act(Sharp(π), BracketApply(koszul, α, β))``, the Lie-algebra
  homomorphism ``π^♯ : (Ω¹, [·,·]_K) → (X(M), [·,·]_VF)`` that
  expresses the integrability of ``π``. Poisson-flag-gated: only fires
  when ``π`` carries the :class:`~jacopy.core.properties.Poisson`
  property in the supplied registry, since the identity is exactly
  ``[π, π]_SN = 0``.

Together with the ``_is_plain_vf`` extension that lets
:class:`~jacopy.calculus.closure_axioms.VfActCommutatorDefinition` fire
on ``Act(Sharp(π), α)`` heads (anchor-image vector fields), these two
rules close the residue from the rel-4 / rel-6 Cartan-magic chain.
"""

from __future__ import annotations

from itertools import combinations
from typing import Optional, Tuple

from jacopy.algebra.derivation import Act
from jacopy.algebra.lie_bracket_vf import LieBracketVF
from jacopy.brackets.base import BracketApply
from jacopy.brackets.koszul import KoszulBracket
from jacopy.calculus.exterior_d import ExteriorDerivative
from jacopy.calculus.hamiltonian_vf import HamiltonianVectorField
from jacopy.calculus.lie_derivative import (
    LieDerivative,
    lie_derivative as default_lie_derivative,
)
from jacopy.calculus.musical import Sharp
from jacopy.calculus.pairing import Pairing
from jacopy.core.expr import Expr, Neg, Sum
from jacopy.core.multi_eval import MultiEval
from jacopy.core.properties import Poisson
from jacopy.core.registry import PropertyRegistry
from jacopy.proof.expansion import Definition


# --------------------------------------------------------------------- #
# Helpers                                                                #
# --------------------------------------------------------------------- #


def _strip_neg(expr: Expr) -> Tuple[bool, Expr]:
    if isinstance(expr, Neg):
        return True, expr.arg
    return False, expr


def _match_lie_lie_slot(
    expr: Expr,
) -> Optional[Tuple[LieDerivative, LieDerivative, Expr]]:
    """Match ``Act(L_X, Act(L_Y, ω))`` for two LieDerivative atoms.

    Returns ``(L_X, L_Y, ω)`` or ``None``. ``L_X != L_Y`` is required,
    otherwise the commutator is trivially zero and there's nothing to
    fold.
    """
    if not isinstance(expr, Act):
        return None
    outer = expr.op
    inner = expr.arg
    if not isinstance(outer, LieDerivative):
        return None
    if not isinstance(inner, Act):
        return None
    if not isinstance(inner.op, LieDerivative):
        return None
    if outer == inner.op:
        return None
    return outer, inner.op, inner.arg


# --------------------------------------------------------------------- #
# Slot-aware Lie commutator                                              #
# --------------------------------------------------------------------- #


class MultiEvalLieCommutatorSlotDefinition(Definition):
    r"""Slot-level ``[L_X, L_Y] ω`` collapse inside a ``MultiEval`` Sum pair.

    Sum-level pair finder. Scans children for a positive
    ``MultiEval(V, ..., Act(L_X, Act(L_Y, ω)), ..., other_args)`` and a
    negated mirror
    ``Neg(MultiEval(V, ..., Act(L_Y, Act(L_X, ω)), ..., other_args))``
    whose ``MultiEval`` shells coincide on every slot except the
    distinguished one. The two children are removed and replaced with
    a single
    ``MultiEval(V, ..., Act(LieDerivative(LieBracketVF(X, Y)), ω),
    ..., other_args)``, the bracket-VF-acting-on-the-form form of the
    operator commutator, lifted into the slot.

    Why this rule is needed: after the Faz 14.E tilde-intrinsic
    expansion, residues like
    ``+V(L_π^♯(α)(L_π^♯(η)(ξ))) − V(L_π^♯(η)(L_π^♯(α)(ξ)))`` appear
    inside ``MultiEval`` slots. The Faz 13.C
    :class:`~jacopy.calculus.vf_axioms.OpCommutatorVfDefinition` only
    fires at the *bare-Act* Sum level; it can't reach into a
    ``MultiEval`` slot. This rule lifts that operator-level identity to
    the slot level, producing
    ``+V(L_{[π^♯(α), π^♯(η)]_VF}(ξ))``, at which point
    :class:`AnchorLieHomomorphismDefinition` rewrites the inner bracket
    via the Poisson identity and the bracket-expansion rule unfolds
    ``[α, η]_K``, lining the term up with the sibling
    ``V([[α,η]_K, ξ]_K)`` term in the residue for cancellation.

    The matched ``LieDerivative`` atoms can be constructed via any
    factory; the rule's ``lie_derivative_factory`` parameter selects
    which factory builds the resulting
    ``LieDerivative(LieBracketVF(X, Y))``, pass a custom one when the
    host calculus uses a non-default ``d`` or interior product.
    """

    name = "V(..., L_X(L_Y(ω)), ...) − V(..., L_Y(L_X(ω)), ...) = V(..., L_[X,Y]_VF(ω), ...)"

    def __init__(self, *, lie_derivative_factory=None) -> None:
        self._lie = (
            lie_derivative_factory
            if lie_derivative_factory is not None
            else default_lie_derivative
        )

    def matches(self, expr: Expr) -> bool:
        return isinstance(expr, Sum) and self._find_pair(expr) is not None

    def rewrite(self, expr: Expr) -> Expr:
        match = self._find_pair(expr)
        assert match is not None, "matches() guarantees a pair"
        i, j, sign, slot_idx, parent, X, Y, omega = match
        bracket_vf = LieBracketVF(X, Y)
        new_slot = Act(self._lie(bracket_vf), omega)
        new_args = list(parent.args)
        new_args[slot_idx] = new_slot
        new_term: Expr = MultiEval(
            parent.head,
            *new_args,
            alternating=parent.alternating,
            slot_kind=parent.slot_kind,
        )
        if sign < 0:
            new_term = Neg(new_term)
        kept = [c for k, c in enumerate(expr.children) if k != i and k != j]
        return Sum.make(new_term, *kept)

    def _find_pair(self, sum_expr: Sum):
        """Locate the first cancelling slot-Lie pair in the Sum.

        Returns ``(i, j, sign, slot_idx, parent_pos, X, Y, ω)`` where:
          * ``i, j``, child indices in the Sum
          * ``sign``, overall sign to attach to the merged result
            (``+1`` if the +child came in canonical order ``L_X∘L_Y``,
            ``-1`` if it came in reversed order, see the docstring of
            the rewrite for the algebra)
          * ``slot_idx``, index of the differing slot in the MultiEval
          * ``parent_pos``, the +child's MultiEval (used to inherit
            head, alternating, slot_kind, and other-slot args)
          * ``X, Y, ω``, operator commutator components, ordered so
            that ``+V(L_X(L_Y(ω))) − V(L_Y(L_X(ω))) = V(L_[X,Y]_VF(ω))``
        """
        children = sum_expr.children
        for i, j in combinations(range(len(children)), 2):
            for a, b in ((i, j), (j, i)):
                neg_a, inner_a = _strip_neg(children[a])
                neg_b, inner_b = _strip_neg(children[b])
                # Must have opposite outer signs (one positive, one
                # negated) for the commutator to net to non-trivial.
                if neg_a == neg_b:
                    continue
                if not isinstance(inner_a, MultiEval):
                    continue
                if not isinstance(inner_b, MultiEval):
                    continue
                if not self._shells_match(inner_a, inner_b):
                    continue
                slot_match = self._find_unique_diff_slot(inner_a, inner_b)
                if slot_match is None:
                    continue
                slot_idx, slot_a, slot_b = slot_match
                pair_a = _match_lie_lie_slot(slot_a)
                pair_b = _match_lie_lie_slot(slot_b)
                if pair_a is None or pair_b is None:
                    continue
                LXa, LYa, omega_a = pair_a
                LYb, LXb, omega_b = pair_b
                # The two inner Lie pairs must be the swap of each other
                # acting on the same form.
                if LXa != LXb or LYa != LYb or omega_a != omega_b:
                    continue
                # Extract the underlying VFs from the LieDerivative atoms.
                X = LXa.vector_field
                Y = LYa.vector_field
                # Decide the merged sign:
                #   neg_a = False, neg_b = True  → +V(L_X L_Y ω) − V(L_Y L_X ω)
                #     = +V(L_[X,Y]_VF ω) → sign = +1
                #   neg_a = True,  neg_b = False → −V(L_X L_Y ω) + V(L_Y L_X ω)
                #     = −V(L_[X,Y]_VF ω) → sign = −1
                sign = +1 if not neg_a else -1
                return (
                    a,
                    b,
                    sign,
                    slot_idx,
                    inner_a,
                    X,
                    Y,
                    omega_a,
                )
        return None

    @staticmethod
    def _shells_match(a: MultiEval, b: MultiEval) -> bool:
        """Same head, arity, alternating flag, slot kind."""
        return (
            a.head == b.head
            and len(a.args) == len(b.args)
            and a.alternating == b.alternating
            and a.slot_kind == b.slot_kind
        )

    @staticmethod
    def _find_unique_diff_slot(
        a: MultiEval, b: MultiEval
    ) -> Optional[Tuple[int, Expr, Expr]]:
        """Return ``(idx, a.args[idx], b.args[idx])`` if exactly one slot differs."""
        diff_indices = [
            k for k in range(len(a.args)) if a.args[k] != b.args[k]
        ]
        if len(diff_indices) != 1:
            return None
        idx = diff_indices[0]
        return idx, a.args[idx], b.args[idx]


# --------------------------------------------------------------------- #
# Anchor Lie-algebra homomorphism                                        #
# --------------------------------------------------------------------- #


class AnchorLieHomomorphismDefinition(Definition):
    r"""``[π^♯α, π^♯β]_VF → π^♯([α, β]_K)``, Poisson-gated.

    Fires on a :class:`LieBracketVF` whose two operands are both
    ``Act(Sharp(π), _)`` for the same ``Sharp(π)``, and rewrites the
    bracket as ``Act(Sharp(π), BracketApply(koszul, α, β))``.

    The identity ``π^♯ : (Ω¹, [·,·]_K) → (X(M), [·,·]_VF)`` being a Lie
    algebra homomorphism is *equivalent* to ``[π, π]_SN = 0``, i.e. to
    ``π`` being a Poisson bivector. The rule is therefore registry-
    gated: it only fires when ``π`` carries the
    :class:`~jacopy.core.properties.Poisson` property, the same flag
    set by :meth:`~jacopy.library.koszul_problem.KoszulProblem.assume_poisson`
    that Faz 14.D's
    :class:`~jacopy.calculus.tilde.aux_axioms.TildeDSquaredPoissonDefinition`
    already consumes.

    Used by the rel-4 / rel-6 closure pipeline: after
    :class:`MultiEvalLieCommutatorSlotDefinition` produces
    ``Act(L_{[π^♯α, π^♯η]_VF}, ω)``, this rule rewrites the inner
    ``LieBracketVF(π^♯α, π^♯η)`` to ``Act(Sharp(π), [α, η]_K)``,
    lining up the merged term with the sibling ``V([[α,η]_K, ξ]_K)``
    bracket residue for cancellation under bracket-expansion.

    Tree-traversal note: the engine descends through :class:`Act` and
    other compound shapes, so a ``LieBracketVF`` sitting as the
    ``vector_field`` of a :class:`LieDerivative` is reached by
    matching at the LieDerivative-as-children level, but this rule
    matches the bare ``LieBracketVF`` node directly, wherever it
    appears. (LieDerivative itself is opaque; that case is handled by
    a sibling rule that matches the LieDerivative shape.)
    """

    def __init__(
        self,
        pi: Expr,
        koszul: KoszulBracket,
        *,
        registry: PropertyRegistry,
        sharp: Optional[Sharp] = None,
    ) -> None:
        if not isinstance(pi, Expr):
            raise TypeError("AnchorLieHomomorphismDefinition pi must be an Expr")
        if not isinstance(koszul, KoszulBracket):
            raise TypeError(
                "AnchorLieHomomorphismDefinition koszul must be a KoszulBracket"
            )
        if not isinstance(registry, PropertyRegistry):
            raise TypeError(
                "AnchorLieHomomorphismDefinition registry must be a PropertyRegistry"
            )
        if sharp is not None and not isinstance(sharp, Sharp):
            raise TypeError(
                "AnchorLieHomomorphismDefinition sharp must be a Sharp instance"
            )
        self._pi = pi
        self._koszul = koszul
        self._registry = registry
        self._sharp = sharp if sharp is not None else Sharp(pi)
        self.name = (
            f"[π^♯α, π^♯β]_VF = π^♯([α,β]_K) [{pi._repr_inner()}]"
        )

    @property
    def pi(self) -> Expr:
        return self._pi

    @property
    def koszul(self) -> KoszulBracket:
        return self._koszul

    @property
    def sharp(self) -> Sharp:
        return self._sharp

    def _is_anchor_image(self, expr: Expr) -> Optional[Expr]:
        """Return ``α`` if ``expr == Act(Sharp(π), α)``, else ``None``."""
        if not isinstance(expr, Act):
            return None
        if expr.op != self._sharp:
            return None
        return expr.arg

    def matches(self, expr: Expr) -> bool:
        if not isinstance(expr, LieBracketVF):
            return False
        if not self._registry.has(self._pi, Poisson):
            return False
        return (
            self._is_anchor_image(expr.X) is not None
            and self._is_anchor_image(expr.Y) is not None
        )

    def rewrite(self, expr: Expr) -> Expr:
        assert isinstance(expr, LieBracketVF), "matches() guarantees LieBracketVF"
        alpha = self._is_anchor_image(expr.X)
        beta = self._is_anchor_image(expr.Y)
        assert alpha is not None and beta is not None
        return Act(self._sharp, BracketApply(self._koszul, alpha, beta))


# --------------------------------------------------------------------- #
# Lie-derivative-of-bracket-VF rewrite                                   #
# --------------------------------------------------------------------- #


class LieDerivativeOfAnchorBracketDefinition(Definition):
    r"""Rewrite ``LieDerivative(LieBracketVF(π^♯α, π^♯β))`` under Poisson.

    The :class:`LieDerivative` is opaque to engine tree traversal, its
    ``vector_field`` doesn't surface as a child, so the bare
    :class:`AnchorLieHomomorphismDefinition` cannot reach a
    ``LieBracketVF`` sitting inside one. This rule fills that gap: it
    matches a :class:`LieDerivative` whose ``vector_field`` is a
    qualifying ``LieBracketVF(Act(Sharp(π), α), Act(Sharp(π), β))`` and
    rebuilds it as ``LieDerivative(Act(Sharp(π), [α, β]_K))``, the
    same shape used by the Koszul bracket-expansion rule's output.

    Poisson-gated for the same reason as
    :class:`AnchorLieHomomorphismDefinition`. Engine match-order matters:
    this rule must register *before* the bare-bracket variant so that
    the LieDerivative wrapper gets handled in one rewrite step instead
    of leaking the inner bracket through tree traversal (which doesn't
    happen anyway, since LieDerivative.vector_field isn't a child).
    """

    def __init__(
        self,
        pi: Expr,
        koszul: KoszulBracket,
        *,
        registry: PropertyRegistry,
        sharp: Optional[Sharp] = None,
        lie_derivative_factory=None,
    ) -> None:
        if not isinstance(pi, Expr):
            raise TypeError(
                "LieDerivativeOfAnchorBracketDefinition pi must be an Expr"
            )
        if not isinstance(koszul, KoszulBracket):
            raise TypeError(
                "LieDerivativeOfAnchorBracketDefinition koszul must be a KoszulBracket"
            )
        if not isinstance(registry, PropertyRegistry):
            raise TypeError(
                "LieDerivativeOfAnchorBracketDefinition registry must be a "
                "PropertyRegistry"
            )
        if sharp is not None and not isinstance(sharp, Sharp):
            raise TypeError(
                "LieDerivativeOfAnchorBracketDefinition sharp must be a Sharp"
            )
        self._pi = pi
        self._koszul = koszul
        self._registry = registry
        self._sharp = sharp if sharp is not None else Sharp(pi)
        self._lie = (
            lie_derivative_factory
            if lie_derivative_factory is not None
            else default_lie_derivative
        )
        self.name = (
            f"L_{{[π^♯α, π^♯β]_VF}} = L_{{π^♯([α,β]_K)}} [{pi._repr_inner()}]"
        )

    def _is_anchor_image(self, expr: Expr) -> Optional[Expr]:
        if not isinstance(expr, Act):
            return None
        if expr.op != self._sharp:
            return None
        return expr.arg

    def matches(self, expr: Expr) -> bool:
        if not isinstance(expr, LieDerivative):
            return False
        if not self._registry.has(self._pi, Poisson):
            return False
        vf = expr.vector_field
        if not isinstance(vf, LieBracketVF):
            return False
        return (
            self._is_anchor_image(vf.X) is not None
            and self._is_anchor_image(vf.Y) is not None
        )

    def rewrite(self, expr: Expr) -> Expr:
        assert isinstance(expr, LieDerivative)
        vf = expr.vector_field
        assert isinstance(vf, LieBracketVF)
        alpha = self._is_anchor_image(vf.X)
        beta = self._is_anchor_image(vf.Y)
        assert alpha is not None and beta is not None
        # Pre-expand the Koszul bracket inline. The Sharp-image wraps the
        # entire bracket-expanded payload, and *that* payload sits inside
        # the new ``LieDerivative``'s opaque ``vector_field``. If we left
        # it as ``BracketApply``, the engine couldn't traverse into the
        # LieDerivative to fire the standard bracket-expansion rule on
        # it. Pre-expanding here matches the shape produced elsewhere by
        # the regular bracket-expansion path (``BracketApply`` from
        # outer slots → bracket-expanded directly), so the merged term
        # lines up with sibling residue terms for cancellation.
        expanded = self._koszul.expand(alpha, beta)
        new_vf = Act(self._sharp, expanded)
        return self._lie(new_vf)


# --------------------------------------------------------------------- #
# [L_X, d] = 0 in any LieDerivative mode (tilde-engine local)            #
# --------------------------------------------------------------------- #


class LieCommutesWithDTildeDefinition(Definition):
    r"""``L_X(d ω) → d(L_X ω)`` for any :class:`LieDerivative` mode.

    Tilde-engine variant of
    :class:`~jacopy.proof.expansion.LieDerivativeCommutesWithDDefinition`
    that drops the flow-mode restriction. The identity ``[L_X, d] = 0``
    is true in both axiomatic modes; the flow-mode gate on the
    proof-engine version was conservative, it kept Cartan-mode
    expansions from racing with their own magic-formula unfold. In the
    tilde engine the L̃/d̃ intrinsic rules carry the heavy lifting and
    the ``L_X(d…)`` shapes left in residues come from Koszul-bracket
    expansion's ``L_πα`` factors. Allowing the rule to fire on those
    cartan-mode atoms is exactly what surfaces ``d(L_πα⟨…⟩)`` for the
    pairing-Leibniz rule to consume next.
    """

    name = "L_X ∘ d = d ∘ L_X (tilde, any mode)"

    def matches(self, expr: Expr) -> bool:
        if not (isinstance(expr, Act) and isinstance(expr.op, LieDerivative)):
            return False
        inner = expr.arg
        if not (isinstance(inner, Act) and isinstance(inner.op, ExteriorDerivative)):
            return False
        L: LieDerivative = expr.op  # type: ignore[assignment]
        if L.d is not None and inner.op != L.d:
            return False
        return True

    def rewrite(self, expr: Expr) -> Expr:
        L = expr.op
        d_inner = expr.arg.op  # type: ignore[union-attr]
        omega = expr.arg.arg  # type: ignore[union-attr]
        return Act(d_inner, Act(L, omega))


# --------------------------------------------------------------------- #
# L_{π^♯α}(π^♯β) → π^♯([α,β]_K), Poisson-gated                          #
# --------------------------------------------------------------------- #


class LieDerivativeOnAnchorImageDefinition(Definition):
    r"""``Act(L_{π^♯α}, Act(Sharp(π), β)) → Act(Sharp(π), [α,β]_K)``.

    Anchor-Lie-homomorphism in *operator-application* form. Under
    Poisson the identity

        L_{π^♯α}(π^♯β) = [π^♯α, π^♯β]_VF = π^♯([α, β]_K)

    holds on vector fields. The closure pipeline reaches this shape
    after :class:`LieCommutesWithDTildeDefinition` pushes ``L_πα`` past
    a ``d`` and :class:`PairingLieLeibnizDefinition` (the existing rule
    in :mod:`jacopy.calculus.pairing_axioms`) opens the pairing's
    Leibniz, leaving an
    ``Act(L_πα, Act(Sharp(π), β))`` term in the first pairing slot.

    Poisson-gated for the same reason as the bracket-VF variants: the
    identity is equivalent to ``[π, π]_SN = 0``. Pre-expands the Koszul
    bracket inline so the resulting ``Act(Sharp(π), …)`` matches the
    shape produced by direct bracket-expansion elsewhere in the
    residue, enabling syntactic cancellation.
    """

    def __init__(
        self,
        pi: Expr,
        koszul: KoszulBracket,
        *,
        registry: PropertyRegistry,
        sharp: Optional[Sharp] = None,
    ) -> None:
        if not isinstance(pi, Expr):
            raise TypeError(
                "LieDerivativeOnAnchorImageDefinition pi must be an Expr"
            )
        if not isinstance(koszul, KoszulBracket):
            raise TypeError(
                "LieDerivativeOnAnchorImageDefinition koszul must be a "
                "KoszulBracket"
            )
        if not isinstance(registry, PropertyRegistry):
            raise TypeError(
                "LieDerivativeOnAnchorImageDefinition registry must be a "
                "PropertyRegistry"
            )
        if sharp is not None and not isinstance(sharp, Sharp):
            raise TypeError(
                "LieDerivativeOnAnchorImageDefinition sharp must be a Sharp"
            )
        self._pi = pi
        self._koszul = koszul
        self._registry = registry
        self._sharp = sharp if sharp is not None else Sharp(pi)
        self.name = (
            f"L_{{π^♯α}}(π^♯β) = π^♯([α,β]_K) [{pi._repr_inner()}]"
        )

    def _is_anchor_image(self, expr: Expr) -> Optional[Expr]:
        if not isinstance(expr, Act):
            return None
        if expr.op != self._sharp:
            return None
        return expr.arg

    def matches(self, expr: Expr) -> bool:
        if not isinstance(expr, Act):
            return False
        L = expr.op
        if not isinstance(L, LieDerivative):
            return False
        if not self._registry.has(self._pi, Poisson):
            return False
        # X in L_X must itself be an anchor-image Act(Sharp(π), α).
        if self._is_anchor_image(L.vector_field) is None:
            return False
        # Argument must be an anchor-image Act(Sharp(π), β).
        return self._is_anchor_image(expr.arg) is not None

    def rewrite(self, expr: Expr) -> Expr:
        assert isinstance(expr, Act)
        L = expr.op
        assert isinstance(L, LieDerivative)
        alpha = self._is_anchor_image(L.vector_field)
        beta = self._is_anchor_image(expr.arg)
        assert alpha is not None and beta is not None
        expanded = self._koszul.expand(alpha, beta)
        return Act(self._sharp, expanded)


# --------------------------------------------------------------------- #
# Hamiltonian-pairing antisymmetry from π, Sum-level cancellation      #
# --------------------------------------------------------------------- #


class HamiltonianAnchorPairingAntisymmetryDefinition(Definition):
    r"""``X_⟨π^♯a, b⟩(c) + X_⟨π^♯b, a⟩(c) → 0``, Sum-level π antisymmetry.

    The bivector ``π`` is antisymmetric, so for any 1-forms ``a, b``

        ⟨π^♯a, b⟩ = π(a, b) = −π(b, a) = −⟨π^♯b, a⟩,

    hence ``X_⟨π^♯a, b⟩ = −X_⟨π^♯b, a⟩`` as Hamiltonian vector fields.
    Their actions on the same operand cancel pairwise. The rule
    finds two children of a :class:`Sum` whose shape is
    ``Act(X_⟨π^♯a, b⟩, c)`` and ``Act(X_⟨π^♯b, a⟩, c)`` with matching
    outer signs (both bare or both ``Neg``-wrapped) and removes them.

    Why a Sum-level rule rather than a Pairing rewrite: a Pairing
    rewrite ``⟨π^♯a, b⟩ → −⟨π^♯b, a⟩`` would oscillate without a
    canonical key. The Sum-level matched-pair cancellation is
    deterministic and idempotent, and it leaves untouched any
    non-cancellable shapes (e.g. ``X_⟨π^♯a, b⟩(c) − X_⟨π^♯b, a⟩(d)``
    where ``c ≠ d``).

    Poisson-gated: the identity is purely about ``π`` antisymmetry,
    which the Poisson property pins. Without the gate the rule would
    fire for arbitrary anchor-image Hamiltonians whose ``Sharp`` happens
    to be the supplied one, and that's exactly the safe scope.
    """

    def __init__(
        self,
        pi: Expr,
        *,
        registry: PropertyRegistry,
        sharp: Optional[Sharp] = None,
    ) -> None:
        if not isinstance(pi, Expr):
            raise TypeError(
                "HamiltonianAnchorPairingAntisymmetryDefinition pi must be Expr"
            )
        if not isinstance(registry, PropertyRegistry):
            raise TypeError(
                "HamiltonianAnchorPairingAntisymmetryDefinition registry must "
                "be a PropertyRegistry"
            )
        if sharp is not None and not isinstance(sharp, Sharp):
            raise TypeError(
                "HamiltonianAnchorPairingAntisymmetryDefinition sharp must be "
                "a Sharp"
            )
        self._pi = pi
        self._registry = registry
        self._sharp = sharp if sharp is not None else Sharp(pi)
        self.name = (
            f"X_⟨π^♯a,b⟩ + X_⟨π^♯b,a⟩ = 0 [{pi._repr_inner()}]"
        )

    def _peel_term(
        self, term: Expr
    ) -> Optional[Tuple[bool, Expr, Expr, Expr]]:
        """Return ``(neg, a, b, c)`` for ``Act(X_⟨π^♯a, b⟩, c)`` shape.

        ``neg`` is the outer-Neg flag. ``a`` is the form whose anchor is
        the first pairing slot, ``b`` is the second pairing slot,
        ``c`` is the operand the Hamiltonian acts on.
        """
        neg, inner = _strip_neg(term)
        if not isinstance(inner, Act):
            return None
        op = inner.op
        c = inner.arg
        if not isinstance(op, HamiltonianVectorField):
            return None
        if op.bivector != self._pi:
            return None
        f = op.function
        if not isinstance(f, Pairing):
            return None
        # First slot must be Act(Sharp(π), a).
        first = f.alpha
        if not (isinstance(first, Act) and first.op == self._sharp):
            return None
        a = first.arg
        b = f.X
        return neg, a, b, c

    def matches(self, expr: Expr) -> bool:
        if not isinstance(expr, Sum):
            return False
        if not self._registry.has(self._pi, Poisson):
            return False
        return self._find_pair(expr) is not None

    def rewrite(self, expr: Expr) -> Expr:
        match = self._find_pair(expr)
        assert match is not None
        i, j = match
        kept = [c for k, c in enumerate(expr.children) if k != i and k != j]
        return Sum.make(*kept)

    def _find_pair(self, sum_expr: Sum) -> Optional[Tuple[int, int]]:
        children = sum_expr.children
        peeled = [(k, self._peel_term(c)) for k, c in enumerate(children)]
        candidates = [(k, p) for k, p in peeled if p is not None]
        for (i, p1), (j, p2) in combinations(candidates, 2):
            neg1, a1, b1, c1 = p1
            neg2, a2, b2, c2 = p2
            # Same outer sign, both X_⟨πa,b⟩(c) bare or both Neg-wrapped.
            if neg1 != neg2:
                continue
            # Operand must match.
            if c1 != c2:
                continue
            # Pairing slots must be swapped: a1=b2 and b1=a2.
            if a1 != b2 or b1 != a2:
                continue
            return i, j
        return None


# --------------------------------------------------------------------- #
# ⟨X, dF⟩ → L_X(F), pairing-with-exact-form identity                    #
# --------------------------------------------------------------------- #


class TildeSnJacobiResidueDefinition(Definition):
    r"""``Σ {5-term SN-Jacobi residue} → 0`` under Poisson.

    The Faz 14.G residual left by the rel-4 and rel-6 tilde-Cartan
    proofs (after :class:`WrappedPairingAnchorAntisymmetryDefinition`
    has cancelled its share) collapses to a 5-term sum that is exactly
    the Schouten-Nijenhuis-Jacobi obstruction
    ``[π,π]_SN(a, b, c) = 0`` evaluated through ``V ∘ d`` against the
    1-form triple at hand. The five children, modulo a shared outer
    wrapping ``W[·]`` (any chain of :class:`Act` ops or
    :class:`MultiEval` evaluation slots) and 1-form labels ``(a, b, c)``:

    * ``+W[⟨X_{⟨π^♯a, b⟩}, c⟩]``                 (Hamiltonian, sign ``+``)
    * ``-W[⟨X_{⟨π^♯c, a⟩}, b⟩]``                 (Hamiltonian, sign ``−``)
    * ``-W[⟨π^♯(L_{π^♯a} b), c⟩]``               (anchor-Lie, sign ``−``)
    * ``+W[⟨π^♯(L_{π^♯b} a), c⟩]``               (anchor-Lie, sign ``+``)
    * ``+W[⟨π^♯a, L_{π^♯b} c⟩]``                 (mixed, sign ``+``)

    Under the :class:`~jacopy.core.properties.Poisson` property the
    sum vanishes; the rule removes all five matched children. Multiple
    matches in the same Sum are handled across successive engine
    iterations, each call removes one quintuple.

    Why a focused recognizer rather than expanding the SN-Jacobi
    identity step-by-step: the constituent identities (Pairing
    Leibniz, ``L_{π^♯a}π^♯b = π^♯[a,b]_K``, ``X_F = π^♯(dF)``) interact
    multiplicatively, installing them all would oscillate without a
    canonical normalization, and the resulting search space dwarfs
    the recognizer cost. The 5-term shape is stable across rel-4 and
    rel-6 (verified empirically with ``assume_poisson()`` on a generic
    multivector ``V``), making this the cleanest closure path.

    Poisson-gated; declines if ``π`` is not registered as Poisson in
    the supplied registry.
    """

    def __init__(
        self,
        pi: Expr,
        *,
        registry: PropertyRegistry,
        sharp: Optional[Sharp] = None,
    ) -> None:
        if not isinstance(pi, Expr):
            raise TypeError(
                "TildeSnJacobiResidueDefinition pi must be Expr"
            )
        if not isinstance(registry, PropertyRegistry):
            raise TypeError(
                "TildeSnJacobiResidueDefinition registry must be a "
                "PropertyRegistry"
            )
        if sharp is not None and not isinstance(sharp, Sharp):
            raise TypeError(
                "TildeSnJacobiResidueDefinition sharp must be a Sharp"
            )
        self._pi = pi
        self._registry = registry
        self._sharp = sharp if sharp is not None else Sharp(pi)
        self.name = (
            f"[π,π]_SN(a,b,c) = 0 residue [{pi._repr_inner()}]"
        )

    # -- shape classifiers ------------------------------------------- #

    def _classify_pairing(
        self, pairing: Pairing
    ) -> Optional[Tuple[str, Tuple[Expr, Expr, Expr]]]:
        """Return ``(tag, (p, q, r))`` for one of the three Pairing shapes,
        or ``None``. Tags: ``"H"``, ``"AL"``, ``"M"``."""
        first, second = pairing.alpha, pairing.X

        # Hamiltonian shape: ⟨X_{⟨π^♯p, q⟩}, r⟩.
        if isinstance(first, HamiltonianVectorField):
            if first.bivector != self._pi:
                return None
            F = first.function
            if not isinstance(F, Pairing):
                return None
            x = F.alpha
            y = F.X
            if not (isinstance(x, Act) and x.op == self._sharp):
                return None
            return "H", (x.arg, y, second)

        # Anchor-image shape on first slot: ⟨π^♯..., ...⟩.
        if isinstance(first, Act) and first.op == self._sharp:
            inner = first.arg
            # Anchor-Lie: π^♯(L_{π^♯p} q).
            if isinstance(inner, Act) and isinstance(inner.op, LieDerivative):
                ld_X = inner.op.vector_field
                if isinstance(ld_X, Act) and ld_X.op == self._sharp:
                    return "AL", (ld_X.arg, inner.arg, second)
            # Mixed: ⟨π^♯p, L_{π^♯q} r⟩.
            p = inner
            if isinstance(second, Act) and isinstance(second.op, LieDerivative):
                ld_X = second.op.vector_field
                if isinstance(ld_X, Act) and ld_X.op == self._sharp:
                    return "M", (p, ld_X.arg, second.arg)

        return None

    def _classify(
        self, term: Expr
    ) -> Optional[Tuple[bool, Tuple, str, Tuple[Expr, Expr, Expr]]]:
        """Strip outer Neg + wrap (Acts / MultiEvals) until reaching a
        Pairing, then dispatch to :meth:`_classify_pairing`."""
        from jacopy.core.multi_eval import MultiEval as _MultiEval

        neg, inner = _strip_neg(term)
        wrap: list = []
        while True:
            if isinstance(inner, Pairing):
                break
            if isinstance(inner, Act):
                wrap.append(("Act", inner.op))
                inner = inner.arg
                continue
            if isinstance(inner, _MultiEval):
                if not inner.args:
                    return None
                wrap.append(
                    (
                        "MultiEval",
                        inner.head,
                        inner.args[1:],
                        inner.alternating,
                        inner.slot_kind,
                    )
                )
                inner = inner.args[0]
                continue
            return None
        classified = self._classify_pairing(inner)
        if classified is None:
            return None
        tag, forms = classified
        return neg, tuple(wrap), tag, forms

    # -- search ----------------------------------------------------- #

    def matches(self, expr: Expr) -> bool:
        if not isinstance(expr, Sum):
            return False
        if not self._registry.has(self._pi, Poisson):
            return False
        return self._find_quintuple(expr) is not None

    def rewrite(self, expr: Expr) -> Expr:
        idxs = self._find_quintuple(expr)
        assert idxs is not None
        kept = [c for k, c in enumerate(expr.children) if k not in idxs]
        return Sum.make(*kept)

    def _find_quintuple(self, sum_expr: Sum) -> Optional[set]:
        children = sum_expr.children
        classified = [
            (i, self._classify(c)) for i, c in enumerate(children)
        ]
        valid = [(i, t) for i, t in classified if t is not None]

        # Group by wrap signature.
        groups: dict = {}
        for i, (neg, wrap, tag, forms) in valid:
            groups.setdefault(wrap, []).append((i, neg, tag, forms))

        for members in groups.values():
            quintuple = self._search_group(members)
            if quintuple is not None:
                return quintuple
        return None

    def _search_group(self, members) -> Optional[set]:
        """Find five members that match the SN-Jacobi pattern for some
        triple ``(a, b, c)``. Tries both polarities, canonical (anchor
        on H+) and overall-sign-flipped (anchor on H−). Both encode the
        same ``[π,π]_SN(a,b,c) = 0`` identity; which one a downstream
        rewrite leaves depends on the wrap chain (e.g. an outer
        ``Neg`` from a ``MultiEval`` slot Leibniz)."""
        # polarity=False: canonical signs.
        # polarity=True: every sign flipped (whole sum still = 0).
        for polarity in (False, True):
            result = self._search_with_polarity(members, flip=polarity)
            if result is not None:
                return result
        return None

    def _search_with_polarity(self, members, *, flip) -> Optional[set]:
        H_anchor_neg = flip
        H_other_neg = not flip
        AL_first_neg = not flip
        AL_second_neg = flip
        M_neg = flip
        for i_anchor, neg_anchor, tag_anchor, forms_anchor in members:
            if tag_anchor != "H" or neg_anchor != H_anchor_neg:
                continue
            a, b, c = forms_anchor
            i_P2 = self._locate(members, neg=H_other_neg, tag="H", forms=(c, a, b))
            i_P3 = self._locate(members, neg=AL_first_neg, tag="AL", forms=(a, b, c))
            i_P4 = self._locate(members, neg=AL_second_neg, tag="AL", forms=(b, a, c))
            i_P5 = self._locate(members, neg=M_neg, tag="M", forms=(a, b, c))
            if None in (i_P2, i_P3, i_P4, i_P5):
                continue
            idxs = {i_anchor, i_P2, i_P3, i_P4, i_P5}
            if len(idxs) == 5:
                return idxs
        return None

    @staticmethod
    def _locate(members, *, neg, tag, forms) -> Optional[int]:
        for i, neg_m, tag_m, forms_m in members:
            if neg_m == neg and tag_m == tag and forms_m == forms:
                return i
        return None


# --------------------------------------------------------------------- #
# Wrapped pairing antisymmetry, Sum-level π antisym through wrappers   #
# --------------------------------------------------------------------- #


class WrappedPairingAnchorAntisymmetryDefinition(Definition):
    r"""``Σ ... + W[⟨π^♯a, b⟩] + W[⟨π^♯b, a⟩] + ... → Σ ... + ...``.

    Sum-level pair-cancellation that lifts the bivector antisymmetry
    ``⟨π^♯a, b⟩ + ⟨π^♯b, a⟩ = 0`` through any common chain of
    :class:`Act` wrappers. Two children of a :class:`Sum` cancel iff:

    * after stripping an outer :class:`Neg` they have the same sign,
    * peeling :class:`Act` operators from each yields the **same**
      sequence of operators (e.g. both wrapped by ``Act(V, Act(d, ·))``
      or both bare), and
    * the innermost shape on each is :class:`Pairing` whose first slot
      is ``Act(Sharp(π), a)`` for some 1-form ``a`` and second slot is
      a 1-form ``b``, with the two pairings related by the swap
      ``a ↔ b``.

    Why a wrapped variant: the residues left by rel-4 / rel-6 wrap
    each pairing in ``V(d(·))`` (or another bundle-specific outer
    chain). A Pairing-only rewrite ``⟨π^♯a, b⟩ → −⟨π^♯b, a⟩`` would
    oscillate without a canonical key; this Sum-level matched-pair
    cancellation is deterministic and idempotent.

    Poisson-gated: the identity is purely about ``π`` antisymmetry,
    which the Poisson property pins. Without the gate the rule would
    fire wherever a ``Sharp`` happens to match.
    """

    def __init__(
        self,
        pi: Expr,
        *,
        registry: PropertyRegistry,
        sharp: Optional[Sharp] = None,
    ) -> None:
        if not isinstance(pi, Expr):
            raise TypeError(
                "WrappedPairingAnchorAntisymmetryDefinition pi must be Expr"
            )
        if not isinstance(registry, PropertyRegistry):
            raise TypeError(
                "WrappedPairingAnchorAntisymmetryDefinition registry must be a "
                "PropertyRegistry"
            )
        if sharp is not None and not isinstance(sharp, Sharp):
            raise TypeError(
                "WrappedPairingAnchorAntisymmetryDefinition sharp must be a Sharp"
            )
        self._pi = pi
        self._registry = registry
        self._sharp = sharp if sharp is not None else Sharp(pi)
        self.name = (
            f"⟨π^♯a,b⟩ + ⟨π^♯b,a⟩ = 0 (wrapped) [{pi._repr_inner()}]"
        )

    def _peel(
        self, term: Expr
    ) -> Optional[Tuple[bool, Tuple[Expr, ...], Expr, Expr]]:
        """Return ``(neg, wrap_key, a, b)`` for ``±W[⟨π^♯a, b⟩]`` shape.

        ``neg`` is the outer-Neg flag. ``wrap_key`` is a tuple-encoded
        signature of the wrapping (Acts and MultiEval evaluations) around
        the pairing, outermost first. Two terms cancel only when their
        ``wrap_key`` tuples are equal. The innermost expression must be a
        :class:`Pairing` whose first slot is ``Act(Sharp(π), a)``.
        """
        from jacopy.core.multi_eval import MultiEval as _MultiEval

        neg, inner = _strip_neg(term)
        wrap: list = []
        while True:
            if isinstance(inner, Pairing):
                break
            if isinstance(inner, Act):
                wrap.append(("Act", inner.op))
                inner = inner.arg
                continue
            if isinstance(inner, _MultiEval):
                # Wrapping shape: MultiEval(head, arg_0, *rest_args).
                # Walk into arg_0 (the slot carrying the Pairing) and
                # encode the head + remaining args + flags so two
                # children only cancel under identical multi-eval wrap.
                if not inner.args:
                    return None
                wrap.append(
                    (
                        "MultiEval",
                        inner.head,
                        inner.args[1:],
                        inner.alternating,
                        inner.slot_kind,
                    )
                )
                inner = inner.args[0]
                continue
            return None
        first = inner.alpha
        if not (isinstance(first, Act) and first.op == self._sharp):
            return None
        a = first.arg
        b = inner.X
        return neg, tuple(wrap), a, b

    def matches(self, expr: Expr) -> bool:
        if not isinstance(expr, Sum):
            return False
        if not self._registry.has(self._pi, Poisson):
            return False
        return self._find_pair(expr) is not None

    def rewrite(self, expr: Expr) -> Expr:
        match = self._find_pair(expr)
        assert match is not None
        i, j = match
        kept = [c for k, c in enumerate(expr.children) if k != i and k != j]
        return Sum.make(*kept)

    def _find_pair(self, sum_expr: Sum) -> Optional[Tuple[int, int]]:
        peeled = [(k, self._peel(c)) for k, c in enumerate(sum_expr.children)]
        candidates = [(k, p) for k, p in peeled if p is not None]
        for (i, p1), (j, p2) in combinations(candidates, 2):
            neg1, w1, a1, b1 = p1
            neg2, w2, a2, b2 = p2
            if neg1 != neg2:
                continue
            if w1 != w2:
                continue
            if a1 != b2 or b1 != a2:
                continue
            return i, j
        return None


# --------------------------------------------------------------------- #
# ⟨X, dF⟩ → L_X(F), pairing-with-exact-form identity                    #
# --------------------------------------------------------------------- #


class PairingWithExactFormDefinition(Definition):
    r"""``⟨X, d F⟩ → L_X(F)``, pairing of a VF with an exact 1-form.

    For any vector field ``X`` and 0-form ``F``:

        ⟨X, dF⟩ = (dF)(X) = X(F) = L_X(F).

    The rule fires on a :class:`Pairing` whose second slot is
    ``Act(d, F)`` for any :class:`ExteriorDerivative` ``d`` (the
    bundle-specific override ``d_E`` is also accepted via structural
    equality). The rewrite emits ``Act(LieDerivative(X), F)`` so that
    downstream :class:`PairingLieLeibnizDefinition` /
    :class:`LieDerivativeOnAnchorImageDefinition` /
    :class:`LieCommutesWithDTildeDefinition` can pick up where this
    rule left off.

    This identity isn't restricted to anchor-image VFs, it's a basic
    pairing identity. Restricting the operator argument to a generic
    :class:`Expr` keeps the rule from firing on ``Pairing(form, dF)``
    where the first slot is itself a 1-form (no VF semantics). The
    matches() check only looks for ``Act(d, F)`` in the second slot;
    the caller's responsibility is to ensure the first slot is a VF.
    In practice the residues we close here have first-slot anchor-
    images or :class:`Sharp` applications, all of which are VFs.
    """

    name = "⟨X, dF⟩ = L_X(F)"

    def matches(self, expr: Expr) -> bool:
        if not isinstance(expr, Pairing):
            return False
        # Second slot must be Act(d, F) for some exterior derivative.
        rhs = expr.X
        if not isinstance(rhs, Act):
            return False
        if not isinstance(rhs.op, ExteriorDerivative):
            return False
        return True

    def rewrite(self, expr: Expr) -> Expr:
        from jacopy.calculus.lie_derivative import (
            lie_derivative as default_lie_derivative,
        )

        X = expr.alpha
        F = expr.X.arg  # type: ignore[union-attr]
        return Act(default_lie_derivative(X), F)
