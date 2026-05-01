"""
Koszul bracket on 1-forms.

Given an anchor ``ρ: T*M → TM`` the classical Koszul bracket on 1-forms
is

    [α, β]_K = L_{ρα} β − L_{ρβ} α − d⟨ρα, β⟩.

It is a degree-``0`` graded-antisymmetric bracket whose Jacobi identity
is *conditional*: it holds iff the anchor is compatible with the Lie
bracket on ``TM`` and the underlying generator (e.g. the Poisson
bivector that induces ``ρ = π^♯``) satisfies ``[Q, Q] = 0``. That
conditionality is declared here via ``satisfies_graded_jacobi=None``,
matching the derived-bracket convention.

The classical form lives here. The *derived* form ``{α, β}_{sn, π}`` and
the theorem stating the two agree when ``ρ = π^♯`` are built on top of
this module (Stage 2B, :class:`DerivedBracket` gains an ``acting_on``
parameter that lifts 1-form operands through an anchor, and a separate
proof-layer helper closes the equivalence as a :class:`ProofChain`).

The constructor mirrors :class:`DorfmanBracket`: the anchor is required,
and the exterior derivative + Lie-derivative factory default to the
smooth-manifold singletons. Lie-algebroid callers substitute their own
Cartan family.
"""

from __future__ import annotations

from typing import Any, Callable, Optional

from jacopy.algebra.derivation import Act, Derivation
from jacopy.brackets.base import GradedBracket
from jacopy.brackets.derived import DerivedBracket, VanishingCondition
from jacopy.brackets.schouten import sn as default_sn
from jacopy.calculus.exterior_d import d as default_d
from jacopy.calculus.lie_derivative import lie_derivative as default_lie_derivative
from jacopy.calculus.pairing import pairing
from jacopy.core.expr import Expr, Neg, Sum
from jacopy.core.registry import PropertyRegistry
from jacopy.core.symbolic_degree import DegreeLike
from jacopy.proof.chain import ProofChain
from jacopy.proof.step import ProofStep


LieDerivativeFactory = Callable[[Expr], Derivation]


class KoszulBracket(GradedBracket):
    """Classical Koszul bracket ``[α, β]_K`` on 1-forms.

    Parameters
    ----------
    anchor
        The anchor ``ρ: T*M → TM``, any degree-0
        :class:`~jacopy.algebra.derivation.Derivation`. The canonical
        choices are a dedicated :class:`~jacopy.calculus.anchor.Anchor`
        on a Lie algebroid and the musical map
        :class:`~jacopy.calculus.musical.Sharp` (``π^♯``) on a Poisson
        manifold; both share the same ``Derivation`` base class and the
        bracket is agnostic between them.
    d
        Exterior derivative operator. Defaults to the
        :data:`jacopy.calculus.exterior_d.d` singleton.
    lie_derivative
        Factory ``X → L_X``. Defaults to
        :func:`jacopy.calculus.lie_derivative.lie_derivative` (Cartan
        definition).
    name
        Display name; defaults to ``"[·,·]_K"``.

    Notes
    -----
    * Degree 0, graded-antisymmetric.
    * Graded Jacobi is declared *conditional* (``None``) rather than
      ``True`` because its proof rests on anchor-compatibility plus a
      vanishing generator obstruction, the same pattern used by
      :class:`DerivedBracket`.
    * Leibniz is declared ``True`` at the type level; on 1-forms the
      "product" is wedge, handled by the usual product-rule layer.
    """

    def __init__(
        self,
        anchor: Derivation,
        *,
        d: Optional[Derivation] = None,
        lie_derivative: Optional[LieDerivativeFactory] = None,
        name: str = "[·,·]_K",
    ) -> None:
        if not isinstance(anchor, Derivation):
            raise TypeError(
                f"KoszulBracket anchor must be a Derivation, got {type(anchor).__name__}"
            )
        super().__init__(
            name,
            degree=0,
            is_graded_antisymmetric=True,
            satisfies_leibniz=True,
            satisfies_graded_jacobi=None,
        )
        self._anchor = anchor
        self._d = d if d is not None else default_d
        self._lie_derivative = (
            lie_derivative if lie_derivative is not None else default_lie_derivative
        )

    @property
    def anchor(self) -> Derivation:
        return self._anchor

    def expand(
        self,
        a: Expr,
        b: Expr,
        registry: Optional[PropertyRegistry] = None,
    ) -> Expr:
        """``[α, β]_K = L_{ρα} β − L_{ρβ} α − d⟨ρα, β⟩``.

        All three Cartan pieces are assembled via :class:`Act` and
        :func:`pairing` so the shape is visible to the proof layer.
        The caller is free to pipe the result through ``simplify`` /
        ``canonicalize`` for a readable form.
        """
        if not isinstance(a, Expr):
            raise TypeError("KoszulBracket first operand must be an Expr")
        if not isinstance(b, Expr):
            raise TypeError("KoszulBracket second operand must be an Expr")

        rho_a = Act(self._anchor, a)
        rho_b = Act(self._anchor, b)
        L_rho_a = self._lie_derivative(rho_a)
        L_rho_b = self._lie_derivative(rho_b)

        return Sum(
            Act(L_rho_a, b),
            Neg(Act(L_rho_b, a)),
            Neg(Act(self._d, pairing(rho_a, b))),
        )

    def _identity_key(self) -> Any:
        return super()._identity_key() + (
            self._anchor,
            self._d,
            self._lie_derivative,
        )

    # ---- Jacobi condition ------------------------------------------ #

    def jacobi_condition(
        self,
        bivector: Expr,
        *,
        degree_bivector: DegreeLike = 1,
        registry: Optional[PropertyRegistry] = None,
    ) -> VanishingCondition:
        """Return the :class:`VanishingCondition` controlling Koszul Jacobi.

        On a Poisson manifold with anchor ``ρ = π^♯``, the classical
        result is that the Koszul bracket on 1-forms satisfies graded
        Jacobi iff ``[π, π]_SN = 0``, the same vanishing condition
        that controls the derived bracket
        :class:`DerivedBracket(sn, π)`. The helper exposes that link
        explicitly, without recomputing the proof each time.

        The anchor stored on this Koszul bracket is *not* used here,
        the returned condition lives on ``π`` directly and is
        anchor-agnostic. Callers working on a Lie algebroid with an
        independent anchor substitute their own generator.
        """
        if not isinstance(bivector, Expr):
            raise TypeError("jacobi_condition bivector must be an Expr")
        obstruction = default_sn.expand(bivector, bivector, registry)
        return VanishingCondition(
            obstruction=obstruction,
            name=f"Koszul Jacobi condition ([·,·]_K via {bivector._repr_inner()})",
        )

    def prove_jacobi_reduction(
        self,
        alpha: Expr,
        beta: Expr,
        gamma: Expr,
        *,
        bivector: Expr,
        degree_bivector: DegreeLike = 1,
        registry: Optional[PropertyRegistry] = None,
    ) -> ProofChain:
        """Reduce triple Koszul Jacobi ``(α, β, γ)`` to ``[π, π]_SN``.

        Returns a :class:`ProofChain` that cites the Derived Bracket
        Theorem and rewrites the cyclic Koszul Jacobi sum to the
        universal obstruction ``[π, π]_SN``. The chain does not discharge
        the obstruction; the caller supplies ``[π, π]_SN = 0`` (the
        Poisson hypothesis) to close it.

        Parameters
        ----------
        alpha, beta, gamma
            1-form operands. The bracket lifts each to a vector field
            via ``self.anchor`` before applying the Koszul formula.
        bivector
            The SN generator ``π``. The reduction theorem applies when
            ``self.anchor = π^♯``, :meth:`prove_jacobi_reduction` does
            not check anchor / bivector compatibility; supplying a
            mismatched pair gives a chain whose obstruction is the
            stated ``[π, π]_SN`` but whose reduction is only valid in
            spirit (the Poisson manifold context).
        degree_bivector
            Symbolic SN degree of ``π``; default ``1`` (i.e. a 2-vector
            in the unshifted grading).

        Companion to the typed handle returned by :meth:`jacobi_condition`:
        same obstruction, but transcribed as a one-step
        :class:`ProofChain` rather than a bare
        :class:`VanishingCondition`. Mirrors
        :meth:`jacopy.library.poisson.PoissonBracket.prove_koszul_jacobi_reduction`,
        which delegates to a Poisson-specific ``Sharp(π)`` anchor; this
        method is the anchor-agnostic version.
        """
        if not isinstance(alpha, Expr) or not isinstance(beta, Expr) or not isinstance(gamma, Expr):
            raise TypeError(
                "prove_jacobi_reduction operands must be Expr instances"
            )
        if not isinstance(bivector, Expr):
            raise TypeError("prove_jacobi_reduction bivector must be an Expr")

        derived = DerivedBracket(
            default_sn,
            bivector,
            degree_Q=degree_bivector,
            acting_on=self._anchor,
            name=self.name,
        )
        jacobi_sum = derived.graded_jacobi_obstruction(
            alpha, beta, gamma, registry,
        )
        obstruction_raw = derived.jacobi_obstruction_raw()
        obstruction = derived.jacobi_obstruction(registry)
        chain = ProofChain()
        chain.append(
            ProofStep(
                jacobi_sum,
                obstruction_raw,
                rule="DerivedBracketTheorem",
                justification=(
                    f"Jacobi on {self.name} ⟺ "
                    f"[{bivector._repr_inner()}, {bivector._repr_inner()}]_SN = 0 "
                    f"(Derived Bracket Theorem)"
                ),
                provenance_tag="theorem",
            )
        )
        if obstruction != obstruction_raw:
            chain.append(
                ProofStep(
                    obstruction_raw,
                    obstruction,
                    rule="sn-expand",
                    justification="apply SN definition to [π, π]",
                )
            )
        return chain
