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
this module (Stage 2B — :class:`DerivedBracket` gains an ``acting_on``
parameter that lifts 1-form operands through an anchor, and a separate
proof-layer helper closes the equivalence as a :class:`ProofChain`).

The constructor mirrors :class:`DorfmanBracket`: the anchor is required,
and the exterior derivative + Lie-derivative factory default to the
smooth-manifold singletons. Lie-algebroid callers substitute their own
Cartan family.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, Optional

from gradalg.algebra.derivation import Act, Derivation
from gradalg.brackets.base import GradedBracket
from gradalg.brackets.derived import VanishingCondition
from gradalg.brackets.schouten import sn as default_sn
from gradalg.calculus.exterior_d import d as default_d
from gradalg.calculus.lie_derivative import lie_derivative as default_lie_derivative
from gradalg.calculus.pairing import pairing
from gradalg.core.expr import Expr, Neg, Sum
from gradalg.core.registry import PropertyRegistry
from gradalg.core.symbolic_degree import DegreeLike

if TYPE_CHECKING:
    # Imported lazily to avoid a circular import: ``calculus.anchor``
    # imports ``brackets.base``, which eagerly triggers this package's
    # ``__init__`` and would then re-enter ``anchor`` mid-initialisation.
    # The runtime ``isinstance`` check below imports ``Anchor`` locally.
    from gradalg.calculus.anchor import Anchor


LieDerivativeFactory = Callable[[Expr], Derivation]


class KoszulBracket(GradedBracket):
    """Classical Koszul bracket ``[α, β]_K`` on 1-forms.

    Parameters
    ----------
    anchor
        The anchor ``ρ: T*M → TM``. In the Poisson case this is the
        musical map ``π^♯`` (supplied as an :class:`Anchor` instance
        named ``"π♯"`` by the caller), but the bracket itself is
        anchor-agnostic — any :class:`Anchor` that lands in ``TM`` will
        do.
    d
        Exterior derivative operator. Defaults to the
        :data:`gradalg.calculus.exterior_d.d` singleton.
    lie_derivative
        Factory ``X → L_X``. Defaults to
        :func:`gradalg.calculus.lie_derivative.lie_derivative` (Cartan
        definition).
    name
        Display name; defaults to ``"[·,·]_K"``.

    Notes
    -----
    * Degree 0, graded-antisymmetric.
    * Graded Jacobi is declared *conditional* (``None``) rather than
      ``True`` because its proof rests on anchor-compatibility plus a
      vanishing generator obstruction — the same pattern used by
      :class:`DerivedBracket`.
    * Leibniz is declared ``True`` at the type level; on 1-forms the
      "product" is wedge, handled by the usual product-rule layer.
    """

    def __init__(
        self,
        anchor: Anchor,
        *,
        d: Optional[Derivation] = None,
        lie_derivative: Optional[LieDerivativeFactory] = None,
        name: str = "[·,·]_K",
    ) -> None:
        from gradalg.calculus.anchor import Anchor as _Anchor

        if not isinstance(anchor, _Anchor):
            raise TypeError(
                f"KoszulBracket anchor must be an Anchor, got {type(anchor).__name__}"
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
    def anchor(self) -> Anchor:
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
        Jacobi iff ``[π, π]_SN = 0`` — the same vanishing condition
        that controls the derived bracket
        :class:`DerivedBracket(sn, π)`. The helper exposes that link
        explicitly, without recomputing the proof each time.

        The anchor stored on this Koszul bracket is *not* used here —
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
