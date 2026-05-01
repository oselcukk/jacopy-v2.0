"""
Cartan–Koszul invariant formula for ``d``.

Evaluated against vector fields, the exterior derivative ``d`` has a
closed-form, coordinate-free expansion, the *Cartan–Koszul invariant
formula*. In full generality, for a ``k``-form ``ω`` and vector fields
``X_0, …, X_k``:

    (dω)(X_0, …, X_k)
        = Σ_i (−1)^i  X_i(ω(X_0, …, X̂_i, …, X_k))
        + Σ_{i<j} (−1)^{i+j}  ω([X_i, X_j], X_0, …, X̂_i, …, X̂_j, …, X_k).

This module ships the simplest interesting case, the **one-form
formula** (``k = 1``, two vector fields):

    (dω)(X, Y) = X(ω(Y)) − Y(ω(X)) − ω([X, Y]).

That identity is the concrete anchor point for every other ``k``: it
exposes the three-term shape (two "directional derivatives" plus one
bracket correction) that every higher-degree version generalises, and
it falls out of the existing Cartan relations

* magic formula ``ι_X d = L_X − d ι_X``,
* commutator rule ``[L_X, ι_Y] = ι_{[X, Y]}``,
* plus the standard pairings ``L_X(f) = X(f)`` on 0-forms and
  ``ι_Y(df) = Y(f)`` on exact 1-forms.

Writing the left-hand side as ``ι_Y ι_X (dω)`` and chasing the
relations gives

    ι_Y ι_X dω
        = ι_Y (L_X − d ι_X) ω                       (magic)
        = (L_X ι_Y − ι_{[X, Y]}) ω − ι_Y d (ι_X ω)  ([L, ι] commutator)
        = X(ι_Y ω) − ι_{[X, Y]} ω − Y(ι_X ω)        (0-form pairings)
        = X(ω(Y)) − Y(ω(X)) − ω([X, Y]).

The module offers two entry points. :func:`invariant_d_one_form`
builds the right-hand-side :class:`Expr` directly, use it when you
want the formula as a ready-made expression to compare against or
substitute into a larger calculation.
:class:`InvariantDOneFormDefinition` wraps the same rewrite as a
:class:`~jacopy.proof.expansion.Definition` so it can be registered
with an :class:`~jacopy.proof.expansion.ExpansionEngine` and fire
bottom-up on ``Act(ι_Y, Act(ι_X, Act(d, ω)))`` subterms, with
``provenance_tag`` selectable between ``"axiom"`` and ``"theorem"``,
the same "choice of presentation" knob the plan asked for alongside
the :class:`~jacopy.proof.expansion.DSquaredZeroDefinition`.

Higher-degree analogues (``k ≥ 2``) follow the same pattern, stack
more interior products on the outside and pick up additional bracket
corrections, but are not implemented here; the ``k = 1`` case is the
non-trivial anchor, and the machinery for arbitrary ``k`` needs
hat-and-sign bookkeeping that is pedagogical rather than load-bearing.
"""

from __future__ import annotations

from typing import Any, Callable, Optional

from jacopy.algebra.derivation import Act, Derivation
from jacopy.brackets.base import GradedBracket
from jacopy.calculus.exterior_d import ExteriorDerivative, d as default_d
from jacopy.calculus.interior import (
    InteriorProduct,
    interior as default_interior,
)
from jacopy.core.expr import Expr, Neg, Sum
from jacopy.core.registry import PropertyRegistry
from jacopy.core.symbolic_degree import Degree
from jacopy.proof.expansion import Definition


InteriorFactory = Callable[[Expr], InteriorProduct]


# --------------------------------------------------------------------- #
# RHS builder                                                            #
# --------------------------------------------------------------------- #


def invariant_d_one_form(
    omega: Expr,
    X: Expr,
    Y: Expr,
    *,
    bracket: GradedBracket,
    interior: Optional[InteriorFactory] = None,
) -> Expr:
    """Return ``X(ω(Y)) − Y(ω(X)) − ω([X, Y])`` as an :class:`Expr`.

    The three :class:`Act` shapes encode, in order:

    * ``Act(X, Act(ι_Y, ω))``, the directional derivative of the
      scalar ``ω(Y)`` along ``X``,
    * ``Act(Y, Act(ι_X, ω))``, the symmetric partner with the sign
      absorbed in an outer :class:`Neg`,
    * ``Act(ι_{[X, Y]}, ω)``, the pairing of ``ω`` with the bracket,
      wrapped in a :class:`Neg`.

    ``bracket`` supplies the vector-field commutator; typically the
    :class:`~jacopy.brackets.lie.LieBracket` singleton. ``interior``
    defaults to :func:`jacopy.calculus.interior.interior` so the
    ``ι_X, ι_Y, ι_{[X, Y]}`` operators share equality semantics with
    the rest of the package.
    """
    if not isinstance(omega, Expr):
        raise TypeError("invariant_d_one_form: ω must be an Expr")
    if not isinstance(X, Expr):
        raise TypeError("invariant_d_one_form: X must be an Expr")
    if not isinstance(Y, Expr):
        raise TypeError("invariant_d_one_form: Y must be an Expr")
    if not isinstance(bracket, GradedBracket):
        raise TypeError(
            "invariant_d_one_form: bracket must be a GradedBracket"
        )
    iota_factory: InteriorFactory = (
        default_interior if interior is None else interior
    )
    iota_X = iota_factory(X)
    iota_Y = iota_factory(Y)
    iota_XY = iota_factory(bracket.expand(X, Y))
    return Sum(
        Act(X, Act(iota_Y, omega)),
        Neg(Act(Y, Act(iota_X, omega))),
        Neg(Act(iota_XY, omega)),
    )


# --------------------------------------------------------------------- #
# Engine rule                                                            #
# --------------------------------------------------------------------- #


#: The two classifications an :class:`InvariantDOneFormDefinition` can carry.
INVARIANT_D_CLASSIFICATIONS = ("axiom", "theorem")


class InvariantDOneFormDefinition(Definition):
    """``ι_Y ι_X (dω) → X(ι_Y ω) − Y(ι_X ω) − ι_{[X, Y]} ω`` for 1-forms.

    Fires on the shape ``Act(ι_Y, Act(ι_X, Act(d, ω)))`` when the
    registry declares ``|ω| = 1``. The vector fields ``X`` and ``Y``
    come from the :class:`InteriorProduct` operators themselves; the
    bracket ``[X, Y]`` is supplied by the ``bracket`` argument passed
    to the constructor.

    ``classification`` selects the provenance recorded in the fired
    :class:`~jacopy.proof.step.ProofStep`:

    * ``"axiom"``, the formula is taken as a primitive characterisation
      of ``d`` on 1-forms; the step is tagged ``"axiom"`` with no
      sub-proof attached.
    * ``"theorem"`` (default), the formula is a theorem derivable
      from the Cartan magic formula and the ``[L_X, ι_Y] = ι_{[X, Y]}``
      commutator; the step is tagged ``"theorem"`` and, in
      ``mode="foundational"``, carries a one-step sub-proof citing
      those two Cartan relations as the foundational input.

    ``d`` pins the rewrite to a specific :class:`ExteriorDerivative`
    (use a distinct instance for e.g. a Lie-algebroid ``d_E``);
    ``interior`` optionally overrides the interior-product factory for
    the output ``ι_{[X, Y]}``.
    """

    name = "invariant d on 1-forms"

    def __init__(
        self,
        bracket: GradedBracket,
        *,
        d: Optional[ExteriorDerivative] = None,
        interior: Optional[InteriorFactory] = None,
        registry: Optional[PropertyRegistry] = None,
        classification: str = "theorem",
    ) -> None:
        if not isinstance(bracket, GradedBracket):
            raise TypeError(
                "InvariantDOneFormDefinition requires a GradedBracket"
            )
        if classification not in INVARIANT_D_CLASSIFICATIONS:
            raise ValueError(
                f"InvariantDOneFormDefinition classification must be one of "
                f"{INVARIANT_D_CLASSIFICATIONS}, got {classification!r}"
            )
        self._bracket = bracket
        self._d = default_d if d is None else d
        self._interior: InteriorFactory = (
            default_interior if interior is None else interior
        )
        self._registry = registry
        self._classification = classification

    @property
    def bracket(self) -> GradedBracket:
        return self._bracket

    @property
    def classification(self) -> str:
        return self._classification

    # ---- match / rewrite -------------------------------------------- #

    def _omega_is_one_form(self, omega: Expr) -> bool:
        """Registry-safe ``|ω| = 1`` check."""
        try:
            from jacopy.algebra.derivation import degree_of

            return degree_of(omega, self._registry) == Degree.const(1)
        except ValueError:
            return False

    def matches(self, expr: Expr) -> bool:
        if not (isinstance(expr, Act) and isinstance(expr.op, InteriorProduct)):
            return False
        inner = expr.arg
        if not (isinstance(inner, Act) and isinstance(inner.op, InteriorProduct)):
            return False
        innermost = inner.arg
        if not (
            isinstance(innermost, Act)
            and isinstance(innermost.op, ExteriorDerivative)
        ):
            return False
        if innermost.op != self._d:
            return False
        return self._omega_is_one_form(innermost.arg)

    def rewrite(self, expr: Expr) -> Expr:
        iota_Y_op: InteriorProduct = expr.op  # type: ignore[assignment]
        iota_X_op: InteriorProduct = expr.arg.op  # type: ignore[assignment]
        omega: Expr = expr.arg.arg.arg
        X = iota_X_op.vector_field
        Y = iota_Y_op.vector_field
        return invariant_d_one_form(
            omega, X, Y, bracket=self._bracket, interior=self._interior
        )

    # ---- theorem sub-proof ------------------------------------------ #

    def theorem_proof_builder(self):
        if self._classification != "theorem":
            return None

        def _build(matched: Expr) -> Any:
            # Single-step honest sub-proof: the formula is a theorem
            # whose only foundational inputs are the Cartan magic
            # formula and the [L_X, ι_Y] = ι_{[X, Y]} commutator. A
            # user who wants the full multi-step derivation composes
            # CartanCalculus.verify('cartan_magic') and verify('lie_iota')
            # and stitches them, see this module's header for the
            # derivation. The sub-proof here cites those two relations
            # as the theorem's foundations.
            from jacopy.proof.chain import ProofChain
            from jacopy.proof.step import ProofStep

            after = self.rewrite(matched)
            chain = ProofChain()
            chain.append(
                ProofStep(
                    matched,
                    after,
                    rule=(
                        "Cartan magic formula + [L_X, ι_Y] = ι_{[X, Y]}"
                    ),
                    justification=(
                        "invariant formula (dω)(X, Y) = X(ω(Y)) − Y(ω(X)) − ω([X, Y]) "
                        "derives from ι_X d = L_X − d ι_X and the "
                        "Lie-derivative / interior-product commutator"
                    ),
                    provenance_tag="axiom",
                )
            )
            return chain

        return _build
