"""
Lie algebroid library.

A :class:`LieAlgebroid` bundles the defining data ``(E, [·,·]_E, ρ)`` of
a Lie algebroid: a vector bundle ``E`` over a manifold, a bracket on
its sections, and an anchor morphism ``ρ: E → TM``. The standard axioms
are:

* ``[·,·]_E`` is graded-antisymmetric, satisfies graded Jacobi, and is
  a Leibniz derivation in each slot (the bracket itself carries those
  axiom flags, the wrapper does not re-derive them);
* **anchor compatibility**, ``ρ([X, Y]_E) = [ρ(X), ρ(Y)]_{TM}``,
  which is *not* implied by the bracket's own axioms and is surfaced
  here as :meth:`LieAlgebroid.anchor_compatibility_obstruction` /
  :meth:`anchor_compatibility_condition` /
  :meth:`prove_anchor_compatibility`.

On top of the data bundle the wrapper exposes an algebroid
:class:`~jacopy.calculus.cartan.CartanCalculus`, the exterior
derivative ``d_E`` on ``Λ*E*``, the algebroid Lie derivative ``L_{E,X}``,
and interior product ``ι_{E,X}`` factories, all wired to ``bracket`` as
the vector bracket. The five Cartan relations (``d² = 0``, magic formula,
``[d, L]``, ``[L, L]``, ``[L, ι]``) hold on this bundle the same way
they do on ``TM``, so :meth:`cartan` is the hook by which downstream
code verifies any of them on the algebroid.

The seeded theorem :data:`THEOREM_LIE_ALGEBROID_ANCHOR_COMPAT` records
the anchor compatibility as a one-step axiomatic chain, on a Lie
algebroid that identity is part of the *definition*, not something you
prove from simpler parts, so the chain cites the axiom directly.
"""

from __future__ import annotations

from typing import Optional

from jacopy.brackets.base import GradedBracket
from jacopy.brackets.derived import VanishingCondition
from jacopy.brackets.lie import LieBracket, lie as _lie_TM
from jacopy.calculus.anchor import Anchor, bracket_compatibility_obstruction
from jacopy.calculus.cartan import CartanCalculus
from jacopy.calculus.exterior_d import ExteriorDerivative
from jacopy.calculus.interior import InteriorProduct, interior
from jacopy.calculus.lie_derivative import LieDerivative, lie_derivative
from jacopy.core.expr import Expr, Integer, Symbol
from jacopy.core.properties import Graded
from jacopy.core.registry import PropertyRegistry
from jacopy.library.theorem_book import Theorem, theorem_book
from jacopy.proof.chain import ProofChain
from jacopy.proof.step import ProofStep


# --------------------------------------------------------------------- #
# LieAlgebroid wrapper                                                   #
# --------------------------------------------------------------------- #


class LieAlgebroid:
    """``(E, [·,·]_E, ρ)``, a Lie algebroid over a manifold.

    Parameters
    ----------
    bundle
        Symbolic name of the vector bundle ``E``. Carried for display
        only, the algebra lives in the bracket and anchor.
    bracket
        :class:`~jacopy.brackets.base.GradedBracket` on sections of
        ``E``. Graded-antisymmetric, Jacobi, Leibniz assumed (flags on
        the bracket record them; we don't second-guess).
    anchor
        :class:`~jacopy.calculus.anchor.Anchor` ``ρ: E → TM``. Degree
        zero; serves both as the compatibility target and as the lift
        that algebroid formulas use to evaluate on functions.
    vector_bracket
        Bracket on ``TM`` used as the compatibility target. Defaults to
        the standard :data:`jacopy.brackets.lie.lie` singleton.
    name
        Optional display name; defaults to ``f"Algebroid({bundle})"``.

    Notes
    -----
    The wrapper deliberately *does not* take ``E`` itself as an
    algebraic object, sections are introduced on demand as
    :class:`~jacopy.core.expr.Expr` operands when a caller invokes
    :meth:`anchor_compatibility_obstruction` or feeds an ``X`` into
    :attr:`cartan`. ``bundle`` is a naming handle only.
    """

    __slots__ = (
        "_bundle",
        "_bracket",
        "_anchor",
        "_vector_bracket",
        "_d_E",
        "_cartan",
        "_name",
    )

    def __init__(
        self,
        bundle: Expr,
        *,
        bracket: GradedBracket,
        anchor: Anchor,
        vector_bracket: Optional[GradedBracket] = None,
        name: Optional[str] = None,
    ) -> None:
        if not isinstance(bundle, Expr):
            raise TypeError("LieAlgebroid bundle must be an Expr")
        if not isinstance(bracket, GradedBracket):
            raise TypeError("LieAlgebroid bracket must be a GradedBracket")
        if not isinstance(anchor, Anchor):
            raise TypeError("LieAlgebroid anchor must be an Anchor")
        if vector_bracket is None:
            vector_bracket = _lie_TM
        elif not isinstance(vector_bracket, GradedBracket):
            raise TypeError(
                "LieAlgebroid vector_bracket must be a GradedBracket"
            )
        self._bundle = bundle
        self._bracket = bracket
        self._anchor = anchor
        self._vector_bracket = vector_bracket
        # Algebroid exterior derivative ``d_E``, a fresh degree-+1
        # derivation named to distinguish it from the ambient ``d`` on
        # TM. Callers who want to verify ``d_E² = 0`` on concrete
        # algebras do so through :attr:`cartan`.
        self._d_E = ExteriorDerivative(name=f"d_{bundle._repr_inner()}")
        self._cartan = CartanCalculus(
            d=self._d_E,
            lie_derivative=self._make_lie_derivative_factory(),
            interior=self._make_interior_factory(),
            vector_bracket=bracket,
        )
        self._name = (
            name if name is not None else f"Algebroid({bundle._repr_inner()})"
        )

    # ---- Cartan factories ------------------------------------------ #

    def _make_lie_derivative_factory(self):
        bundle_tag = self._bundle._repr_inner()
        iota_factory = self._make_interior_factory()
        d_E = self._d_E

        def factory(X: Expr) -> LieDerivative:
            if not isinstance(X, Expr):
                raise TypeError("algebroid L factory requires an Expr section")
            # Plumbing: the algebroid ``L_{E,X}`` has to carry its bundle's
            # ``d_E`` and ``ι_{E,·}`` factory so that the expansion engine's
            # Cartan rewrite produces ``d_E ∘ ι_{E,X} + ι_{E,X} ∘ d_E``
            # instead of the TM default ``d ∘ ι_X + ι_X ∘ d``. Without this
            # the algebroid magic formula residual can't close, the two
            # sides use mismatched operator names.
            return lie_derivative(
                X,
                name=f"L_{bundle_tag},{X._repr_inner()}",
                d=d_E,
                iota_factory=iota_factory,
            )

        return factory

    def _make_interior_factory(self):
        bundle_tag = self._bundle._repr_inner()

        def factory(X: Expr) -> InteriorProduct:
            if not isinstance(X, Expr):
                raise TypeError("algebroid ι factory requires an Expr section")
            return interior(X, name=f"ι_{bundle_tag},{X._repr_inner()}")

        return factory

    # ---- accessors -------------------------------------------------- #

    @property
    def bundle(self) -> Expr:
        return self._bundle

    @property
    def bracket(self) -> GradedBracket:
        return self._bracket

    @property
    def anchor(self) -> Anchor:
        return self._anchor

    @property
    def vector_bracket(self) -> GradedBracket:
        return self._vector_bracket

    @property
    def d(self) -> ExteriorDerivative:
        """The algebroid exterior derivative ``d_E``."""
        return self._d_E

    @property
    def cartan(self) -> CartanCalculus:
        """Algebroid Cartan bundle, ``(d_E, L_{E,·}, ι_{E,·}, [·,·]_E)``.

        Same :class:`CartanCalculus` API as the ``TM`` one. The factories
        name their outputs with the bundle tag so ``L_{E,X}`` and
        ``ι_{E,X}`` are distinguishable from the ambient manifold
        operators when both appear in a single expression.
        """
        return self._cartan

    @property
    def name(self) -> str:
        return self._name

    # ---- anchor compatibility -------------------------------------- #

    def anchor_compatibility_obstruction(
        self,
        X: Expr,
        Y: Expr,
        registry: Optional[PropertyRegistry] = None,
    ) -> Expr:
        """``ρ([X, Y]_E) − [ρ(X), ρ(Y)]_{TM}``, the axiom as an Expr.

        Thin forwarder to
        :func:`~jacopy.calculus.anchor.bracket_compatibility_obstruction`
        with the algebroid's own brackets and anchor.
        """
        if not isinstance(X, Expr) or not isinstance(Y, Expr):
            raise TypeError(
                "anchor_compatibility_obstruction requires Expr operands"
            )
        return bracket_compatibility_obstruction(
            self._anchor,
            self._bracket,
            self._vector_bracket,
            X,
            Y,
            registry,
        )

    def anchor_compatibility_condition(
        self,
        X: Expr,
        Y: Expr,
        registry: Optional[PropertyRegistry] = None,
    ) -> VanishingCondition:
        """Wrap the compatibility obstruction as a :class:`VanishingCondition`."""
        return VanishingCondition(
            obstruction=self.anchor_compatibility_obstruction(X, Y, registry),
            name=f"anchor compatibility on {self._name}",
        )

    def prove_anchor_compatibility(
        self,
        X: Expr,
        Y: Expr,
        *,
        registry: Optional[PropertyRegistry] = None,
    ) -> ProofChain:
        """Record the compatibility axiom as a single-step
        :class:`ProofChain`.

        On a Lie algebroid, ``ρ([X, Y]_E) = [ρ(X), ρ(Y)]_{TM}`` is part
        of the *definition*, there is nothing to prove from simpler
        parts; the chain has a single ``axiom``-tagged step that
        discharges the obstruction directly. Callers that want to
        expand both sides and watch the cancellation happen should
        skip this method and run
        :meth:`anchor_compatibility_obstruction` through the standard
        :func:`~jacopy.proof.strategies.ExpandAndSimplify`; that path
        is *not* in general zero, because the bracket on ``TM`` is
        atomic and no rewrite bridges the two sides without the axiom.
        """
        obs = self.anchor_compatibility_obstruction(X, Y, registry)
        chain = ProofChain()
        chain.append(
            ProofStep(
                obs,
                Integer(0),
                rule="LieAlgebroidAnchorCompat",
                justification=(
                    f"ρ([X, Y]_E) = [ρ(X), ρ(Y)]_{{TM}}, Lie algebroid "
                    f"anchor compatibility axiom on {self._name}"
                ),
                provenance_tag="axiom",
            )
        )
        return chain

    # ---- dunder ---------------------------------------------------- #

    def __repr__(self) -> str:
        return (
            f"LieAlgebroid(bundle={self._bundle._repr_inner()}, "
            f"bracket={self._bracket.name}, anchor={self._anchor.name})"
        )


# --------------------------------------------------------------------- #
# Factory                                                                #
# --------------------------------------------------------------------- #


def lie_algebroid(
    bundle: Expr,
    *,
    bracket: GradedBracket,
    anchor: Anchor,
    vector_bracket: Optional[GradedBracket] = None,
    name: Optional[str] = None,
) -> LieAlgebroid:
    """Functional mirror of :class:`LieAlgebroid`."""
    return LieAlgebroid(
        bundle,
        bracket=bracket,
        anchor=anchor,
        vector_bracket=vector_bracket,
        name=name,
    )


# --------------------------------------------------------------------- #
# Seeded anchor-compatibility theorem                                    #
# --------------------------------------------------------------------- #


def _build_lie_algebroid_anchor_compat_theorem() -> Theorem:
    """Construct the canonical ``lie_algebroid_anchor_compat`` theorem.

    Generic witness: ``E`` a symbolic bundle with LieBracket ``[·,·]_E``
    and anchor ``ρ`` into the standard ``TM``-Lie bracket. The proof
    chain is a single ``axiom``-tagged step, the Lie algebroid
    definition postulates compatibility, and the theorem record
    catalogues it as a citable result.
    """
    E = Symbol("E")
    X = Symbol("X")
    Y = Symbol("Y")
    reg = PropertyRegistry()
    reg.declare(X, Graded(degree=0))
    reg.declare(Y, Graded(degree=0))
    bracket_E = LieBracket(name="[·,·]_E")
    rho = Anchor(name="ρ")
    algebroid = LieAlgebroid(E, bracket=bracket_E, anchor=rho)
    chain = algebroid.prove_anchor_compatibility(X, Y, registry=reg)
    return Theorem(
        name="lie_algebroid_anchor_compat",
        statement="ρ([X, Y]_E) = [ρ(X), ρ(Y)]_{TM}",
        from_axioms=(
            "Lie algebroid anchor compatibility axiom",
        ),
        proof=chain,
        notes=(
            "On a Lie algebroid the anchor is required by definition to "
            "intertwine the algebroid bracket with the TM Lie bracket. "
            "The theorem is catalogued so downstream proofs (algebroid "
            "Cartan calculus, Courant-Dorfman bridge) can cite it as a "
            "single axiomatic step rather than re-posting the obstruction."
        ),
    )


#: The Lie algebroid anchor-compatibility axiom, catalogued as a
#: :class:`Theorem` so downstream library modules can cite it by name.
THEOREM_LIE_ALGEBROID_ANCHOR_COMPAT = (
    _build_lie_algebroid_anchor_compat_theorem()
)


if "lie_algebroid_anchor_compat" not in theorem_book:
    theorem_book.add(THEOREM_LIE_ALGEBROID_ANCHOR_COMPAT)
