"""
Lie derivative ``L_X``.

The Lie derivative along a vector field ``X`` is the degree ``0``
graded derivation on the exterior algebra. Two standard definitions:

* **Flow definition (axiom mode ``"flow"``)**, ``L_X(ω) = d/dt|_0 φ_t^* ω``
  where ``φ_t`` is the flow of ``X``. Cartan's magic formula then
  becomes a *theorem*:
  ``L_X = d ∘ ι_X + ι_X ∘ d``.
* **Cartan definition (axiom mode ``"cartan"``)**, take the magic
  formula itself as the definition:
  ``L_X := d ∘ ι_X + ι_X ∘ d``.
  Cartan's magic formula is then a tautology.

The package supports both definitions because the choice affects what
has to be proved downstream (e.g. Leibniz is a theorem in flow mode
but follows from the derivation structure of ``d`` and ``ι_X`` in
Cartan mode). The :class:`LieDerivative` instance records which mode
it was constructed under via :attr:`definition`.

As with the interior product there is no global singleton, ``L_X``
is a family indexed by the vector field ``X``. Use :func:`lie_derivative`
to build one. The class is a :class:`Derivation` of degree 0 so
:mod:`product_rule` handles its Leibniz behaviour directly; the Cartan
identity itself is represented by :func:`cartan_expansion`, which
returns the ``d ∘ ι_X + ι_X ∘ d`` composition as an :class:`Expr` the
caller can splice in or compare against.
"""

from __future__ import annotations

from typing import Callable, Optional

from jacopy.algebra.derivation import Act, Derivation, compose
from jacopy.calculus.exterior_d import ExteriorDerivative, d as default_d
from jacopy.calculus.interior import InteriorProduct, interior
from jacopy.core.expr import Expr, Sum
from jacopy.core.registry import PropertyRegistry


#: Type alias, a callable ``X -> ι_X`` that builds an
#: :class:`InteriorProduct` for a given vector field. Used by
#: :class:`LieDerivative` to remember which interior-product factory its
#: Cartan expansion should use (so algebroid bundles get ``ι_{E,X}``
#: rather than the default ``ι_X``).
IotaFactory = Callable[[Expr], InteriorProduct]


# Supported axiomatic definitions. Kept as a tuple of literals rather
# than an Enum, there are exactly two options and the strings serve
# as both the API-facing choice and the internal key.
DEFINITIONS = ("flow", "cartan")


class LieDerivative(Derivation):
    """Lie derivative ``L_X``, degree ``0`` graded derivation.

    Carries the vector field ``X`` and the axiomatic definition mode
    (``"flow"`` or ``"cartan"``). Equality / hash reduce to
    :class:`Derivation`'s structural ``(name, degree)`` key, so two
    Lie derivatives are equal iff their display names match, the
    default name includes ``X`` which makes that line up with equality
    of the vector field.
    """

    __slots__ = ("_vector_field", "_definition", "_d", "_iota_factory")

    def __init__(
        self,
        X: Expr,
        *,
        definition: str = "cartan",
        name: Optional[str] = None,
        d: Optional[ExteriorDerivative] = None,
        iota_factory: Optional[IotaFactory] = None,
    ) -> None:
        if not isinstance(X, Expr):
            raise TypeError("Lie derivative requires an Expr vector field")
        if definition not in DEFINITIONS:
            raise ValueError(
                f"definition must be one of {DEFINITIONS}, got {definition!r}"
            )
        if d is not None and not isinstance(d, ExteriorDerivative):
            raise TypeError(
                "LieDerivative d override must be an ExteriorDerivative"
            )
        if iota_factory is not None and not callable(iota_factory):
            raise TypeError("LieDerivative iota_factory must be callable")
        display_name = name if name is not None else f"L_{X._repr_inner()}"
        super().__init__(display_name, degree=0)
        self._vector_field = X
        self._definition = definition
        self._d = d
        self._iota_factory = iota_factory

    @property
    def vector_field(self) -> Expr:
        return self._vector_field

    @property
    def definition(self) -> str:
        return self._definition

    @property
    def d(self) -> Optional[ExteriorDerivative]:
        """Bundle-specific exterior derivative, or ``None`` for the TM default.

        When non-``None`` the Cartan expansion driven by the expansion
        engine uses this ``d`` in place of the default :mod:`exterior_d`
        singleton, this is how a Lie-algebroid ``L_{E,X}`` keeps its
        ``d_E`` glued to its own bundle instead of falling back to the
        ambient manifold ``d``.
        """
        return self._d

    @property
    def iota_factory(self) -> Optional[IotaFactory]:
        """Bundle-specific ``ι_X`` factory, or ``None`` for the default.

        Parallels :attr:`d`: when non-``None`` the engine's Cartan
        expansion builds the interior product through this factory
        (yielding e.g. ``ι_{E,X}`` rather than the default ``ι_X``) so
        that composed forms match the algebroid's own operator names.
        """
        return self._iota_factory


def lie_derivative(
    X: Expr,
    *,
    definition: str = "cartan",
    name: Optional[str] = None,
    d: Optional[ExteriorDerivative] = None,
    iota_factory: Optional[IotaFactory] = None,
) -> LieDerivative:
    """Build the Lie-derivative operator ``L_X``."""
    return LieDerivative(
        X,
        definition=definition,
        name=name,
        d=d,
        iota_factory=iota_factory,
    )


# --------------------------------------------------------------------- #
# Cartan's magic formula                                                 #
# --------------------------------------------------------------------- #


def cartan_expansion(
    X: Expr,
    *,
    d: Optional[ExteriorDerivative] = None,
    iota: Optional[InteriorProduct] = None,
) -> Expr:
    """Return ``d ∘ ι_X + ι_X ∘ d`` as an :class:`Expr`.

    The shape is a :class:`Sum` of two :class:`Product` compositions,
    the form that :func:`jacopy.algebra.derivation.compose` produces.
    Applying this to an operand gives the magic formula's RHS; for
    ``definition="cartan"`` it *is* the definition of ``L_X``, and for
    ``definition="flow"`` it is what has to be derived.

    ``d`` defaults to the module singleton from :mod:`exterior_d`.
    ``iota`` defaults to a fresh :class:`InteriorProduct` for ``X``;
    pass an explicit instance when the caller needs to share the same
    interior-product object elsewhere in the expression (equality on
    ``Derivation`` is structural, so a freshly-constructed one still
    compares equal to any other built with the same name and degree,
    but sharing makes intent clearer in proof output).
    """
    dop = default_d if d is None else d
    iop = interior(X) if iota is None else iota
    return Sum(
        compose(dop, iop),
        compose(iop, dop),
    )


def cartan_obstruction(
    L: LieDerivative,
    arg: Expr,
    *,
    d: Optional[ExteriorDerivative] = None,
    iota: Optional[InteriorProduct] = None,
    registry: Optional[PropertyRegistry] = None,  # noqa: ARG001
) -> Expr:
    """Return ``L_X(arg) − (d ∘ ι_X + ι_X ∘ d)(arg)`` as an :class:`Expr`.

    Zero-*expression* (i.e. the result :class:`simplify`'s down to
    :class:`Integer` ``0``) iff Cartan's magic formula holds for this
    ``L``, ``X``, and ``arg``. The helper is registry-accepting even
    though the shape itself is registry-free, expanding and
    simplifying downstream will need the registry for the Koszul
    signs in :mod:`product_rule`, so keeping the argument here keeps
    the whole obstruction flow registry-aware.
    """
    from jacopy.core.expr import Neg  # local to keep top clean
    lhs = Act(L, arg)
    rhs = Act(cartan_expansion(L.vector_field, d=d, iota=iota), arg)
    return Sum(lhs, Neg(rhs))
