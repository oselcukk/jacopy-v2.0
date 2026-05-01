"""
Dorfman bracket on sections of ``TM ⊕ T*M``.

The Dorfman bracket is the non-skew-symmetric cousin of the Courant
bracket. On section pairs ``(X, α), (Y, β)`` it is

    [(X, α), (Y, β)]_D = ( [X, Y],  L_X β − ι_Y dα ).

Unlike its Courant counterpart, the Dorfman bracket satisfies the
Leibniz (Jacobi-for-Loday) identity exactly, at the cost of graded
antisymmetry, which only holds up to an exact term. That asymmetry is
encoded here by declaring ``is_graded_antisymmetric=False`` and
``satisfies_graded_jacobi=True``: Dorfman is a Leibniz algebra, which
is the structure the downstream derived-bracket machinery actually
relies on.

Operands live as :class:`SectionPair` nodes, a small :class:`Expr`
wrapper carrying a vector-field half and a form half. The bracket's
:meth:`~DorfmanBracket.expand` unpacks both sides, runs the vector
Lie bracket on the ``X, Y`` components, and assembles the form
component as ``L_X β − ι_Y dα`` using the caller-supplied Cartan
operators (exterior derivative, Lie derivative factory, interior
product factory).
"""

from __future__ import annotations

from typing import Any, Callable, Optional, Tuple

from jacopy.algebra.derivation import Act, Derivation
from jacopy.brackets.base import GradedBracket
from jacopy.brackets.lie import LieBracket
from jacopy.calculus.exterior_d import d as default_d
from jacopy.calculus.interior import interior as default_interior
from jacopy.calculus.lie_derivative import lie_derivative as default_lie_derivative
from jacopy.core.expr import Expr, Neg, Sum
from jacopy.core.registry import PropertyRegistry


# --------------------------------------------------------------------- #
# Section pair node                                                      #
# --------------------------------------------------------------------- #


class SectionPair(Expr):
    """A ``(vector, form)`` section pair, the operand type for the Dorfman bracket.

    The pair is inert: it has no algebraic structure of its own beyond
    being a two-child :class:`Expr`. The Dorfman bracket reads ``vector``
    and ``form`` back out at expansion time and routes each component
    through the appropriate Cartan operator.
    """

    __slots__ = ("_vector", "_form")

    def __init__(self, vector: Expr, form: Expr) -> None:
        if not isinstance(vector, Expr):
            raise TypeError("SectionPair vector component must be an Expr")
        if not isinstance(form, Expr):
            raise TypeError("SectionPair form component must be an Expr")
        self._vector = vector
        self._form = form

    @property
    def vector(self) -> Expr:
        return self._vector

    @property
    def form(self) -> Expr:
        return self._form

    @property
    def children(self) -> Tuple[Expr, ...]:
        return (self._vector, self._form)

    def _key(self) -> Any:
        return (self._vector, self._form)

    def _repr_inner(self) -> str:
        return f"({self._vector._repr_inner()}, {self._form._repr_inner()})"


# --------------------------------------------------------------------- #
# Bracket                                                                #
# --------------------------------------------------------------------- #


LieDerivativeFactory = Callable[[Expr], Derivation]
InteriorFactory = Callable[[Expr], Derivation]


class DorfmanBracket(GradedBracket):
    """Dorfman bracket on :class:`SectionPair` operands.

    Parameters
    ----------
    name
        Display name; defaults to ``"[·,·]_D"``.
    vector_bracket
        Bracket used on the vector-field halves. Defaults to
        :class:`LieBracket`.
    d
        Exterior derivative operator. Defaults to the
        :data:`jacopy.calculus.exterior_d.d` singleton.
    lie_derivative
        Factory ``X → L_X``. Defaults to
        :func:`jacopy.calculus.lie_derivative.lie_derivative` (Cartan
        definition).
    interior
        Factory ``Y → ι_Y``. Defaults to
        :func:`jacopy.calculus.interior.interior`.

    The defaults wire up the standard smooth-manifold case. Callers
    targeting a Lie algebroid substitute their own Cartan family and
    vector bracket to build the algebroid-valued Dorfman bracket.
    """

    def __init__(
        self,
        name: str = "[·,·]_D",
        *,
        vector_bracket: Optional[GradedBracket] = None,
        d: Optional[Derivation] = None,
        lie_derivative: Optional[LieDerivativeFactory] = None,
        interior: Optional[InteriorFactory] = None,
    ) -> None:
        # Leibniz (Loday) holds exactly; graded antisymmetry does not,
        # it's broken by an exact correction ``d⟨X, β⟩ + d⟨Y, α⟩`` so the
        # Dorfman pair is the asymmetric-but-Jacobi twin of Courant.
        super().__init__(
            name,
            degree=0,
            is_graded_antisymmetric=False,
            satisfies_leibniz=True,
            satisfies_graded_jacobi=True,
        )
        self._vector_bracket = vector_bracket if vector_bracket is not None else LieBracket()
        self._d = d if d is not None else default_d
        self._lie_derivative = (
            lie_derivative if lie_derivative is not None else default_lie_derivative
        )
        self._interior = interior if interior is not None else default_interior

    @property
    def vector_bracket(self) -> GradedBracket:
        return self._vector_bracket

    def expand(
        self,
        a: Expr,
        b: Expr,
        registry: Optional[PropertyRegistry] = None,
    ) -> Expr:
        """``[(X, α), (Y, β)]_D = ( [X, Y],  L_X β − ι_Y dα )``.

        Raises :class:`TypeError` if either operand is not a
        :class:`SectionPair`, the Dorfman bracket is defined on the
        product bundle and has no meaningful action on raw vectors or
        raw forms alone.
        """
        if not isinstance(a, SectionPair):
            raise TypeError(
                f"DorfmanBracket left operand must be a SectionPair, "
                f"got {type(a).__name__}"
            )
        if not isinstance(b, SectionPair):
            raise TypeError(
                f"DorfmanBracket right operand must be a SectionPair, "
                f"got {type(b).__name__}"
            )
        X, alpha = a.vector, a.form
        Y, beta = b.vector, b.form

        vector_part = self._vector_bracket.expand(X, Y, registry)
        L_X = self._lie_derivative(X)
        iota_Y = self._interior(Y)
        # L_X β − ι_Y(dα). Built via Act + Neg so the shape is visible to
        # the proof layer; downstream simplify / product_rule can turn
        # it into whatever canonical form the caller needs.
        form_part = Sum(
            Act(L_X, beta),
            Neg(Act(iota_Y, Act(self._d, alpha))),
        )
        return SectionPair(vector_part, form_part)

    def _identity_key(self) -> Any:
        # Two DorfmanBrackets are equal iff they share their axiom
        # profile AND the underlying vector bracket / Cartan operators.
        # The factories compare by identity, two calls to
        # ``lie_derivative`` producing distinct Derivation instances are
        # OK as long as the factories themselves are the same callable.
        return super()._identity_key() + (
            self._vector_bracket,
            self._d,
            self._lie_derivative,
            self._interior,
        )
