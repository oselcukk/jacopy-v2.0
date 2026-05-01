r"""
Affine connection ∇, Faz 16.A.

An affine connection ``∇`` on a smooth manifold is a binary operator
that takes a vector field ``X`` and a vector field ``Y`` and produces
a vector field ``∇_X Y``. It is :math:`C^\infty`-linear in ``X``,
:math:`\mathbb{R}`-linear and Leibniz in ``Y``:

* ``∇_{fX + gY} Z = f ∇_X Z + g ∇_Y Z``
* ``∇_X (Y + Z) = ∇_X Y + ∇_X Z``
* ``∇_X (f Y) = X(f) Y + f ∇_X Y``

This module ships:

* :class:`AffineConnection`, an :class:`~jacopy.core.expr.Atom`
  carrying just a display name, used as the connection identifier
  inside :class:`ConnectionEvalExpr`.
* :class:`ConnectionEvalExpr`, the ``∇_X Y`` evaluation node,
  parametric in the connection. Children are ``(X, Y)`` so the
  expansion engine walks naturally into both slots, no
  operator-atom-index pre-pass needed (unlike Lie / interior /
  exterior-d, whose vector-field slot is private, see
  ``operator_atom_index_opacity.md``).
* Three engine :class:`~jacopy.proof.expansion.Definition` rules
  (X-additivity, Y-additivity, Y-Leibniz) that turn ``∇`` into a
  fully usable building block for Faz 16's torsion / curvature /
  Bianchi machinery.

The connection on functions is by definition the directional
derivative ``∇_X f := X(f)``, the Y-Leibniz rule's ``X(f)`` term
relies on this convention. For an algebroid connection
``∇̃`` carrying an anchor ``ρ`` the convention generalises to
``∇̃_σ f := ρ(σ)(f)`` (Q9 / Math 595, Poisson side ``ρ = π^♯``),
routed through :meth:`AffineConnection.function_action`.
"""

from __future__ import annotations

from typing import Any, Optional, Tuple

from jacopy.algebra.derivation import Act, degree_of
from jacopy.calculus.anchor import Anchor, AnchoredVectorField
from jacopy.core.expr import Atom, Expr, Neg, Product, Sum
from jacopy.core.registry import PropertyRegistry
from jacopy.core.symbolic_degree import Degree
from jacopy.proof.expansion import Definition


# --------------------------------------------------------------------- #
# Connection identifier atom                                             #
# --------------------------------------------------------------------- #


class AffineConnection(Atom):
    r"""An affine connection ``∇``, opaque named atom.

    Carries a display name and an optional :class:`Anchor`. Two
    :class:`AffineConnection` instances with the same ``(name, anchor)``
    pair compare equal. The atom itself does not appear inside
    :class:`Act` nodes, vector-field application is the role of
    :class:`ConnectionEvalExpr`. It exists so that engine rules can
    dispatch on a specific connection (e.g. "this rule fires only on
    ``∇ = nabla``"), in line with how
    :class:`~jacopy.calculus.musical.Sharp` carries a specific
    bivector.

    The optional ``anchor`` slot generalises the directional-derivative
    convention ``∇_X f := X(f)`` to the algebroid setting
    ``∇̃_σ f := ρ(σ)(f)``. When an anchor is present
    :meth:`function_action` produces an :class:`AnchoredVectorField`
    wrapper instead of a bare :class:`Act`.

    The optional ``bracket`` slot generalises the section-level Lie
    bracket used inside ``T(∇)(X, Y) = ∇_X Y − ∇_Y X − [X, Y]`` and the
    third term of ``R(∇)(X, Y) Z = … − ∇_{[X, Y]} Z``. When
    ``bracket=None`` the default :class:`LieBracketVF` (vector-field
    Lie bracket on ``TM``) is used; when set to a
    :class:`~jacopy.brackets.base.GradedBracket` instance the Torsion
    and Curvature defining rules emit a ``BracketApply(bracket, X, Y)``
    instead. This is the Q9 (Math 595) setup: the connection on
    ``T*M`` carries ``ρ = π^♯`` *and* ``bracket = [·,·]_K`` (Koszul).
    """

    __slots__ = ("_name", "_anchor", "_bracket")

    def __init__(
        self,
        name: str,
        *,
        anchor: Optional[Anchor] = None,
        bracket: Optional[Any] = None,
    ) -> None:
        if not isinstance(name, str):
            raise TypeError("AffineConnection name must be a str")
        if not name:
            raise ValueError("AffineConnection name must be non-empty")
        if anchor is not None and not isinstance(anchor, Anchor):
            raise TypeError("AffineConnection anchor must be an Anchor")
        if bracket is not None:
            from jacopy.brackets.base import GradedBracket
            if not isinstance(bracket, GradedBracket):
                raise TypeError(
                    "AffineConnection bracket must be a GradedBracket"
                )
        self._name = name
        self._anchor = anchor
        self._bracket = bracket

    @property
    def name(self) -> str:
        return self._name

    @property
    def anchor(self) -> Optional[Anchor]:
        return self._anchor

    @property
    def bracket(self) -> Optional[Any]:
        return self._bracket

    def _key(self) -> Any:
        return (self._name, self._anchor, self._bracket)

    def _repr_inner(self) -> str:
        return self._name

    def eval(self, X: Expr, Y: Expr) -> "ConnectionEvalExpr":
        r"""Build ``∇_X Y`` as a :class:`ConnectionEvalExpr`."""
        return ConnectionEvalExpr(self, X, Y)

    def function_action(self, X: Expr, f: Expr) -> Expr:
        r"""Return the directional-derivative term ``∇_X f`` on a 0-form ``f``.

        For a plain affine connection: ``Act(X, f)``, i.e. the vector
        field ``X`` acting on ``f`` as a derivation. For an algebroid
        connection with ``anchor = ρ`` set: ``Act(ρ(X), f)`` where
        ``ρ(X)`` is wrapped as an :class:`AnchoredVectorField`. The
        Y-Leibniz rule routes its ``X(f)``-style term through this
        method so the same axiom serves both ``TM`` and ``T*M``.
        """
        if self._anchor is None:
            return Act(X, f)
        return Act(AnchoredVectorField(self._anchor, X), f)

    def vector_bracket(self, X: Expr, Y: Expr) -> Expr:
        r"""Return the bracket ``[X, Y]`` used inside torsion / curvature.

        For a plain affine connection (``bracket=None``): the
        vector-field Lie bracket :class:`LieBracketVF`. For an
        algebroid connection carrying a custom bracket: a
        :class:`BracketApply` against that bracket. Routed through this
        method so :class:`TorsionDefinitionDefinition` and
        :class:`CurvatureDefinitionDefinition` are bracket-agnostic.
        """
        if self._bracket is None:
            from jacopy.algebra.lie_bracket_vf import LieBracketVF
            return LieBracketVF(X, Y)
        from jacopy.brackets.base import BracketApply
        return BracketApply(self._bracket, X, Y)


def connection(
    name: str = "∇",
    *,
    anchor: Optional[Anchor] = None,
    bracket: Optional[Any] = None,
) -> AffineConnection:
    r"""Functional constructor for :class:`AffineConnection`."""
    return AffineConnection(name, anchor=anchor, bracket=bracket)


def koszul_connection(
    name: str = "∇̃",
    *,
    anchor: Optional[Anchor] = None,
    anchor_name: str = "π^♯",
    bracket: Optional[Any] = None,
) -> AffineConnection:
    r"""Build an algebroid connection ``∇̃`` on ``T*M`` for the Q9 setup.

    Equivalent to ``connection(name, anchor=…, bracket=…)`` with
    Poisson-side defaults: a pre-wired connection whose function-action
    goes through the Poisson anchor ``π^♯`` and whose
    torsion / curvature definitions use the Koszul bracket
    ``[·,·]_K``. The caller may supply a pre-built ``KoszulBracket``
    instance (recommended when you want the same bracket shared with
    explicit ``BracketApply`` constructions) or let the factory build
    a fresh anchor + Koszul bracket pair.

    Used as the ``∇̃`` instance in Q9 (Math 595), the cotangent-bundle
    counterpart to the tangent-bundle ``∇`` of Q7 / Q8.
    """
    if anchor is None:
        anchor = Anchor(anchor_name)
    if bracket is None:
        from jacopy.brackets.koszul import KoszulBracket
        bracket = KoszulBracket(anchor)
    return AffineConnection(name, anchor=anchor, bracket=bracket)


# --------------------------------------------------------------------- #
# Evaluation node                                                        #
# --------------------------------------------------------------------- #


class ConnectionEvalExpr(Expr):
    r"""Connection evaluation ``∇_X Y``.

    Stores the connection as a parametric slot (not a child) and the
    two vector-field arguments as children. This keeps the
    connection identity opaque (engine rules dispatch on it) while
    letting bottom-up rewriting walk freely into ``X`` and ``Y``,
    the slot-opacity workaround that the Lie-derivative-style atoms
    require (``AtomSlotLift``, see Faz 15.C) is unnecessary here by
    construction.
    """

    __slots__ = ("_connection", "_X", "_Y")

    def __init__(
        self, connection: AffineConnection, X: Expr, Y: Expr
    ) -> None:
        if not isinstance(connection, AffineConnection):
            raise TypeError(
                "ConnectionEvalExpr requires an AffineConnection"
            )
        if not isinstance(X, Expr):
            raise TypeError("ConnectionEvalExpr X must be an Expr")
        if not isinstance(Y, Expr):
            raise TypeError("ConnectionEvalExpr Y must be an Expr")
        self._connection = connection
        self._X = X
        self._Y = Y

    @property
    def connection(self) -> AffineConnection:
        return self._connection

    @property
    def X(self) -> Expr:
        return self._X

    @property
    def Y(self) -> Expr:
        return self._Y

    @property
    def children(self) -> Tuple[Expr, ...]:
        return (self._X, self._Y)

    def _rebuild(self, new_children: Tuple[Expr, ...]) -> "ConnectionEvalExpr":
        if len(new_children) != 2:
            raise ValueError(
                "ConnectionEvalExpr._rebuild expects exactly two children"
            )
        return ConnectionEvalExpr(
            self._connection, new_children[0], new_children[1]
        )

    def _key(self) -> Any:
        return (self._connection, self._X, self._Y)

    def _repr_inner(self) -> str:
        return (
            f"{self._connection._repr_inner()}_"
            f"{self._X._repr_inner()}({self._Y._repr_inner()})"
        )


# --------------------------------------------------------------------- #
# Engine axioms                                                          #
# --------------------------------------------------------------------- #


def _is_degree_zero(
    expr: Expr, registry: Optional[PropertyRegistry]
) -> bool:
    """Safe degree-zero check: returns False on any undecidable case."""
    try:
        return degree_of(expr, registry) == Degree.const(0)
    except ValueError:
        return False


class ConnectionXLinearityDefinition(Definition):
    r"""``∇_{A + B + …} Y → ∇_A Y + ∇_B Y + …`` and ``∇_{-A} Y → -∇_A Y``.

    Distributes :class:`Sum` and :class:`Neg` in the X-slot. Scoped to
    a specific :class:`AffineConnection` so two connections in the
    same proof don't cross-fire.
    """

    def __init__(self, conn: AffineConnection) -> None:
        if not isinstance(conn, AffineConnection):
            raise TypeError(
                "ConnectionXLinearityDefinition requires an AffineConnection"
            )
        self._conn = conn
        self.name = f"∇_X X-linearity [{conn._repr_inner()}]"

    def matches(self, expr: Expr) -> bool:
        return (
            isinstance(expr, ConnectionEvalExpr)
            and expr.connection == self._conn
            and isinstance(expr.X, (Sum, Neg))
        )

    def rewrite(self, expr: Expr) -> Expr:
        Y = expr.Y
        x_slot = expr.X
        if isinstance(x_slot, Neg):
            return Neg(ConnectionEvalExpr(self._conn, x_slot.arg, Y))
        terms = []
        for c in x_slot.children:
            if isinstance(c, Neg):
                terms.append(
                    Neg(ConnectionEvalExpr(self._conn, c.arg, Y))
                )
            else:
                terms.append(ConnectionEvalExpr(self._conn, c, Y))
        return Sum.make(*terms)


class ConnectionXScalarPullDefinition(Definition):
    r"""``∇_{f · X} Y → f · ∇_X Y``, :math:`C^\infty`-linearity in X.

    Fires when the X-slot is a :class:`Product` of two or more factors.
    The leading factors are folded into the scalar prefactor ``f`` and
    the last factor is the underlying vector ``X``. Same convention as
    :class:`~jacopy.calculus.non_metricity.NonMetricityVScalarPullDefinition`
    and the other ScalarPull rules in the codebase: last factor is the
    geometric object, leading factors are scalars.

    Scoped to a specific :class:`AffineConnection`.
    """

    def __init__(self, conn: AffineConnection) -> None:
        if not isinstance(conn, AffineConnection):
            raise TypeError(
                "ConnectionXScalarPullDefinition requires an AffineConnection"
            )
        self._conn = conn
        self.name = f"∇_X X-scalar pull [{conn._repr_inner()}]"

    def matches(self, expr: Expr) -> bool:
        return (
            isinstance(expr, ConnectionEvalExpr)
            and expr.connection == self._conn
            and isinstance(expr.X, Product)
            and len(expr.X.children) >= 2
        )

    def rewrite(self, expr: Expr) -> Expr:
        prod = expr.X
        assert isinstance(prod, Product)
        *scalars, head = prod.children
        f = Product.make(*scalars)
        return Product(f, ConnectionEvalExpr(self._conn, head, expr.Y))


class ConnectionYAdditivityDefinition(Definition):
    r"""``∇_X (A + B + …) → ∇_X A + ∇_X B + …`` and ``∇_X (-A) → -∇_X A``.

    Sum / Neg distribution in the Y-slot. Scoped to a specific
    connection. Together with the X-linearity rule and Y-Leibniz, this
    completes the textbook ``(∇_X)`` linearity package.
    """

    def __init__(self, conn: AffineConnection) -> None:
        if not isinstance(conn, AffineConnection):
            raise TypeError(
                "ConnectionYAdditivityDefinition requires an AffineConnection"
            )
        self._conn = conn
        self.name = f"∇_X Y-additivity [{conn._repr_inner()}]"

    def matches(self, expr: Expr) -> bool:
        return (
            isinstance(expr, ConnectionEvalExpr)
            and expr.connection == self._conn
            and isinstance(expr.Y, (Sum, Neg))
        )

    def rewrite(self, expr: Expr) -> Expr:
        X = expr.X
        y_slot = expr.Y
        if isinstance(y_slot, Neg):
            return Neg(ConnectionEvalExpr(self._conn, X, y_slot.arg))
        terms = []
        for c in y_slot.children:
            if isinstance(c, Neg):
                terms.append(
                    Neg(ConnectionEvalExpr(self._conn, X, c.arg))
                )
            else:
                terms.append(ConnectionEvalExpr(self._conn, X, c))
        return Sum.make(*terms)


class ConnectionYLeibnizDefinition(Definition):
    r"""``∇_X (f · Y) → X(f) · Y + f · ∇_X Y`` for 0-form ``f``.

    Fires when ``Y``-slot is a :class:`~jacopy.core.expr.Product`
    whose first factor resolves to degree zero in the registry.
    Anything more complicated (multi-factor products, mixed-degree
    factors) is left alone, let the X-additivity / Y-additivity
    rules canonicalise the slot first.

    The ``X(f)`` term is emitted via the connection's
    :meth:`AffineConnection.function_action`: a bare ``Act(X, f)`` for
    plain affine connections, ``Act(ρ(X), f)`` for algebroid
    connections that carry an anchor (Q9 setup). This bridges the
    connection's Leibniz on functions (``∇_X f = X(f)`` resp.
    ``∇̃_X f = ρ(X)(f)``) into the standard derivation pipeline.
    """

    def __init__(
        self,
        conn: AffineConnection,
        *,
        registry: Optional[PropertyRegistry] = None,
    ) -> None:
        if not isinstance(conn, AffineConnection):
            raise TypeError(
                "ConnectionYLeibnizDefinition requires an AffineConnection"
            )
        self._conn = conn
        self._registry = registry
        self.name = f"∇_X Y-Leibniz [{conn._repr_inner()}]"

    def matches(self, expr: Expr) -> bool:
        if not isinstance(expr, ConnectionEvalExpr):
            return False
        if expr.connection != self._conn:
            return False
        if not isinstance(expr.Y, Product):
            return False
        if len(expr.Y.children) < 2:
            return False
        # First factor must be a 0-form (function); second factor
        # carries the rest of the product (typically a vector field).
        return _is_degree_zero(expr.Y.children[0], self._registry)

    def rewrite(self, expr: Expr) -> Expr:
        X = expr.X
        prod = expr.Y
        f = prod.children[0]
        rest = prod.children[1:]
        Y_part = (
            rest[0] if len(rest) == 1 else Product.make(*rest)
        )
        return Sum.make(
            Product.make(self._conn.function_action(X, f), Y_part),
            Product.make(f, ConnectionEvalExpr(self._conn, X, Y_part)),
        )
