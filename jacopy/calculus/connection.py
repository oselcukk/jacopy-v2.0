r"""
Affine connection ∇ — Faz 16.A.

An affine connection ``∇`` on a smooth manifold is a binary operator
that takes a vector field ``X`` and a vector field ``Y`` and produces
a vector field ``∇_X Y``. It is :math:`C^\infty`-linear in ``X``,
:math:`\mathbb{R}`-linear and Leibniz in ``Y``:

* ``∇_{fX + gY} Z = f ∇_X Z + g ∇_Y Z``
* ``∇_X (Y + Z) = ∇_X Y + ∇_X Z``
* ``∇_X (f Y) = X(f) Y + f ∇_X Y``

This module ships:

* :class:`AffineConnection` — an :class:`~jacopy.core.expr.Atom`
  carrying just a display name, used as the connection identifier
  inside :class:`ConnectionEvalExpr`.
* :class:`ConnectionEvalExpr` — the ``∇_X Y`` evaluation node,
  parametric in the connection. Children are ``(X, Y)`` so the
  expansion engine walks naturally into both slots — no
  operator-atom-index pre-pass needed (unlike Lie / interior /
  exterior-d, whose vector-field slot is private — see
  ``operator_atom_index_opacity.md``).
* Three engine :class:`~jacopy.proof.expansion.Definition` rules
  (X-additivity, Y-additivity, Y-Leibniz) that turn ``∇`` into a
  fully usable building block for Faz 16's torsion / curvature /
  Bianchi machinery.

The connection on functions is by definition the directional
derivative ``∇_X f := X(f)`` — the Y-Leibniz rule's ``X(f)`` term
relies on this convention. The companion rule
:class:`ConnectionOnFunctionDefinition` makes it an explicit
engine step.
"""

from __future__ import annotations

from typing import Any, Optional, Tuple

from jacopy.algebra.derivation import Act, degree_of
from jacopy.core.expr import Atom, Expr, Neg, Product, Sum
from jacopy.core.registry import PropertyRegistry
from jacopy.core.symbolic_degree import Degree
from jacopy.proof.expansion import Definition


# --------------------------------------------------------------------- #
# Connection identifier atom                                             #
# --------------------------------------------------------------------- #


class AffineConnection(Atom):
    r"""An affine connection ``∇`` — opaque named atom.

    Carries just a display name. Two :class:`AffineConnection`
    instances with the same name compare equal. The atom itself does
    not appear inside :class:`Act` nodes — vector-field application is
    the role of :class:`ConnectionEvalExpr`. It exists so that engine
    rules can dispatch on a specific connection (e.g.
    "this rule fires only on ``∇ = nabla``"), in line with how
    :class:`~jacopy.calculus.musical.Sharp` carries a specific
    bivector.
    """

    __slots__ = ("_name",)

    def __init__(self, name: str) -> None:
        if not isinstance(name, str):
            raise TypeError("AffineConnection name must be a str")
        if not name:
            raise ValueError("AffineConnection name must be non-empty")
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    def _key(self) -> Any:
        return self._name

    def _repr_inner(self) -> str:
        return self._name

    def eval(self, X: Expr, Y: Expr) -> "ConnectionEvalExpr":
        r"""Build ``∇_X Y`` as a :class:`ConnectionEvalExpr`."""
        return ConnectionEvalExpr(self, X, Y)


def connection(name: str = "∇") -> AffineConnection:
    r"""Functional constructor for :class:`AffineConnection`."""
    return AffineConnection(name)


# --------------------------------------------------------------------- #
# Evaluation node                                                        #
# --------------------------------------------------------------------- #


class ConnectionEvalExpr(Expr):
    r"""Connection evaluation ``∇_X Y``.

    Stores the connection as a parametric slot (not a child) and the
    two vector-field arguments as children. This keeps the
    connection identity opaque (engine rules dispatch on it) while
    letting bottom-up rewriting walk freely into ``X`` and ``Y`` —
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
    factors) is left alone — let the X-additivity / Y-additivity
    rules canonicalise the slot first.

    The ``X(f)`` term is emitted as ``Act(X, f)``: ``X`` itself is
    treated as a derivation on functions (which it is — every vector
    field is). This bridges the connection's Leibniz on functions
    (``∇_X f = X(f)``) into the standard derivation pipeline.
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
            Product.make(Act(X, f), Y_part),
            Product.make(f, ConnectionEvalExpr(self._conn, X, Y_part)),
        )
