r"""
Torsion and curvature of an affine connection — Faz 16.B.

For an :class:`~jacopy.calculus.connection.AffineConnection` ``∇`` the
two structural tensors are:

* **Torsion** ``T(∇)(X, Y) := ∇_X Y − ∇_Y X − [X, Y]_VF`` —
  antisymmetric in ``(X, Y)``, vanishes for the Levi-Civita
  connection on a Riemannian manifold.
* **Curvature** ``R(∇)(X, Y) Z := ∇_X ∇_Y Z − ∇_Y ∇_X Z − ∇_{[X,Y]_VF} Z``
  — antisymmetric in the first two slots, the obstruction to ``∇``
  having flat parallel transport.

Both shapes are encoded as :class:`~jacopy.core.expr.Expr` nodes
with the connection as an opaque parametric slot and the vector-field
arguments exposed as :attr:`~Expr.children`. That keeps the
expansion engine walking freely into ``X``, ``Y``, ``Z`` for
linearity / additivity / Leibniz rewrites, without needing the
``AtomSlotLift`` workaround that the Lie-derivative-style atoms
require (see ``operator_atom_index_opacity.md``).

Two engine rules ship here, the textbook *defining* axioms:

* :class:`TorsionDefinitionDefinition` — rewrites
  ``Torsion(∇, X, Y) → ∇_X Y − ∇_Y X − [X, Y]_VF``.
* :class:`CurvatureDefinitionDefinition` — rewrites
  ``Curvature(∇, X, Y, Z) → ∇_X ∇_Y Z − ∇_Y ∇_X Z − ∇_{[X, Y]_VF} Z``.

Antisymmetry of ``T`` and of ``R``'s first two slots is a
mathematical *consequence* of these definitions plus
:class:`~jacopy.algebra.lie_bracket_vf.LieBracketVF`'s antisymmetry,
not a separate axiom — Faz 16.D's Bianchi closure exercises it
through the LBVF-antisymmetry rule already in the engine bundle.
"""

from __future__ import annotations

from typing import Any, Tuple

from jacopy.algebra.lie_bracket_vf import LieBracketVF
from jacopy.calculus.connection import AffineConnection, ConnectionEvalExpr
from jacopy.core.expr import Expr, Neg, Sum
from jacopy.proof.expansion import Definition


# --------------------------------------------------------------------- #
# Torsion node                                                           #
# --------------------------------------------------------------------- #


class Torsion(Expr):
    r"""``T(∇)(X, Y)`` — torsion tensor evaluated on two vector fields.

    Children are ``(X, Y)``; the connection slot is parametric. Two
    instances with the same connection and the same arguments compare
    equal regardless of where they were constructed.
    """

    __slots__ = ("_connection", "_X", "_Y")

    def __init__(
        self, connection: AffineConnection, X: Expr, Y: Expr
    ) -> None:
        if not isinstance(connection, AffineConnection):
            raise TypeError("Torsion requires an AffineConnection")
        if not isinstance(X, Expr):
            raise TypeError("Torsion X must be an Expr")
        if not isinstance(Y, Expr):
            raise TypeError("Torsion Y must be an Expr")
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

    def _rebuild(self, new_children: Tuple[Expr, ...]) -> "Torsion":
        if len(new_children) != 2:
            raise ValueError(
                "Torsion._rebuild expects exactly two children"
            )
        return Torsion(self._connection, new_children[0], new_children[1])

    def _key(self) -> Any:
        return (self._connection, self._X, self._Y)

    def _repr_inner(self) -> str:
        return (
            f"T({self._connection._repr_inner()})"
            f"({self._X._repr_inner()},{self._Y._repr_inner()})"
        )


# --------------------------------------------------------------------- #
# Curvature node                                                         #
# --------------------------------------------------------------------- #


class Curvature(Expr):
    r"""``R(∇)(X, Y) Z`` — curvature tensor evaluated on three vector fields.

    Children are ``(X, Y, Z)``; the connection slot is parametric.
    The convention here matches Tu/Cattaneo: ``R(X, Y) Z`` is the
    commutator obstruction ``∇_X ∇_Y Z − ∇_Y ∇_X Z − ∇_{[X, Y]_VF} Z``,
    antisymmetric in ``(X, Y)``, ``Z`` slot is the operand.
    """

    __slots__ = ("_connection", "_X", "_Y", "_Z")

    def __init__(
        self,
        connection: AffineConnection,
        X: Expr,
        Y: Expr,
        Z: Expr,
    ) -> None:
        if not isinstance(connection, AffineConnection):
            raise TypeError("Curvature requires an AffineConnection")
        for slot, val in (("X", X), ("Y", Y), ("Z", Z)):
            if not isinstance(val, Expr):
                raise TypeError(f"Curvature {slot} must be an Expr")
        self._connection = connection
        self._X = X
        self._Y = Y
        self._Z = Z

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
    def Z(self) -> Expr:
        return self._Z

    @property
    def children(self) -> Tuple[Expr, ...]:
        return (self._X, self._Y, self._Z)

    def _rebuild(self, new_children: Tuple[Expr, ...]) -> "Curvature":
        if len(new_children) != 3:
            raise ValueError(
                "Curvature._rebuild expects exactly three children"
            )
        return Curvature(
            self._connection,
            new_children[0],
            new_children[1],
            new_children[2],
        )

    def _key(self) -> Any:
        return (self._connection, self._X, self._Y, self._Z)

    def _repr_inner(self) -> str:
        return (
            f"R({self._connection._repr_inner()})"
            f"({self._X._repr_inner()},{self._Y._repr_inner()})"
            f"({self._Z._repr_inner()})"
        )


# --------------------------------------------------------------------- #
# Engine axioms                                                          #
# --------------------------------------------------------------------- #


class TorsionDefinitionDefinition(Definition):
    r"""``T(∇)(X, Y) → ∇_X Y − ∇_Y X − [X, Y]_VF``.

    The textbook torsion definition. Scoped to a specific connection:
    two connections coexisting in the same proof don't cross-fire.
    """

    def __init__(self, conn: AffineConnection) -> None:
        if not isinstance(conn, AffineConnection):
            raise TypeError(
                "TorsionDefinitionDefinition requires an AffineConnection"
            )
        self._conn = conn
        self.name = (
            f"T(∇)(X,Y) = ∇_X Y − ∇_Y X − [X,Y]_VF [{conn._repr_inner()}]"
        )

    def matches(self, expr: Expr) -> bool:
        return isinstance(expr, Torsion) and expr.connection == self._conn

    def rewrite(self, expr: Expr) -> Expr:
        X, Y = expr.X, expr.Y
        return Sum.make(
            ConnectionEvalExpr(self._conn, X, Y),
            Neg(ConnectionEvalExpr(self._conn, Y, X)),
            Neg(LieBracketVF(X, Y)),
        )


class CurvatureDefinitionDefinition(Definition):
    r"""``R(∇)(X, Y) Z → ∇_X ∇_Y Z − ∇_Y ∇_X Z − ∇_{[X, Y]_VF} Z``.

    The textbook curvature definition. The third term ``∇_{[X, Y]_VF} Z``
    is emitted as ``ConnectionEvalExpr(∇, [X, Y]_VF, Z)`` — the LBVF
    sits in ∇'s X-slot, where the X-linearity / additivity rules walk
    naturally. (LBVF is itself a :class:`Derivation`, so degree-of
    treats it as a vector field, which matches its mathematical role.)
    """

    def __init__(self, conn: AffineConnection) -> None:
        if not isinstance(conn, AffineConnection):
            raise TypeError(
                "CurvatureDefinitionDefinition requires an AffineConnection"
            )
        self._conn = conn
        self.name = (
            f"R(∇)(X,Y)Z = ∇_X ∇_Y Z − ∇_Y ∇_X Z − ∇_[X,Y]_VF Z "
            f"[{conn._repr_inner()}]"
        )

    def matches(self, expr: Expr) -> bool:
        return (
            isinstance(expr, Curvature) and expr.connection == self._conn
        )

    def rewrite(self, expr: Expr) -> Expr:
        X, Y, Z = expr.X, expr.Y, expr.Z
        first = ConnectionEvalExpr(
            self._conn, X, ConnectionEvalExpr(self._conn, Y, Z)
        )
        second = Neg(
            ConnectionEvalExpr(
                self._conn, Y, ConnectionEvalExpr(self._conn, X, Z)
            )
        )
        third = Neg(
            ConnectionEvalExpr(self._conn, LieBracketVF(X, Y), Z)
        )
        return Sum.make(first, second, third)


# --------------------------------------------------------------------- #
# Covariant-derivative-of-tensor nodes                                   #
# --------------------------------------------------------------------- #


class TorsionCovariantDerivative(Expr):
    r"""``(∇_U T)(V, W)`` — covariant derivative of the torsion tensor.

    Children are ``(U, V, W)``; the connection slot is parametric. The
    underlying mathematical object ``∇_U T`` is itself a ``(1,2)``-tensor;
    this node represents its evaluation on ``V, W``.
    """

    __slots__ = ("_connection", "_U", "_V", "_W")

    def __init__(
        self,
        connection: AffineConnection,
        U: Expr,
        V: Expr,
        W: Expr,
    ) -> None:
        if not isinstance(connection, AffineConnection):
            raise TypeError(
                "TorsionCovariantDerivative requires an AffineConnection"
            )
        for slot, val in (("U", U), ("V", V), ("W", W)):
            if not isinstance(val, Expr):
                raise TypeError(
                    f"TorsionCovariantDerivative {slot} must be an Expr"
                )
        self._connection = connection
        self._U = U
        self._V = V
        self._W = W

    @property
    def connection(self) -> AffineConnection:
        return self._connection

    @property
    def U(self) -> Expr:
        return self._U

    @property
    def V(self) -> Expr:
        return self._V

    @property
    def W(self) -> Expr:
        return self._W

    @property
    def children(self) -> Tuple[Expr, ...]:
        return (self._U, self._V, self._W)

    def _rebuild(
        self, new_children: Tuple[Expr, ...]
    ) -> "TorsionCovariantDerivative":
        if len(new_children) != 3:
            raise ValueError(
                "TorsionCovariantDerivative._rebuild expects three children"
            )
        return TorsionCovariantDerivative(
            self._connection,
            new_children[0],
            new_children[1],
            new_children[2],
        )

    def _key(self) -> Any:
        return (self._connection, self._U, self._V, self._W)

    def _repr_inner(self) -> str:
        return (
            f"(∇_{self._U._repr_inner()} T({self._connection._repr_inner()}))"
            f"({self._V._repr_inner()},{self._W._repr_inner()})"
        )


class CurvatureCovariantDerivative(Expr):
    r"""``(∇_U R)(V, W) Z`` — covariant derivative of the curvature tensor.

    Children are ``(U, V, W, Z)``; the connection slot is parametric.
    The underlying mathematical object ``∇_U R`` is a ``(1,3)``-tensor;
    this node represents its evaluation on ``V, W, Z``.
    """

    __slots__ = ("_connection", "_U", "_V", "_W", "_Z")

    def __init__(
        self,
        connection: AffineConnection,
        U: Expr,
        V: Expr,
        W: Expr,
        Z: Expr,
    ) -> None:
        if not isinstance(connection, AffineConnection):
            raise TypeError(
                "CurvatureCovariantDerivative requires an AffineConnection"
            )
        for slot, val in (("U", U), ("V", V), ("W", W), ("Z", Z)):
            if not isinstance(val, Expr):
                raise TypeError(
                    f"CurvatureCovariantDerivative {slot} must be an Expr"
                )
        self._connection = connection
        self._U = U
        self._V = V
        self._W = W
        self._Z = Z

    @property
    def connection(self) -> AffineConnection:
        return self._connection

    @property
    def U(self) -> Expr:
        return self._U

    @property
    def V(self) -> Expr:
        return self._V

    @property
    def W(self) -> Expr:
        return self._W

    @property
    def Z(self) -> Expr:
        return self._Z

    @property
    def children(self) -> Tuple[Expr, ...]:
        return (self._U, self._V, self._W, self._Z)

    def _rebuild(
        self, new_children: Tuple[Expr, ...]
    ) -> "CurvatureCovariantDerivative":
        if len(new_children) != 4:
            raise ValueError(
                "CurvatureCovariantDerivative._rebuild expects four children"
            )
        return CurvatureCovariantDerivative(
            self._connection,
            new_children[0],
            new_children[1],
            new_children[2],
            new_children[3],
        )

    def _key(self) -> Any:
        return (self._connection, self._U, self._V, self._W, self._Z)

    def _repr_inner(self) -> str:
        return (
            f"(∇_{self._U._repr_inner()} R({self._connection._repr_inner()}))"
            f"({self._V._repr_inner()},{self._W._repr_inner()})"
            f"({self._Z._repr_inner()})"
        )


# --------------------------------------------------------------------- #
# Tensor-Leibniz axioms                                                  #
# --------------------------------------------------------------------- #


class TorsionCovariantDerivativeDefinition(Definition):
    r"""``(∇_U T)(V, W) → ∇_U T(V, W) − T(∇_U V, W) − T(V, ∇_U W)``.

    The standard tensor-Leibniz rule applied to the torsion ``(1,2)``-tensor.
    Scoped to a specific connection so two connections in the same proof
    don't cross-fire.
    """

    def __init__(self, conn: AffineConnection) -> None:
        if not isinstance(conn, AffineConnection):
            raise TypeError(
                "TorsionCovariantDerivativeDefinition requires an AffineConnection"
            )
        self._conn = conn
        self.name = (
            f"(∇_U T)(V,W) = ∇_U T(V,W) − T(∇_U V,W) − T(V,∇_U W) "
            f"[{conn._repr_inner()}]"
        )

    def matches(self, expr: Expr) -> bool:
        return (
            isinstance(expr, TorsionCovariantDerivative)
            and expr.connection == self._conn
        )

    def rewrite(self, expr: Expr) -> Expr:
        U, V, W = expr.U, expr.V, expr.W
        first = ConnectionEvalExpr(self._conn, U, Torsion(self._conn, V, W))
        second = Neg(
            Torsion(self._conn, ConnectionEvalExpr(self._conn, U, V), W)
        )
        third = Neg(
            Torsion(self._conn, V, ConnectionEvalExpr(self._conn, U, W))
        )
        return Sum.make(first, second, third)


class CurvatureCovariantDerivativeDefinition(Definition):
    r"""``(∇_U R)(V, W) Z → ∇_U R(V,W)Z − R(∇_U V, W) Z − R(V, ∇_U W) Z − R(V, W) ∇_U Z``.

    The tensor-Leibniz rule applied to the curvature ``(1,3)``-tensor.
    Scoped to a specific connection.
    """

    def __init__(self, conn: AffineConnection) -> None:
        if not isinstance(conn, AffineConnection):
            raise TypeError(
                "CurvatureCovariantDerivativeDefinition requires an AffineConnection"
            )
        self._conn = conn
        self.name = (
            f"(∇_U R)(V,W)Z = ∇_U R(V,W)Z − R(∇_U V,W)Z − R(V,∇_U W)Z "
            f"− R(V,W) ∇_U Z [{conn._repr_inner()}]"
        )

    def matches(self, expr: Expr) -> bool:
        return (
            isinstance(expr, CurvatureCovariantDerivative)
            and expr.connection == self._conn
        )

    def rewrite(self, expr: Expr) -> Expr:
        U, V, W, Z = expr.U, expr.V, expr.W, expr.Z
        first = ConnectionEvalExpr(
            self._conn, U, Curvature(self._conn, V, W, Z)
        )
        second = Neg(
            Curvature(
                self._conn, ConnectionEvalExpr(self._conn, U, V), W, Z
            )
        )
        third = Neg(
            Curvature(
                self._conn, V, ConnectionEvalExpr(self._conn, U, W), Z
            )
        )
        fourth = Neg(
            Curvature(
                self._conn, V, W, ConnectionEvalExpr(self._conn, U, Z)
            )
        )
        return Sum.make(first, second, third, fourth)
