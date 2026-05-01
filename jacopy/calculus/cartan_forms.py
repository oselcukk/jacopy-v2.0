r"""
Cartan-form Expr nodes, Faz 17.C.

For an :class:`~jacopy.calculus.connection.AffineConnection` ``∇``,
a (Faz 17.A) :class:`~jacopy.calculus.local_frame.LocalFrame` ``F``,
and (where relevant) a :class:`~jacopy.calculus.metric.MetricTensor`
``g``, four local-component forms appear naturally:

.. math::

    \omega^a{}_b(\nabla)(V)        &:= e^a(\nabla_V X_b),         \\
    Q_{ab}(\nabla, g)(V)           &:= Q(\nabla, g)(V, X_a, X_b), \\
    T^a(\nabla)(U, V)              &:= e^a(T(\nabla)(U, V)),      \\
    R^a{}_b(\nabla)(U, V)          &:= e^a(R(\nabla)(U, V) X_b).

This module ships each as a dedicated :class:`Atom` keyed on its
parameter tuple, plus the four engine rules that open the eval shape:

* :class:`ConnectionForm` (1-form), scoped to ``(connection, frame)``
  with two indices. Eval rule rewrites the
  :class:`~jacopy.calculus.pairing.Pairing` ``⟨ω^a_b, V⟩`` into
  ``⟨e^a, ∇_V X_b⟩``, after which Pairing C∞-linearity (Faz 12.B)
  and ConnectionEval X-linearity (Faz 16.A) carry the proof.
* :class:`NonMetricityForm` (1-form), scoped to
  ``(connection, metric, frame)`` with two indices. Eval rule rewrites
  ``⟨Q_{ab}, V⟩`` into :class:`~jacopy.calculus.non_metricity.NonMetricityEvalExpr`
  applied to ``(V, X_a, X_b)``.
* :class:`TorsionForm` (2-form), scoped to ``(connection, frame)``
  with one index. Eval rule rewrites the
  :class:`~jacopy.core.multi_eval.MultiEval` ``T^a(U, V)`` into
  ``⟨e^a, T(∇)(U, V)⟩``.
* :class:`CurvatureForm` (2-form), scoped to ``(connection, frame)``
  with two indices. Eval rule rewrites ``R^a_b(U, V)`` into
  ``⟨e^a, R(∇)(U, V) X_b⟩``.

Convention: 2-form heads carry ``alternating=True`` in their
:class:`MultiEval` evaluations so the standard alternating-normal /
repeat-arg-zero engine passes apply automatically. 1-forms route
through :class:`Pairing` and need no flag, Pairing is bilinear in
the ordinary sense.

Each form node is an opaque :class:`Atom`: it has no children, no
evaluation by construction, and only enters the engine through the
companion ``…FormDefinition`` rule. That keeps these nodes inert
inside any walk that does *not* include the corresponding axiom in
its bundle, so a notebook reasoning *about* ``ω^a_b`` symbolically
(without unfolding) is fine.
"""

from __future__ import annotations

from typing import Any, Tuple, Union

from jacopy.calculus.connection import AffineConnection, ConnectionEvalExpr
from jacopy.calculus.local_frame import (
    FrameIndex,
    LocalFrame,
    _coerce_index,
)
from jacopy.calculus.metric import MetricTensor
from jacopy.calculus.non_metricity import NonMetricityEvalExpr
from jacopy.calculus.pairing import Pairing
from jacopy.calculus.torsion_curvature import Curvature, Torsion
from jacopy.core.expr import Atom, Expr
from jacopy.core.multi_eval import MultiEval
from jacopy.proof.expansion import Definition


# --------------------------------------------------------------------- #
# ConnectionForm  ω^a_b(∇)                                              #
# --------------------------------------------------------------------- #


class ConnectionForm(Atom):
    r"""Connection 1-form ``ω^a{}_b(∇)`` on a local frame.

    Identified by ``(connection, frame, upper, lower)``. The
    ``frame`` is compared by its identity tuple (not just its name)
    so two frames sharing a name but differing in dimension or display
    symbols stay distinguishable.
    """

    __slots__ = ("_connection", "_frame", "_upper", "_lower")

    def __init__(
        self,
        connection: AffineConnection,
        frame: LocalFrame,
        upper: Union[FrameIndex, str],
        lower: Union[FrameIndex, str],
    ) -> None:
        if not isinstance(connection, AffineConnection):
            raise TypeError("ConnectionForm requires an AffineConnection")
        if not isinstance(frame, LocalFrame):
            raise TypeError("ConnectionForm requires a LocalFrame")
        self._connection = connection
        self._frame = frame
        self._upper = _coerce_index(upper)
        self._lower = _coerce_index(lower)

    @property
    def connection(self) -> AffineConnection:
        return self._connection

    @property
    def frame(self) -> LocalFrame:
        return self._frame

    @property
    def upper(self) -> FrameIndex:
        return self._upper

    @property
    def lower(self) -> FrameIndex:
        return self._lower

    def _key(self) -> Any:
        return (self._connection, self._frame, self._upper, self._lower)

    def _repr_inner(self) -> str:
        return (
            f"ω^{self._upper._repr_inner()}_"
            f"{self._lower._repr_inner()}"
            f"({self._connection._repr_inner()})"
        )

    def substitute_atom(self, dummy: Expr, target: Expr) -> Expr:
        if self == dummy:
            return target
        if not (isinstance(dummy, FrameIndex) and isinstance(target, FrameIndex)):
            return self
        new_upper = target if self._upper == dummy else self._upper
        new_lower = target if self._lower == dummy else self._lower
        if new_upper is self._upper and new_lower is self._lower:
            return self
        return ConnectionForm(
            self._connection, self._frame, new_upper, new_lower
        )


class ConnectionFormDefinition(Definition):
    r"""``⟨ω^a_b(∇), V⟩ → ⟨e^a, ∇_V X_b⟩``.

    Fires on a :class:`~jacopy.calculus.pairing.Pairing` whose first
    slot is a :class:`ConnectionForm` matching this rule's
    ``(connection, frame)`` pair. The second slot ``V`` is left alone,
    Pairing C∞-linearity / Sum / Neg distribution handle non-atomic
    ``V`` before this rule fires (bottom-up walk).
    """

    def __init__(
        self, connection: AffineConnection, frame: LocalFrame
    ) -> None:
        if not isinstance(connection, AffineConnection):
            raise TypeError(
                "ConnectionFormDefinition requires an AffineConnection"
            )
        if not isinstance(frame, LocalFrame):
            raise TypeError(
                "ConnectionFormDefinition requires a LocalFrame"
            )
        self._connection = connection
        self._frame = frame
        self.name = (
            f"ω definition [{connection._repr_inner()},"
            f"{frame.name}]"
        )

    def matches(self, expr: Expr) -> bool:
        if not isinstance(expr, Pairing):
            return False
        head = expr.alpha
        if not isinstance(head, ConnectionForm):
            return False
        return (
            head.connection == self._connection
            and head.frame == self._frame
        )

    def rewrite(self, expr: Expr) -> Expr:
        assert isinstance(expr, Pairing)
        head = expr.alpha
        assert isinstance(head, ConnectionForm)
        V = expr.X
        return Pairing(
            self._frame.coframe(head.upper),
            ConnectionEvalExpr(
                self._connection, V, self._frame.X(head.lower)
            ),
        )


# --------------------------------------------------------------------- #
# NonMetricityForm  Q_{ab}(∇, g)                                        #
# --------------------------------------------------------------------- #


class NonMetricityForm(Atom):
    r"""Non-metricity 1-form ``Q_{ab}(∇, g)`` on a local frame.

    Identified by ``(connection, metric, frame, lower_a, lower_b)``;
    both indices sit in the *lower* slot mirroring the textbook
    ``Q_{ab}`` notation.
    """

    __slots__ = (
        "_connection",
        "_metric",
        "_frame",
        "_lower_a",
        "_lower_b",
    )

    def __init__(
        self,
        connection: AffineConnection,
        metric: MetricTensor,
        frame: LocalFrame,
        lower_a: Union[FrameIndex, str],
        lower_b: Union[FrameIndex, str],
    ) -> None:
        if not isinstance(connection, AffineConnection):
            raise TypeError("NonMetricityForm requires an AffineConnection")
        if not isinstance(metric, MetricTensor):
            raise TypeError("NonMetricityForm requires a MetricTensor")
        if not isinstance(frame, LocalFrame):
            raise TypeError("NonMetricityForm requires a LocalFrame")
        self._connection = connection
        self._metric = metric
        self._frame = frame
        self._lower_a = _coerce_index(lower_a)
        self._lower_b = _coerce_index(lower_b)

    @property
    def connection(self) -> AffineConnection:
        return self._connection

    @property
    def metric(self) -> MetricTensor:
        return self._metric

    @property
    def frame(self) -> LocalFrame:
        return self._frame

    @property
    def lower_a(self) -> FrameIndex:
        return self._lower_a

    @property
    def lower_b(self) -> FrameIndex:
        return self._lower_b

    def _key(self) -> Any:
        return (
            self._connection,
            self._metric,
            self._frame,
            self._lower_a,
            self._lower_b,
        )

    def _repr_inner(self) -> str:
        return (
            f"Q_{self._lower_a._repr_inner()}"
            f"{self._lower_b._repr_inner()}"
            f"({self._connection._repr_inner()},"
            f"{self._metric._repr_inner()})"
        )

    def substitute_atom(self, dummy: Expr, target: Expr) -> Expr:
        if self == dummy:
            return target
        if not (isinstance(dummy, FrameIndex) and isinstance(target, FrameIndex)):
            return self
        new_a = target if self._lower_a == dummy else self._lower_a
        new_b = target if self._lower_b == dummy else self._lower_b
        if new_a is self._lower_a and new_b is self._lower_b:
            return self
        return NonMetricityForm(
            self._connection, self._metric, self._frame, new_a, new_b
        )


class NonMetricityFormDefinition(Definition):
    r"""``⟨Q_{ab}(∇, g), V⟩ → Q(∇, g)(V, X_a, X_b)``.

    Scoped to a specific ``(connection, metric, frame)`` triple so
    two metrics or two frames in the same proof never cross-fire.
    """

    def __init__(
        self,
        connection: AffineConnection,
        metric: MetricTensor,
        frame: LocalFrame,
    ) -> None:
        if not isinstance(connection, AffineConnection):
            raise TypeError(
                "NonMetricityFormDefinition requires an AffineConnection"
            )
        if not isinstance(metric, MetricTensor):
            raise TypeError(
                "NonMetricityFormDefinition requires a MetricTensor"
            )
        if not isinstance(frame, LocalFrame):
            raise TypeError(
                "NonMetricityFormDefinition requires a LocalFrame"
            )
        self._connection = connection
        self._metric = metric
        self._frame = frame
        self.name = (
            f"Q form definition [{connection._repr_inner()},"
            f"{metric._repr_inner()},{frame.name}]"
        )

    def matches(self, expr: Expr) -> bool:
        if not isinstance(expr, Pairing):
            return False
        head = expr.alpha
        if not isinstance(head, NonMetricityForm):
            return False
        return (
            head.connection == self._connection
            and head.metric == self._metric
            and head.frame == self._frame
        )

    def rewrite(self, expr: Expr) -> Expr:
        assert isinstance(expr, Pairing)
        head = expr.alpha
        assert isinstance(head, NonMetricityForm)
        V = expr.X
        return NonMetricityEvalExpr(
            self._connection,
            self._metric,
            V,
            self._frame.X(head.lower_a),
            self._frame.X(head.lower_b),
        )


# --------------------------------------------------------------------- #
# TorsionForm  T^a(∇)                                                   #
# --------------------------------------------------------------------- #


class TorsionForm(Atom):
    r"""Torsion 2-form ``T^a(∇)`` on a local frame.

    Identified by ``(connection, frame, upper)``. Evaluated through
    :class:`~jacopy.core.multi_eval.MultiEval` with two vector-field
    arguments; the head carries ``alternating=True`` by convention so
    the existing alternating-normal / repeat-arg-zero engine rules
    fire.
    """

    __slots__ = ("_connection", "_frame", "_upper")

    def __init__(
        self,
        connection: AffineConnection,
        frame: LocalFrame,
        upper: Union[FrameIndex, str],
    ) -> None:
        if not isinstance(connection, AffineConnection):
            raise TypeError("TorsionForm requires an AffineConnection")
        if not isinstance(frame, LocalFrame):
            raise TypeError("TorsionForm requires a LocalFrame")
        self._connection = connection
        self._frame = frame
        self._upper = _coerce_index(upper)

    @property
    def connection(self) -> AffineConnection:
        return self._connection

    @property
    def frame(self) -> LocalFrame:
        return self._frame

    @property
    def upper(self) -> FrameIndex:
        return self._upper

    def _key(self) -> Any:
        return (self._connection, self._frame, self._upper)

    def _repr_inner(self) -> str:
        return (
            f"T^{self._upper._repr_inner()}"
            f"({self._connection._repr_inner()})"
        )

    def substitute_atom(self, dummy: Expr, target: Expr) -> Expr:
        if self == dummy:
            return target
        if not (isinstance(dummy, FrameIndex) and isinstance(target, FrameIndex)):
            return self
        if self._upper == dummy:
            return TorsionForm(self._connection, self._frame, target)
        return self


class TorsionFormDefinition(Definition):
    r"""``T^a(∇)(U, V) → ⟨e^a, T(∇)(U, V)⟩``.

    Fires on a :class:`MultiEval` of arity 2 whose head is a
    :class:`TorsionForm` matching this rule's ``(connection, frame)``
    pair.
    """

    def __init__(
        self, connection: AffineConnection, frame: LocalFrame
    ) -> None:
        if not isinstance(connection, AffineConnection):
            raise TypeError(
                "TorsionFormDefinition requires an AffineConnection"
            )
        if not isinstance(frame, LocalFrame):
            raise TypeError(
                "TorsionFormDefinition requires a LocalFrame"
            )
        self._connection = connection
        self._frame = frame
        self.name = (
            f"T form definition [{connection._repr_inner()},"
            f"{frame.name}]"
        )

    def matches(self, expr: Expr) -> bool:
        if not isinstance(expr, MultiEval):
            return False
        if expr.arity != 2:
            return False
        head = expr.head
        if not isinstance(head, TorsionForm):
            return False
        return (
            head.connection == self._connection
            and head.frame == self._frame
        )

    def rewrite(self, expr: Expr) -> Expr:
        assert isinstance(expr, MultiEval)
        head = expr.head
        assert isinstance(head, TorsionForm)
        U, V = expr.args
        return Pairing(
            self._frame.coframe(head.upper),
            Torsion(self._connection, U, V),
        )


# --------------------------------------------------------------------- #
# CurvatureForm  R^a_b(∇)                                               #
# --------------------------------------------------------------------- #


class CurvatureForm(Atom):
    r"""Curvature 2-form ``R^a{}_b(∇)`` on a local frame.

    Identified by ``(connection, frame, upper, lower)``. Same
    :class:`MultiEval`-driven evaluation pattern as
    :class:`TorsionForm`.
    """

    __slots__ = ("_connection", "_frame", "_upper", "_lower")

    def __init__(
        self,
        connection: AffineConnection,
        frame: LocalFrame,
        upper: Union[FrameIndex, str],
        lower: Union[FrameIndex, str],
    ) -> None:
        if not isinstance(connection, AffineConnection):
            raise TypeError("CurvatureForm requires an AffineConnection")
        if not isinstance(frame, LocalFrame):
            raise TypeError("CurvatureForm requires a LocalFrame")
        self._connection = connection
        self._frame = frame
        self._upper = _coerce_index(upper)
        self._lower = _coerce_index(lower)

    @property
    def connection(self) -> AffineConnection:
        return self._connection

    @property
    def frame(self) -> LocalFrame:
        return self._frame

    @property
    def upper(self) -> FrameIndex:
        return self._upper

    @property
    def lower(self) -> FrameIndex:
        return self._lower

    def _key(self) -> Any:
        return (self._connection, self._frame, self._upper, self._lower)

    def _repr_inner(self) -> str:
        return (
            f"R^{self._upper._repr_inner()}_"
            f"{self._lower._repr_inner()}"
            f"({self._connection._repr_inner()})"
        )

    def substitute_atom(self, dummy: Expr, target: Expr) -> Expr:
        if self == dummy:
            return target
        if not (isinstance(dummy, FrameIndex) and isinstance(target, FrameIndex)):
            return self
        new_upper = target if self._upper == dummy else self._upper
        new_lower = target if self._lower == dummy else self._lower
        if new_upper is self._upper and new_lower is self._lower:
            return self
        return CurvatureForm(
            self._connection, self._frame, new_upper, new_lower
        )


class CurvatureFormDefinition(Definition):
    r"""``R^a{}_b(∇)(U, V) → ⟨e^a, R(∇)(U, V) X_b⟩``.

    Fires on a :class:`MultiEval` of arity 2 whose head is a
    :class:`CurvatureForm` matching this rule's ``(connection, frame)``
    pair.
    """

    def __init__(
        self, connection: AffineConnection, frame: LocalFrame
    ) -> None:
        if not isinstance(connection, AffineConnection):
            raise TypeError(
                "CurvatureFormDefinition requires an AffineConnection"
            )
        if not isinstance(frame, LocalFrame):
            raise TypeError(
                "CurvatureFormDefinition requires a LocalFrame"
            )
        self._connection = connection
        self._frame = frame
        self.name = (
            f"R form definition [{connection._repr_inner()},"
            f"{frame.name}]"
        )

    def matches(self, expr: Expr) -> bool:
        if not isinstance(expr, MultiEval):
            return False
        if expr.arity != 2:
            return False
        head = expr.head
        if not isinstance(head, CurvatureForm):
            return False
        return (
            head.connection == self._connection
            and head.frame == self._frame
        )

    def rewrite(self, expr: Expr) -> Expr:
        assert isinstance(expr, MultiEval)
        head = expr.head
        assert isinstance(head, CurvatureForm)
        U, V = expr.args
        return Pairing(
            self._frame.coframe(head.upper),
            Curvature(
                self._connection, U, V, self._frame.X(head.lower)
            ),
        )
