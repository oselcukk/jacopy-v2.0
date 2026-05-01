r"""
Non-metricity tensor ``Q(∇, g)``, Faz 17.B.

For an affine connection ``∇`` and a metric ``g`` the non-metricity
tensor is the ``(0, 3)``-tensor

.. math::

    Q(\nabla, g)(V, X, Y) \;:=\; (\nabla_V g)(X, Y).

For the local-component / form-property reading the only fact
needed downstream is that ``Q(∇, g)(V, X, Y)`` is
:math:`C^\infty`-linear in ``V``, the
``V``-slot is exactly what becomes the local 1-form ``Q_{ab}(∇, g)``
when ``X = X_a`` and ``Y = X_b``. The compatibility-style expansion
``Q = ∂g - g(∇·, ·) - g(·, ∇·)`` is also mechanised here as an
**opt-in** rule: include
:class:`NonMetricityCompatibilityDefinition` in the engine bundle
to open every ``Q``, leave it out to keep ``Q`` opaque. The two
modes don't interfere, the closure axioms (V-linearity, X↔Y
symmetry) and the opener live in the same registry, but the engine
only fires what's been registered.

This module ships:

* :class:`NonMetricityEvalExpr`, the ``Q(∇, g)(V, X, Y)``
  evaluation node. Children are ``(V, X, Y)`` so the engine walks
  freely; the connection and metric sit in parametric slots, exactly
  as :class:`~jacopy.calculus.torsion_curvature.Torsion` does for
  ``∇``.
* :class:`NonMetricityVLinearityDefinition`, distributes
  :class:`Sum` and :class:`Neg` in the ``V``-slot. Same shape as
  :class:`~jacopy.calculus.connection.ConnectionXLinearityDefinition`.
* :class:`NonMetricityVScalarPullDefinition`,
  ``Q(f · V, X, Y) → f · Q(V, X, Y)`` with ``f`` a scalar prefactor.
  Models the C∞-linearity of the V-slot at the engine level so the
  local-component identity ``Q_{ab}(fV) = f · Q_{ab}(V)`` reduces in
  two rewrites (Sum → distribute → ScalarPull on a singleton Sum).
* :class:`NonMetricityXYSymmetryDefinition`, canonicalises
  ``Q(V, X, Y) → Q(V, Y, X)`` whenever the last two slots are out of
  ``repr``-order. Q inherits ``X ↔ Y`` symmetry from ``g``'s symmetry;
  taken primitive here so a ``Q_{ab} = Q_{ba}`` rewrite never has to
  open the compatibility expansion. Same shape as
  :class:`~jacopy.calculus.metric.MetricEvalSymmetryDefinition`.
* :class:`NonMetricityCompatibilityDefinition`, opt-in opener
  ``Q(∇, g)(V, X, Y) → V(g(X, Y)) − g(∇_V X, Y) − g(X, ∇_V Y)``.
  The first term ``V(g(X, Y))`` is encoded as
  :class:`~jacopy.algebra.derivation.Act` of ``V`` on
  :class:`~jacopy.calculus.metric.MetricEvalExpr`; the other two
  terms feed :class:`~jacopy.calculus.connection.ConnectionEvalExpr`
  into the metric eval slots. Termination: the right-hand side
  contains no ``Q``, so the rule cannot loop with the closure axioms.
"""

from __future__ import annotations

from typing import Any, Tuple

from jacopy.algebra.derivation import Act, Derivation
from jacopy.calculus.connection import AffineConnection, ConnectionEvalExpr
from jacopy.calculus.metric import MetricEvalExpr, MetricTensor
from jacopy.core.expr import Expr, Neg, Product, Sum
from jacopy.proof.expansion import Definition


# --------------------------------------------------------------------- #
# Evaluation node                                                        #
# --------------------------------------------------------------------- #


class NonMetricityEvalExpr(Expr):
    r"""``Q(∇, g)(V, X, Y)``, non-metricity evaluation node.

    Children are ``(V, X, Y)``; the connection ``∇`` and metric ``g``
    sit in parametric slots so engine rules can dispatch on a specific
    pair (e.g. "this rule fires only on ``Q(∇₁, g₁)``") and two
    distinct connections / metrics in the same proof never cross-fire.
    """

    __slots__ = ("_connection", "_metric", "_V", "_X", "_Y")

    def __init__(
        self,
        connection: AffineConnection,
        metric: MetricTensor,
        V: Expr,
        X: Expr,
        Y: Expr,
    ) -> None:
        if not isinstance(connection, AffineConnection):
            raise TypeError(
                "NonMetricityEvalExpr requires an AffineConnection"
            )
        if not isinstance(metric, MetricTensor):
            raise TypeError(
                "NonMetricityEvalExpr requires a MetricTensor"
            )
        for slot, val in (("V", V), ("X", X), ("Y", Y)):
            if not isinstance(val, Expr):
                raise TypeError(
                    f"NonMetricityEvalExpr {slot} must be an Expr"
                )
        self._connection = connection
        self._metric = metric
        self._V = V
        self._X = X
        self._Y = Y

    @property
    def connection(self) -> AffineConnection:
        return self._connection

    @property
    def metric(self) -> MetricTensor:
        return self._metric

    @property
    def V(self) -> Expr:
        return self._V

    @property
    def X(self) -> Expr:
        return self._X

    @property
    def Y(self) -> Expr:
        return self._Y

    @property
    def children(self) -> Tuple[Expr, ...]:
        return (self._V, self._X, self._Y)

    def _rebuild(
        self, new_children: Tuple[Expr, ...]
    ) -> "NonMetricityEvalExpr":
        if len(new_children) != 3:
            raise ValueError(
                "NonMetricityEvalExpr._rebuild expects three children"
            )
        return NonMetricityEvalExpr(
            self._connection,
            self._metric,
            new_children[0],
            new_children[1],
            new_children[2],
        )

    def _key(self) -> Any:
        return (
            self._connection,
            self._metric,
            self._V,
            self._X,
            self._Y,
        )

    def _repr_inner(self) -> str:
        return (
            f"Q({self._connection._repr_inner()},"
            f"{self._metric._repr_inner()})"
            f"({self._V._repr_inner()},"
            f"{self._X._repr_inner()},"
            f"{self._Y._repr_inner()})"
        )


# --------------------------------------------------------------------- #
# Engine axioms                                                          #
# --------------------------------------------------------------------- #


class NonMetricityVLinearityDefinition(Definition):
    r"""``Q(A + B + …, X, Y) → Q(A,X,Y) + Q(B,X,Y) + …`` and ``Q(−A, X, Y) → −Q(A, X, Y)``.

    Distributes :class:`Sum` and :class:`Neg` in the ``V``-slot.
    Scoped to a specific ``(connection, metric)`` pair so
    non-metricities for two different metrics on the same manifold
    don't cross-fire.
    """

    def __init__(
        self, conn: AffineConnection, metric: MetricTensor
    ) -> None:
        if not isinstance(conn, AffineConnection):
            raise TypeError(
                "NonMetricityVLinearityDefinition requires an AffineConnection"
            )
        if not isinstance(metric, MetricTensor):
            raise TypeError(
                "NonMetricityVLinearityDefinition requires a MetricTensor"
            )
        self._conn = conn
        self._metric = metric
        self.name = (
            f"Q V-linearity [{conn._repr_inner()},"
            f"{metric._repr_inner()}]"
        )

    def matches(self, expr: Expr) -> bool:
        return (
            isinstance(expr, NonMetricityEvalExpr)
            and expr.connection == self._conn
            and expr.metric == self._metric
            and isinstance(expr.V, (Sum, Neg))
        )

    def rewrite(self, expr: Expr) -> Expr:
        X, Y = expr.X, expr.Y
        v_slot = expr.V
        if isinstance(v_slot, Neg):
            return Neg(
                NonMetricityEvalExpr(
                    self._conn, self._metric, v_slot.arg, X, Y
                )
            )
        terms = []
        for c in v_slot.children:
            if isinstance(c, Neg):
                terms.append(
                    Neg(
                        NonMetricityEvalExpr(
                            self._conn, self._metric, c.arg, X, Y
                        )
                    )
                )
            else:
                terms.append(
                    NonMetricityEvalExpr(
                        self._conn, self._metric, c, X, Y
                    )
                )
        return Sum.make(*terms)


class NonMetricityVScalarPullDefinition(Definition):
    r"""``Q(f · V, X, Y) → f · Q(V, X, Y)``, :math:`C^\infty`-linearity in V.

    Fires when the V-slot is a :class:`Product` of two or more factors.
    The leading factors are folded into the scalar prefactor ``f`` and
    the last factor is the underlying vector ``V``. Same convention as
    :class:`~jacopy.calculus.multi_eval_scalar_axioms.MultiEvalScalarPullDefinition`
    and :class:`~jacopy.calculus.pairing_linearity_axioms.PairingScalarPullDefinition`:
    last factor is the geometric object, leading factors are scalars.

    Scoped to a specific ``(connection, metric)`` pair.
    """

    def __init__(
        self, conn: AffineConnection, metric: MetricTensor
    ) -> None:
        if not isinstance(conn, AffineConnection):
            raise TypeError(
                "NonMetricityVScalarPullDefinition requires an AffineConnection"
            )
        if not isinstance(metric, MetricTensor):
            raise TypeError(
                "NonMetricityVScalarPullDefinition requires a MetricTensor"
            )
        self._conn = conn
        self._metric = metric
        self.name = (
            f"Q V-scalar pull [{conn._repr_inner()},"
            f"{metric._repr_inner()}]"
        )

    def matches(self, expr: Expr) -> bool:
        if not isinstance(expr, NonMetricityEvalExpr):
            return False
        if expr.connection != self._conn or expr.metric != self._metric:
            return False
        return (
            isinstance(expr.V, Product)
            and len(expr.V.children) >= 2
        )

    def rewrite(self, expr: Expr) -> Expr:
        prod = expr.V
        assert isinstance(prod, Product)
        *scalars, head = prod.children
        f = Product.make(*scalars)
        inner = NonMetricityEvalExpr(
            self._conn, self._metric, head, expr.X, expr.Y
        )
        return Product(f, inner)


class NonMetricityXYSymmetryDefinition(Definition):
    r"""``Q(V, X, Y) → Q(V, Y, X)``, canonical-order swap on the symmetric pair.

    Fires on a :class:`NonMetricityEvalExpr` whose ``(X, Y)`` are out
    of ``repr``-order. After the rewrite the pair is sorted so the
    rule applies at most once per node, same termination story as
    :class:`~jacopy.calculus.metric.MetricEvalSymmetryDefinition`.

    Mathematically the symmetry is a consequence of ``g``'s symmetry
    through the compatibility expansion ``Q(V, X, Y) = (∇_V g)(X, Y)``
    plus the fact that ``∇_V`` is a derivation on tensors and
    therefore preserves symmetry of its argument. Here we take it
    primitive: the local-component identity ``Q_{ab} = Q_{ba}``
    should not require a detour through the full ``∇g`` Leibniz rule
    when the compat expansion isn't even mechanised yet.

    Scoped to a specific ``(connection, metric)`` pair.
    """

    def __init__(
        self, conn: AffineConnection, metric: MetricTensor
    ) -> None:
        if not isinstance(conn, AffineConnection):
            raise TypeError(
                "NonMetricityXYSymmetryDefinition requires an AffineConnection"
            )
        if not isinstance(metric, MetricTensor):
            raise TypeError(
                "NonMetricityXYSymmetryDefinition requires a MetricTensor"
            )
        self._conn = conn
        self._metric = metric
        self.name = (
            f"Q X-Y symmetry [{conn._repr_inner()},"
            f"{metric._repr_inner()}]"
        )

    def matches(self, expr: Expr) -> bool:
        return (
            isinstance(expr, NonMetricityEvalExpr)
            and expr.connection == self._conn
            and expr.metric == self._metric
            and repr(expr.X) > repr(expr.Y)
        )

    def rewrite(self, expr: Expr) -> Expr:
        return NonMetricityEvalExpr(
            self._conn, self._metric, expr.V, expr.Y, expr.X
        )


class NonMetricityCompatibilityDefinition(Definition):
    r"""``Q(∇, g)(V, X, Y) → V(g(X, Y)) − g(∇_V X, Y) − g(X, ∇_V Y)``.

    Opens the compatibility-style definition of the non-metricity
    tensor: ``Q = (∇_V g)(X, Y)`` expanded by the Leibniz rule for
    ``∇`` on the ``(0, 2)``-tensor ``g``. The first term
    ``V(g(X, Y))`` is the derivation ``V`` acting on the scalar
    ``g(X, Y)``, represented as :class:`Act` of ``V`` (which must be a
    :class:`Derivation`) on
    :class:`~jacopy.calculus.metric.MetricEvalExpr`. The other two
    terms compose ``∇`` with the metric eval through
    :class:`~jacopy.calculus.connection.ConnectionEvalExpr`.

    **Opt-in.** Unlike the closure axioms (V-linearity, X↔Y symmetry)
    this rule unfolds the definition, so a notebook keeping ``Q``
    opaque should not include it in the engine bundle. Termination:
    the RHS contains no :class:`NonMetricityEvalExpr`, so the rule
    cannot loop with the closure axioms.

    Match guard: the ``V``-slot must be a :class:`Derivation` so the
    :class:`Act` node on the first term is well-typed. Sums / scalar
    multiples in the ``V``-slot should be processed by
    :class:`NonMetricityVLinearityDefinition` /
    :class:`NonMetricityVScalarPullDefinition` first; the bottom-up
    engine walk does this naturally.

    Scoped to a specific ``(connection, metric)`` pair.
    """

    def __init__(
        self, conn: AffineConnection, metric: MetricTensor
    ) -> None:
        if not isinstance(conn, AffineConnection):
            raise TypeError(
                "NonMetricityCompatibilityDefinition requires an AffineConnection"
            )
        if not isinstance(metric, MetricTensor):
            raise TypeError(
                "NonMetricityCompatibilityDefinition requires a MetricTensor"
            )
        self._conn = conn
        self._metric = metric
        self.name = (
            f"Q compatibility [{conn._repr_inner()},"
            f"{metric._repr_inner()}]"
        )

    def matches(self, expr: Expr) -> bool:
        if not isinstance(expr, NonMetricityEvalExpr):
            return False
        if expr.connection != self._conn or expr.metric != self._metric:
            return False
        return isinstance(expr.V, Derivation)

    def rewrite(self, expr: Expr) -> Expr:
        V, X, Y = expr.V, expr.X, expr.Y
        first = Act(V, MetricEvalExpr(self._metric, X, Y))
        second = MetricEvalExpr(
            self._metric, ConnectionEvalExpr(self._conn, V, X), Y
        )
        third = MetricEvalExpr(
            self._metric, X, ConnectionEvalExpr(self._conn, V, Y)
        )
        return Sum.make(first, Neg(second), Neg(third))
