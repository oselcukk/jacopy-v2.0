r"""
Metric tensor and its evaluation node, Faz 17.B.

A *metric tensor* on a smooth manifold is a smooth, symmetric,
non-degenerate ``(0, 2)``-tensor field ``g``. Faz 17 needs three
distinct surfaces:

* an *identifier*, :class:`MetricTensor`, an opaque named atom that
  carries "this is the metric ``g``" through proofs as a parametric
  slot of the non-metricity tensor and (later) of compatibility-style
  expansions;
* an *evaluation node*, :class:`MetricEvalExpr`, the rank-2
  evaluation ``g(X, Y)`` whose children are ``(X, Y)`` so the engine
  walks freely;
* a *symmetry axiom*, :class:`MetricEvalSymmetryDefinition`, which
  canonicalises ``g(X, Y) → g(Y, X)`` whenever the args are out of
  ``repr``-order, mirroring
  :class:`~jacopy.calculus.antisym_axioms.RegistryAntiSymCanonicalDefinition`'s
  canonicalise-only-out-of-order strategy.
* a *bilinearity axiom*, :class:`MetricEvalLinearityDefinition`,
  distributing :class:`Sum` and :class:`Neg` in either slot.
* a *scalar-pull axiom*, :class:`MetricEvalScalarPullDefinition`,
  pulling a scalar factor out of either slot. Same shape as
  :class:`~jacopy.calculus.pairing_linearity_axioms.PairingScalarPullDefinition`.

The Cartan-form mechanisations don't exercise ``g`` directly, the
non-metricity 1-form proofs route through ``Q``'s primitive
``V``-linearity and the Cartan structure equations carry no ``g``
, but the C∞-bilinearity machinery lands now so ``MetricEvalExpr``
is fully calculational the moment a future proof (``∇``
compatibility, Levi-Civita, …) reaches for it.
"""

from __future__ import annotations

from typing import Any, Tuple

from jacopy.core.expr import Expr, Atom, Neg, Product, Sum
from jacopy.proof.expansion import Definition


class MetricTensor(Atom):
    """An opaque metric tensor identifier ``g``.

    Carries just a display name. Equality is structural over the name
    so any two constructions with the same name agree. The atom does
    not appear inside :class:`~jacopy.algebra.derivation.Act` nodes,
    its mathematical evaluation ``g(X, Y)`` lives in
    :class:`MetricEvalExpr` so the engine can rewrite the eval shape
    without disturbing identity comparisons that route through the
    atom.
    """

    __slots__ = ("_name",)

    def __init__(self, name: str) -> None:
        if not isinstance(name, str):
            raise TypeError("MetricTensor name must be a str")
        if not name:
            raise ValueError("MetricTensor name must be non-empty")
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    def _key(self) -> Any:
        return self._name

    def _repr_inner(self) -> str:
        return self._name


def metric(name: str = "g") -> MetricTensor:
    """Functional constructor for :class:`MetricTensor`."""
    return MetricTensor(name)


# --------------------------------------------------------------------- #
# Evaluation node                                                        #
# --------------------------------------------------------------------- #


class MetricEvalExpr(Expr):
    r"""``g(X, Y)``, metric evaluated on two vector fields.

    Children are ``(X, Y)``; the metric sits in a parametric slot so
    engine rules can dispatch on a specific metric (e.g. "this rule
    fires only on ``g₁``") and two metrics never cross-fire. Same
    convention as :class:`~jacopy.calculus.torsion_curvature.Torsion`
    for the connection.
    """

    __slots__ = ("_metric", "_X", "_Y")

    def __init__(self, metric: MetricTensor, X: Expr, Y: Expr) -> None:
        if not isinstance(metric, MetricTensor):
            raise TypeError("MetricEvalExpr requires a MetricTensor")
        if not isinstance(X, Expr):
            raise TypeError("MetricEvalExpr X must be an Expr")
        if not isinstance(Y, Expr):
            raise TypeError("MetricEvalExpr Y must be an Expr")
        self._metric = metric
        self._X = X
        self._Y = Y

    @property
    def metric(self) -> MetricTensor:
        return self._metric

    @property
    def X(self) -> Expr:
        return self._X

    @property
    def Y(self) -> Expr:
        return self._Y

    @property
    def children(self) -> Tuple[Expr, ...]:
        return (self._X, self._Y)

    def _rebuild(self, new_children: Tuple[Expr, ...]) -> "MetricEvalExpr":
        if len(new_children) != 2:
            raise ValueError(
                "MetricEvalExpr._rebuild expects two children"
            )
        return MetricEvalExpr(
            self._metric, new_children[0], new_children[1]
        )

    def _key(self) -> Any:
        return (self._metric, self._X, self._Y)

    def _repr_inner(self) -> str:
        return (
            f"{self._metric._repr_inner()}"
            f"({self._X._repr_inner()},{self._Y._repr_inner()})"
        )


# --------------------------------------------------------------------- #
# Symmetry axiom                                                         #
# --------------------------------------------------------------------- #


class MetricEvalSymmetryDefinition(Definition):
    r"""``g(X, Y) → g(Y, X)``, canonical-order swap on a symmetric metric.

    Fires on a :class:`MetricEvalExpr` whose args are out of
    ``repr``-order. After the rewrite the args are sorted, so the
    rule applies at most once per node. Mirrors
    :class:`~jacopy.calculus.antisym_axioms.RegistryAntiSymCanonicalDefinition`
    in shape but without the :class:`Neg` wrapper, ``g`` is symmetric,
    not antisymmetric.

    Scoped to a specific :class:`MetricTensor` so two distinct metrics
    on the same manifold never cross-fire.
    """

    def __init__(self, metric: MetricTensor) -> None:
        if not isinstance(metric, MetricTensor):
            raise TypeError(
                "MetricEvalSymmetryDefinition requires a MetricTensor"
            )
        self._metric = metric
        self.name = f"g symmetry [{metric._repr_inner()}]"

    def matches(self, expr: Expr) -> bool:
        return (
            isinstance(expr, MetricEvalExpr)
            and expr.metric == self._metric
            and repr(expr.X) > repr(expr.Y)
        )

    def rewrite(self, expr: Expr) -> Expr:
        return MetricEvalExpr(self._metric, expr.Y, expr.X)


# --------------------------------------------------------------------- #
# C∞-bilinearity axioms                                                  #
# --------------------------------------------------------------------- #


class MetricEvalLinearityDefinition(Definition):
    r"""``g(A + B + …, Y) → g(A,Y) + g(B,Y) + …`` and dual; ``g(−A, Y) → −g(A, Y)``.

    Distributes :class:`Sum` and :class:`Neg` in either slot of a
    :class:`MetricEvalExpr`. Shape mirrors
    :class:`~jacopy.calculus.non_metricity.NonMetricityVLinearityDefinition`,
    but inspects both slots (left first; the engine's bottom-up walk
    catches the right slot in a follow-up pass).

    Scoped to a specific :class:`MetricTensor` so two distinct metrics
    never cross-fire.
    """

    def __init__(self, metric: MetricTensor) -> None:
        if not isinstance(metric, MetricTensor):
            raise TypeError(
                "MetricEvalLinearityDefinition requires a MetricTensor"
            )
        self._metric = metric
        self.name = f"g linearity [{metric._repr_inner()}]"

    def matches(self, expr: Expr) -> bool:
        if not isinstance(expr, MetricEvalExpr):
            return False
        if expr.metric != self._metric:
            return False
        return isinstance(expr.X, (Sum, Neg)) or isinstance(
            expr.Y, (Sum, Neg)
        )

    def rewrite(self, expr: Expr) -> Expr:
        if isinstance(expr.X, Neg):
            return Neg(MetricEvalExpr(self._metric, expr.X.arg, expr.Y))
        if isinstance(expr.X, Sum):
            return Sum.make(
                *(self._lift_left(c, expr.Y) for c in expr.X.children)
            )
        if isinstance(expr.Y, Neg):
            return Neg(MetricEvalExpr(self._metric, expr.X, expr.Y.arg))
        # expr.Y is Sum
        assert isinstance(expr.Y, Sum)
        return Sum.make(
            *(self._lift_right(expr.X, c) for c in expr.Y.children)
        )

    def _lift_left(self, child: Expr, Y: Expr) -> Expr:
        if isinstance(child, Neg):
            return Neg(MetricEvalExpr(self._metric, child.arg, Y))
        return MetricEvalExpr(self._metric, child, Y)

    def _lift_right(self, X: Expr, child: Expr) -> Expr:
        if isinstance(child, Neg):
            return Neg(MetricEvalExpr(self._metric, X, child.arg))
        return MetricEvalExpr(self._metric, X, child)


class MetricEvalScalarPullDefinition(Definition):
    r"""``g(f·X, Y) → f·g(X, Y)`` and ``g(X, f·Y) → f·g(X, Y)``.

    Fires whenever either slot is a :class:`Product` of two or more
    factors. Convention matches
    :class:`~jacopy.calculus.pairing_linearity_axioms.PairingScalarPullDefinition`:
    leading factors are scalars, the last factor is the underlying
    vector field. When *both* slots are products the rule pulls from
    the left slot first; the bottom-up engine walk catches the right.

    Scoped to a specific :class:`MetricTensor`.
    """

    def __init__(self, metric: MetricTensor) -> None:
        if not isinstance(metric, MetricTensor):
            raise TypeError(
                "MetricEvalScalarPullDefinition requires a MetricTensor"
            )
        self._metric = metric
        self.name = f"g scalar pull [{metric._repr_inner()}]"

    def matches(self, expr: Expr) -> bool:
        if not isinstance(expr, MetricEvalExpr):
            return False
        if expr.metric != self._metric:
            return False
        return self._slot_to_pull(expr) is not None

    def rewrite(self, expr: Expr) -> Expr:
        slot = self._slot_to_pull(expr)
        assert slot is not None
        if slot == "X":
            prod = expr.X
            assert isinstance(prod, Product)
            *scalars, head = prod.children
            f = Product.make(*scalars)
            return Product(f, MetricEvalExpr(self._metric, head, expr.Y))
        prod = expr.Y
        assert isinstance(prod, Product)
        *scalars, head = prod.children
        f = Product.make(*scalars)
        return Product(f, MetricEvalExpr(self._metric, expr.X, head))

    @staticmethod
    def _slot_to_pull(expr: "MetricEvalExpr") -> str | None:
        if isinstance(expr.X, Product) and len(expr.X.children) >= 2:
            return "X"
        if isinstance(expr.Y, Product) and len(expr.Y.children) >= 2:
            return "Y"
        return None
