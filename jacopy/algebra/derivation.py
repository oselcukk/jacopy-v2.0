"""
Graded derivations as first-class Exprs.

A :class:`Derivation` is a named operator that carries a degree and
participates in the expression tree like any other atom. Derivations
compose as :class:`Product` children (operator composition), appear
inside :class:`Commutator` nodes, and apply to an operand via
:class:`Act`.

:class:`Act` is the *uninterpreted* application node ``D(x)``, keeping
it inert lets the algorithms layer decide when to expand. The
graded-Leibniz expansion lives in :mod:`jacopy.algorithms.product_rule`.

Degree is a full :class:`Degree` polynomial, so generic p-form
derivations like ``d`` (degree 1) and ``ι_X`` (degree −1) coexist with
symbolic-degree derivations that only decide their parity at proof
time.
"""

from __future__ import annotations

from typing import Any, Optional, Tuple

from jacopy.core.expr import Atom, Expr, Integer, Neg, Product, Rational
from jacopy.core.properties import Graded, Scalar
from jacopy.core.registry import PropertyRegistry
from jacopy.core.symbolic_degree import Degree, DegreeLike, as_degree


# --------------------------------------------------------------------- #
# Derivation atom                                                        #
# --------------------------------------------------------------------- #


class Derivation(Atom):
    """A named graded derivation operator.

    ``name`` is a display handle; equality is purely structural over
    ``(name, degree)`` so two ``Derivation("d", 1)`` instances compare
    equal. Grading follows the Koszul convention: a derivation of degree
    ``d`` applied to a product satisfies
    ``D(a*b) = D(a)*b + (-1)^{d*|a|} a*D(b)``.
    """

    __slots__ = ("_name", "_degree")

    def __init__(self, name: str, degree: DegreeLike = 0) -> None:
        if not isinstance(name, str):
            raise TypeError("Derivation name must be a str")
        if not name:
            raise ValueError("Derivation name must be non-empty")
        self._name = name
        self._degree = as_degree(degree)

    @property
    def name(self) -> str:
        return self._name

    @property
    def degree(self) -> Degree:
        return self._degree

    def _key(self) -> Any:
        return (self._name, self._degree)

    def _repr_inner(self) -> str:
        return self._name

    def __call__(self, arg: Expr) -> "Act":
        """Shortcut for ``Act(self, arg)``."""
        return Act(self, arg)


# --------------------------------------------------------------------- #
# Application node                                                       #
# --------------------------------------------------------------------- #


class Act(Expr):
    """Application of an operator to an argument: ``op(arg)``.

    This is inert, it does not apply Leibniz at construction. The
    :mod:`jacopy.algorithms.product_rule` pass is the one that rewrites
    ``Act(D, Product(...))`` into the expanded Leibniz sum.
    """

    __slots__ = ("_op", "_arg")

    def __init__(self, op: Expr, arg: Expr) -> None:
        if not isinstance(op, Expr):
            raise TypeError("Act operator must be an Expr")
        if not isinstance(arg, Expr):
            raise TypeError("Act argument must be an Expr")
        self._op = op
        self._arg = arg

    @property
    def op(self) -> Expr:
        return self._op

    @property
    def arg(self) -> Expr:
        return self._arg

    @property
    def children(self) -> Tuple[Expr, ...]:
        return (self._op, self._arg)

    def _key(self) -> Any:
        return (self._op, self._arg)

    def _repr_inner(self) -> str:
        return f"{self._op._repr_inner()}({self._arg._repr_inner()})"


# --------------------------------------------------------------------- #
# Degree lookup                                                          #
# --------------------------------------------------------------------- #


def degree_of(
    expr: Expr, registry: Optional[PropertyRegistry] = None
) -> Degree:
    """Return the degree of ``expr``, or raise if undetermined.

    Resolution order:

    * :class:`Derivation`, its own ``degree``.
    * Numeric literal (:class:`Integer`, :class:`Rational`), degree 0.
    * :class:`Product`, sum of child degrees. This covers two uses:
      a graded tensor product ``|a*b| = |a|+|b|`` and operator
      composition ``|D1∘D2| = |D1|+|D2|`` (the two are represented by
      the same ``Product`` node).
    * :class:`Neg`, same as its argument; negation is a scalar sign.
    * :class:`Act`, ``|D(x)| = |D| + |x|``.
    * :class:`jacopy.brackets.base.BracketApply`,
      ``|[a, b]| = |a| + |b| + bracket.degree``. Resolved via a late
      import so the algebra layer does not depend on the brackets
      layer at import time.
    * Registry :class:`Scalar`, degree 0.
    * Registry :class:`Graded`, its own degree.
    * Otherwise, :class:`ValueError`.

    :class:`Sum` is intentionally *not* walked: a Sum has a degree only
    when every term's degree agrees, and that policy is cleaner kept in
    the caller that needs it.
    """
    if isinstance(expr, Derivation):
        return expr.degree
    if isinstance(expr, (Integer, Rational)):
        return Degree.const(0)
    if isinstance(expr, Product):
        total = Degree.const(0)
        for c in expr.children:
            total = total + degree_of(c, registry)
        return total
    # Wedge: degree law ``|α ∧ β| = |α| + |β|``. Late import to avoid
    # forcing the wedge module on every algebra import; the algebra
    # layer otherwise has no wedge dependency.
    from jacopy.core.wedge import Wedge  # noqa: WPS433
    if isinstance(expr, Wedge):
        total = Degree.const(0)
        for c in expr.children:
            total = total + degree_of(c, registry)
        return total
    if isinstance(expr, Neg):
        return degree_of(expr.arg, registry)
    if isinstance(expr, Act):
        return degree_of(expr.op, registry) + degree_of(
            expr.arg, registry
        )
    # Brackets are a higher layer; resolve via late import to avoid
    # an algebra → brackets cycle.
    from jacopy.brackets.base import BracketApply  # noqa: WPS433
    if isinstance(expr, BracketApply):
        return (
            degree_of(expr.a, registry)
            + degree_of(expr.b, registry)
            + expr.bracket.degree
        )
    # The canonical pairing ⟨α, X⟩ produces a scalar regardless of its
    # inputs. Late import keeps the algebra → calculus direction clean.
    from jacopy.calculus.pairing import Pairing  # noqa: WPS433
    if isinstance(expr, Pairing):
        return Degree.const(0)
    # Multilinear evaluation ω(Y_1, …, Y_p) likewise produces a scalar.
    # Late import: ``MultiEval`` itself late-imports ``degree_of`` for
    # arity validation, so resolving eagerly at the top of this module
    # would cycle.
    from jacopy.core.multi_eval import MultiEval  # noqa: WPS433
    if isinstance(expr, MultiEval):
        return Degree.const(0)
    if registry is not None:
        if registry.has(expr, Scalar):
            return Degree.const(0)
        graded = registry.get(expr, Graded)
        if graded is not None:
            return graded.degree
    raise ValueError(
        f"Degree of {expr!r} is not determined; register it as Scalar "
        "or Graded, or wrap it in a Derivation"
    )


# --------------------------------------------------------------------- #
# Composition                                                            #
# --------------------------------------------------------------------- #


def compose(*ops: Expr) -> Expr:
    """Compose operators left-to-right: ``compose(D1, D2)(x) = D1(D2(x))``.

    Returns a :class:`Product` node whose children are the operators in
    the order given, composition and element-wise product share the
    same non-commutative :class:`Product` representation. Single-operator
    compositions collapse to the operator itself; a zero-argument call
    is rejected because the identity operator has no canonical
    representation here.

    The :mod:`jacopy.algorithms.product_rule` pass is the one that
    unfolds ``Act(compose(D1, D2), x)`` into nested ``Act`` nodes and
    then applies Leibniz to each layer.
    """
    if not ops:
        raise ValueError("compose() requires at least one operator")
    for op in ops:
        if not isinstance(op, Expr):
            raise TypeError("compose() arguments must be Expr")
    if len(ops) == 1:
        return ops[0]
    return Product(*ops)
