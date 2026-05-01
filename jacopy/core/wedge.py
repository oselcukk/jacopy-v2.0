r"""
Wedge product Expr node, the graded-antisymmetric counterpart of
:class:`~jacopy.core.expr.Product`.

In differential geometry the wedge ``α ∧ β`` is distinct from a plain
product even when both factors are forms: it carries graded antisymmetry
``α ∧ β = (−1)^{|α|·|β|} β ∧ α`` and a degree law
``|α ∧ β| = |α| + |β|``. In this codebase :class:`~jacopy.core.expr.Product`
is reserved for the non-commutative scalar/operator product (used for
both ``f · g`` and ``D_1 ∘ D_2``), so we need a separate node to keep
the two semantics from leaking into one another.

Earlier stages (Faz 17.E.4) deferred the dedicated Wedge node on the
basis that wedges that surface inside an :class:`IndexedSum` body
already behave like a structural product for the rules that fire
there. That deferral is local, the moment Cartan I/II structure
equations expose ``α ∧ β`` to a :class:`~jacopy.core.multi_eval.MultiEval`
and ask "what is its action on two vector fields?", a real Wedge node
is needed: the alternating expansion ``(α ∧ β)(U, V) = α(U)·β(V) −
α(V)·β(U)`` cannot be derived from Product semantics without further
input. So Faz 17.F.1.5 introduces this node.

The Expr node itself is purely structural, the action axiom
``MultiEval(Wedge(α₁, …, α_p), X_1, …, X_p) → det[α_i(X_j)]`` lives in
:mod:`jacopy.calculus.wedge_axioms`. Smart-constructor handling here
covers only the trivial identities a Product would (associativity,
``0`` absorption, ``1`` removal); graded-antisymmetric reordering and
``α ∧ α = 0`` for degree-1 ``α`` need degree information and live in
the algorithms layer.
"""

from __future__ import annotations

from typing import Any, Tuple

from jacopy.core.expr import Expr, One, Zero, _is_integer_value


class Wedge(Expr):
    r"""n-ary wedge product ``α_1 ∧ α_2 ∧ … ∧ α_p``.

    Children are stored in the order given. No graded reordering is
    applied at construction, that requires degree information and
    lives in :mod:`jacopy.algorithms.sort_product` (or a dedicated
    wedge-sort pass added when needed).

    The smart constructor :meth:`make` flattens nested wedges
    (associativity), absorbs any ``0`` factor, drops integer-``1``
    factors (the wedge unit), and collapses to a single factor when
    only one remains. Degree-aware identities such as ``α ∧ α = 0``
    for a degree-1 ``α`` are not applied here; a graded-canonicalize
    pass is responsible for those.
    """

    __slots__ = ("_children",)

    def __init__(self, *children: Expr) -> None:
        if len(children) < 2:
            raise ValueError(
                "Wedge requires at least two factors; use the factor "
                "directly for a single-element 'wedge'"
            )
        for c in children:
            if not isinstance(c, Expr):
                raise TypeError("Wedge children must be Expr")
        self._children = tuple(children)

    @property
    def children(self) -> Tuple[Expr, ...]:
        return self._children

    @classmethod
    def make(cls, *args: Expr) -> Expr:
        """Smart constructor.

        * Flattens nested wedges (associativity).
        * Returns :data:`~jacopy.core.expr.Zero` if any factor is the
          integer ``0``.
        * Drops integer-``1`` factors (the wedge unit).
        * Collapses an empty wedge to :data:`~jacopy.core.expr.One`
          and a one-factor wedge to that factor.
        """
        flat: list[Expr] = []
        for a in args:
            if isinstance(a, Wedge):
                flat.extend(a._children)
            else:
                flat.append(a)
        for x in flat:
            if _is_integer_value(x, 0):
                return Zero
        flat = [x for x in flat if not _is_integer_value(x, 1)]
        if not flat:
            return One
        if len(flat) == 1:
            return flat[0]
        return cls(*flat)

    def _key(self) -> Any:
        return self._children

    def _repr_inner(self) -> str:
        return "(" + " ∧ ".join(c._repr_inner() for c in self._children) + ")"
