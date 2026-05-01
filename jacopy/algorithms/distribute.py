"""
Distributivity: turn ``A * (B + C) * D`` into ``A*B*D + A*C*D``.

Products here are non-commutative by default, so the algorithm must
preserve factor order on both sides of the sum. This is the crucial
difference from naive distributivity: expanding ``X * (Y + Z) * W``
must produce ``(X*Y*W) + (X*Z*W)``, not reorder factors.

Multiple sum factors are expanded left-to-right, expanding the first
sum factor yields a Sum whose terms still contain the remaining sum
factors, and the recursive call on each term takes care of those. This
is the ordinary multilinear expansion: ``(a+b)*(c+d)`` becomes
``a*c + a*d + b*c + b*d`` with factor order preserved everywhere.

:class:`Neg` is treated as a sign, not a distributive barrier:
``A * (-B)`` passes through the product unchanged, and ``-1 * (B + C)``
composes the way you'd expect once :func:`flatten` has flattened the
outer product.
"""

from __future__ import annotations

from typing import List, Tuple

from jacopy.algorithms.base import Algorithm
from jacopy.core.expr import Expr, One, Product, Sum


def distribute(expr: Expr) -> Expr:
    """Fully distribute products over sums, bottom-up.

    Idempotent on results: once every Product contains no Sum factor,
    :func:`distribute` returns the input.
    """
    if expr.is_atom:
        return expr

    new_children = tuple(distribute(c) for c in expr.children)

    if isinstance(expr, Product):
        return _distribute_product(new_children)
    return expr._rebuild(new_children)


def _distribute_product(factors: Tuple[Expr, ...]) -> Expr:
    """Expand the first Sum factor; recurse on each resulting term."""
    for i, f in enumerate(factors):
        if isinstance(f, Sum):
            left = factors[:i]
            right = factors[i + 1:]
            terms: List[Expr] = []
            for term in f.children:
                new_factors = list(left) + [term] + list(right)
                if len(new_factors) == 1:
                    sub = new_factors[0]
                else:
                    sub = Product(*new_factors)
                # Recurse: `sub` may still contain Sum factors further
                # right, and `term` itself could be a Product that
                # benefits from a second sweep.
                terms.append(distribute(sub))
            if not terms:
                # f was an empty Sum, algebraically zero, so the whole
                # product is zero. Empty sums shouldn't normally reach
                # here (smart-ctor collapses them), but handle defensively.
                from jacopy.core.expr import Zero
                return Zero
            if len(terms) == 1:
                return terms[0]
            # Use Sum.make so nested Sums produced by the recursive
            # distribute() calls on multi-sum products flatten into a
            # single top-level Sum, ``(a+b)*(c+d)`` should give a
            # flat 4-term sum, not a 2-level tree.
            return Sum.make(*terms)

    if not factors:
        return One
    if len(factors) == 1:
        return factors[0]
    return Product(*factors)


class Distribute(Algorithm):
    """:class:`Algorithm` wrapper around :func:`distribute`."""

    def can_apply(self, expr: Expr) -> bool:
        for node in expr.walk():
            if isinstance(node, Product):
                if any(isinstance(c, Sum) for c in node.children):
                    return True
        return False

    def apply(self, expr: Expr) -> Expr:
        return distribute(expr)
