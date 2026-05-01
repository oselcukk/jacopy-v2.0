"""
Associativity: flatten nested sums and products.

``Sum`` and ``Product`` are n-ary but expressions built via direct
constructors (especially by rewrite rules) can carry nested shapes like
``Sum(Sum(x, y), z)``. The smart constructors ``Sum.make`` and
``Product.make`` already flatten, but rewrite passes that build trees
positionally, :func:`replace_at`, hand-built patterns, substitutions
that preserve rhs structure, leave nesting in place. :func:`flatten`
is the explicit pass that normalizes that nesting without touching
anything else.

Bottom-up; idempotent. Singleton sums collapse to their element;
empty sums become :data:`Zero`, empty products :data:`One`. Neg and
Power aren't associative in the relevant sense and are traversed
transparently.
"""

from __future__ import annotations

from typing import List

from jacopy.algorithms.base import Algorithm
from jacopy.core.expr import Expr, One, Product, Sum, Zero


def flatten(expr: Expr) -> Expr:
    """Return ``expr`` with nested Sums and Products flattened.

    Idempotent and structure-preserving otherwise: atoms pass through,
    other compound nodes are rebuilt with flattened children.
    """
    if expr.is_atom:
        return expr

    new_children = tuple(flatten(c) for c in expr.children)

    if isinstance(expr, Sum):
        return _flatten_sum(new_children)
    if isinstance(expr, Product):
        return _flatten_product(new_children)
    return expr._rebuild(new_children)


def _flatten_sum(children) -> Expr:
    flat: List[Expr] = []
    for c in children:
        if isinstance(c, Sum):
            flat.extend(c.children)
        else:
            flat.append(c)
    if not flat:
        return Zero
    if len(flat) == 1:
        return flat[0]
    return Sum(*flat)


def _flatten_product(children) -> Expr:
    flat: List[Expr] = []
    for c in children:
        if isinstance(c, Product):
            flat.extend(c.children)
        else:
            flat.append(c)
    if not flat:
        return One
    if len(flat) == 1:
        return flat[0]
    return Product(*flat)


class Flatten(Algorithm):
    """:class:`Algorithm` wrapper around :func:`flatten` for proof unroll."""

    def can_apply(self, expr: Expr) -> bool:
        for node in expr.walk():
            if isinstance(node, (Sum, Product)):
                for c in node.children:
                    if type(c) is type(node):
                        return True
        return False

    def apply(self, expr: Expr) -> Expr:
        return flatten(expr)
