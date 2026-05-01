"""
Sum-level like-term collection.

Merges structurally-equal summands into a single term with the
combined coefficient: ``x + x → 2*x``, ``2*x + 3*x → 5*x``,
``x - x → 0``. This is the narrowest slice of what
:func:`jacopy.algorithms.canonicalize.canonicalize` does, the rest
(numeric folding in Products, Power rules, sorting) lives elsewhere.

The shared helpers ``_coeff_and_core`` and ``_combine_coeff`` are
reused from :mod:`canonicalize`: duplicating them would mean
maintaining two parallel implementations of a subtle sign/coefficient
split.
"""

from __future__ import annotations

from fractions import Fraction
from typing import Dict, List

from jacopy.algorithms.base import Algorithm
from jacopy.algorithms.canonicalize import (
    _coeff_and_core,
    _combine_coeff,
    _from_fraction,
)
from jacopy.core.expr import Expr, One, Sum, Zero


def collect_terms(expr: Expr) -> Expr:
    """Collect structurally-equal terms in every :class:`Sum` subtree.

    Traverses bottom-up. Non-Sum nodes are reassembled with collected
    children. Idempotent.
    """
    if expr.is_atom:
        return expr

    new_children = tuple(collect_terms(c) for c in expr.children)

    if isinstance(expr, Sum):
        return _collect_sum(new_children)
    return expr._rebuild(new_children)


def _collect_sum(children) -> Expr:
    # Flatten one level so like terms across nested Sums still merge.
    flat: List[Expr] = []
    for c in children:
        if isinstance(c, Sum):
            flat.extend(c.children)
        else:
            flat.append(c)

    groups: Dict[Expr, Fraction] = {}
    order: List[Expr] = []
    for t in flat:
        coeff, core = _coeff_and_core(t)
        if core in groups:
            groups[core] = groups[core] + coeff
        else:
            groups[core] = coeff
            order.append(core)

    const = groups.pop(One, Fraction(0))

    terms: List[Expr] = []
    for core in order:
        if core == One:
            continue
        coeff = groups[core]
        if coeff == 0:
            continue
        terms.append(_combine_coeff(coeff, core))
    if const != 0:
        terms.append(_from_fraction(const))

    if not terms:
        return Zero
    if len(terms) == 1:
        return terms[0]
    return Sum(*terms)


class CollectTerms(Algorithm):
    """:class:`Algorithm` wrapper around :func:`collect_terms`."""

    def can_apply(self, expr: Expr) -> bool:
        # A Sum is a candidate whenever it has at least two children;
        # the pass is cheap enough that we don't pre-scan for actual
        # duplicates.
        for node in expr.walk():
            if isinstance(node, Sum) and len(node.children) >= 2:
                return True
        return False

    def apply(self, expr: Expr) -> Expr:
        return collect_terms(expr)
