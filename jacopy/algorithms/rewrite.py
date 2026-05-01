"""
Rule-based rewriting.

A :class:`Rule` pairs a pattern (``lhs``) with a rewrite (``rhs``).
Applying a rule finds bindings that match the lhs, optionally checks
a guard predicate, and substitutes into the rhs.

Rules compose via small strategies:

* :func:`apply_once_at_root`, try each rule at the root exactly once.
* :func:`apply_bottomup`, rewrite children first, then the node.
* :func:`apply_topdown`, rewrite the node first, then the new children.
* :func:`normalize`, iterate a strategy until a fixed point.

Strategies here are deliberately simple. The "no reordering" stance from
:mod:`jacopy.core.wildcards` is preserved: if a rule wants commutative
matching, the caller is expected to canonicalize first.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence, Tuple

from jacopy.core.expr import Expr
from jacopy.core.registry import PropertyRegistry
from jacopy.core.wildcards import Bindings, match, substitute


Guard = Callable[[Bindings, Optional[PropertyRegistry]], bool]


# --------------------------------------------------------------------- #
# Internal rebuilding                                                    #
# --------------------------------------------------------------------- #


def _rebuild(old: Expr, new_children: Tuple[Expr, ...]) -> Expr:
    """Rebuild ``old`` with ``new_children``, preferring smart constructors.

    When the underlying class exposes a ``make`` classmethod (Sum,
    Product), use it so a child rewrite that produces, say, a nested
    Product collapses the way it would have if the user had built the
    expression from scratch with ``*``. For classes without ``make``
    (Neg, Power), fall back to the direct constructor.
    """
    make = getattr(type(old), "make", None)
    if make is not None:
        return make(*new_children)
    return old._rebuild(new_children)


# --------------------------------------------------------------------- #
# Rule                                                                   #
# --------------------------------------------------------------------- #


@dataclass(frozen=True)
class Rule:
    """A single rewrite rule: if ``lhs`` matches, produce ``rhs``.

    ``guard`` is an optional predicate on bindings; it lets a rule fire
    only when a side condition holds ("only if ?n is an odd integer",
    "only if ?f is scalar"). It receives the bindings dict and the
    registry, and returning ``False`` rejects the match.
    """

    lhs: Expr
    rhs: Expr
    name: str = ""
    guard: Optional[Guard] = field(default=None, compare=False)

    def try_at(
        self, expr: Expr, registry: Optional[PropertyRegistry] = None
    ) -> Optional[Expr]:
        """Attempt to rewrite ``expr`` at its root. ``None`` on no match."""
        bindings = match(self.lhs, expr, registry)
        if bindings is None:
            return None
        if self.guard is not None and not self.guard(bindings, registry):
            return None
        return substitute(self.rhs, bindings)


# --------------------------------------------------------------------- #
# Strategies                                                             #
# --------------------------------------------------------------------- #


def apply_once_at_root(
    rules: Sequence[Rule],
    expr: Expr,
    registry: Optional[PropertyRegistry] = None,
) -> Optional[Expr]:
    """Try each rule in order at the root. Return first rewrite, else ``None``."""
    for rule in rules:
        result = rule.try_at(expr, registry)
        if result is not None:
            return result
    return None


def apply_bottomup(
    rules: Sequence[Rule],
    expr: Expr,
    registry: Optional[PropertyRegistry] = None,
) -> Expr:
    """Rewrite children first, then the node itself, once.

    A single bottom-up pass. No fixed-point iteration, use
    :func:`normalize` when a rule set may cascade.
    """
    if not expr.is_atom:
        new_children = tuple(
            apply_bottomup(rules, c, registry) for c in expr.children
        )
        if any(a is not b for a, b in zip(new_children, expr.children)):
            expr = _rebuild(expr, new_children)
    rewritten = apply_once_at_root(rules, expr, registry)
    return rewritten if rewritten is not None else expr


def apply_topdown(
    rules: Sequence[Rule],
    expr: Expr,
    registry: Optional[PropertyRegistry] = None,
) -> Expr:
    """Rewrite the node first, then recurse into the (possibly new) children."""
    rewritten = apply_once_at_root(rules, expr, registry)
    if rewritten is not None:
        expr = rewritten
    if expr.is_atom:
        return expr
    new_children = tuple(
        apply_topdown(rules, c, registry) for c in expr.children
    )
    if any(a is not b for a, b in zip(new_children, expr.children)):
        expr = _rebuild(expr, new_children)
    return expr


# --------------------------------------------------------------------- #
# Fixed-point iteration                                                  #
# --------------------------------------------------------------------- #


def normalize(
    rules: Sequence[Rule],
    expr: Expr,
    registry: Optional[PropertyRegistry] = None,
    *,
    max_iter: int = 100,
    strategy: str = "bottomup",
) -> Expr:
    """Apply ``rules`` until a fixed point is reached or ``max_iter`` exceeds.

    ``strategy`` selects the per-iteration pass: ``"bottomup"`` or
    ``"topdown"``. Non-converging rule sets raise :class:`RuntimeError`
    rather than looping silently, catching divergence early is more
    useful than hanging a proof.
    """
    if strategy == "bottomup":
        step = apply_bottomup
    elif strategy == "topdown":
        step = apply_topdown
    else:
        raise ValueError(f"Unknown strategy: {strategy!r}")

    for _ in range(max_iter):
        new = step(rules, expr, registry)
        if new == expr:
            return new
        expr = new
    raise RuntimeError(
        f"normalize did not converge after {max_iter} iterations"
    )
