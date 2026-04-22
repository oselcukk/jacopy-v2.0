"""
Graded-commutative sorting of Product factors.

A :class:`Product` in the core layer is non-commutative — factor order is
preserved verbatim. Many proofs need the *graded-commutative* reading
instead: adjacent factors may be swapped, each swap producing a Koszul
sign ``(-1)^{|a||b|}``. :func:`sort_product` performs that re-ordering
and returns the accumulated sign as a :class:`Degree` polynomial so the
caller can inspect it (even when the sign is symbolic like
``(-1)^{|α||β|}``).

Every factor must be registered as :class:`Scalar` or :class:`Graded`.
Scalars sort to the front and contribute no sign — they commute with
everything in the Koszul rule. Graded factors sort by a stable
structural key and contribute ``|a||b|`` per swap. Unclassified factors
raise :class:`ValueError`; the fix is to declare their grading, not to
silently paper over the missing information.

The sign is returned as a *polynomial in the degree variables*. Call
:meth:`Degree.parity` to collapse to ``0``/``1`` when the parity is
decidable; use :func:`apply_sign` to fold a decidable sign into an
:class:`Expr` directly. When the parity is symbolic — as in
Schouten/Courant Jacobi proofs — the Degree polynomial is the right
thing to keep.
"""

from __future__ import annotations

from typing import List, Tuple

from gradalg.core.expr import Expr, Integer, Neg, Product, Rational
from gradalg.core.properties import (
    AntiCommuting,
    Graded,
    GradedCommutative,
    NonCommuting,
    Scalar,
)
from gradalg.core.registry import PropertyRegistry
from gradalg.core.symbolic_degree import Degree


# --------------------------------------------------------------------- #
# Classification helpers                                                 #
# --------------------------------------------------------------------- #


def _is_scalar(factor: Expr, registry: PropertyRegistry) -> bool:
    # Numeric literals are always scalars — they commute with
    # everything and carry no grading. Declaring Scalar on every
    # Integer(k) would be noise.
    if isinstance(factor, (Integer, Rational)):
        return True
    return registry.has(factor, Scalar)


def _degree_of(factor: Expr, registry: PropertyRegistry) -> Degree:
    """Return the grading of ``factor``, or raise if unclassified.

    Scalars have degree ``0``; :class:`Graded` factors carry their own
    Degree. Anything else is an error at this layer.
    """
    if _is_scalar(factor, registry):
        return Degree.const(0)
    prop = registry.get(factor, Graded)
    if prop is None:
        raise ValueError(
            f"Factor {factor!r} is neither Scalar nor Graded; "
            f"declare its grading before sorting"
        )
    return prop.degree


def _factor_key(factor: Expr, registry: PropertyRegistry) -> Tuple[int, str]:
    """Ordering key: scalars first (bucket 0), others by repr."""
    bucket = 0 if _is_scalar(factor, registry) else 1
    return (bucket, repr(factor))


def _should_swap(
    a: Expr, b: Expr, registry: PropertyRegistry
) -> bool:
    return _factor_key(a, registry) > _factor_key(b, registry)


def _swap_behavior(
    a: Expr, b: Expr, registry: PropertyRegistry
) -> Tuple[bool, Degree]:
    """Return ``(can_swap, sign_exp_delta)`` for adjacent ``a``, ``b``.

    The commutativity law between the pair governs whether the swap is
    permitted and, if so, the Koszul exponent it contributes:

    * Either factor carries :class:`Scalar` — free swap, no sign.
    * Either factor carries :class:`NonCommuting` — swap forbidden.
    * Both carry :class:`AntiCommuting` — swap produces a flat ``-1``
      (parity delta ``1``).
    * Both carry :class:`GradedCommutative`, or both are
      :class:`Graded` (the implicit Koszul default) — swap produces
      ``(-1)^{|a||b|}``.
    """
    if _is_scalar(a, registry) or _is_scalar(b, registry):
        return (True, Degree.const(0))
    if registry.has(a, NonCommuting) or registry.has(b, NonCommuting):
        return (False, Degree.const(0))
    if registry.has(a, AntiCommuting) and registry.has(b, AntiCommuting):
        return (True, Degree.const(1))
    if registry.has(a, GradedCommutative) and registry.has(b, GradedCommutative):
        return (True, _degree_of(a, registry) * _degree_of(b, registry))
    # Implicit default: both Graded → Koszul. Preserves backward-compat
    # with the pre-marker tests and matches the common case where the
    # user declared grading without separately declaring the sign rule.
    return (True, _degree_of(a, registry) * _degree_of(b, registry))


# --------------------------------------------------------------------- #
# Public API                                                             #
# --------------------------------------------------------------------- #


def sort_product(
    expr: Expr, registry: PropertyRegistry
) -> Tuple[Expr, Degree]:
    """Sort a Product's factors into canonical order with a Koszul sign.

    Returns ``(sorted_expr, sign_exponent)``. The sign is the exponent
    of ``-1``, held as a :class:`Degree` polynomial so symbolic cases
    like ``|α|*|β|`` survive intact.

    Non-:class:`Product` inputs pass through unchanged with exponent
    ``0``. Single-factor products collapse to the lone factor.
    """
    if not isinstance(expr, Product):
        return (expr, Degree.const(0))

    factors: List[Expr] = list(expr.children)
    # Up-front classification check so an unclassified factor is
    # reported even when it happens to already be in canonical
    # position — missing a grading is a modelling error, not a perf issue.
    for f in factors:
        _degree_of(f, registry)
    sign_exp = Degree.const(0)
    n = len(factors)
    # Bubble sort: stable enough for our size, and each swap's sign
    # contribution is easy to track. A NonCommuting pair blocks the
    # swap entirely — the factors stay adjacent in whatever order they
    # started, and later passes can still re-run the sort after the
    # blocker is removed.
    for i in range(n):
        for j in range(n - 1 - i):
            a, b = factors[j], factors[j + 1]
            if not _should_swap(a, b, registry):
                continue
            can_swap, delta = _swap_behavior(a, b, registry)
            if not can_swap:
                continue
            factors[j], factors[j + 1] = b, a
            sign_exp = sign_exp + delta

    if not factors:
        return (expr, sign_exp)
    if len(factors) == 1:
        return (factors[0], sign_exp)
    return (Product(*factors), sign_exp)


def apply_sign(expr: Expr, sign_exp: Degree) -> Expr:
    """Fold a decidable sign exponent into the expression.

    Returns ``expr`` for even parity, ``Neg(expr)`` for odd. Raises
    :class:`ValueError` when parity depends on unknown variables — the
    caller should keep the :class:`Degree` around in that case (or
    eventually wrap in a ``(-1)^{…}`` node once the package has one).
    """
    parity = sign_exp.parity()
    if parity is None:
        raise ValueError(
            f"Parity of {sign_exp!r} is not decidable; cannot fold sign"
        )
    if parity == 0:
        return expr
    return Neg(expr)
