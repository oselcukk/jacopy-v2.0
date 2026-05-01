"""
Graded-commutative sorting of Product factors.

A :class:`Product` in the core layer is non-commutative, factor order is
preserved verbatim. Many proofs need the *graded-commutative* reading
instead: adjacent factors may be swapped, each swap producing a Koszul
sign ``(-1)^{|a||b|}``. :func:`sort_product` performs that re-ordering
and returns the accumulated sign as a :class:`Degree` polynomial so the
caller can inspect it (even when the sign is symbolic like
``(-1)^{|α||β|}``).

Every factor must be registered as :class:`Scalar` or :class:`Graded`.
Scalars sort to the front and contribute no sign, they commute with
everything in the Koszul rule. Graded factors sort by a stable
structural key and contribute ``|a||b|`` per swap. Unclassified factors
raise :class:`ValueError`; the fix is to declare their grading, not to
silently paper over the missing information.

The sign is returned as a *polynomial in the degree variables*. Call
:meth:`Degree.parity` to collapse to ``0``/``1`` when the parity is
decidable; use :func:`apply_sign` to fold a decidable sign into an
:class:`Expr` directly. When the parity is symbolic, as in
Schouten/Courant Jacobi proofs, the Degree polynomial is the right
thing to keep.
"""

from __future__ import annotations

from typing import List, Tuple

from jacopy.algebra.derivation import Derivation
from jacopy.core.expr import Expr, Integer, Neg, Product, Rational
from jacopy.core.properties import (
    AntiCommuting,
    Graded,
    GradedCommutative,
    NonCommuting,
    Scalar,
)
from jacopy.core.registry import PropertyRegistry
from jacopy.core.symbolic_degree import Degree


# --------------------------------------------------------------------- #
# Classification helpers                                                 #
# --------------------------------------------------------------------- #


def _peel_neg(factor: Expr) -> Tuple[Expr, int]:
    """Strip leading :class:`Neg` wrappers, returning ``(inner, parity)``.

    A :class:`Neg` node has the same degree as its argument and the same
    commutativity behaviour, it is a pure scalar sign in front of the
    factor. Pulling it out lets the rest of :func:`sort_product` run on
    the bare factor (which the registry knows how to grade) and fold the
    accumulated sign back into the Koszul exponent at the end. Double
    negations simply cancel their parity contribution.
    """
    parity = 0
    while isinstance(factor, Neg):
        parity ^= 1
        factor = factor.arg
    return factor, parity


def _explode_factor(factor: Expr) -> Tuple[List[Expr], int]:
    """Return ``(flat_factors, parity)`` for a factor appearing under a Product.

    Peels any :class:`Neg` layers and splices an inner :class:`Product`
    into the surrounding factor list. The ordinary :mod:`flatten` pass
    merges nested :class:`Product` nodes only when they appear
    directly; a :class:`Neg` between the outer and inner product acts
    as a barrier that flatten leaves alone. :func:`sort_product` needs
    to see bare factors for registry lookup, so we handle the
    flatten-through-Neg case here inline, recursing through both
    sides until every emitted factor is a non-Product, non-Neg atom
    (or at worst a composite node that the registry has explicitly
    graded).
    """
    inner, parity = _peel_neg(factor)
    if not isinstance(inner, Product):
        return ([inner], parity)
    result: List[Expr] = []
    for child in inner.children:
        sub_factors, sub_parity = _explode_factor(child)
        result.extend(sub_factors)
        parity ^= sub_parity
    return (result, parity)


def _is_scalar(factor: Expr, registry: PropertyRegistry) -> bool:
    # Numeric literals are always scalars, they commute with
    # everything and carry no grading. Declaring Scalar on every
    # Integer(k) would be noise.
    if isinstance(factor, (Integer, Rational)):
        return True
    return registry.has(factor, Scalar)


def _degree_of(factor: Expr, registry: PropertyRegistry) -> Degree:
    """Return the grading of ``factor``, or raise if unclassified.

    Scalars have degree ``0``; :class:`Graded` factors carry their own
    Degree. A :class:`Derivation` used as an operator factor carries
    its grading intrinsically. Compound factors that aren't directly
    registered, ``Act(d, f)``, ``Act(ι_X, ω)``, nested ``Product``
    operator chains, ``Pairing``, ``BracketApply``, delegate to
    :func:`jacopy.algebra.derivation.degree_of`, which walks the tree
    and resolves ``|Act(D, x)| = |D| + |x|`` and friends. Anything
    that still can't be classified raises with a hint about the
    missing declaration.
    """
    if _is_scalar(factor, registry):
        return Degree.const(0)
    if isinstance(factor, Derivation):
        return factor.degree
    prop = registry.get(factor, Graded)
    if prop is not None:
        return prop.degree
    # Compound factor, delegate to the algebra-layer walker. It
    # handles Act / Product / Neg / Pairing / BracketApply by
    # recursing into children and summing degrees.
    from jacopy.algebra.derivation import degree_of
    try:
        return degree_of(factor, registry)
    except ValueError:
        raise ValueError(
            f"Factor {factor!r} is neither Scalar nor Graded; "
            f"declare its grading before sorting"
        ) from None


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

    * Either factor carries :class:`Scalar`, free swap, no sign.
    * Either factor carries :class:`NonCommuting`, swap forbidden.
    * Both carry :class:`AntiCommuting`, swap produces a flat ``-1``
      (parity delta ``1``).
    * Both carry :class:`GradedCommutative`, or both are
      :class:`Graded` (the implicit Koszul default), swap produces
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

    raw_factors: List[Expr] = list(expr.children)
    # Peel off any :class:`Neg` wrappers and splice nested
    # :class:`Product` children up into the parent factor list. Negs
    # only contribute a sign, so their parity is tallied and folded
    # back into the Koszul exponent at the end. Flattening through the
    # Neg barrier is what lets bracket-expanded shapes (which routinely
    # produce ``Product(a, Neg(Product(b, c)))``) reach the sort layer
    # as ``Product(a, b, c)`` with a ``−1`` sign attached.
    factors: List[Expr] = []
    neg_parity = 0
    for f in raw_factors:
        sub_factors, parity = _explode_factor(f)
        factors.extend(sub_factors)
        neg_parity ^= parity

    # Up-front classification check so an unclassified factor is
    # reported even when it happens to already be in canonical
    # position, missing a grading is a modelling error, not a perf issue.
    for f in factors:
        _degree_of(f, registry)
    sign_exp = Degree.const(neg_parity)
    n = len(factors)
    # Bubble sort: stable enough for our size, and each swap's sign
    # contribution is easy to track. A NonCommuting pair blocks the
    # swap entirely, the factors stay adjacent in whatever order they
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
    :class:`ValueError` when parity depends on unknown variables, the
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
