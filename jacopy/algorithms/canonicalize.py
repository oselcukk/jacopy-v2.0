"""
Canonical form for expressions.

``canonicalize`` is the post-processing step that normalizes an
expression to a single representative of its equivalence class under a
small, conservative set of identities:

* Constants fold: ``2 + 3 → 5``, ``2 * 3 → 6``, ``(2/3) + (1/6) → 5/6``.
* Negation normalizes: ``-(-x) → x``, ``Neg(Integer(3)) → Integer(-3)``.
* Like terms in a Sum merge: ``x + x → 2*x``, ``2*x + 3*x → 5*x``.
* Numeric factors in a Product move to the front and merge.
* Trivial powers collapse: ``x**0 → 1``, ``x**1 → x``, ``2**3 → 8``.
* Sum children are sorted by a stable repr-based key for deterministic
  output.

Non-commutative reordering of *non-numeric* Product factors is
deliberately not performed here, that lives in
:mod:`jacopy.algorithms.sort_product`, which needs grading information.
The goal of this module is to give proofs a deterministic, readable
pre-form; anything that needs the Koszul sign rule is a separate pass.
"""

from __future__ import annotations

from fractions import Fraction
from typing import Dict, List, Optional, Tuple

from jacopy.core.expr import (
    Expr,
    Integer,
    Neg,
    One,
    Power,
    Product,
    Rational,
    Sum,
    Zero,
)


# --------------------------------------------------------------------- #
# Public entry point                                                    #
# --------------------------------------------------------------------- #


def canonicalize(expr: Expr) -> Expr:
    """Return the canonical form of ``expr``.

    Idempotent: ``canonicalize(canonicalize(x)) == canonicalize(x)``.
    The traversal is bottom-up; each node is normalized against its
    already-canonical children.
    """
    if expr.is_atom:
        return expr

    new_children = tuple(canonicalize(c) for c in expr.children)

    if isinstance(expr, Neg):
        return _canon_neg(new_children[0])
    if isinstance(expr, Sum):
        return _canon_sum(new_children)
    if isinstance(expr, Product):
        return _canon_product(new_children)
    if isinstance(expr, Power):
        return _canon_power(new_children[0], new_children[1])

    # Unknown compound, rebuild preserving structure.
    return expr._rebuild(new_children)


# --------------------------------------------------------------------- #
# Numeric helpers                                                       #
# --------------------------------------------------------------------- #


def _as_fraction(x: Expr) -> Optional[Fraction]:
    """Return ``x`` as a :class:`Fraction` if it's a numeric atom."""
    if isinstance(x, Integer):
        return Fraction(x.value)
    if isinstance(x, Rational):
        return Fraction(x.p, x.q)
    return None


def _from_fraction(f: Fraction) -> Expr:
    """Materialize a :class:`Fraction` as the tightest numeric Expr."""
    if f.denominator == 1:
        return Integer(f.numerator)
    return Rational(f.numerator, f.denominator)


# --------------------------------------------------------------------- #
# Neg                                                                   #
# --------------------------------------------------------------------- #


def _canon_neg(arg: Expr) -> Expr:
    """Normalize ``Neg(arg)`` given an already-canonical ``arg``."""
    if isinstance(arg, Neg):
        return arg.arg
    if isinstance(arg, Integer):
        return Integer(-arg.value)
    if isinstance(arg, Rational):
        return Rational(-arg.p, arg.q)
    if isinstance(arg, Sum):
        # Push Neg through Sum: −(a + b + c) → (−a) + (−b) + (−c).
        # This lets like-term collection cancel across signs (e.g.
        # `X − (X − Y) → Y`). Each child is already canonical, so
        # recurse through _canon_neg for proper Integer/Neg folding,
        # then re-canonicalize the rebuilt Sum so sign-flipped terms
        # merge with their opposites.
        return _canon_sum(
            tuple(_canon_neg(c) for c in arg.children)
        )
    return Neg(arg)


# --------------------------------------------------------------------- #
# Sum: like-term collection                                             #
# --------------------------------------------------------------------- #


def _coeff_and_core(term: Expr) -> Tuple[Fraction, Expr]:
    """Split ``term`` into ``(numeric_coefficient, non-numeric core)``.

    The rule:

    * ``Neg(t)`` contributes a ``-1`` factor to the coefficient.
    * A pure numeric atom yields ``(value, One)``.
    * Numeric factors inside a :class:`Product` are pulled into the
      coefficient; the rest becomes the core.
    * Anything else is considered the core with coefficient ``1``.
    """
    if isinstance(term, Neg):
        c, core = _coeff_and_core(term.arg)
        return (-c, core)
    frac = _as_fraction(term)
    if frac is not None:
        return (frac, One)
    if isinstance(term, Product):
        coeff = Fraction(1)
        rest: List[Expr] = []
        for f in term.children:
            if isinstance(f, Neg):
                inner = f.arg
                ff = _as_fraction(inner)
                if ff is not None:
                    coeff *= -ff
                    continue
                coeff = -coeff
                rest.append(inner)
                continue
            ff = _as_fraction(f)
            if ff is not None:
                coeff *= ff
            else:
                rest.append(f)
        if not rest:
            return (coeff, One)
        if len(rest) == 1:
            return (coeff, rest[0])
        return (coeff, Product(*rest))
    return (Fraction(1), term)


def _combine_coeff(coeff: Fraction, core: Expr) -> Expr:
    """Rebuild ``coeff * core`` with proper sign/zero handling."""
    if coeff == 0:
        return Zero
    if core == One:
        return _from_fraction(coeff)
    if coeff == 1:
        return core
    if coeff == -1:
        return Neg(core)
    c_expr = _from_fraction(coeff)
    if isinstance(core, Product):
        return Product(c_expr, *core.children)
    return Product(c_expr, core)


def _canon_sum(children: Tuple[Expr, ...]) -> Expr:
    """Collect like terms, fold constants, sort by stable repr key."""
    # Flatten nested Sums (defensive; smart-ctor usually does this).
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

    non_const = [c for c in order if c != One]
    non_const.sort(key=lambda e: repr(e))

    terms: List[Expr] = []
    for core in non_const:
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


# --------------------------------------------------------------------- #
# Product: numeric factor extraction                                    #
# --------------------------------------------------------------------- #


def _canon_product(children: Tuple[Expr, ...]) -> Expr:
    """Fold numeric factors, propagate sign, preserve non-numeric order.

    We deliberately do *not* reorder non-numeric factors, products are
    non-commutative here. The Koszul-signed sort is
    :mod:`jacopy.algorithms.sort_product`.
    """
    # Flatten nested Products (child canonicalization may leave them).
    flat: List[Expr] = []
    for c in children:
        if isinstance(c, Product):
            flat.extend(c.children)
        else:
            flat.append(c)

    coeff = Fraction(1)
    rest: List[Expr] = []
    for c in flat:
        if isinstance(c, Neg):
            coeff = -coeff
            c = c.arg
        ff = _as_fraction(c)
        if ff is not None:
            coeff *= ff
            continue
        rest.append(c)

    if coeff == 0:
        return Zero
    if not rest:
        return _from_fraction(coeff)
    if coeff == 1:
        return rest[0] if len(rest) == 1 else Product(*rest)
    if coeff == -1:
        inner = rest[0] if len(rest) == 1 else Product(*rest)
        return Neg(inner)
    return Product(_from_fraction(coeff), *rest)


# --------------------------------------------------------------------- #
# Power                                                                 #
# --------------------------------------------------------------------- #


def _canon_power(base: Expr, exp: Expr) -> Expr:
    """Fold trivial powers and concrete numeric powers with small exponents."""
    if isinstance(exp, Integer):
        if exp.value == 0:
            return One
        if exp.value == 1:
            return base
    if base == One:
        return One
    if base == Zero:
        if isinstance(exp, Integer) and exp.value > 0:
            return Zero
    if isinstance(exp, Integer) and exp.value > 0:
        frac = _as_fraction(base)
        if frac is not None:
            return _from_fraction(frac ** exp.value)
    return Power(base, exp)


# --------------------------------------------------------------------- #
# Semantic equality / hashing                                            #
# --------------------------------------------------------------------- #
#
# These helpers live here (not in ``jacopy.core.equality``) because
# canonical form depends on algorithms-layer logic. Putting them in the
# core would reverse the dependency direction.


def semantically_equal(a: Expr, b: Expr) -> bool:
    """Return ``True`` iff ``a`` and ``b`` canonicalize to the same tree.

    Coarser than structural ``==`` (``x + x`` equals ``2 * x``), but
    still conservative: anything requiring grading information, Koszul
    sign rule, symmetric/antisymmetric bracket collapse, is out of
    scope. Use :func:`jacopy.algorithms.sort_product` downstream when
    those are needed.
    """
    return canonicalize(a) == canonicalize(b)


def canonical_hash(expr: Expr) -> int:
    """Hash that agrees with :func:`semantically_equal`.

    ``semantically_equal(a, b)`` implies
    ``canonical_hash(a) == canonical_hash(b)``; the converse is not
    guaranteed (hash collisions are possible, as always).
    """
    return hash(canonicalize(expr))
