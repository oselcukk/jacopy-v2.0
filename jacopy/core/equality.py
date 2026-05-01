"""
Equality utilities.

The default ``a == b`` on expressions is *structural*: same type,
same children in the same order. That's the right default, it makes
hashing well-defined and keeps the core layer free of semantic
assumptions. But proofs need a few other notions of sameness, and
this module collects the ones that are not deep enough to warrant
the canonicalization machinery of Faz 2.

* :func:`structural_equal`, a named alias for ``==``, for readable
  proof transcripts.

* :func:`alpha_equal`, two *patterns* are alpha-equivalent if they
  agree modulo a consistent renaming of wildcard names. Used to
  deduplicate rewrite rules that differ only in how they name their
  holes.

* :func:`sum_bag_equal`, top-level :class:`Sum` children compared
  as a multiset. Useful for sanity-checking an intermediate step
  without committing to a canonical term order; deeper commutative
  equality is the job of :mod:`jacopy.algorithms.canonicalize`.
"""

from __future__ import annotations

from collections import Counter
from typing import Dict

from jacopy.core.expr import Expr, Sum
from jacopy.core.wildcards import SeqWildcard, Wildcard


# --------------------------------------------------------------------- #
# Structural                                                            #
# --------------------------------------------------------------------- #


def structural_equal(a: Expr, b: Expr) -> bool:
    """Return ``a == b``, structural, position-sensitive equality."""
    return a == b


# --------------------------------------------------------------------- #
# Alpha equivalence                                                     #
# --------------------------------------------------------------------- #


def alpha_equal(p: Expr, q: Expr) -> bool:
    """True iff ``p`` and ``q`` are equal up to wildcard renaming.

    Two patterns like ``?A + x`` and ``?B + x`` are alpha-equivalent:
    both describe "a sum of something with x". The renaming must be
    a *bijection*, two distinct wildcards in ``p`` cannot collapse
    into one in ``q``, and vice versa. Type filters must match
    exactly: a ``?A:Scalar`` is not alpha-equivalent to a plain
    ``?A`` or a ``?A:Graded``.

    Wildcards on one side with non-wildcards on the other are never
    alpha-equal, even if the non-wildcard "fits", alpha is about
    shape, not inhabitation.
    """
    return _alpha(p, q, {}, {})


def _alpha(
    p: Expr,
    q: Expr,
    fwd: Dict[str, str],
    bwd: Dict[str, str],
) -> bool:
    if isinstance(p, Wildcard) and isinstance(q, Wildcard):
        if p.type_filter is not q.type_filter:
            return False
        return _bind_name(p.name, q.name, fwd, bwd)
    if isinstance(p, SeqWildcard) and isinstance(q, SeqWildcard):
        return _bind_name(p.name, q.name, fwd, bwd)
    # Mixed kinds, wildcard on one side only.
    if isinstance(p, (Wildcard, SeqWildcard)) or isinstance(
        q, (Wildcard, SeqWildcard)
    ):
        return False

    if type(p) is not type(q):
        return False
    if p.is_atom:
        return p == q
    if len(p.children) != len(q.children):
        return False
    for pc, qc in zip(p.children, q.children):
        if not _alpha(pc, qc, fwd, bwd):
            return False
    return True


def _bind_name(a: str, b: str, fwd: Dict[str, str], bwd: Dict[str, str]) -> bool:
    if a in fwd:
        return fwd[a] == b
    if b in bwd:
        # ``b`` is already claimed by some other name, not a bijection.
        return False
    fwd[a] = b
    bwd[b] = a
    return True


# --------------------------------------------------------------------- #
# Multiset equality on top-level Sum                                    #
# --------------------------------------------------------------------- #


def sum_bag_equal(a: Expr, b: Expr) -> bool:
    """True iff ``a`` and ``b`` are Sums with the same multiset of terms.

    Compares only the top level. Inner Sums are still compared
    structurally, so ``Sum(x+y, z)`` and ``Sum(y+x, z)`` are *not*
    bag-equal by this function, reach for
    :mod:`jacopy.algorithms.canonicalize` if you want the deep
    version. For non-Sum inputs, degenerates to ``==``.
    """
    if type(a) is not type(b):
        return False
    if not isinstance(a, Sum):
        return a == b
    if len(a.children) != len(b.children):
        return False
    return Counter(a.children) == Counter(b.children)
