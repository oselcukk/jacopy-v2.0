"""
Vector-field axioms, Faz 13.C.

Two engine-level rewrite rules that turn iterated Lie-derivative shapes
into named ``L_{[X,Y]_VF}`` atoms and discharge the cyclic Jacobi
identity for the vector-field Lie bracket:

* :class:`OpCommutatorVfDefinition`, fires on a :class:`Sum` and
  collapses any pair ``L_X(L_Y(ω)) − L_Y(L_X(ω))`` (positive-then-
  negated, in any positional order) into ``L_{[X,Y]_VF}(ω)``. The
  operator-commutator-to-vector-field-Lie identity is what lets the
  2f-deep cancellation chain replace iterated derivative pairs by a
  single Lie derivative against a Lie-bracket vector field.
* :class:`LieVfJacobiDefinition`, fires on a :class:`Sum` containing
  three cyclically-permuted ``L_{[X,[Y,Z]_VF]_VF}(ω)`` terms (with the
  same operand ``ω``) and zeros them out. The vector-field Lie
  bracket satisfies the ordinary (un-graded) Jacobi identity, so
  the cyclic triple is mathematically zero, recognising it as such
  is the engine-level expression of that fact.

Both rules scan the children of a Sum to find the matching pattern;
they are deliberately permissive about positional order so the
upstream pipeline (product_rule + simplify) doesn't have to canonicalise
before they fire. The match returns the *first* viable shape; the
engine fix-point loop reapplies each rule until no further pair /
triple is found.
"""

from __future__ import annotations

from itertools import combinations, permutations
from typing import Optional, Tuple

from jacopy.algebra.derivation import Act
from jacopy.algebra.lie_bracket_vf import LieBracketVF
from jacopy.calculus.lie_derivative import (
    LieDerivative,
    lie_derivative as default_lie_derivative,
)
from jacopy.core.expr import Expr, Integer, Neg, Sum
from jacopy.proof.expansion import Definition


# --------------------------------------------------------------------- #
# Helpers                                                                #
# --------------------------------------------------------------------- #


def _strip_neg(expr: Expr) -> Tuple[bool, Expr]:
    """Return (has_neg, inner), peel a single :class:`Neg` if present."""
    if isinstance(expr, Neg):
        return True, expr.arg
    return False, expr


def _as_lie_outer_inner(
    expr: Expr,
) -> Optional[Tuple[LieDerivative, LieDerivative, Expr]]:
    """Match ``Act(L_X, Act(L_Y, w))`` and return ``(L_X, L_Y, w)``.

    Returns ``None`` if either layer is not a :class:`LieDerivative`
    application.
    """
    if not isinstance(expr, Act):
        return None
    outer = expr.op
    inner_act = expr.arg
    if not isinstance(outer, LieDerivative):
        return None
    if not isinstance(inner_act, Act):
        return None
    if not isinstance(inner_act.op, LieDerivative):
        return None
    return outer, inner_act.op, inner_act.arg


# --------------------------------------------------------------------- #
# Axiom 5, operator commutator → vector-field Lie bracket               #
# --------------------------------------------------------------------- #


class OpCommutatorVfDefinition(Definition):
    """``L_X(L_Y(ω)) − L_Y(L_X(ω)) → L_{[X,Y]_VF}(ω)`` inside a Sum.

    Scans a :class:`Sum`'s children for a positive ``Act(L_X,
    Act(L_Y, ω))`` paired with its sign-flipped twin ``Neg(Act(L_Y,
    Act(L_X, ω)))``. The two terms are removed and replaced with a
    single ``Act(L_{[X,Y]_VF}, ω)``, the named-bracket vector-field
    Lie derivative on the same operand.

    Arguments may sit in either positional order; the matcher tries
    both orderings before declaring no pair available.

    The resulting :class:`~jacopy.calculus.lie_derivative.LieDerivative`
    inherits its construction options from the rule's
    ``lie_derivative_factory`` parameter, pass a custom factory when
    the host calculus uses a non-default ``d`` or interior product.
    """

    name = "[L_X, L_Y] = L_{[X,Y]_VF}"

    def __init__(self, *, lie_derivative_factory=None) -> None:
        self._lie = (
            lie_derivative_factory
            if lie_derivative_factory is not None
            else default_lie_derivative
        )

    def matches(self, expr: Expr) -> bool:
        return isinstance(expr, Sum) and self._find_pair(expr) is not None

    def rewrite(self, expr: Expr) -> Expr:
        match = self._find_pair(expr)
        assert match is not None, "matches() guarantees a pair"
        i, j, X, Y, omega = match
        bracket_vf = LieBracketVF(X, Y)
        new_term = Act(self._lie(bracket_vf), omega)
        kept = [c for k, c in enumerate(expr.children) if k != i and k != j]
        return Sum.make(new_term, *kept)

    def _find_pair(
        self, sum_expr: Sum
    ) -> Optional[Tuple[int, int, Expr, Expr, Expr]]:
        """Return ``(i, j, X, Y, ω)`` for the first cancelling pair, else ``None``.

        ``i`` is the index of the positive term ``Act(L_X, Act(L_Y, ω))``
        and ``j`` of its negated twin ``Neg(Act(L_Y, Act(L_X, ω)))``.
        """
        children = sum_expr.children
        for i, j in combinations(range(len(children)), 2):
            for a, b in ((i, j), (j, i)):
                pos_neg, pos_inner = _strip_neg(children[a])
                neg_neg, neg_inner = _strip_neg(children[b])
                if pos_neg or not neg_neg:
                    continue
                pos = _as_lie_outer_inner(pos_inner)
                neg = _as_lie_outer_inner(neg_inner)
                if pos is None or neg is None:
                    continue
                L_X, L_Y, omega_pos = pos
                L_Y2, L_X2, omega_neg = neg
                if (
                    L_X == L_X2
                    and L_Y == L_Y2
                    and omega_pos == omega_neg
                    and L_X != L_Y
                ):
                    return (
                        a,
                        b,
                        L_X.vector_field,
                        L_Y.vector_field,
                        omega_pos,
                    )
        return None


# --------------------------------------------------------------------- #
# Axiom 6, Lie-Jacobi for the vector-field Lie bracket                  #
# --------------------------------------------------------------------- #


class LieVfJacobiDefinition(Definition):
    """``[X,[Y,Z]_VF]_VF + [Y,[Z,X]_VF]_VF + [Z,[X,Y]_VF]_VF = 0`` (cyclic).

    Fires on a :class:`Sum` whose children include three terms of the
    shape ``Act(L_{[X,[Y,Z]_VF]_VF}, ω)`` (and the two cyclic
    permutations of ``X, Y, Z``) sharing a single operand ``ω``. The
    three cancelling terms are stripped from the Sum and the whole
    cyclic block reduces to :class:`Integer` ``0``.

    Why apply on ``Act(L_..., ω)`` rather than on the bare
    ``LieBracketVF`` chain
    --------------------------------------------------------------

    The 2f-deep cyclic-Jacobi cancellation reaches its closing form
    after operator-commutator collapse: each iterated commutator pair
    has been folded into ``L_{[·,·]_VF}``, so the residual cyclic
    triple lives at the *Act* level, not at the bare-derivation
    level. Matching the ``Act(L_..., ω)`` shape is what lets the
    rule fire at the place the chain actually produces the cyclic
    sum.
    """

    name = "Lie-Jacobi for vector fields"

    def matches(self, expr: Expr) -> bool:
        return (
            isinstance(expr, Sum)
            and self._find_cyclic_triple(expr) is not None
        )

    def rewrite(self, expr: Expr) -> Expr:
        match = self._find_cyclic_triple(expr)
        assert match is not None, "matches() guarantees a triple"
        idxs = match
        kept = [c for k, c in enumerate(expr.children) if k not in idxs]
        if not kept:
            return Integer(0)
        if len(kept) == 1:
            return kept[0]
        return Sum.make(*kept)

    def _peel_act_lie_jacobi(
        self, term: Expr
    ) -> Optional[Tuple[Expr, Expr, Expr, Expr]]:
        """Return ``(X, Y, Z, ω)`` for ``Act(L_{[X,[Y,Z]_VF]_VF}, ω)``,
        else ``None``. ``Neg``-wrapped variants return ``None``,
        Lie-Jacobi is the *positive* cyclic sum.
        """
        if isinstance(term, Neg):
            return None
        if not isinstance(term, Act):
            return None
        L = term.op
        omega = term.arg
        if not isinstance(L, LieDerivative):
            return None
        outer_brkt = L.vector_field
        if not isinstance(outer_brkt, LieBracketVF):
            return None
        X = outer_brkt.X
        inner_brkt = outer_brkt.Y
        if not isinstance(inner_brkt, LieBracketVF):
            return None
        Y, Z = inner_brkt.X, inner_brkt.Y
        return X, Y, Z, omega

    def _find_cyclic_triple(
        self, sum_expr: Sum
    ) -> Optional[Tuple[int, int, int]]:
        """Locate three cyclically-permuted ``[X,[Y,Z]_VF]_VF``
        derivative terms sharing one operand ``ω``.
        """
        children = sum_expr.children
        peeled = [(i, self._peel_act_lie_jacobi(c)) for i, c in enumerate(children)]
        candidates = [(i, p) for i, p in peeled if p is not None]
        if len(candidates) < 3:
            return None
        for combo in combinations(candidates, 3):
            (i1, (X1, Y1, Z1, w1)), (i2, (X2, Y2, Z2, w2)), (i3, (X3, Y3, Z3, w3)) = combo
            if w1 != w2 or w2 != w3:
                continue
            triple = ((X1, Y1, Z1), (X2, Y2, Z2), (X3, Y3, Z3))
            # Three triples must be the three cyclic permutations of
            # some (A, B, C) with each carrying (A, B, C) → (X, [Y,Z])
            # = (A, [B,C]) on its respective rotation.
            anchor = triple[0]
            A, B, C = anchor
            cyclic = {
                (A, B, C),
                (B, C, A),
                (C, A, B),
            }
            actual = set(triple)
            if actual == cyclic:
                return (i1, i2, i3)
        return None
