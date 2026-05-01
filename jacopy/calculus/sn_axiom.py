"""
Schouten-Nijenhuis bivector formula, Faz 13.D.

After OpCommutator collapse (Faz 13.C axiom 5) the cyclic Koszul Jacobi
sum's iterated-Lie group folds into three terms of the shape

    L_{[π^♯a, π^♯b]_VF}(c)     for cyclic (a, b, c),

each one a Lie derivative against the vector-field Lie bracket of two
sharps of the same bivector ``π``. This module's
:class:`SnBivectorFormulaDefinition` is the engine-level expression of
the classical evaluator formula

    Σ_cyc L_{[π^♯a, π^♯b]_VF}(c) = ½⟨[π,π]_SN, a∧b∧c⟩, represented
    here as the inert ``BracketApply([·,·]_SN, π, π)`` node, which
    the surrounding strategy can pair against ``a∧b∧c`` downstream.

The rule is a sibling of :class:`LieVfJacobiDefinition`: same triple-
finder, but it requires each ``X, Y`` in the inner ``LieBracketVF(X, Y)``
to be ``Act(Sharp(π), ·)`` for a single shared :class:`Sharp` atom, and
it rewrites the three terms to the SN obstruction handle (rather than
to ``Integer(0)``).

Why this is the closing step of 2f-deep
---------------------------------------

Faz 13.A-C axioms drive the term-level cascade: Sharp distributes over
sums, Pairing distributes bilinearly, ``L_X`` commutes through pairings,
and operator commutators fold into ``L_{[·,·]_VF}``. What remains after
those rewrites is precisely the cyclic ``L_{[π^♯·, π^♯·]_VF}(·)`` triple
plus a residue of bookkeeping terms (nested ``L_π^♯(L_π^♯ ·)``,
``d⟨·, ·⟩``, ``L_X(d⟨·, ·⟩)``) that the strategy layer can simplify or
the user can interpret as the leftover ``[π,π]_SN``-pairing pieces. The
axiom captures the *named-bracket* part of the cancellation,
``[π^♯·, π^♯·]_VF`` is what carries the SN content; the rest is
bilinear reshape.
"""

from __future__ import annotations

from itertools import combinations
from typing import Optional, Tuple

from jacopy.algebra.derivation import Act
from jacopy.algebra.lie_bracket_vf import LieBracketVF
from jacopy.brackets.base import BracketApply
from jacopy.brackets.schouten import sn as default_sn
from jacopy.calculus.lie_derivative import LieDerivative
from jacopy.calculus.musical import Sharp
from jacopy.core.expr import Expr, Integer, Neg, Sum
from jacopy.proof.expansion import Definition


# --------------------------------------------------------------------- #
# Helpers                                                                #
# --------------------------------------------------------------------- #


def _peel_sharp(expr: Expr, sharp_atom: Sharp) -> Optional[Expr]:
    """Return ``α`` if ``expr`` is ``Act(sharp_atom, α)``, else ``None``."""
    if isinstance(expr, Act) and expr.op == sharp_atom:
        return expr.arg
    return None


# --------------------------------------------------------------------- #
# Axiom 6, SN bivector formula                                          #
# --------------------------------------------------------------------- #


class SnBivectorFormulaDefinition(Definition):
    """``Σ_cyc L_{[π^♯a, π^♯b]_VF}(c) → BracketApply([·,·]_SN, π, π)``.

    Fires on a :class:`Sum` containing three terms of the shape
    ``Act(L_{[Act(π^♯, a), Act(π^♯, b)]_VF}, c)`` whose inner
    ``(a, b, c)`` triples form a cyclic permutation. The three matched
    terms are stripped from the Sum and replaced by a single
    ``BracketApply(sn, π, π)``, the inert SN self-bracket node that
    Faz 9 Stage B treats as the universal Poisson obstruction.

    Parameters
    ----------
    sharp
        The musical map ``π^♯`` whose bivector identifies which SN
        self-bracket the rule resolves to. Each matched
        ``LieBracketVF(X, Y)`` must have ``X = Act(sharp, ·)`` and
        ``Y = Act(sharp, ·)`` for this same ``sharp`` instance.
    sn_bracket
        Optional :class:`~jacopy.brackets.schouten.SchoutenBracket`,
        defaults to :data:`jacopy.brackets.schouten.sn`. Override to
        thread a non-default SN bracket through the rewrite.

    Notes
    -----
    Pattern-matching strictness: each candidate term must be
    ``Act(LieDerivative(LieBracketVF(Act(π^♯, a), Act(π^♯, b))), c)``
    with the same ``π^♯`` instance for both inner Acts and across all
    three terms. Mismatched sharps signal a different geometric
    setting and the rule declines to fire.
    """

    name = "[π,π]_SN bivector formula"

    def __init__(
        self,
        sharp: Sharp,
        *,
        sn_bracket=None,
    ) -> None:
        if not isinstance(sharp, Sharp):
            raise TypeError(
                "SnBivectorFormulaDefinition requires a Sharp atom"
            )
        self._sharp = sharp
        self._sn = sn_bracket if sn_bracket is not None else default_sn

    def matches(self, expr: Expr) -> bool:
        return (
            isinstance(expr, Sum)
            and self._find_cyclic_triple(expr) is not None
        )

    def rewrite(self, expr: Expr) -> Expr:
        match = self._find_cyclic_triple(expr)
        assert match is not None, "matches() guarantees a triple"
        idxs = match
        kept = [
            c for k, c in enumerate(expr.children) if k not in idxs
        ]
        bivector = self._sharp.bivector
        sn_apply = BracketApply(self._sn, bivector, bivector)
        if not kept:
            return sn_apply
        return Sum.make(sn_apply, *kept)

    def _peel_term(
        self, term: Expr
    ) -> Optional[Tuple[Expr, Expr, Expr]]:
        """Return ``(a, b, c)`` for ``Act(L_{[π^♯a, π^♯b]_VF}, c)``,
        else ``None``. ``Neg``-wrapped terms decline, the cyclic
        formula is the *positive* triple."""
        if isinstance(term, Neg):
            return None
        if not isinstance(term, Act):
            return None
        L = term.op
        c = term.arg
        if not isinstance(L, LieDerivative):
            return None
        brkt = L.vector_field
        if not isinstance(brkt, LieBracketVF):
            return None
        a = _peel_sharp(brkt.X, self._sharp)
        b = _peel_sharp(brkt.Y, self._sharp)
        if a is None or b is None:
            return None
        return a, b, c

    def _find_cyclic_triple(
        self, sum_expr: Sum
    ) -> Optional[Tuple[int, int, int]]:
        children = sum_expr.children
        peeled = [
            (i, self._peel_term(c)) for i, c in enumerate(children)
        ]
        candidates = [(i, p) for i, p in peeled if p is not None]
        if len(candidates) < 3:
            return None
        for combo in combinations(candidates, 3):
            (i1, t1), (i2, t2), (i3, t3) = combo
            triple = (t1, t2, t3)
            anchor = triple[0]
            A, B, C = anchor
            cyclic = {(A, B, C), (B, C, A), (C, A, B)}
            if set(triple) == cyclic:
                return (i1, i2, i3)
        return None
