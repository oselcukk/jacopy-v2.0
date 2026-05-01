"""
Function-level Poisson axioms, Faz 13.E.

Two engine-level rewrite rules that drive the 2g-deep cancellation
chain, the function-level analog of the 2f-deep / 13.D pass, from
the cyclic Poisson Jacobi sum

    Σ_cyc {f, {g, h}_π}_π

down to the SN self-bracket handle ``BracketApply([·,·]_SN, π, π)``.
The reduction proceeds in two steps after the LHS is formed via
:meth:`DerivedBracket.graded_jacobi_obstruction`:

* :class:`PoissonAsHamiltonianDefinition`, replaces a pinned
  Poisson-derived ``BracketApply(P, f, g)`` with ``Act(X_f, g)``,
  collapsing the bracket to its Hamiltonian vector-field action.
  Applies bottom-up to inner and outer brackets in turn, so a triply
  nested Jacobi term ``{f, {g, h}}`` rewrites to the iterated
  derivation ``X_f(X_g(h))``.
* :class:`HamiltonianCyclicSnFormulaDefinition`, the function-level
  sister of :class:`SnBivectorFormulaDefinition` (Faz 13.D). Fires on
  a :class:`Sum` containing three cyclic ``Act(X_a, Act(X_b, c))``
  terms over the same bivector and rewrites them to a single
  :class:`BracketApply` ``[·,·]_SN(π, π)``. Tolerates a global ``Neg``
  on each term, the Koszul-signed Jacobi obstruction on functions
  enters the chain with each cyclic term wrapped in :class:`Neg`,
  and the rule preserves that polarity in its emitted SN handle.

Together these collapse the function-level cyclic Poisson Jacobi to
the same universal SN obstruction the form-level (2f-deep) chain
produces, the function-level evaluator formula of the Schouten-
Nijenhuis bracket, without citing the Derived Bracket Theorem.
"""

from __future__ import annotations

from itertools import combinations
from typing import Optional, Tuple

from jacopy.algebra.derivation import Act
from jacopy.brackets.base import BracketApply
from jacopy.brackets.derived import DerivedBracket
from jacopy.brackets.schouten import sn as default_sn
from jacopy.calculus.hamiltonian_vf import (
    HamiltonianVectorField,
    hamiltonian_vf as default_hamiltonian_vf,
)
from jacopy.core.expr import Expr, Neg, Sum
from jacopy.proof.expansion import Definition


# --------------------------------------------------------------------- #
# Axiom 2g-1, Poisson bracket as Hamiltonian action                     #
# --------------------------------------------------------------------- #


class PoissonAsHamiltonianDefinition(Definition):
    """``{f, g}_π → X_f(g)`` for a pinned Poisson DerivedBracket.

    Fires on :class:`BracketApply` nodes whose bracket is the specific
    :class:`DerivedBracket` instance supplied at construction. The
    rewrite emits ``Act(HamiltonianVectorField(f, bivector=π), g)``,
    the standard identification of the Poisson bracket with the
    Hamiltonian vector field's action on the second argument.

    Pinning the bracket instance keeps unrelated derived brackets in
    the same proof from being swept up: a Courant or Schouten bracket
    sharing a name with the Poisson one would otherwise match this
    rule. The rule deliberately does not consult the bracket's name or
    base, only object identity (``is``) and structural equality
    (``==``), so the caller controls the scope precisely.

    Parameters
    ----------
    bracket
        The :class:`DerivedBracket` whose applications should be
        rewritten. Typically obtained from
        :attr:`jacopy.library.poisson.PoissonBracket.derived` for the
        bivector of interest.
    bivector
        Optional override for the bivector that parametrises the
        emitted Hamiltonian vector field. Defaults to the bracket's
        own ``Q`` (i.e. the bivector ``π`` already pinned by the
        DerivedBracket). Passing an explicit value is mainly useful
        when threading a bivector that is structurally equal but not
        identical to ``bracket.Q``.

    Raises
    ------
    TypeError
        If ``bracket`` is not a :class:`DerivedBracket`.
    """

    def __init__(
        self,
        bracket: DerivedBracket,
        *,
        bivector: Optional[Expr] = None,
    ) -> None:
        if not isinstance(bracket, DerivedBracket):
            raise TypeError(
                "PoissonAsHamiltonianDefinition requires a DerivedBracket"
            )
        self._bracket = bracket
        self._bivector = bivector if bivector is not None else bracket.Q
        self.name = f"{{f,g}}_π = X_f(g) [{bracket.name}]"

    def matches(self, expr: Expr) -> bool:
        return (
            isinstance(expr, BracketApply) and expr.bracket is self._bracket
        )

    def rewrite(self, expr: Expr) -> Expr:
        f = expr.a
        g = expr.b
        Xf = default_hamiltonian_vf(f, bivector=self._bivector)
        return Act(Xf, g)


# --------------------------------------------------------------------- #
# Axiom 2g-2, function-level SN bivector formula                        #
# --------------------------------------------------------------------- #


class HamiltonianCyclicSnFormulaDefinition(Definition):
    """``Σ_cyc X_a(X_b(c)) → BracketApply([·,·]_SN, π, π)`` (functions).

    Fires on a :class:`Sum` containing three terms of the shape
    ``Act(X_a, Act(X_b, c))`` (each optionally wrapped in :class:`Neg`)
    whose ``(a, b, c)`` triples form a cyclic permutation, with each
    Hamiltonian vector field built over the same bivector. The three
    matched terms are stripped from the Sum and replaced with a single
    ``BracketApply(sn, π, π)``, the inert SN self-bracket node that
    the Faz 9 Stage B machinery treats as the universal Poisson
    obstruction.

    Polarity handling. The Koszul-signed cyclic Jacobi obstruction on
    functions of shifted SN-degree ``-1`` enters the chain with each
    cyclic term wrapped in :class:`Neg`. Three matched terms must
    share the same polarity (all positive or all negated); the emitted
    SN handle inherits that sign. Mixed-polarity triples decline.

    Parameters
    ----------
    bivector
        The Poisson bivector ``π``. Each matched
        :class:`HamiltonianVectorField` must report ``bivector == π``
        for this same instance, otherwise the rule declines.
    sn_bracket
        Optional override for the SN bracket used in the emitted
        :class:`BracketApply`. Defaults to
        :data:`jacopy.brackets.schouten.sn`.

    Raises
    ------
    TypeError
        If ``bivector`` is not an :class:`Expr`.
    """

    name = "[π,π]_SN function-level formula"

    def __init__(
        self,
        bivector: Expr,
        *,
        sn_bracket=None,
    ) -> None:
        if not isinstance(bivector, Expr):
            raise TypeError(
                "HamiltonianCyclicSnFormulaDefinition requires an Expr bivector"
            )
        self._bivector = bivector
        self._sn = sn_bracket if sn_bracket is not None else default_sn

    def matches(self, expr: Expr) -> bool:
        return (
            isinstance(expr, Sum)
            and self._find_cyclic_triple(expr) is not None
        )

    def rewrite(self, expr: Expr) -> Expr:
        match = self._find_cyclic_triple(expr)
        assert match is not None, "matches() guarantees a triple"
        i1, i2, i3, sign = match
        idxs = {i1, i2, i3}
        kept = [c for k, c in enumerate(expr.children) if k not in idxs]
        sn_apply: Expr = BracketApply(self._sn, self._bivector, self._bivector)
        if sign == -1:
            sn_apply = Neg(sn_apply)
        if not kept:
            return sn_apply
        return Sum.make(sn_apply, *kept)

    def _peel_term(
        self, term: Expr
    ) -> Optional[Tuple[int, Expr, Expr, Expr]]:
        """Return ``(sign, a, b, c)`` for ``Act(X_a, Act(X_b, c))``,
        or its :class:`Neg`-wrapped variant. ``None`` if the shape
        doesn't match or either Hamiltonian's bivector differs."""
        sign = 1
        if isinstance(term, Neg):
            sign = -1
            term = term.arg
        if not isinstance(term, Act):
            return None
        outer = term.op
        inner_act = term.arg
        if not isinstance(outer, HamiltonianVectorField):
            return None
        if outer.bivector != self._bivector:
            return None
        if not isinstance(inner_act, Act):
            return None
        inner = inner_act.op
        if not isinstance(inner, HamiltonianVectorField):
            return None
        if inner.bivector != self._bivector:
            return None
        return sign, outer.function, inner.function, inner_act.arg

    def _find_cyclic_triple(
        self, sum_expr: Sum
    ) -> Optional[Tuple[int, int, int, int]]:
        """Locate three cyclically-permuted ``Act(X_a, Act(X_b, c))``
        terms with consistent polarity. Return
        ``(i1, i2, i3, sign)`` on success."""
        children = sum_expr.children
        peeled = [
            (i, self._peel_term(c)) for i, c in enumerate(children)
        ]
        candidates = [(i, p) for i, p in peeled if p is not None]
        if len(candidates) < 3:
            return None
        for combo in combinations(candidates, 3):
            (i1, p1), (i2, p2), (i3, p3) = combo
            s1, a1, b1, c1 = p1
            s2, a2, b2, c2 = p2
            s3, a3, b3, c3 = p3
            if s1 != s2 or s2 != s3:
                continue
            triple = ((a1, b1, c1), (a2, b2, c2), (a3, b3, c3))
            anchor = triple[0]
            A, B, C = anchor
            cyclic = {(A, B, C), (B, C, A), (C, A, B)}
            if set(triple) == cyclic:
                return (i1, i2, i3, s1)
        return None
