r"""
BracketApply-side axioms, Q9 Stage 9.C.

Parallel of the Lie-bracket-of-vector-fields closure family
(:mod:`jacopy.calculus.closure_axioms` plus the LBVF rules in
:mod:`jacopy.calculus.sn_function_axiom`) for an opaque
:class:`~jacopy.brackets.base.BracketApply` headed by a specific
:class:`~jacopy.brackets.base.GradedBracket` instance. Lets the
Bianchi-identity engine close ``cycl R̃(α, β) γ`` for a connection whose
"vector bracket" is the Koszul bracket on T*M (or any other
graded-antisymmetric, Jacobi-satisfying graded bracket) without
collapsing it to ``LieBracketVF`` first.

Five rules, mirroring the LBVF set:

* :class:`BracketApplySumLinearityDefinition`, Sum distribution through
  either bracket slot.
* :class:`BracketApplyNegLinearityDefinition`, Neg pull-out through
  either bracket slot.
* :class:`BracketApplyArgAntisymmetryDefinition`, atom-level
  arg-canonicalization ``[b, a] → −[a, b]`` via deterministic ``repr``
  ordering. Gated on ``is_graded_antisymmetric=True`` and degree-0
  (the only case the sign collapses cleanly without graded parity
  tracking).
* :class:`BracketApplyAntiSymmetryDefinition`, Sum-level pair
  cancellation ``[a, b] + [b, a] → 0`` inside a structurally-shared
  wrapper. Same gate.
* :class:`BracketApplyJacobiDefinition`, Sum-level cyclic-triple finder
  for graded Jacobi at degree 0; same shape as the LBVF Jacobi rule but
  scoped to a specific bracket head.

Each rule scopes on a single :class:`GradedBracket` instance, two
coexisting brackets in a proof don't cross-fire, mirroring the
``Sharp``-scoped rules in :mod:`sn_function_axiom`.
"""

from __future__ import annotations

from itertools import combinations, permutations
from typing import Optional, Tuple

from jacopy.brackets.base import BracketApply, GradedBracket
from jacopy.calculus.closure_axioms import (
    _BARE_BRACKET_KEY,
    _BRACKET_PLACEHOLDER,
    _SubstituteSentinel,
    _strip_neg,
)
from jacopy.core.expr import Expr, Integer, Neg, Sum
from jacopy.proof.expansion import Definition


# --------------------------------------------------------------------- #
# Linearity                                                              #
# --------------------------------------------------------------------- #


class BracketApplySumLinearityDefinition(Definition):
    r"""``[Sum(a, b, …), z]_B → Σ [a_i, z]_B`` (and right-slot mirror).

    Parametric on a :class:`GradedBracket` so two coexisting brackets
    (e.g. Koszul vs SN) keep distinct distributors. Scoped match: the
    rule fires only when ``expr.bracket == self._bracket``.
    """

    def __init__(self, bracket: GradedBracket) -> None:
        if not isinstance(bracket, GradedBracket):
            raise TypeError(
                "BracketApplySumLinearityDefinition requires a GradedBracket"
            )
        self._bracket = bracket
        self.name = f"[Sum(a,…), z]_{bracket.name} → Σ [a_i, z]"

    def matches(self, expr: Expr) -> bool:
        if not isinstance(expr, BracketApply):
            return False
        if expr.bracket != self._bracket:
            return False
        return isinstance(expr.a, Sum) or isinstance(expr.b, Sum)

    def rewrite(self, expr: Expr) -> Expr:
        a, b = expr.a, expr.b
        if isinstance(a, Sum):
            return Sum.make(
                *(BracketApply(self._bracket, c, b) for c in a.children)
            )
        return Sum.make(
            *(BracketApply(self._bracket, a, c) for c in b.children)
        )


class BracketApplyNegLinearityDefinition(Definition):
    r"""``[Neg(a), b]_B / [a, Neg(b)]_B → Neg([a, b]_B)``.

    Pushes a ``Neg`` out of either bracket slot so downstream
    pair-finders see the canonical un-negated bracket. Scoped to one
    :class:`GradedBracket`. Sign-stable when both slots are negated:
    the two flips cancel and the result is the bare bracket.
    """

    def __init__(self, bracket: GradedBracket) -> None:
        if not isinstance(bracket, GradedBracket):
            raise TypeError(
                "BracketApplyNegLinearityDefinition requires a GradedBracket"
            )
        self._bracket = bracket
        self.name = f"[(−a), b]_{bracket.name} → −[a, b]"

    def matches(self, expr: Expr) -> bool:
        if not isinstance(expr, BracketApply):
            return False
        if expr.bracket != self._bracket:
            return False
        return isinstance(expr.a, Neg) or isinstance(expr.b, Neg)

    def rewrite(self, expr: Expr) -> Expr:
        a, b = expr.a, expr.b
        sign_neg = False
        if isinstance(a, Neg):
            a = a.arg
            sign_neg = not sign_neg
        if isinstance(b, Neg):
            b = b.arg
            sign_neg = not sign_neg
        new = BracketApply(self._bracket, a, b)
        return Neg(new) if sign_neg else new


# --------------------------------------------------------------------- #
# Atom-level arg-antisymmetry                                            #
# --------------------------------------------------------------------- #


def _require_graded_antisym(bracket: GradedBracket, ctx: str) -> None:
    """Guard for the antisym / Jacobi rules.

    All three rules require ``is_graded_antisymmetric=True``, without
    it, the swap and Jacobi-term sign hooks return ``None`` and the
    rules can't fire safely.

    The bracket's :meth:`~GradedBracket.pair_swap_sign` and
    :meth:`~GradedBracket.jacobi_term_sign` hooks supply the actual
    sign convention; the default implementations encode literal
    antisymmetry (``[a, b] = −[b, a]``) and the literal cyclic Jacobi
    (no per-term Koszul factor), which is what existing degree-0
    brackets (Koszul on 1-forms, Lie bracket on TM) want. Higher-degree
    or fully-graded brackets override the hooks; no degree-based
    gate is needed here.
    """
    if not bracket.is_graded_antisymmetric:
        raise ValueError(
            f"{ctx} requires a graded-antisymmetric bracket; "
            f"{bracket.name} declares is_graded_antisymmetric=False"
        )


class BracketApplyArgAntisymmetryDefinition(Definition):
    r"""``[b, a]_B → swap_sign · [a, b]_B`` for ``repr(b) > repr(a)``.

    Atom-level canonicalization: line up bracket arg order via
    deterministic ``repr`` ordering so two paths that produce the same
    bracket with swapped args end up structurally identical (modulo a
    sign the engine then folds via ``collect_terms``). The rewrite sign
    is supplied by :meth:`GradedBracket.pair_swap_sign`; the rule
    declines when the hook returns ``None`` (undecidable parity).

    Default behavior is literal antisymmetry (``swap_sign = -1``), so
    ``[b, a] → −[a, b]`` for the existing degree-0 brackets. A subclass
    overriding :meth:`~GradedBracket.pair_swap_sign` to encode full
    graded behavior gets the rewrite sign automatically.
    """

    def __init__(self, bracket: GradedBracket) -> None:
        if not isinstance(bracket, GradedBracket):
            raise TypeError(
                "BracketApplyArgAntisymmetryDefinition requires a GradedBracket"
            )
        _require_graded_antisym(
            bracket, "BracketApplyArgAntisymmetryDefinition"
        )
        self._bracket = bracket
        self.name = f"[b, a]_{bracket.name} → ±[a, b] [repr-canonical]"

    def matches(self, expr: Expr) -> bool:
        if not isinstance(expr, BracketApply):
            return False
        if expr.bracket != self._bracket:
            return False
        if expr.a == expr.b:
            return False
        if repr(expr.a) <= repr(expr.b):
            return False
        return self._bracket.pair_swap_sign(expr.b, expr.a) is not None

    def rewrite(self, expr: Expr) -> Expr:
        sign = self._bracket.pair_swap_sign(expr.b, expr.a)
        canonical = BracketApply(self._bracket, expr.b, expr.a)
        return Neg(canonical) if sign == -1 else canonical


# --------------------------------------------------------------------- #
# Wrapper extraction, parallel of closure_axioms._extract_…             #
# --------------------------------------------------------------------- #


def _extract_bracket_apply_with_wrapper(
    expr: Expr, target_bracket: GradedBracket
) -> Tuple[Optional[BracketApply], Optional[Expr]]:
    """Find a :class:`BracketApply` payload + structural wrapper key.

    Mirror of
    :func:`jacopy.calculus.closure_axioms._extract_bracket_with_wrapper`
    but matching :class:`BracketApply` whose ``.bracket`` equals
    ``target_bracket``. Returns ``(None, None)`` when no scoped bracket
    payload sits inside a recognized single-arg wrapper (Act,
    MultiEval with one bracket-bearing slot, ``ConnectionEvalExpr``
    with bracket in exactly one slot).

    Why a parallel function instead of a parametric refactor of the
    LBVF version: the two payload types have disjoint structural roles
    (LieBracketVF is a Derivation atom, BracketApply carries an
    explicit bracket head) and downstream consumers, the cyclic-triple
    finder, the antisym pair finder, also stay in their own
    bracket-specific lane. Sharing a walker would force a runtime
    discriminant on the payload type with no real reuse benefit.
    """
    from jacopy.algebra.derivation import Act
    from jacopy.calculus.connection import ConnectionEvalExpr
    from jacopy.core.multi_eval import MultiEval

    if isinstance(expr, BracketApply) and expr.bracket == target_bracket:
        return expr, _BARE_BRACKET_KEY

    if isinstance(expr, Act):
        op, arg = expr.op, expr.arg
        # Act head can be a BracketApply (e.g. ``Act([U,V]_K, f)`` after
        # an outer scalar commutator has been folded). We don't expect
        # this for Q9, connection rules emit BracketApply only inside
        # ConnectionEvalExpr.X, but keep the symmetry with the LBVF
        # walker.
        if isinstance(op, BracketApply) and op.bracket == target_bracket:
            wrapper = Act(_BRACKET_PLACEHOLDER, arg)
            return op, _compose_wrapper(wrapper, _BARE_BRACKET_KEY)
        inner_bracket, inner_key = _extract_bracket_apply_with_wrapper(
            arg, target_bracket
        )
        if inner_bracket is None:
            return None, None
        wrapper = Act(op, _BRACKET_PLACEHOLDER)
        return inner_bracket, _compose_wrapper(wrapper, inner_key)

    if isinstance(expr, MultiEval):
        bracket_slots = [
            i
            for i, a in enumerate(expr.args)
            if _extract_bracket_apply_with_wrapper(a, target_bracket)[0]
            is not None
        ]
        if len(bracket_slots) != 1:
            return None, None
        slot = bracket_slots[0]
        inner_bracket, inner_key = _extract_bracket_apply_with_wrapper(
            expr.args[slot], target_bracket
        )
        new_args = tuple(
            _BRACKET_PLACEHOLDER if i == slot else a
            for i, a in enumerate(expr.args)
        )
        wrapper = MultiEval(
            expr.head,
            *new_args,
            alternating=expr.alternating,
            slot_kind=expr.slot_kind,
        )
        return inner_bracket, _compose_wrapper(wrapper, inner_key)

    if isinstance(expr, ConnectionEvalExpr):
        x_bracket = _extract_bracket_apply_with_wrapper(
            expr.X, target_bracket
        )[0]
        y_bracket = _extract_bracket_apply_with_wrapper(
            expr.Y, target_bracket
        )[0]
        if (x_bracket is None) == (y_bracket is None):
            return None, None
        if x_bracket is not None:
            inner_bracket, inner_key = _extract_bracket_apply_with_wrapper(
                expr.X, target_bracket
            )
            wrapper = ConnectionEvalExpr(
                expr.connection, _BRACKET_PLACEHOLDER, expr.Y
            )
        else:
            inner_bracket, inner_key = _extract_bracket_apply_with_wrapper(
                expr.Y, target_bracket
            )
            wrapper = ConnectionEvalExpr(
                expr.connection, expr.X, _BRACKET_PLACEHOLDER
            )
        return inner_bracket, _compose_wrapper(wrapper, inner_key)

    return None, None


def _compose_wrapper(outer: Expr, inner_key: Optional[Expr]) -> Expr:
    """Same composition rule as the LBVF version, re-exported here.

    Splices the inner key tree back into the placeholder slot when the
    outer wrapper sits above a non-bare inner wrapper. Bare-bracket keys
    use the shared sentinel from
    :mod:`jacopy.calculus.closure_axioms` so the two wrapper schemes
    can never collide on the same residue.
    """
    if inner_key is _BARE_BRACKET_KEY:
        return outer
    return _SubstituteSentinel(outer, inner_key).build()


# --------------------------------------------------------------------- #
# Sum-level antisymmetry                                                 #
# --------------------------------------------------------------------- #


class BracketApplyAntiSymmetryDefinition(Definition):
    r"""Sum-level cancellation: ``[a, b]_B + s · [b, a]_B → 0`` (same wrapper).

    Mirror of
    :class:`~jacopy.calculus.closure_axioms.LieBracketVfAntiSymmetryDefinition`
    for opaque :class:`BracketApply` payloads scoped to one bracket.
    Two children of a Sum cancel when their wrapper-redacted shapes
    coincide, the underlying bracket pair is ``([a, b], [b, a])`` (with
    ``a != b``), and the outer Sum signs ``(s1, s2)`` satisfy
    ``s1 + s2 · swap_sign(b, a) = 0``, i.e. ``s2 = −s1 · swap_sign``.

    For the default literal-antisym convention (``swap_sign = −1``)
    that's just ``s1 == s2``: two outer-positive copies (or two
    outer-negative copies) cancel. For a graded-symmetric pair
    (``swap_sign = +1``, override-only) ``s2 = −s1`` cancels instead.
    """

    def __init__(self, bracket: GradedBracket) -> None:
        if not isinstance(bracket, GradedBracket):
            raise TypeError(
                "BracketApplyAntiSymmetryDefinition requires a GradedBracket"
            )
        _require_graded_antisym(
            bracket, "BracketApplyAntiSymmetryDefinition"
        )
        self._bracket = bracket
        self.name = f"[a, b]_{bracket.name} + s · [b, a]_{bracket.name} = 0"

    def matches(self, expr: Expr) -> bool:
        return isinstance(expr, Sum) and self._find_pair(expr) is not None

    def rewrite(self, expr: Expr) -> Expr:
        match = self._find_pair(expr)
        assert match is not None, "matches() guarantees a pair"
        i, j = match
        kept = [c for k, c in enumerate(expr.children) if k != i and k != j]
        if not kept:
            return Integer(0)
        if len(kept) == 1:
            return kept[0]
        return Sum.make(*kept)

    def _find_pair(self, sum_expr: Sum) -> Optional[Tuple[int, int]]:
        children = sum_expr.children
        candidates = []
        for idx, child in enumerate(children):
            outer_neg, inner = _strip_neg(child)
            outer_sign = -1 if outer_neg else 1
            bracket, wrapper_key = _extract_bracket_apply_with_wrapper(
                inner, self._bracket
            )
            if bracket is None:
                continue
            candidates.append((idx, outer_sign, wrapper_key, bracket))

        for (i1, s1, w1, b1), (i2, s2, w2, b2) in combinations(candidates, 2):
            if w1 != w2:
                continue
            if b1.a == b2.b and b1.b == b2.a and b1.a != b1.b:
                swap = self._bracket.pair_swap_sign(b1.b, b1.a)
                if swap is None:
                    continue
                if s1 + s2 * swap == 0:
                    return i1, i2
        return None


# --------------------------------------------------------------------- #
# Sum-level Jacobi                                                       #
# --------------------------------------------------------------------- #


def _peel_bracket_apply_jacobi(
    bracket: Expr, target_bracket: GradedBracket
) -> Tuple[Tuple[int, Expr, Expr, Expr], ...]:
    r"""Decompose a once-nested :class:`BracketApply` into ``[A, [B, C]]`` forms.

    Same enumeration pattern as
    :func:`jacopy.calculus.closure_axioms._peel_lie_bracket_jacobi`:
    returns ``(sign, A, B, C)`` tuples covering the four
    outer/inner-anti-symmetry orientations of the same nested bracket.
    Variant signs are sourced from
    :meth:`GradedBracket.pair_swap_sign` (for the inner swap on ``[B, C]``
    and the outer swap that flips ``[[B, C], A] → [A, [B, C]]``); for the
    default literal-antisym convention the swap sign is ``−1``,
    reproducing the original four (``+1, −1, −1, +1``) variant signs.
    Scoped on ``target_bracket``, both the outer and inner brackets
    must be the same :class:`GradedBracket` instance.
    """
    if not isinstance(bracket, BracketApply):
        return ()
    if bracket.bracket != target_bracket:
        return ()

    P, Q = bracket.a, bracket.b
    variants: list[Tuple[int, Expr, Expr, Expr]] = []

    if isinstance(Q, BracketApply) and Q.bracket == target_bracket:
        A, B, C = P, Q.a, Q.b
        inner_swap = target_bracket.pair_swap_sign(C, B)
        variants.append((+1, A, B, C))
        if inner_swap is not None:
            variants.append((inner_swap, A, C, B))

    if isinstance(P, BracketApply) and P.bracket == target_bracket:
        A, B, C = Q, P.a, P.b
        outer_swap = target_bracket.pair_swap_sign(P, A)
        if outer_swap is not None:
            variants.append((outer_swap, A, B, C))
            inner_swap = target_bracket.pair_swap_sign(C, B)
            if inner_swap is not None:
                variants.append((outer_swap * inner_swap, A, C, B))

    return tuple(variants)


class BracketApplyJacobiDefinition(Definition):
    r"""Cyclic Jacobi for a scoped :class:`BracketApply`.

    Sum-level analog of
    :class:`~jacopy.calculus.closure_axioms.LieBracketVfJacobiDefinition`.
    Recognises three children whose nested-bracket payloads form the
    cyclic triple ``[A,[B,C]] + [B,[C,A]] + [C,[A,B]] = 0``, in any
    sign-permuted algebraic guise (Leibniz form, negated Leibniz form,
    outer-anti-symmetrised cyclic form), all of which collapse to the
    same canonical ``(A, B, C)`` triple under the four-variant
    enumeration. The three children must share a structurally-identical
    single-arg wrapper.

    Gated on degree-0 graded-antisymmetric. The sign-flip enumeration
    inside :func:`_peel_bracket_apply_jacobi` is exactly the inner
    anti-symmetry rule, so applying it without ``is_graded_antisymmetric=True``
    would produce silently-incorrect cancellations.

    For brackets whose Jacobi is *conditional* (``KoszulBracket``
    declares ``satisfies_graded_jacobi=None``, leaving the truth on
    ``[π, π]_SN = 0``) the rule still fires, engaging the rule is the
    user's assertion that the Jacobi hypothesis holds in the proof
    context. Mirror of the LBVF Jacobi rule, which fires
    unconditionally even though Lie-bracket Jacobi is itself a theorem.
    """

    def __init__(self, bracket: GradedBracket) -> None:
        if not isinstance(bracket, GradedBracket):
            raise TypeError(
                "BracketApplyJacobiDefinition requires a GradedBracket"
            )
        _require_graded_antisym(
            bracket, "BracketApplyJacobiDefinition"
        )
        self._bracket = bracket
        self.name = f"Jacobi for [·,·]_{bracket.name}"

    def matches(self, expr: Expr) -> bool:
        return (
            isinstance(expr, Sum)
            and self._find_triple(expr) is not None
        )

    def rewrite(self, expr: Expr) -> Expr:
        match = self._find_triple(expr)
        assert match is not None, "matches() guarantees a triple"
        idxs = match
        kept = [c for k, c in enumerate(expr.children) if k not in idxs]
        if not kept:
            return Integer(0)
        if len(kept) == 1:
            return kept[0]
        return Sum.make(*kept)

    def _find_triple(self, sum_expr: Sum) -> Optional[Tuple[int, int, int]]:
        children = sum_expr.children

        candidates: list[
            Tuple[int, int, Expr, Tuple[Tuple[int, Expr, Expr, Expr], ...]]
        ] = []
        for idx, child in enumerate(children):
            outer_neg, inner = _strip_neg(child)
            outer_sign = -1 if outer_neg else 1
            bracket, wrapper_key = _extract_bracket_apply_with_wrapper(
                inner, self._bracket
            )
            if bracket is None:
                continue
            variants = _peel_bracket_apply_jacobi(bracket, self._bracket)
            if not variants:
                continue
            candidates.append((idx, outer_sign, wrapper_key, variants))

        if len(candidates) < 3:
            return None

        for trio in combinations(candidates, 3):
            (i1, s1, w1, v1), (i2, s2, w2, v2), (i3, s3, w3, v3) = trio
            if w1 != w2 or w2 != w3:
                continue
            for var1 in v1:
                for var2 in v2:
                    for var3 in v3:
                        if self._is_jacobi_match(
                            (s1, var1), (s2, var2), (s3, var3)
                        ):
                            return (i1, i2, i3)
        return None

    def _is_jacobi_match(
        self,
        c1: Tuple[int, Tuple[int, Expr, Expr, Expr]],
        c2: Tuple[int, Tuple[int, Expr, Expr, Expr]],
        c3: Tuple[int, Tuple[int, Expr, Expr, Expr]],
    ) -> bool:
        s1_outer, (s1_var, A1, B1, C1) = c1
        s2_outer, (s2_var, A2, B2, C2) = c2
        s3_outer, (s3_var, A3, B3, C3) = c3
        f1 = self._bracket.jacobi_term_sign(A1, B1, C1)
        f2 = self._bracket.jacobi_term_sign(A2, B2, C2)
        f3 = self._bracket.jacobi_term_sign(A3, B3, C3)
        if f1 is None or f2 is None or f3 is None:
            return False
        net1 = s1_outer * s1_var * f1
        net2 = s2_outer * s2_var * f2
        net3 = s3_outer * s3_var * f3
        if not (net1 == net2 == net3):
            return False
        triple = ((A1, B1, C1), (A2, B2, C2), (A3, B3, C3))
        for perm in permutations((A1, B1, C1)):
            P, Q, R = perm
            cyclic = {(P, Q, R), (Q, R, P), (R, P, Q)}
            if set(triple) == cyclic:
                return True
        return False
