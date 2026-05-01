r"""
Closure axioms, Faz 12.A.6.

Two engine-level rewrite rules that close the three Cartan relations
that 12.A.4's intrinsic + canonicalize pipeline left open on a 1- or
2-form (``[d, L_X] = 0``, ``d² = 0``, ``[L_X, L_Y] = L_{[X,Y]_VF}``):

* :class:`VfActCommutatorDefinition`, the *generic* version of
  Faz 13.C's :class:`~jacopy.calculus.vf_axioms.OpCommutatorVfDefinition`.
  Whereas the latter fires only on
  ``Act(L_X, Act(L_Y, ω)) − Act(L_Y, Act(L_X, ω))``, this rule fires
  on any pair of bare-Derivation Acts:
  ``Act(X, Act(Y, f)) − Act(Y, Act(X, f)) → Act([X,Y]_VF, f)``.
  Why both rules: after the L-intrinsic axiom rewrites
  ``Act(L_X, ω)`` into ``X(ω(…))``-shaped towers, the residual
  commutator sits at the *vector-field-on-scalar* level, not at the
  Lie-derivative-on-form level. The L-specific rule no longer
  matches; this generic rule does.

* :class:`LieBracketVfJacobiDefinition`, Sum-level cyclic-triple
  finder for the unwrapped Lie-bracket Jacobi identity, recognising
  any of the algebraically-equivalent forms (cyclic
  ``[X,[Y,Z]] + [Y,[Z,X]] + [Z,[X,Y]] = 0``, the Leibniz form
  ``[X,[Y,Z]] − [Y,[X,Z]] − [[X,Y],Z] = 0``, and the outer-
  anti-symmetrised d²=0 form). The match enumerates the four
  outer/inner sign variants of each child so that the three
  permutations of ``(X, Y, Z)`` are recognised regardless of which
  algebraic form they happen to occupy in the Sum.

Both rules need ``product_rule`` to have run first, the residual
commutator pairs only surface as bare ``Act(X, Act(Y, f))`` once Act
has been distributed through Sum / Neg layers.
:func:`jacopy.calculus.intrinsic_engine.intrinsic_engine_with_closure`
returns a factory that bundles these two rules with the existing
seven, and
:func:`jacopy.calculus.intrinsic_engine.prove_intrinsic_equivalence`
already runs an expand → product_rule loop when the engine is built
that way.

Why a separate module: the rules belong conceptually to Faz 12.A.6
(intrinsic-formula closure) but reach into 13.C-level machinery
(LieBracketVF). Splitting them out keeps :mod:`vf_axioms` focused on
the original L-specific Faz 13 axioms and lets the closure pass live
beside the intrinsic axioms it completes.
"""

from __future__ import annotations

from itertools import combinations, permutations
from typing import Optional, Tuple

from jacopy.algebra.derivation import Act, Derivation
from jacopy.algebra.lie_bracket_vf import LieBracketVF
from jacopy.calculus.exterior_d import ExteriorDerivative
from jacopy.calculus.interior import InteriorProduct
from jacopy.calculus.lie_derivative import LieDerivative
from jacopy.core.expr import Expr, Integer, Neg, Sum
from jacopy.core.multi_eval import MultiEval
from jacopy.proof.expansion import Definition


# --------------------------------------------------------------------- #
# Helpers                                                                #
# --------------------------------------------------------------------- #


def _strip_neg(expr: Expr) -> Tuple[bool, Expr]:
    """Return ``(has_neg, inner)``, peeling a single :class:`Neg` if present."""
    if isinstance(expr, Neg):
        return True, expr.arg
    return False, expr


_RESERVED_OPS = (LieDerivative, InteriorProduct, ExteriorDerivative)


_TILDE_RESERVED_OPS: Optional[Tuple[type, ...]] = None


def _tilde_reserved_ops() -> Tuple[type, ...]:
    """Lazy lookup of the tilde Cartan-operator classes.

    Resolved on first use to avoid a top-level import cycle:
    :mod:`jacopy.calculus.tilde.intrinsic_engine` imports this module,
    so a top-level ``from .tilde.operators import …`` here would
    re-enter the partially-initialised package.
    """
    global _TILDE_RESERVED_OPS
    if _TILDE_RESERVED_OPS is None:
        from jacopy.calculus.tilde.operators import (
            TildeExteriorDerivative,
            TildeInteriorProduct,
            TildeLieDerivative,
        )

        _TILDE_RESERVED_OPS = (
            TildeLieDerivative,
            TildeInteriorProduct,
            TildeExteriorDerivative,
        )
    return _TILDE_RESERVED_OPS


def _is_plain_vf(op: Expr) -> bool:
    """``op`` is a vector-field-like atom, not a Cartan operator.

    The three Cartan operators (``L_X``, ``ι_X``, ``d``) have their
    own intrinsic axioms; the closure pass deliberately doesn't try
    to combine them via VF-commutator. Anything else with degree-0
    :class:`Derivation` typing, a plain vector field or a
    :class:`LieBracketVF` atom, is a valid commutator participant.

    Anchor-image extension: ``Act(Sharp(π), α)`` is also accepted as
    a plain VF. A ``Sharp(π)`` is a degree-``+1`` derivation that
    converts a 1-form into a vector field; the resulting compound
    ``Act(Sharp(π), α)`` is therefore a vector-field expression even
    though structurally it is an :class:`Act`, not a bare Derivation.
    Recognising it here lets
    :class:`VfActCommutatorDefinition` fold residues like
    ``π^♯(α)(π^♯(η)(f)) − π^♯(η)(π^♯(α)(f))`` into
    ``[π^♯(α), π^♯(η)]_VF(f)``, the bridge that
    :class:`~jacopy.calculus.tilde.closure_axioms.AnchorLieHomomorphismDefinition`
    then converts to ``π^♯([α, η]_K)(f)``.
    """
    from jacopy.calculus.musical import Sharp  # late import: musical → derivation

    if isinstance(op, Act) and isinstance(op.op, Sharp):
        return True
    if not isinstance(op, Derivation):
        return False
    if isinstance(op, _RESERVED_OPS):
        return False
    if isinstance(op, _tilde_reserved_ops()):
        return False
    return True


def _match_act_act(
    expr: Expr,
) -> Optional[Tuple[Derivation, Derivation, Expr]]:
    """Match ``Act(X, Act(Y, f))`` for plain VF X, Y; return ``(X, Y, f)``.

    Returns ``None`` if either layer is missing, the Acts wrap a
    Cartan operator, or X equals Y (the commutator vanishes trivially
    so there's nothing to fold).
    """
    if not isinstance(expr, Act):
        return None
    outer = expr.op
    inner_act = expr.arg
    if not isinstance(inner_act, Act):
        return None
    inner_op = inner_act.op
    if not (_is_plain_vf(outer) and _is_plain_vf(inner_op)):
        return None
    if outer == inner_op:
        return None
    return outer, inner_op, inner_act.arg


# --------------------------------------------------------------------- #
# Generic VF-commutator                                                  #
# --------------------------------------------------------------------- #


class VfActCommutatorDefinition(Definition):
    r"""Generic vector-field commutator: ``X(Y(f)) − Y(X(f)) → [X,Y]_VF(f)``.

    Sum-level pair finder. Scans children for a positive
    ``Act(X, Act(Y, f))`` and a negated mirror
    ``Neg(Act(Y, Act(X, f)))`` (in either positional order, and with
    either sign on either child as long as one is positive and one
    negative). The two children are removed and replaced with
    ``Act([X,Y]_VF, f)``.

    The rule deliberately excludes Cartan operators
    (:class:`LieDerivative`, :class:`InteriorProduct`,
    :class:`ExteriorDerivative`) so it doesn't compete with the
    intrinsic axioms. Faz 13.C's
    :class:`~jacopy.calculus.vf_axioms.OpCommutatorVfDefinition`
    handles the L-on-L commutator at the operator level; this rule
    handles the *post-intrinsic-expansion* level where the commutator
    has surfaced as plain vector-field Acts on a scalar.
    """

    name = "X(Y(f)) − Y(X(f)) = [X,Y]_VF(f)"

    def matches(self, expr: Expr) -> bool:
        return isinstance(expr, Sum) and self._find_pair(expr) is not None

    def rewrite(self, expr: Expr) -> Expr:
        match = self._find_pair(expr)
        assert match is not None, "matches() guarantees a pair"
        i, j, X, Y, f = match
        bracket = LieBracketVF(X, Y)
        new_term = Act(bracket, f)
        kept = [c for k, c in enumerate(expr.children) if k != i and k != j]
        return Sum.make(new_term, *kept)

    def _find_pair(
        self, sum_expr: Sum
    ) -> Optional[Tuple[int, int, Expr, Expr, Expr]]:
        """Return ``(i, j, X, Y, f)`` for the first cancelling pair, else ``None``.

        ``i`` is the index of the positive ``Act(X, Act(Y, f))`` and
        ``j`` of its negated mirror ``Neg(Act(Y, Act(X, f)))``.
        """
        children = sum_expr.children
        for i, j in combinations(range(len(children)), 2):
            for a, b in ((i, j), (j, i)):
                neg_a, inner_a = _strip_neg(children[a])
                neg_b, inner_b = _strip_neg(children[b])
                if neg_a or not neg_b:
                    continue
                pos = _match_act_act(inner_a)
                neg = _match_act_act(inner_b)
                if pos is None or neg is None:
                    continue
                X1, Y1, f1 = pos
                Y2, X2, f2 = neg
                if X1 == X2 and Y1 == Y2 and f1 == f2:
                    return a, b, X1, Y1, f1
        return None


# --------------------------------------------------------------------- #
# Bare ι_X(ω) used as a scalar, bridge to MultiEval                     #
# --------------------------------------------------------------------- #


class IotaActAsScalarDefinition(Definition):
    r"""Bridge bare ``Act(ι_X, ω)`` into ``MultiEval(ω, X)`` when used as a value.

    Fires on ``Act(D, Act(ι_X, ω))`` where ``D`` is a plain vector field
    (a :class:`Derivation` that is not one of the three Cartan
    operators). The outer ``Act(D, _)`` syntactically asserts that its
    argument is a 0-form scalar, which means ``ι_X(ω)`` must be a
    0-form, which in turn forces ``ω`` to be a 1-form, so the
    rewrite

    .. math::

        D\bigl(\iota_X \omega\bigr) \;\longrightarrow\; D\bigl(\omega(X)\bigr)

    is type-safe at exactly the residues where it triggers.

    Why this rule is needed: :class:`InteriorProductIntrinsicDefinition`
    only fires when ``Act(ι_X, ω)`` sits as the head of an enclosing
    :class:`MultiEval`. The arity-1 branch of
    :class:`ExteriorDIntrinsicDefinition` (``(d f)(Y) = Y(f)``)
    produces residues like ``Act(Y, Act(ι_X, ω))`` where the inner
    ``Act(ι_X, ω)`` is bare, no enclosing MultiEval, so the iota
    rule can't reach it. Without this bridge the 1-form Cartan magic
    relation ``(d ι_X + ι_X d) ω = L_X ω``, evaluated at a single
    vector field ``Y``, leaves the residue ``Y(ι_X ω) − Y(ω(X))``.

    The guard ``_is_plain_vf(D)`` is essential: it excludes the cases
    ``Act(d, Act(ι_X, ω))``, ``Act(L_Y, Act(ι_X, ω))``,
    ``Act(ι_Y, Act(ι_X, ω))``, none of which treat their argument as
    a scalar, so the rule never converts an honest operator
    composition into a malformed MultiEval. ``LieBracketVF`` is
    accepted (it acts as a vector field on scalars).
    """

    name = "bare ι_X(ω) inside Act(D, _) → Act(D, MultiEval(ω, X))"

    def matches(self, expr: Expr) -> bool:
        if not isinstance(expr, Act):
            return False
        if not _is_plain_vf(expr.op):
            return False
        inner = expr.arg
        return (
            isinstance(inner, Act)
            and isinstance(inner.op, InteriorProduct)
        )

    def rewrite(self, expr: Expr) -> Expr:
        outer_op = expr.op
        inner_act = expr.arg
        iota = inner_act.op
        omega = inner_act.arg
        return Act(
            outer_op,
            MultiEval(
                omega,
                iota.vector_field,
                alternating=True,
                slot_kind="vector",
            ),
        )


# --------------------------------------------------------------------- #
# Lie-bracket anti-symmetry                                              #
# --------------------------------------------------------------------- #


class LieBracketVfAntiSymmetryDefinition(Definition):
    r"""Sum-level cancellation: ``[X,Y]_VF(…) + [Y,X]_VF(…) → 0``.

    :class:`LieBracketVF` is opaque, ``LieBracketVF(X, Y)`` and
    ``LieBracketVF(Y, X)`` are distinct atoms with no built-in
    anti-symmetry. After the closure pipeline distributes operators
    through the Cartan obstructions of ``d² = 0`` and ``[d, L_X] = 0``
    on a 2-form, residues like
    ``[X,Y]_VF(ω(W, Z)) + [Y,X]_VF(ω(W, Z))`` show up; without this
    rule they sit forever even though they are algebraically zero.

    The rule scans a :class:`Sum`'s children for two terms whose
    "carrier" (the structurally-shared wrapper around a bracket
    payload, computed by
    :func:`_extract_bracket_with_wrapper`) coincides and whose
    bracket payloads are ``LieBracketVF(X, Y)`` versus
    ``LieBracketVF(Y, X)`` with opposite outer Sum signs (one positive,
    one negated, OR one positive in either order, anti-symmetry alone
    means they always cancel regardless of how the outer signs sit).
    Both children are removed.
    """

    name = "[X,Y]_VF + [Y,X]_VF = 0"

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
        """Find two children whose bracket payloads are ``[X,Y]`` and ``[Y,X]``.

        Wrapper structures must match (so ``ω(X, [Y,Z]) + ω(W, [Z,Y])``
        won't trigger, different positions). Outer signs must have
        opposite parity so the pair sums to zero rather than ``2[X,Y]``.
        """
        children = sum_expr.children
        candidates = []
        for idx, child in enumerate(children):
            outer_neg, inner = _strip_neg(child)
            outer_sign = -1 if outer_neg else 1
            bracket, wrapper_key = _extract_bracket_with_wrapper(inner)
            if not isinstance(bracket, LieBracketVF):
                continue
            candidates.append((idx, outer_sign, wrapper_key, bracket))

        for (i1, s1, w1, b1), (i2, s2, w2, b2) in combinations(candidates, 2):
            if w1 != w2:
                continue
            if b1.X == b2.Y and b1.Y == b2.X and b1.X != b1.Y:
                if s1 == s2:
                    return i1, i2
        return None


# --------------------------------------------------------------------- #
# Lie-bracket Jacobi                                                     #
# --------------------------------------------------------------------- #


def _peel_lie_bracket_jacobi(
    bracket: Expr,
) -> Tuple[Tuple[int, Expr, Expr, Expr], ...]:
    r"""Decompose a once-nested :class:`LieBracketVF` into ``[A, [B, C]]`` forms.

    The Lie-bracket Jacobi identity is most naturally stated on the
    ``[A, [B, C]]`` shape (outer head A, inner pair (B, C)). A bracket
    that physically reads ``[[P, Q], R]`` is equivalent to ``-[R, [P, Q]]``
    by outer anti-symmetry, and a `[A, [B, C]]` reading is equivalent
    to ``-[A, [C, B]]`` by inner anti-symmetry. This helper enumerates
    all four sign / orientation variants the same nested bracket can
    inhabit, giving downstream cyclic-triple matching the freedom to
    line up the residue's algebraic form (which depends on which
    Cartan relation produced it) without committing the rule to a
    single canonical normal form.

    Returns a tuple of ``(sign, A, B, C)`` such that the original
    bracket equals ``sign * [A, [B, C]]`` for each entry. An empty
    tuple is returned when ``bracket`` doesn't have the once-nested
    shape (any depth, any non-bracket children).
    """
    if not isinstance(bracket, LieBracketVF):
        return ()

    P, Q = bracket.X, bracket.Y

    # Form 1: bracket reads [P, [Q.X, Q.Y]], already in [A, [B, C]] shape.
    inner_right_variants: list[Tuple[int, Expr, Expr, Expr]] = []
    if isinstance(Q, LieBracketVF):
        A, B, C = P, Q.X, Q.Y
        inner_right_variants.append((+1, A, B, C))
        inner_right_variants.append((-1, A, C, B))  # inner anti-sym

    # Form 2: bracket reads [[P.X, P.Y], Q] = -[Q, [P.X, P.Y]].
    inner_left_variants: list[Tuple[int, Expr, Expr, Expr]] = []
    if isinstance(P, LieBracketVF):
        A, B, C = Q, P.X, P.Y
        inner_left_variants.append((-1, A, B, C))
        inner_left_variants.append((+1, A, C, B))  # inner anti-sym

    return tuple(inner_right_variants + inner_left_variants)


class LieBracketVfJacobiDefinition(Definition):
    r"""``[X,[Y,Z]_VF]_VF + [Y,[Z,X]_VF]_VF + [Z,[X,Y]_VF]_VF = 0`` (cyclic).

    Sum-level cyclic-triple finder. Looks for three children whose
    underlying nested-bracket payload, possibly inside an arbitrary
    structurally-shared single-arg wrapper, forms the Jacobi cyclic
    triple in any of its algebraically-equivalent sign-permuted guises.
    The three matched children are removed and the wrapped triple is
    replaced with ``0``.

    Why "any algebraically-equivalent guise": the three open Cartan
    relations produce Jacobi residues with quite different sign
    structures,

    * ``[d, L_X] = 0`` :    ``+ω([X,[Y,Z]]) − ω([Y,[X,Z]]) − ω([[X,Y],Z])``
                            (Leibniz form)
    * ``[L_X, L_Y] = L_{[X,Y]}`` :
                            ``−ω([X,[Y,Z]]) + ω([Y,[X,Z]]) + ω([[X,Y],Z])``
                            (negated Leibniz form)
    * ``d² = 0`` :          ``+ω([[X,Y],Z]) − ω([[X,Z],Y]) + ω([[Y,Z],X])``
                            (outer-anti-symmetrised cyclic form)

    All three are equivalent to ``[A,[B,C]] + [B,[C,A]] + [C,[A,B]]``
    after applying outer/inner Lie-bracket anti-symmetry. Rather than
    committing a residue to a single canonical form before matching,
    the rule enumerates the four ``(sign, A, B, C)`` variants of each
    child's bracket and looks for a triple whose tuples are the three
    cyclic permutations of one underlying ``(X, Y, Z)`` with all signs
    aligned (all the same sign after factoring in outer ``Neg``
    wrappers). That makes the rule single-shot for every Cartan
    relation; the residue's signs don't have to be massaged first.

    Wrapping: each child must have the form ``W(B)`` for some single
    ``W`` (the wrapper), where ``W`` is structurally identical across
    the three. A bare ``LieBracketVF`` (no wrapper) qualifies; so does
    ``MultiEval(ω, …)`` with a single bracket arg, or
    ``Act(L_op, ω)`` shapes. The "structurally identical" check is
    Expr equality on the wrapper after redacting the bracket payload,
    so different ω's or different surrounding scalars block the match
    by design.
    """

    name = "Lie-bracket Jacobi for vector fields"

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
        """Locate three Sum children that combine into a Jacobi cyclic zero.

        Each candidate child is reduced to its outer Sum-sign (``±1``)
        and its bracket payload (the deepest :class:`LieBracketVF`
        node found inside any single-arg wrapper). Three children
        qualify when there exist sign-aligned variants of their
        ``(A, B, C)`` decompositions that are exactly the three
        cyclic permutations of one underlying triple.
        """
        children = sum_expr.children

        # Per child: (outer_sign, wrapper_redacted, variants of bracket payload)
        candidates: list[Tuple[int, int, Expr, Tuple[Tuple[int, Expr, Expr, Expr], ...]]] = []
        for idx, child in enumerate(children):
            outer_neg, inner = _strip_neg(child)
            outer_sign = -1 if outer_neg else 1
            bracket, wrapper_key = _extract_bracket_with_wrapper(inner)
            if bracket is None:
                continue
            variants = _peel_lie_bracket_jacobi(bracket)
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

    @staticmethod
    def _is_jacobi_match(
        c1: Tuple[int, Tuple[int, Expr, Expr, Expr]],
        c2: Tuple[int, Tuple[int, Expr, Expr, Expr]],
        c3: Tuple[int, Tuple[int, Expr, Expr, Expr]],
    ) -> bool:
        """Three (sum-sign, (variant-sign, A, B, C)) tuples → cyclic Jacobi?

        After multiplying each child's outer Sum sign with its
        bracket-variant sign, the three nets must agree (all +1 or
        all −1), and the three (A, B, C) tuples must be the three
        cyclic permutations of one underlying ordered triple.
        """
        s1_outer, (s1_var, A1, B1, C1) = c1
        s2_outer, (s2_var, A2, B2, C2) = c2
        s3_outer, (s3_var, A3, B3, C3) = c3
        net1 = s1_outer * s1_var
        net2 = s2_outer * s2_var
        net3 = s3_outer * s3_var
        if not (net1 == net2 == net3):
            return False
        triple = ((A1, B1, C1), (A2, B2, C2), (A3, B3, C3))
        for perm in permutations((A1, B1, C1)):
            P, Q, R = perm
            cyclic = {(P, Q, R), (Q, R, P), (R, P, Q)}
            if set(triple) == cyclic:
                return True
        return False


def _extract_bracket_with_wrapper(
    expr: Expr,
) -> Tuple[Optional[LieBracketVF], Optional[Expr]]:
    """Find a single :class:`LieBracketVF` payload + a structural wrapper key.

    Walks one layer at a time through single-argument wrappers (Act,
    MultiEval with one arg) until a :class:`LieBracketVF` is hit, then
    returns the bracket along with a key that captures the surrounding
    wrapper for cross-child equality. Returns ``(None, None)`` for
    payloads that aren't wrapped brackets.

    The wrapper key is the original ``expr`` with the bracket node
    replaced by a sentinel placeholder, built via ``Expr._rebuild`` so
    that two children sharing the same wrapper compare equal under
    standard expression equality. Plain (no-wrapper) brackets hash to
    a uniform sentinel so they group together.
    """
    if isinstance(expr, LieBracketVF):
        return expr, _BARE_BRACKET_KEY

    # Single-layer unwrap: Act(op, arg). The bracket can sit either as
    # the operator (``Act([X,[Y,Z]]_VF, f)`` after VfActCommutator has
    # folded a scalar commutator) or inside the argument (``Act(L_X,
    # MultiEval(ω, …, [Y,Z]_VF, …))`` shapes); we test both.
    if isinstance(expr, Act):
        op, arg = expr.op, expr.arg
        if isinstance(op, LieBracketVF):
            wrapper = Act(_BRACKET_PLACEHOLDER, arg)
            return op, _compose_wrapper(wrapper, _BARE_BRACKET_KEY)
        inner_bracket, inner_key = _extract_bracket_with_wrapper(arg)
        if inner_bracket is None:
            return None, None
        wrapper = Act(op, _BRACKET_PLACEHOLDER)
        return inner_bracket, _compose_wrapper(wrapper, inner_key)

    from jacopy.core.multi_eval import MultiEval

    # Single bracket-bearing arg slot in an arbitrary-arity MultiEval.
    # ``ω(W, [[X,Y],Z])``, bracket in slot 1, W in slot 0; the wrapper
    # key carries (head, W, placeholder) so siblings with the same head
    # and the same fixed slot match. A residue with brackets in
    # *different* slots (``ω([X,Y], Z)`` vs ``ω(W, [X,Y])``) is rejected
    # by the wrapper-equality check downstream, same-slot is what
    # matters for cyclic-triple recognition.
    if isinstance(expr, MultiEval):
        bracket_slots = [
            i
            for i, a in enumerate(expr.args)
            if _extract_bracket_with_wrapper(a)[0] is not None
        ]
        if len(bracket_slots) != 1:
            return None, None
        slot = bracket_slots[0]
        inner_bracket, inner_key = _extract_bracket_with_wrapper(
            expr.args[slot]
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

    # Affine connection evaluation ``∇_X(Y)``, two-arg wrapper. The
    # bracket can sit in either slot; require exactly one to qualify so
    # cross-child wrapper-equality stays unambiguous (a residue with
    # brackets in different slots across siblings gets rejected here).
    # Imported locally to avoid a top-of-module dependency cycle:
    # connection.py imports nothing from closure_axioms, but this module
    # is imported by intrinsic_engine.py which connection.py does not
    # see, so a top-level import would still be safe, keeping it local
    # mirrors the MultiEval branch above and avoids hoisting calculus
    # imports earlier than they're needed.
    from jacopy.calculus.connection import ConnectionEvalExpr

    if isinstance(expr, ConnectionEvalExpr):
        x_bracket = _extract_bracket_with_wrapper(expr.X)[0]
        y_bracket = _extract_bracket_with_wrapper(expr.Y)[0]
        if (x_bracket is None) == (y_bracket is None):
            return None, None
        if x_bracket is not None:
            inner_bracket, inner_key = _extract_bracket_with_wrapper(expr.X)
            wrapper = ConnectionEvalExpr(
                expr.connection, _BRACKET_PLACEHOLDER, expr.Y
            )
        else:
            inner_bracket, inner_key = _extract_bracket_with_wrapper(expr.Y)
            wrapper = ConnectionEvalExpr(
                expr.connection, expr.X, _BRACKET_PLACEHOLDER
            )
        return inner_bracket, _compose_wrapper(wrapper, inner_key)

    return None, None


def _compose_wrapper(outer: Expr, inner_key: Optional[Expr]) -> Expr:
    """Combine outer wrapper with inner wrapper key for cross-child compare.

    The inner key already encodes whatever wrapping sat between this
    layer and the bracket; the outer layer wraps that key with its own
    operator / multi-eval shell. ``inner_key == _BARE_BRACKET_KEY``
    means the next inward layer was the bracket itself, in which case
    the composed wrapper is just ``outer``.
    """
    if inner_key is _BARE_BRACKET_KEY:
        return outer
    return _SubstituteSentinel(outer, inner_key).build()


# --------------------------------------------------------------------- #
# Sentinels for wrapper-equality                                         #
# --------------------------------------------------------------------- #
#
# ``_BRACKET_PLACEHOLDER`` is a unique :class:`Symbol` used to occupy the
# bracket's position when building a wrapper key, two wrappers compare
# equal iff their operator / multi-eval structure (excluding the bracket
# payload) coincides. ``_BARE_BRACKET_KEY`` is the sentinel used when no
# wrapper is present at all.

from jacopy.core.expr import Symbol  # noqa: E402  (intentional late import)

_BRACKET_PLACEHOLDER = Symbol("__bracket_payload_placeholder__")
_BARE_BRACKET_KEY = Symbol("__bare_bracket_no_wrapper__")


class _SubstituteSentinel:
    """Tiny helper to splice an inner-key tree into a placeholder slot.

    Walks ``outer`` looking for ``_BRACKET_PLACEHOLDER`` and rebuilds
    with ``inner_key`` inserted at that spot. The walker is one-shot
    and bottom-up; it doesn't memoise across calls. Only used during
    ``_extract_bracket_with_wrapper`` composition.
    """

    def __init__(self, outer: Expr, inner_key: Expr) -> None:
        self._outer = outer
        self._inner_key = inner_key

    def build(self) -> Expr:
        return self._walk(self._outer)

    def _walk(self, e: Expr) -> Expr:
        if e is _BRACKET_PLACEHOLDER:
            return self._inner_key
        if e.is_atom:
            return e
        new_children = tuple(self._walk(c) for c in e.children)
        return e._rebuild(new_children)
