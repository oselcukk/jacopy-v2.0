r"""
Engine rules for :class:`~jacopy.core.multi_eval.MultiEval`.

Three structural rewrite rules, together they make a ``MultiEval``
node behave the way a textbook ``ω(Y_1, …, Y_p)`` does:

* :class:`MultiEvalRepeatArgZeroDefinition`, alternating evaluation
  with two equal arguments collapses to ``0``.
* :class:`MultiEvalArgLinearityDefinition`, distributes
  :class:`~jacopy.core.expr.Sum` and
  :class:`~jacopy.core.expr.Neg` in any argument slot, producing the
  expected "expand-the-second-arg" rewrite step.
* :class:`MultiEvalHeadLinearityDefinition`, distributes the same
  Sum / Neg pattern in the *head* slot, so
  ``MultiEval(α + β, X)`` opens to ``MultiEval(α, X) + MultiEval(β, X)``.

These three are the rank-``p`` analogues of the linearity / antisymmetry
that :class:`~jacopy.calculus.pairing.Pairing` already enjoys at rank
``1``. Intrinsic-formula rewrites (``ι``, ``L``, ``d`` on multilinear
arguments) live in a separate module and build on top of these.
"""

from __future__ import annotations

from jacopy.core.expr import Expr, Integer, Neg, Sum, Zero
from jacopy.core.multi_eval import MultiEval, has_repeated_arg
from jacopy.proof.expansion import Definition


def _has_distributable_slot(args: tuple) -> int:
    """Return the index of the first ``Sum``/``Neg`` arg, or ``-1``.

    A separate helper because both arg-linearity and head-linearity want
    to scan their own list of children for the same shape.
    """
    for i, a in enumerate(args):
        if isinstance(a, (Sum, Neg)):
            return i
    return -1


def _distribute_in_slot(
    parent: MultiEval, slot_index: int, target: str
) -> Expr:
    """Distribute a ``Sum``/``Neg`` sitting at ``slot_index``.

    ``target`` selects whether the slot is the head (index ``-1``
    semantically, but represented by ``"head"``) or one of the args
    (``"arg"``). The two cases are structurally identical apart from
    which child gets the distributing children copied over.
    """
    if target == "head":
        focus = parent.head
        rebuild = lambda new: parent.__class__(
            new,
            *parent.args,
            alternating=parent.alternating,
            slot_kind=parent.slot_kind,
        )
    elif target == "arg":
        focus = parent.args[slot_index]
        def rebuild(new: Expr) -> MultiEval:
            new_args = list(parent.args)
            new_args[slot_index] = new
            return parent.__class__(
                parent.head,
                *new_args,
                alternating=parent.alternating,
                slot_kind=parent.slot_kind,
            )
    else:  # pragma: no cover - guarded by callers
        raise ValueError(f"unknown distribution target {target!r}")

    if isinstance(focus, Neg):
        return Neg(rebuild(focus.arg))
    # Sum: distribute, preserving Neg-children.
    terms = []
    for c in focus.children:
        if isinstance(c, Neg):
            terms.append(Neg(rebuild(c.arg)))
        else:
            terms.append(rebuild(c))
    return Sum.make(*terms)


# --------------------------------------------------------------------- #
# Repeat-arg → 0                                                         #
# --------------------------------------------------------------------- #


class MultiEvalRepeatArgZeroDefinition(Definition):
    """``ω(…, X, …, X, …) = 0`` for an alternating multilinear node.

    Fires only when ``alternating=True``; the symmetric / unflagged
    case is left untouched (a future symmetric-bivector pass would add
    its own rule).
    """

    name = "MultiEval repeat-arg → 0 (alternating)"

    def matches(self, expr: Expr) -> bool:
        return (
            isinstance(expr, MultiEval)
            and expr.alternating
            and has_repeated_arg(expr)
        )

    def rewrite(self, expr: Expr) -> Expr:
        return Zero


# --------------------------------------------------------------------- #
# Zero-head → 0                                                          #
# --------------------------------------------------------------------- #


class MultiEvalZeroHeadDefinition(Definition):
    """``0(X_1, …, X_p) → 0``, multilinear evaluation of the zero head.

    Universal multilinearity collapses ``MultiEval(Integer(0), …)`` to
    ``Integer(0)`` regardless of ``alternating`` flag or slot kind. Lets
    a residue from upstream cancellation (e.g. a ``Sum`` whose terms
    canceled to 0 then percolated up as a head) reduce instead of
    surviving as ``0(X_1, …, X_p)``.
    """

    name = "MultiEval zero-head → 0"

    def matches(self, expr: Expr) -> bool:
        return isinstance(expr, MultiEval) and expr.head == Zero

    def rewrite(self, expr: Expr) -> Expr:
        return Zero


# --------------------------------------------------------------------- #
# Linearity in arg slots                                                 #
# --------------------------------------------------------------------- #


class MultiEvalArgLinearityDefinition(Definition):
    """``ω(…, a + b, …) → ω(…, a, …) + ω(…, b, …)`` (and ``Neg``).

    Distributes a single ``Sum``/``Neg`` arg slot at a time; nested
    distribution (multiple Sum slots) unfolds across successive engine
    iterations.
    """

    name = "MultiEval arg slot linearity"

    def matches(self, expr: Expr) -> bool:
        return (
            isinstance(expr, MultiEval)
            and _has_distributable_slot(expr.args) >= 0
        )

    def rewrite(self, expr: Expr) -> Expr:
        i = _has_distributable_slot(expr.args)
        return _distribute_in_slot(expr, i, target="arg")


# --------------------------------------------------------------------- #
# Linearity in head slot                                                 #
# --------------------------------------------------------------------- #


class MultiEvalHeadLinearityDefinition(Definition):
    """``(α + β)(X_1, …, X_p) → α(X_1, …, X_p) + β(X_1, …, X_p)``.

    The head's R-linearity. ``Neg`` heads are likewise distributed:
    ``(-α)(X_1, …, X_p) → -(α(X_1, …, X_p))``.
    """

    name = "MultiEval head linearity"

    def matches(self, expr: Expr) -> bool:
        return isinstance(expr, MultiEval) and isinstance(
            expr.head, (Sum, Neg)
        )

    def rewrite(self, expr: Expr) -> Expr:
        return _distribute_in_slot(expr, slot_index=-1, target="head")


# --------------------------------------------------------------------- #
# Alternating canonicalization (bubble swap toward repr-sorted args)     #
# --------------------------------------------------------------------- #


class MultiEvalAlternatingNormalDefinition(Definition):
    r"""Bubble one out-of-order adjacent arg pair toward canonical order.

    When ``alternating=True`` and the args are not in lexicographic
    ``repr``-order, performs a *single* adjacent transposition on the
    leftmost out-of-order pair and wraps the result in a
    :class:`~jacopy.core.expr.Neg` so the antisymmetry sign is tracked
    explicitly. Engine iteration drives this to fix-point, bubble sort
    in disguise, at which point the arg tuple is sorted and the
    cumulative sign sits as a stack of nested ``Neg`` wrappers. A
    subsequent :func:`~jacopy.algorithms.simplify.simplify` pass
    collapses the ``Neg`` cascade into a single global ``±``.

    Why this matters: Cartan identities like
    ``(ι_X d + d ι_X)(ω) = L_X(ω)`` only close *modulo alternation* on
    a ``p ≥ 2`` form, the intrinsic Koszul / Cartan rewrites produce
    residuals like ``ω([X, Z]_VF, Y) + ω(Y, [X, Z]_VF)`` that vanish
    only after recognizing the swap-symmetry. This rule supplies that
    recognition as a sequence of one-swap engine steps, keeping the
    proof Cadabra-style explicit rather than buried inside a hidden
    canonicalize pass.

    Termination: each rewrite strictly decreases the inversion count
    of the arg tuple under ``repr``-order, so the engine fix-points in
    at most ``n*(n-1)/2`` swaps for arity ``n``. Equal-adjacent args
    never trigger a swap (``repr(a) > repr(a)`` is false); the
    :class:`MultiEvalRepeatArgZeroDefinition` rule handles that case
    independently.
    """

    name = "alternating canonicalize: bubble swap toward repr-sorted args"

    @staticmethod
    def _first_inversion(args: tuple) -> int:
        """Index ``i`` of the leftmost ``args[i] > args[i+1]`` pair, or ``-1``."""
        for i in range(len(args) - 1):
            if repr(args[i]) > repr(args[i + 1]):
                return i
        return -1

    def matches(self, expr: Expr) -> bool:
        return (
            isinstance(expr, MultiEval)
            and expr.alternating
            and self._first_inversion(expr.args) >= 0
        )

    def rewrite(self, expr: Expr) -> Expr:
        i = self._first_inversion(expr.args)
        new_args = list(expr.args)
        new_args[i], new_args[i + 1] = new_args[i + 1], new_args[i]
        swapped = MultiEval(
            expr.head,
            *new_args,
            alternating=expr.alternating,
            slot_kind=expr.slot_kind,
        )
        return Neg(swapped)
