r"""
Multilinear evaluation node ``MultiEval(head, *args)``.

Represents the textbook expression ``ω(Y_1, …, Y_p)``, a ``p``-form
``ω`` evaluated against ``p`` vector fields, and its dual
``π(α, β)``, a bivector evaluated against two covectors. Both share
the same algebraic shape: a graded multilinear contract that swallows
``len(args)`` slots and produces a scalar (degree ``0``).

The node intentionally mirrors :class:`~jacopy.calculus.pairing.Pairing`
in spirit: it's an inert structural container. Antisymmetry, slot
linearity, and intrinsic-formula rewrites live in
:mod:`jacopy.calculus.multi_eval_axioms` as engine
:class:`~jacopy.proof.expansion.Definition`'s.

Slot semantics
--------------

* ``slot_kind="vector"``, args are vector fields; ``head`` is a form.
  Plain "form-on-vectors" evaluation, the rank-``p`` generalisation of
  :class:`Pairing`.
* ``slot_kind="covector"``, args are 1-forms; ``head`` is a
  bivector / multivector. Same antisymmetry, dual contract.

The slot kind is purely declarative, it documents the user's intent
and lets renderers decide on bracket conventions, but the structural
algebra (antisymmetry, linearity) is identical in both cases.

Antisymmetry
------------

When ``alternating=True`` (the default) the node is graded-antisymmetric
in its argument slots: swapping two args picks up a sign of ``-1``;
two identical args force the whole expression to ``0``. When
``alternating=False`` the slots are treated as plain multilinear with
no symmetry, useful for pedagogical demonstrations or for evaluation
contracts that aren't antisymmetric (rare; included for completeness).

Arity
-----

Arity validation is *not* enforced at construction. A user assembling
``MultiEval(omega, X, Y, Z)`` must themselves know whether ``omega`` is
a 3-form. The reason: ``head`` may be a compound expression like
``Act(d, omega)`` whose degree depends on a registry lookup, and we
don't want the structural Expr layer to take a registry dependency.
A separate helper :func:`validate_arity` exposes the check on demand.
"""

from __future__ import annotations

from typing import Any, Optional, Tuple

from jacopy.core.expr import Expr, Integer, Zero
from jacopy.core.registry import PropertyRegistry


_ALLOWED_SLOT_KINDS = ("vector", "covector")


class MultiEval(Expr):
    """Multilinear evaluation ``head(arg_1, …, arg_p)``.

    Parameters
    ----------
    head:
        The form or multivector being evaluated.
    *args:
        Vector fields (for ``slot_kind="vector"``) or 1-forms (for
        ``slot_kind="covector"``). At least one arg is required,
        a zero-arg "evaluation" is just ``head`` itself; constructing
        a :class:`MultiEval` with no args is a category error.
    alternating:
        When ``True`` (default), the node is graded-antisymmetric in
        its argument slots. The :mod:`jacopy.calculus.multi_eval_axioms`
        engine rules use this flag to decide whether repeat-arg
        zeroing fires.
    slot_kind:
        ``"vector"`` (default) when args are vector fields, or
        ``"covector"`` when args are 1-forms. Documentation only,
        does not affect structural algebra.
    """

    __slots__ = ("_head", "_args", "_alternating", "_slot_kind")

    def __init__(
        self,
        head: Expr,
        *args: Expr,
        alternating: bool = True,
        slot_kind: str = "vector",
    ) -> None:
        if not isinstance(head, Expr):
            raise TypeError("MultiEval head must be an Expr")
        if not args:
            raise ValueError(
                "MultiEval requires at least one argument; a zero-arg "
                "evaluation is just the head itself"
            )
        for a in args:
            if not isinstance(a, Expr):
                raise TypeError("MultiEval arguments must be Expr")
        if not isinstance(alternating, bool):
            raise TypeError("alternating must be a bool")
        if slot_kind not in _ALLOWED_SLOT_KINDS:
            raise ValueError(
                f"slot_kind must be one of {_ALLOWED_SLOT_KINDS}, "
                f"got {slot_kind!r}"
            )
        self._head = head
        self._args = tuple(args)
        self._alternating = alternating
        self._slot_kind = slot_kind

    # ---- accessors -------------------------------------------------- #

    @property
    def head(self) -> Expr:
        return self._head

    @property
    def args(self) -> Tuple[Expr, ...]:
        return self._args

    @property
    def arity(self) -> int:
        return len(self._args)

    @property
    def alternating(self) -> bool:
        return self._alternating

    @property
    def slot_kind(self) -> str:
        return self._slot_kind

    # ---- Expr protocol ---------------------------------------------- #

    @property
    def children(self) -> Tuple[Expr, ...]:
        return (self._head,) + self._args

    def _rebuild(self, new_children: Tuple[Expr, ...]) -> "MultiEval":
        if len(new_children) < 2:
            raise ValueError(
                "MultiEval._rebuild needs head plus at least one arg"
            )
        head, *args = new_children
        return MultiEval(
            head,
            *args,
            alternating=self._alternating,
            slot_kind=self._slot_kind,
        )

    def _key(self) -> Any:
        return (self._head, self._args, self._alternating, self._slot_kind)

    def _repr_inner(self) -> str:
        head = self._head._repr_inner()
        arglist = ", ".join(a._repr_inner() for a in self._args)
        return f"{head}({arglist})"

    # ---- ergonomic constructors ------------------------------------- #

    def with_args(self, *new_args: Expr) -> "MultiEval":
        """Return a copy with the same head/flags but new argument tuple."""
        return MultiEval(
            self._head,
            *new_args,
            alternating=self._alternating,
            slot_kind=self._slot_kind,
        )

    def swapped(self, i: int, j: int) -> Tuple["MultiEval", int]:
        """Return ``(new_node, sign)`` for a swap of args ``i`` and ``j``.

        For ``alternating=True`` the sign is ``-1`` when ``i != j`` and
        ``+1`` when ``i == j``. For ``alternating=False`` the sign is
        always ``+1`` (slot-linear with no symmetry).

        Out-of-range indices raise :class:`IndexError`.
        """
        n = len(self._args)
        if not (0 <= i < n and 0 <= j < n):
            raise IndexError(
                f"MultiEval.swapped: indices ({i}, {j}) out of range for "
                f"arity {n}"
            )
        new_args = list(self._args)
        new_args[i], new_args[j] = new_args[j], new_args[i]
        sign = -1 if (self._alternating and i != j) else 1
        return self.with_args(*new_args), sign


# --------------------------------------------------------------------- #
# Helpers                                                                #
# --------------------------------------------------------------------- #


def multi_eval(
    head: Expr,
    *args: Expr,
    alternating: bool = True,
    slot_kind: str = "vector",
) -> MultiEval:
    """Functional constructor for :class:`MultiEval`.

    Mirrors :func:`jacopy.calculus.pairing.pairing`, call sites read
    ``multi_eval(omega, X, Y)`` rather than the capitalised class name.
    """
    return MultiEval(
        head,
        *args,
        alternating=alternating,
        slot_kind=slot_kind,
    )


def has_repeated_arg(expr: MultiEval) -> bool:
    """True when ``expr`` has two structurally equal arguments.

    Used by :class:`jacopy.calculus.multi_eval_axioms.MultiEvalRepeatArgZeroDefinition`
    and exposed publicly so user code can spot the same condition
    without re-importing the engine rule.
    """
    if not isinstance(expr, MultiEval):
        raise TypeError("has_repeated_arg: expected a MultiEval")
    seen: list[Expr] = []
    for a in expr.args:
        if any(a == s for s in seen):
            return True
        seen.append(a)
    return False


def validate_arity(
    expr: MultiEval,
    *,
    registry: Optional[PropertyRegistry] = None,
) -> Optional[int]:
    """Check that ``expr.arity`` matches the head's degree.

    Returns the head's degree when it can be resolved against
    ``registry`` (or in registry-free dispatch). Raises
    :class:`ValueError` when the head's degree is determinate and does
    not equal ``arity``. Returns ``None`` when the head's degree cannot
    be determined, callers in symbolic-degree contexts may then
    proceed without arity enforcement.
    """
    # Late import: ``degree_of`` lives in :mod:`jacopy.algebra.derivation`,
    # which already late-imports ``MultiEval`` (this module), so resolving
    # eagerly at module import would cycle.
    from jacopy.algebra.derivation import degree_of

    try:
        head_degree = degree_of(expr.head, registry)
    except ValueError:
        return None
    head_const = head_degree.as_int()
    if head_const is None:
        return None
    if head_const != expr.arity:
        raise ValueError(
            f"MultiEval arity mismatch: head {expr.head!r} has degree "
            f"{head_const}, but {expr.arity} arg(s) supplied"
        )
    return head_const
