"""
Index-slot pre-pass for §3.1.5 derivator identities (Faz 15.C).

Operator atoms in this codebase (``LieDerivative``, ``InteriorProduct``,
``CartanRemainder``, the four tilde ops) store their indexing field on a
private slot rather than as a child. The :class:`ExpansionEngine`'s
walk descends only into ``children`` so any rewrite shape buried inside
an index slot is unreachable. Section 3.1.5's identities reach for
heads like ``L_{K̃_η U}``, ``K_{K̃_μ U}``, etc., heads whose index
*is* a rewrite target, and bottom out without progress unless we open
the index up first.

This module ships two passes that, together, fully canonicalize index
slots before the :func:`prove_derivator_identity` driver wraps the LHS
/ RHS in a ``MultiEval``:

* :func:`expand_operator_indices`, walks ``expr`` post-order; for each
  recognised operator atom, runs the engine on its index slot, rebuilds
  the atom with the canonicalized index, and recurses to a fixed point.
* :func:`distribute_act_over_index_sums`, for each ``Act(Op, arg)``
  whose ``Op`` has a ``Sum`` (or ``Neg``) index, distributes outward to
  ``Sum(Act(Op_with_summand, arg), …)``. Indices are R-linear for every
  operator handled here, so distribution is a no-op semantically.

Both passes are scoped to §3.1.5; they are *not* a fix for the broader
opacity problem (see ``operator_atom_index_opacity.md`` memory note).

The intended call sequence is::

    expr = expand_operator_indices(expr, engine, registry)
    expr = distribute_act_over_index_sums(expr)
    expr = expand_operator_indices(expr, engine, registry)  # idempotent

so that distribution surfaces fresh atom-shaped indices that the engine
pass can chew on a second time.
"""

from __future__ import annotations

from typing import Optional, Tuple

from jacopy.algebra.derivation import Act
from jacopy.algebra.lie_bracket_vf import LieBracketVF
from jacopy.calculus.cartan_remainder import CartanRemainder
from jacopy.calculus.hamiltonian_vf import HamiltonianVectorField
from jacopy.calculus.interior import InteriorProduct
from jacopy.calculus.lie_derivative import LieDerivative
from jacopy.calculus.tilde.cartan_remainder import TildeCartanRemainder
from jacopy.calculus.tilde.operators import (
    TildeExteriorDerivative,
    TildeInteriorProduct,
    TildeLieDerivative,
)
from jacopy.core.expr import Expr, Neg, Sum
from jacopy.core.registry import PropertyRegistry
from jacopy.proof.expansion import Definition, ExpansionEngine


# --------------------------------------------------------------------- #
# Operator atom rebuild table                                            #
# --------------------------------------------------------------------- #


def _rebuild_op_with_index(op: Expr, new_index: Expr) -> Expr:
    """Return a fresh op atom of the same type with index replaced.

    Returns ``op`` unchanged if the operator class is not on the
    recognised list, callers should treat that as "no recursion needed
    here". Tilde ops keep their bivector slot; only the form slot is
    treated as the rewriteable index (the bivector is a fixed ``π``).
    """
    if isinstance(op, LieDerivative):
        return LieDerivative(
            new_index,
            definition=op.definition,
            d=op.d,
            iota_factory=op.iota_factory,
        )
    if isinstance(op, InteriorProduct):
        return InteriorProduct(new_index)
    if isinstance(op, CartanRemainder):
        return CartanRemainder(new_index)
    if isinstance(op, TildeLieDerivative):
        return TildeLieDerivative(new_index, op.bivector)
    if isinstance(op, TildeInteriorProduct):
        return TildeInteriorProduct(new_index)
    if isinstance(op, TildeCartanRemainder):
        return TildeCartanRemainder(new_index, op.bivector)
    if isinstance(op, HamiltonianVectorField):
        # Function-indexed; rebuild requires the existing bivector /
        # symplectic-form / sign attributes.
        return HamiltonianVectorField(
            new_index,
            bivector=op.bivector,
            symplectic_form=op.symplectic_form,
            sign=op.sign,
        )
    # TildeExteriorDerivative: index is bivector (π), not rewritten in
    # the §3.1.5 setting, so don't touch it. Same for any unknown op.
    return op


def _index_of(op: Expr) -> Optional[Expr]:
    """Return the rewriteable index slot, or ``None`` if the op has none."""
    if isinstance(op, (LieDerivative, InteriorProduct, CartanRemainder)):
        return op.vector_field
    if isinstance(op, TildeCartanRemainder):
        return op.form
    if isinstance(op, (TildeLieDerivative, TildeInteriorProduct)):
        return op.form
    if isinstance(op, HamiltonianVectorField):
        return op.function
    return None


def _open_bracket_atom(op: Expr) -> Optional[Tuple[Expr, Expr]]:
    """Return ``(X, Y)`` for a binary index-bearing atom, ``None`` otherwise."""
    if isinstance(op, LieBracketVF):
        return (op.X, op.Y)
    return None


def _rebuild_bracket_atom(op: Expr, new_X: Expr, new_Y: Expr) -> Expr:
    if isinstance(op, LieBracketVF):
        return LieBracketVF(new_X, new_Y)
    return op


# --------------------------------------------------------------------- #
# Pass 1, recursive index expansion                                     #
# --------------------------------------------------------------------- #


def expand_operator_indices(
    expr: Expr,
    engine: ExpansionEngine,
    registry: Optional[PropertyRegistry] = None,
) -> Expr:
    """Post-order walk; canonicalize each operator atom's index via ``engine``.

    Each recognised atom's index is first transformed recursively (so a
    nested ``Act(K_tilde, Act(K_tilde, V))`` index reduces innermost
    first), then run through ``engine.expand`` to apply the K̃/K
    defining axioms. The atom is rebuilt with the canonicalized index;
    ``op._key()`` is structural over the index so equality tracks the
    rewrite.

    Idempotent at the fixed point, re-applying it after distribution
    catches indices that newly surface as atoms.
    """
    if not isinstance(expr, Expr):
        raise TypeError("expand_operator_indices: expr must be an Expr")
    if not isinstance(engine, ExpansionEngine):
        raise TypeError("expand_operator_indices: engine must be an ExpansionEngine")

    # Recurse into children first (post-order).
    if expr.children:
        new_children = tuple(
            expand_operator_indices(c, engine, registry) for c in expr.children
        )
        if new_children != expr.children:
            expr = expr._rebuild(new_children)

    # Now if expr itself is an operator atom with an index, rewrite the
    # index. Recurse into the index too (it may itself contain operator
    # atoms whose indices need expansion).
    idx = _index_of(expr)
    if idx is not None:
        new_idx = expand_operator_indices(idx, engine, registry)
        canonical, _ = engine.expand(new_idx)
        if canonical != idx:
            expr = _rebuild_op_with_index(expr, canonical)

    # Binary atoms (LieBracketVF) carry two Expr slots; rewrite both.
    pair = _open_bracket_atom(expr)
    if pair is not None:
        X, Y = pair
        new_X = expand_operator_indices(X, engine, registry)
        canonical_X, _ = engine.expand(new_X)
        new_Y = expand_operator_indices(Y, engine, registry)
        canonical_Y, _ = engine.expand(new_Y)
        if canonical_X != X or canonical_Y != Y:
            expr = _rebuild_bracket_atom(expr, canonical_X, canonical_Y)

    return expr


# --------------------------------------------------------------------- #
# Pass 2, distribute Act over Sum/Neg index                             #
# --------------------------------------------------------------------- #


def _distribute_one(op: Expr, arg: Expr) -> Optional[Expr]:
    """If ``op``'s index is a ``Sum`` or ``Neg``, return the distributed shape.

    Returns ``None`` if no distribution applies (op not recognised, or
    index is already an atom-shaped Expr).
    """
    idx = _index_of(op)
    if idx is None:
        return None
    if isinstance(idx, Neg):
        op_inner = _rebuild_op_with_index(op, idx.arg)
        return Neg(Act(op_inner, arg))
    if isinstance(idx, Sum):
        terms = []
        for c in idx.children:
            if isinstance(c, Neg):
                op_inner = _rebuild_op_with_index(op, c.arg)
                terms.append(Neg(Act(op_inner, arg)))
            else:
                op_inner = _rebuild_op_with_index(op, c)
                terms.append(Act(op_inner, arg))
        return Sum.make(*terms)
    return None


def distribute_act_over_index_sums(expr: Expr) -> Expr:
    """Walk ``expr`` post-order; distribute ``Act(Op_{Sum/Neg}, arg)`` outward.

    For every recognised ``Act(Op, arg)`` whose Op's index is a Sum or
    Neg, splits into ``Sum(Act(Op_{i}, arg))`` (resp. ``Neg(...)``).
    Repeats until a fixed point, distribution surfaces fresh Acts that
    may themselves carry index Sums (e.g. a triple-nested chain).
    """
    if not isinstance(expr, Expr):
        raise TypeError("distribute_act_over_index_sums: expr must be an Expr")

    # Recurse into children first.
    if expr.children:
        new_children = tuple(
            distribute_act_over_index_sums(c) for c in expr.children
        )
        if new_children != expr.children:
            expr = expr._rebuild(new_children)

    if isinstance(expr, Act):
        distributed = _distribute_one(expr.op, expr.arg)
        if distributed is not None:
            # The distributed shape may itself contain undistributed
            # Acts (e.g. when the original index was nested), recurse.
            return distribute_act_over_index_sums(distributed)

    return expr


# --------------------------------------------------------------------- #
# Combined driver                                                        #
# --------------------------------------------------------------------- #


# --------------------------------------------------------------------- #
# Engine-time lift, open atom slots from inside the proof loop          #
# --------------------------------------------------------------------- #


class AtomSlotLiftDefinition(Definition):
    r"""Apply ``inner_engine`` to opaque atom slots inside ``Act`` heads.

    The :class:`ExpansionEngine`'s default walk descends only into
    :attr:`children` and skips Expr-typed slots stored on
    :class:`~jacopy.algebra.derivation.Derivation` atoms
    (e.g. ``LieDerivative.vector_field``,
    ``LieBracketVF.X / .Y``). When §3.1.5 expansion produces an
    intermediate ``Act(LieBracketVF(π^♯(df), Y), arg)`` the
    ``π^♯(df) → X_f`` rewrite never fires because the first slot is
    inside the atom.

    This rule lifts that wall: it matches any ``Act(op, arg)`` whose
    ``op`` is a recognised atom and whose slots are *not yet*
    fixed-points of ``inner_engine``. The rewrite re-runs the inner
    engine on each slot and rebuilds the atom with the canonicalized
    pieces, leaving the surrounding ``Act`` shape intact, so the main
    engine's bottom-up walk picks up where it left off.

    The ``inner_engine`` typically carries
    :class:`~jacopy.calculus.sharp_axioms.SharpOnExactDefinition`,
    :class:`~jacopy.calculus.sn_function_axiom.SnBracketOfFunctionDefinition`,
    and any K̃ / K-remainder rules whose presence is needed inside
    atom slots, but emphatically *not* itself, to avoid an infinite
    loop. A separate engine instance keeps the lift idempotent: once
    the inner engine reports no change, the lift's :meth:`matches`
    returns ``False`` and the surrounding loop terminates.

    Parameters
    ----------
    inner_engine
        An :class:`ExpansionEngine` carrying the rewrite rules to apply
        inside atom slots. Must not contain
        :class:`AtomSlotLiftDefinition` itself.
    """

    def __init__(self, inner_engine: ExpansionEngine) -> None:
        if not isinstance(inner_engine, ExpansionEngine):
            raise TypeError(
                "AtomSlotLiftDefinition requires an ExpansionEngine"
            )
        self._inner = inner_engine
        self.name = "atom-slot lift"

    def _slots(self, op: Expr):
        idx = _index_of(op)
        if idx is not None:
            yield ("idx", idx)
        pair = _open_bracket_atom(op)
        if pair is not None:
            yield ("X", pair[0])
            yield ("Y", pair[1])

    def matches(self, expr: Expr) -> bool:
        if not isinstance(expr, Act):
            return False
        op = expr.op
        for _, slot in self._slots(op):
            new, _ = self._inner.expand(slot)
            if new != slot:
                return True
        return False

    def rewrite(self, expr: Expr) -> Expr:
        op = expr.op
        idx = _index_of(op)
        if idx is not None:
            new_idx, _ = self._inner.expand(idx)
            new_op = _rebuild_op_with_index(op, new_idx)
            return Act(new_op, expr.arg)
        pair = _open_bracket_atom(op)
        if pair is not None:
            X, Y = pair
            new_X, _ = self._inner.expand(X)
            new_Y, _ = self._inner.expand(Y)
            new_op = _rebuild_bracket_atom(op, new_X, new_Y)
            return Act(new_op, expr.arg)
        return expr


class BareAtomSlotLiftDefinition(Definition):
    r"""Apply ``inner_engine`` to opaque atom slots when the atom appears bare.

    Sibling of :class:`AtomSlotLiftDefinition` for the case when an
    opaque atom (currently :class:`LieBracketVF`,
    :class:`HamiltonianVectorField`, or any operator atom carrying a
    rewriteable index) appears as a child of an outer node *other* than
    an :class:`Act` head, typically as an arg of a
    :class:`~jacopy.core.multi_eval.MultiEval`, or as a child of a
    :class:`Sum` / :class:`Neg` etc.

    The original :class:`AtomSlotLiftDefinition` only fires on
    ``Act(op, arg)``: when the atom *is* the operator, its slots are
    opened and the surrounding ``Act`` shape is preserved. But §3.1.5
    (1') residues produce ``MultiEval(ξ, LBVF(U, π^♯(K_V η)), …)`` where
    the LBVF is the arg, not the op, so the K_V η inside the LBVF.Y
    slot is unreachable to the standard engine walk *and* to the Act-
    shaped lift.

    This rule fires on the bare atom directly and rebuilds it with
    canonicalized slots; the engine's bottom-up walk reaches the atom
    via its ordinary child of MultiEval / Sum / Neg, so the rewrite
    propagates.
    """

    def __init__(self, inner_engine: ExpansionEngine) -> None:
        if not isinstance(inner_engine, ExpansionEngine):
            raise TypeError(
                "BareAtomSlotLiftDefinition requires an ExpansionEngine"
            )
        self._inner = inner_engine
        self.name = "bare-atom-slot lift"

    def _slots(self, expr: Expr):
        # LBVF, both X and Y slots.
        if isinstance(expr, LieBracketVF):
            yield ("X", expr.X)
            yield ("Y", expr.Y)
            return
        # Operator atoms with a rewriteable index slot, when they appear
        # bare (e.g. inside a Sum or alongside a Symbol). The index_of
        # helper already covers the indexable subset.
        idx = _index_of(expr)
        if idx is not None:
            yield ("idx", idx)

    def _rebuild(self, expr: Expr, new_slots) -> Expr:
        if isinstance(expr, LieBracketVF):
            return LieBracketVF(new_slots["X"], new_slots["Y"])
        if "idx" in new_slots:
            return _rebuild_op_with_index(expr, new_slots["idx"])
        return expr

    def matches(self, expr: Expr) -> bool:
        # Avoid double-firing through AtomSlotLiftDefinition: when the
        # atom IS an op of an outer Act, the Act-shaped lift owns it.
        # Here we only consider the atom in isolation.
        if not (
            isinstance(expr, LieBracketVF)
            or _index_of(expr) is not None
        ):
            return False
        for _, slot in self._slots(expr):
            new, _ = self._inner.expand(slot)
            if new != slot:
                return True
        return False

    def rewrite(self, expr: Expr) -> Expr:
        new_slots = {}
        for tag, slot in self._slots(expr):
            new, _ = self._inner.expand(slot)
            new_slots[tag] = new
        return self._rebuild(expr, new_slots)


def canonicalize_indices(
    expr: Expr,
    engine: ExpansionEngine,
    registry: Optional[PropertyRegistry] = None,
    *,
    max_passes: int = 4,
) -> Expr:
    """Run :func:`expand_operator_indices` and
    :func:`distribute_act_over_index_sums` in alternation until the
    expression stabilises (or ``max_passes`` rounds elapse).

    The default cap is generous for §3.1.5, the deepest identity
    bottoms out in two rounds, but not unbounded so a pathological
    input cannot loop forever.
    """
    prev = None
    out = expr
    for _ in range(max_passes):
        if out == prev:
            break
        prev = out
        out = expand_operator_indices(out, engine, registry)
        out = distribute_act_over_index_sums(out)
    return out
