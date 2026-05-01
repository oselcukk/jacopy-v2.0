r"""
Engine rules for :class:`~jacopy.core.indexed_sum.IndexedSum`, Faz 17.E.3-E.6.

The companion module to :mod:`jacopy.core.indexed_sum`. The Expr node
itself is purely structural; every meaningful manipulation lives here
as an :class:`~jacopy.proof.expansion.Definition`:

Sum / sign distribution (17.E.3)
    * :class:`IndexedSumSumDistributeDefinition`,
      :math:`\Sigma_d (X + Y) \to \Sigma_d X + \Sigma_d Y`.
    * :class:`IndexedSumNegPullDefinition`,
      :math:`\Sigma_d (-X) \to -\Sigma_d X`.
    * :class:`IndexedSumScalarPullDefinition`, pull every dummy-free
      factor out of a Product body.

Pairing pull-in (17.E.5)
    * :class:`IndexedSumPairingPushInRightDefinition`,
      :math:`\langle \alpha,\, \Sigma_d \mathrm{body}(d) \rangle \to
      \Sigma_d\langle \alpha, \mathrm{body}(d) \rangle` when ``α`` is
      dummy-free.
    * :class:`IndexedSumPairingPushInLeftDefinition`, symmetric
      left-slot variant for completeness; rarely used in practice.

Connection-eval push-in (17.F.1)
    * :class:`ConnectionEvalIndexedSumPushInDefinition`,
      :math:`\nabla_X (\Sigma_d \mathrm{body}(d)) \to
      \Sigma_d \nabla_X \mathrm{body}(d)` when ``X`` is dummy-free.
      Required by Cartan I/II reductions to expose Y-Leibniz inside
      the binder.

MultiEval push-in (17.F.2)
    * :class:`MultiEvalIndexedSumPushInDefinition`,
      :math:`\mathrm{MultiEval}(\Sigma_d \mathrm{body}(d), X_1, \ldots, X_p)
      \to \Sigma_d \mathrm{MultiEval}(\mathrm{body}(d), X_1, \ldots, X_p)`
      when every argument is dummy-free. Lifts the IndexedSum out of a
      MultiEval head so the wedge alternating expansion can fire on
      the body, needed by ``Σ_b (ω^a_b ∧ e^b)(U, V)`` reductions.

Kronecker contraction (17.E.6)
    * :class:`IndexedSumKroneckerContractDefinition`,
      :math:`\Sigma_d \delta(a, d)\cdot \mathrm{rest}(d) \to \mathrm{rest}(a)`
      when ``a`` is a free index. The proof-engine omurgası: every
      Cartan reduction terminates here.

The 17.E.4 wedge interaction is currently subsumed by the
:class:`IndexedSumScalarPullDefinition` since this codebase encodes
wedge products structurally via :class:`~jacopy.core.expr.Product`. A
dedicated wedge rule would be added later if a use case ever exposes
a wedge node with semantics distinct from non-commutative product.

Frame-decomposition axioms (17.E.7) are *not* in this module, they
pull a vector field or a covariant derivative into an :class:`IndexedSum`
shape, which is one-directional and shouldn't be auto-fired by the
engine. They live in :mod:`jacopy.calculus.frame_decomposition`.
"""

from __future__ import annotations

from typing import Optional

from jacopy.calculus.connection import AffineConnection, ConnectionEvalExpr
from jacopy.calculus.local_frame import FrameIndex, KroneckerDelta
from jacopy.calculus.pairing import Pairing
from jacopy.core.expr import Expr, Neg, One, Product, Sum
from jacopy.core.indexed_sum import IndexedSum, dummy_in
from jacopy.core.multi_eval import MultiEval
from jacopy.proof.expansion import Definition


# --------------------------------------------------------------------- #
# 17.E.3, Sum / Neg distribute over IndexedSum                          #
# --------------------------------------------------------------------- #


class IndexedSumSumDistributeDefinition(Definition):
    r""":math:`\Sigma_d (X + Y) \to \Sigma_d X + \Sigma_d Y`."""

    name = "Σ over Sum: distribute"

    def matches(self, expr: Expr) -> bool:
        return isinstance(expr, IndexedSum) and isinstance(expr.body, Sum)

    def rewrite(self, expr: Expr) -> Expr:
        assert isinstance(expr, IndexedSum)
        assert isinstance(expr.body, Sum)
        terms = tuple(
            IndexedSum(expr.dummy, expr.range_, c) for c in expr.body.children
        )
        return Sum.make(*terms)


class IndexedSumNegPullDefinition(Definition):
    r""":math:`\Sigma_d (-X) \to -\Sigma_d X`."""

    name = "Σ over Neg: pull out"

    def matches(self, expr: Expr) -> bool:
        return isinstance(expr, IndexedSum) and isinstance(expr.body, Neg)

    def rewrite(self, expr: Expr) -> Expr:
        assert isinstance(expr, IndexedSum)
        assert isinstance(expr.body, Neg)
        return Neg(IndexedSum(expr.dummy, expr.range_, expr.body.arg))


class IndexedSumScalarPullDefinition(Definition):
    r"""Pull every dummy-free factor out of a :class:`~jacopy.core.expr.Product` body.

    Fires on :math:`\Sigma_d (c_1 \cdots c_k \cdot X(d))` whenever at
    least one factor is dummy-free. The freed factors are gathered to
    the left of the residual sum:

    .. math::

        \Sigma_d (c \cdot X(d)) \to c \cdot \Sigma_d X(d)
        \quad (c\ \text{dummy-free}).

    All-or-nothing: the rule extracts every dummy-free factor in one
    rewrite, so the engine does not need to iterate over a partially-
    pulled state. If the body's dependent residue is empty (every
    factor was dummy-free) the rule still fires, leaving an
    :math:`\Sigma_d 1` shape that downstream rules may simplify. In
    practice a non-trivial Cartan body always has at least one
    dependent factor.
    """

    name = "Σ over Product: pull dummy-free factors out"

    def matches(self, expr: Expr) -> bool:
        if not isinstance(expr, IndexedSum):
            return False
        body = expr.body
        if not isinstance(body, Product):
            return False
        return any(not dummy_in(c, expr.dummy) for c in body.children)

    def rewrite(self, expr: Expr) -> Expr:
        assert isinstance(expr, IndexedSum)
        body = expr.body
        assert isinstance(body, Product)
        free: list[Expr] = []
        dep: list[Expr] = []
        for c in body.children:
            if dummy_in(c, expr.dummy):
                dep.append(c)
            else:
                free.append(c)
        inner_body = Product.make(*dep) if dep else One
        new_is = IndexedSum(expr.dummy, expr.range_, inner_body)
        return Product.make(*free, new_is)


# --------------------------------------------------------------------- #
# 17.E.5, Pairing push-in                                               #
# --------------------------------------------------------------------- #


class IndexedSumPairingPushInRightDefinition(Definition):
    r""":math:`\langle \alpha,\, \Sigma_d \mathrm{body}(d) \rangle \to
    \Sigma_d \langle \alpha, \mathrm{body}(d) \rangle`."""

    name = "⟨α, Σ_d body⟩ → Σ_d ⟨α, body⟩"

    def matches(self, expr: Expr) -> bool:
        if not isinstance(expr, Pairing):
            return False
        if not isinstance(expr.X, IndexedSum):
            return False
        return not dummy_in(expr.alpha, expr.X.dummy)

    def rewrite(self, expr: Expr) -> Expr:
        assert isinstance(expr, Pairing)
        is_node = expr.X
        assert isinstance(is_node, IndexedSum)
        new_body = Pairing(expr.alpha, is_node.body)
        return IndexedSum(is_node.dummy, is_node.range_, new_body)


class IndexedSumPairingPushInLeftDefinition(Definition):
    r""":math:`\langle \Sigma_d \mathrm{body}(d),\, X \rangle \to
    \Sigma_d \langle \mathrm{body}(d), X \rangle`."""

    name = "⟨Σ_d body, X⟩ → Σ_d ⟨body, X⟩"

    def matches(self, expr: Expr) -> bool:
        if not isinstance(expr, Pairing):
            return False
        if not isinstance(expr.alpha, IndexedSum):
            return False
        return not dummy_in(expr.X, expr.alpha.dummy)

    def rewrite(self, expr: Expr) -> Expr:
        assert isinstance(expr, Pairing)
        is_node = expr.alpha
        assert isinstance(is_node, IndexedSum)
        new_body = Pairing(is_node.body, expr.X)
        return IndexedSum(is_node.dummy, is_node.range_, new_body)


# --------------------------------------------------------------------- #
# 17.F.1, Connection-eval push-in over IndexedSum                       #
# --------------------------------------------------------------------- #


class ConnectionEvalIndexedSumPushInDefinition(Definition):
    r""":math:`\nabla_X (\Sigma_d \mathrm{body}(d)) \to
    \Sigma_d \nabla_X \mathrm{body}(d)` when ``X`` is dummy-free.

    Companion to :class:`IndexedSumPairingPushInRightDefinition` for
    the connection-eval shape that arises in Cartan I/II reductions:
    after frame-decomposing an outside vector field ``W = Σ_c e^c(W)·X_c``
    and pushing ``∇_U`` through the Σ, we still need this rule to
    actually move ``∇_U`` past the binder before Y-Leibniz can fire on
    the body.

    Bound to a specific :class:`AffineConnection` so the rewrite can be
    enabled per-connection without affecting unrelated derivations.
    """

    def __init__(self, connection: AffineConnection) -> None:
        if not isinstance(connection, AffineConnection):
            raise TypeError(
                "ConnectionEvalIndexedSumPushInDefinition requires an "
                "AffineConnection"
            )
        self._connection = connection
        self.name = (
            f"∇_X Σ_d body → Σ_d ∇_X body "
            f"[{connection._repr_inner()}]"
        )

    @property
    def connection(self) -> AffineConnection:
        return self._connection

    def matches(self, expr: Expr) -> bool:
        if not isinstance(expr, ConnectionEvalExpr):
            return False
        if expr.connection != self._connection:
            return False
        if not isinstance(expr.Y, IndexedSum):
            return False
        return not dummy_in(expr.X, expr.Y.dummy)

    def rewrite(self, expr: Expr) -> Expr:
        assert isinstance(expr, ConnectionEvalExpr)
        is_node = expr.Y
        assert isinstance(is_node, IndexedSum)
        new_body = ConnectionEvalExpr(self._connection, expr.X, is_node.body)
        return IndexedSum(is_node.dummy, is_node.range_, new_body)


# --------------------------------------------------------------------- #
# 17.F.2, MultiEval push-in over IndexedSum                             #
# --------------------------------------------------------------------- #


class MultiEvalIndexedSumPushInDefinition(Definition):
    r""":math:`\mathrm{MultiEval}(\Sigma_d \mathrm{body}(d), X_1, \ldots, X_p)
    \to \Sigma_d \mathrm{MultiEval}(\mathrm{body}(d), X_1, \ldots, X_p)`
    when every argument is dummy-free.

    Required by Cartan I/II reductions: the RHS shape
    :math:`\Sigma_b (\omega^a{}_b \wedge e^b)(U, V)` parses as
    ``MultiEval(IndexedSum(b, F, Wedge(ω^a_b, e^b)), U, V)``. To fire
    the wedge alternating expansion on the body, the
    :class:`~jacopy.core.indexed_sum.IndexedSum` binder must be lifted
    out of the :class:`~jacopy.core.multi_eval.MultiEval`'s head first.

    Loop-safe: after the rewrite, the IndexedSum sits *outside* the
    MultiEval, and the body's MultiEval has a non-IS head, so the
    rule cannot fire on its own output.

    The rewrite preserves both ``alternating`` and ``slot_kind`` flags
    so downstream wedge / pairing axioms see the original shape.
    """

    name = "MultiEval(Σ_d body, X_1, …, X_p) → Σ_d MultiEval(body, X_1, …, X_p)"

    def matches(self, expr: Expr) -> bool:
        if not isinstance(expr, MultiEval):
            return False
        if not isinstance(expr.head, IndexedSum):
            return False
        d = expr.head.dummy
        return all(not dummy_in(a, d) for a in expr.args)

    def rewrite(self, expr: Expr) -> Expr:
        assert isinstance(expr, MultiEval)
        is_node = expr.head
        assert isinstance(is_node, IndexedSum)
        new_body = MultiEval(
            is_node.body,
            *expr.args,
            alternating=expr.alternating,
            slot_kind=expr.slot_kind,
        )
        return IndexedSum(is_node.dummy, is_node.range_, new_body)


# --------------------------------------------------------------------- #
# 17.E.6, Kronecker contraction                                         #
# --------------------------------------------------------------------- #


def _find_contracting_partner(
    body: Expr, dummy: Expr
) -> Optional[FrameIndex]:
    """Return a free :class:`FrameIndex` ``a`` such that ``δ(a, dummy)``
    or ``δ(dummy, a)`` appears in ``body``, or :data:`None` if no such
    Kronecker delta is present.

    Respects :class:`IndexedSum` shadowing, when the walk meets an
    inner binder whose dummy matches ``dummy`` (by name + kind), the
    walk stops there.
    """
    if isinstance(body, KroneckerDelta):
        if body.i == dummy and isinstance(body.j, FrameIndex) and body.j.is_free:
            return body.j
        if body.j == dummy and isinstance(body.i, FrameIndex) and body.i.is_free:
            return body.i
        return None
    if isinstance(body, IndexedSum):
        if body._dummy == dummy:
            return None  # inner shadows outer; nothing to contract from here
        return _find_contracting_partner(body._body, dummy)
    if body.is_atom:
        return None
    for c in body.children:
        partner = _find_contracting_partner(c, dummy)
        if partner is not None:
            return partner
    return None


class IndexedSumKroneckerContractDefinition(Definition):
    r"""Kronecker contraction:

    .. math::

        \Sigma_d \delta(a, d)\cdot \mathrm{rest}(d) \to \mathrm{rest}(a),

    where ``a`` is any free :class:`FrameIndex`. The dummy is replaced
    by ``a`` throughout the body, and the freed
    :class:`KroneckerDelta` ``δ(a, a)`` collapses to
    :data:`~jacopy.core.expr.One` automatically (free-free same-name
    rule on KroneckerDelta).

    This is the proof-engine omurgası of Faz 17, every Cartan
    reduction terminates by removing an :class:`IndexedSum` via this
    rewrite.
    """

    name = "Σ over δ(a,d)·rest(d): contract"

    def matches(self, expr: Expr) -> bool:
        if not isinstance(expr, IndexedSum):
            return False
        return _find_contracting_partner(expr.body, expr.dummy) is not None

    def rewrite(self, expr: Expr) -> Expr:
        assert isinstance(expr, IndexedSum)
        partner = _find_contracting_partner(expr.body, expr.dummy)
        assert partner is not None
        return expr.substitute_dummy_with(partner)
