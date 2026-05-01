r"""
Frame-decomposition axioms, Faz 17.E.7 + Faz 17.F.2.

Three **one-directional** rewrites that introduce an :class:`IndexedSum`
shape on the right-hand side. They are *opt-in*: not registered in any
default engine bundle, since pairing them with the inverse-direction
rules (Pairing duality, ``ConnectionFormDefinition``) creates a loop.
The Cartan structure proofs (:class:`~jacopy.library.cartan_structure.CartanStructureProblem`)
turn them on for the specific reduction sub-pass that needs them.

* :class:`FrameDecompositionDefinition`, ``W → Σ_a e^a(W)·X_a`` for
  any vector field ``W`` over a fixed local frame. Fires on every
  :class:`~jacopy.algebra.derivation.Derivation` instance whose frame
  identity is *not* the wrapper's own frame, i.e. it expands an
  outside-frame vector field into the basis of *this* frame. The dummy
  index is alpha-fresh on every match, so the engine's cache stays
  sane.

* :class:`ConnectionEvalYFrameDecompositionDefinition` (Faz 17.F.2),
  the *positional* counterpart of the previous rule. Decomposes only
  the ``Y`` slot of a :class:`~jacopy.calculus.connection.ConnectionEvalExpr`,
  rewriting

  .. math::

      \nabla_X Y \to \nabla_X \big(\Sigma_a \, e^a(Y)\cdot X_a\big)

  when ``Y`` is a non-frame :class:`~jacopy.algebra.derivation.Derivation`.
  Avoids the global-position loop concerns of
  :class:`FrameDecompositionDefinition` (which fires on every
  Derivation, including ``X`` of the same connection-eval, ``U`` of
  ``Act(U, …)``, etc.), needed by Cartan I/II reductions where
  :class:`FramePairingDualityDefinition` is also in the bundle.

* :class:`ConnectionFormDecompositionDefinition`, the special-case
  rewrite the Cartan-structure proof actually needs:

  .. math::

      \nabla_V X_b \to \Sigma_c \,\omega^c{}_b(\nabla)(V)\cdot X_c.

  Fires on a :class:`~jacopy.calculus.connection.ConnectionEvalExpr`
  whose ``Y`` slot is a :class:`~jacopy.calculus.local_frame.FrameVectorField`
  belonging to *this* rule's frame, with a *free* index ``b``. The
  body uses :class:`~jacopy.calculus.cartan_forms.ConnectionForm`
  paired with ``V`` so the residue is no longer a
  :class:`ConnectionEvalExpr`, adding
  :class:`ConnectionFormDefinition` to the same bundle would loop.

The three rules cover the three distinct shapes Cartan I/II reductions
produce:

1. ``e^a(∇_U V)`` after frame-expanding ``V`` collapses through
   :class:`FrameDecompositionDefinition` (or, in a Cartan-bundle
   context, through :class:`ConnectionEvalYFrameDecompositionDefinition`
   acting on the ``∇_U V`` shape directly).
2. ``∇_U(e^c(V)·X_c)`` after Y-Leibniz produces ``∇_U X_c`` which
   collapses through :class:`ConnectionFormDecompositionDefinition`.

After any rule fires, the standard 17.E linearity / contraction
rules carry the residue forward.
"""

from __future__ import annotations

from typing import Set

from jacopy.algebra.derivation import Derivation
from jacopy.calculus.cartan_forms import ConnectionForm
from jacopy.calculus.connection import AffineConnection, ConnectionEvalExpr
from jacopy.calculus.local_frame import (
    FrameIndex,
    FrameVectorField,
    LocalFrame,
)
from jacopy.calculus.pairing import Pairing
from jacopy.core.expr import Expr, Product
from jacopy.core.indexed_sum import IndexedSum
from jacopy.proof.expansion import Definition


# --------------------------------------------------------------------- #
# Fresh-bound-index minting                                             #
# --------------------------------------------------------------------- #


def _collect_index_names(expr: Expr, acc: Set[str]) -> None:
    """Walk ``expr`` and record every :class:`FrameIndex` name."""
    if isinstance(expr, FrameIndex):
        acc.add(expr.name)
        return
    if isinstance(expr, IndexedSum):
        if isinstance(expr.dummy, FrameIndex):
            acc.add(expr.dummy.name)
        _collect_index_names(expr.body, acc)
        return
    if expr.is_atom:
        # Frame-aware atoms hide a FrameIndex in private slots, pick
        # them up via the canonical accessors when available.
        for slot in ("idx", "upper", "lower", "lower_a", "lower_b"):
            v = getattr(expr, slot, None)
            if isinstance(v, FrameIndex):
                acc.add(v.name)
        return
    for c in expr.children:
        _collect_index_names(c, acc)


def _fresh_bound_index(prefix: str, expr: Expr) -> FrameIndex:
    """Mint a bound :class:`FrameIndex` whose name does not collide with
    any free or bound index already present in ``expr``."""
    used: Set[str] = set()
    _collect_index_names(expr, used)
    if prefix not in used:
        return FrameIndex(prefix, "bound")
    i = 1
    while f"{prefix}{i}" in used:
        i += 1
    return FrameIndex(f"{prefix}{i}", "bound")


# --------------------------------------------------------------------- #
# 17.E.7, FrameDecomposition                                           #
# --------------------------------------------------------------------- #


class FrameDecompositionDefinition(Definition):
    r"""Rewrite an outside-frame vector field into the local basis:

    .. math::

        W \to \Sigma_a \, e^a(W) \cdot X_a.

    The sum is over basis indices of the wrapper's :class:`LocalFrame`.
    The dummy ``a`` is alpha-fresh on every match.

    **Direction & loop avoidance.** This rule is one-directional, the
    inverse step ``Σ_a e^a(W)·X_a → W`` is not an engine rule.
    Pairing :class:`FramePairingDualityDefinition`
    (``e^a(X_b) → δ^a_b``) would unfold the residue back into a sum of
    Kronecker deltas, but it never reproduces the original ``W`` shape
    (which is an arbitrary outside-frame VF). So the loop concern only
    arises if a caller registers the rule on a VF that is *itself* a
    :class:`FrameVectorField` of this frame, the matcher excludes
    that case to keep the rule monotone.

    **Scope.** Match shape is "any :class:`Derivation` instance whose
    frame is not this wrapper's frame". A common pattern: a
    :class:`Symbol`-named hypothesis vector field ``V`` gets decomposed
    so the proof can apply ``Σ``-distribute / ``∇`` Y-Leibniz
    underneath. Frame VFs of the same frame are skipped to avoid the
    obvious tautology ``X_b → Σ_a δ^a_b · X_a``.
    """

    def __init__(self, frame: LocalFrame) -> None:
        if not isinstance(frame, LocalFrame):
            raise TypeError(
                "FrameDecompositionDefinition requires a LocalFrame"
            )
        self._frame = frame
        self.name = f"Frame decomposition [{frame.name}]"

    @property
    def frame(self) -> LocalFrame:
        return self._frame

    def matches(self, expr: Expr) -> bool:
        if not isinstance(expr, Derivation):
            return False
        if isinstance(expr, FrameVectorField) and expr.frame == self._frame:
            return False
        return True

    def rewrite(self, expr: Expr) -> Expr:
        assert isinstance(expr, Derivation)
        a = _fresh_bound_index("a", expr)
        body = Product.make(
            Pairing(self._frame.coframe(a), expr),
            self._frame.X(a),
        )
        return IndexedSum(a, self._frame, body)


# --------------------------------------------------------------------- #
# 17.F.2, ConnectionEvalYFrameDecomposition (positional)                #
# --------------------------------------------------------------------- #


class ConnectionEvalYFrameDecompositionDefinition(Definition):
    r"""Decompose only the ``Y`` slot of a
    :class:`~jacopy.calculus.connection.ConnectionEvalExpr`:

    .. math::

        \nabla_X Y \to \nabla_X \big(\Sigma_a \, e^a(Y) \cdot X_a\big),

    where ``Y`` is a non-frame
    :class:`~jacopy.algebra.derivation.Derivation`. The rewrite leaves
    ``∇_X`` outside the binder; downstream
    :class:`~jacopy.calculus.indexed_sum_axioms.ConnectionEvalIndexedSumPushInDefinition`
    plus :class:`~jacopy.calculus.connection.ConnectionYLeibnizDefinition`
    plus :class:`ConnectionFormDecompositionDefinition` carry the
    residue forward.

    **Why a separate positional rule?** :class:`FrameDecompositionDefinition`
    fires on *every* :class:`Derivation` in the expression, including
    ``X`` of the same :class:`ConnectionEvalExpr`, the bare ``U`` in
    ``Act(U, …)`` shapes, and so on. Bundling it with
    :class:`~jacopy.calculus.local_frame.FramePairingDualityDefinition`
    (``e^a(X_b) → δ^a_b``) creates loops on those positions. The
    positional version restricts the match to the ``Y`` slot of a
    specific connection's :class:`ConnectionEvalExpr`, leaving every
    other Derivation in the expression untouched.

    **Match guards.**

    * head is :class:`ConnectionEvalExpr` for *this* connection;
    * ``Y`` is a :class:`Derivation`;
    * ``Y`` is *not* a :class:`FrameVectorField` of this rule's frame
      (that case belongs to
      :class:`ConnectionFormDecompositionDefinition`).

    **Loop safety.** After the rewrite, ``Y`` is an
    :class:`~jacopy.core.indexed_sum.IndexedSum`, not a
    :class:`Derivation`, so the matcher does not re-fire on its own
    output.
    """

    def __init__(
        self, connection: AffineConnection, frame: LocalFrame
    ) -> None:
        if not isinstance(connection, AffineConnection):
            raise TypeError(
                "ConnectionEvalYFrameDecompositionDefinition requires an "
                "AffineConnection"
            )
        if not isinstance(frame, LocalFrame):
            raise TypeError(
                "ConnectionEvalYFrameDecompositionDefinition requires a "
                "LocalFrame"
            )
        self._connection = connection
        self._frame = frame
        self.name = (
            f"∇_X Y → ∇_X (Σ_a e^a(Y)·X_a) "
            f"[{connection._repr_inner()},{frame.name}]"
        )

    @property
    def connection(self) -> AffineConnection:
        return self._connection

    @property
    def frame(self) -> LocalFrame:
        return self._frame

    def matches(self, expr: Expr) -> bool:
        if not isinstance(expr, ConnectionEvalExpr):
            return False
        if expr.connection != self._connection:
            return False
        Y = expr.Y
        if not isinstance(Y, Derivation):
            return False
        if isinstance(Y, FrameVectorField) and Y.frame == self._frame:
            return False
        return True

    def rewrite(self, expr: Expr) -> Expr:
        assert isinstance(expr, ConnectionEvalExpr)
        Y = expr.Y
        a = _fresh_bound_index("a", expr)
        body = Product.make(
            Pairing(self._frame.coframe(a), Y),
            self._frame.X(a),
        )
        new_Y = IndexedSum(a, self._frame, body)
        return ConnectionEvalExpr(self._connection, expr.X, new_Y)


# --------------------------------------------------------------------- #
# 17.E.7, ConnectionFormDecomposition                                  #
# --------------------------------------------------------------------- #


class ConnectionFormDecompositionDefinition(Definition):
    r"""Special-case rewrite for the connection eval on a basis VF:

    .. math::

        \nabla_V X_b \to \Sigma_c \,\omega^c{}_b(\nabla)(V) \cdot X_c.

    Fires on a :class:`ConnectionEvalExpr` whose ``Y`` slot is a
    :class:`FrameVectorField` of this rule's :class:`LocalFrame`. The
    dummy ``c`` is alpha-fresh on every match, relative to the entire
    expression, so it never collides with an outer-bound ``b``
    introduced by an enclosing :class:`IndexedSum`.

    **Direction & loop avoidance.** The body uses
    :class:`ConnectionForm` paired with ``V`` (i.e. ``ω^c_b(∇)(V)``),
    not :class:`ConnectionEvalExpr`. So the residue contains no
    matching shape for this rule, no loop. It does, however, contain
    an ``ω`` that could be unfolded by
    :class:`~jacopy.calculus.cartan_forms.ConnectionFormDefinition`
    back into ``Pairing(e^c, ∇_V X_b)``, and **that would loop**. A
    bundle that registers this rule must therefore *not* register
    ``ConnectionFormDefinition`` on the same connection/frame, or vice
    versa. The Cartan-structure problem wrappers in 17.F/G keep this
    invariant.

    **Both free and bound b accepted.** The Cartan I/II reductions
    apply this rule both at the user-facing free-``b`` shape and at
    the inner ``∇_V X_c`` shape that arises *inside* a frame-
    decomposition IS body (where ``c`` is bound by the enclosing
    binder). The fresh-``d`` minting collects every index name in the
    target expression so it never collides with the outer ``c``,
    nested IndexedSums correctly nest under standard alpha-shadowing
    rules.
    """

    def __init__(
        self, connection: AffineConnection, frame: LocalFrame
    ) -> None:
        if not isinstance(connection, AffineConnection):
            raise TypeError(
                "ConnectionFormDecompositionDefinition requires an "
                "AffineConnection"
            )
        if not isinstance(frame, LocalFrame):
            raise TypeError(
                "ConnectionFormDecompositionDefinition requires a LocalFrame"
            )
        self._connection = connection
        self._frame = frame
        self.name = (
            f"∇_V X_b → Σ_c ω^c_b(V)·X_c "
            f"[{connection._repr_inner()},{frame.name}]"
        )

    @property
    def connection(self) -> AffineConnection:
        return self._connection

    @property
    def frame(self) -> LocalFrame:
        return self._frame

    def matches(self, expr: Expr) -> bool:
        if not isinstance(expr, ConnectionEvalExpr):
            return False
        if expr.connection != self._connection:
            return False
        Y = expr.Y
        if not isinstance(Y, FrameVectorField):
            return False
        return Y.frame == self._frame

    def rewrite(self, expr: Expr) -> Expr:
        assert isinstance(expr, ConnectionEvalExpr)
        Y = expr.Y
        assert isinstance(Y, FrameVectorField)
        b = Y.idx
        V = expr.X
        c = _fresh_bound_index("c", expr)
        body = Product.make(
            Pairing(
                ConnectionForm(self._connection, self._frame, c, b), V
            ),
            self._frame.X(c),
        )
        return IndexedSum(c, self._frame, body)
