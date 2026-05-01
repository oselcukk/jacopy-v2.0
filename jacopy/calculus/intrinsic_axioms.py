r"""
Intrinsic (coordinate-free) definitions of the Cartan operators on a
:class:`~jacopy.core.multi_eval.MultiEval` evaluation.

Engine rewrite rules that *open up* the textbook formulas

* ``(ι_X ω)(Y_1, …, Y_{p-1}) = ω(X, Y_1, …, Y_{p-1})``
* ``(L_X ω)(Y_1, …, Y_p) = X(ω(Y_1, …, Y_p)) − Σ_i ω(…, [X, Y_i], …)``
* ``(d ω)(X_0, …, X_p) = Σ_i (−1)^i X_i(ω(…, hat_i, …))
       + Σ_{i<j} (−1)^{i+j} ω([X_i, X_j], …, hat_i, …, hat_j, …)``

so that whenever the user wraps an operator-valued head inside a
:class:`MultiEval`, the engine can take a single rewrite step that
expands the operator into its multilinear definition. Faz 12.A.1
ships the simplest of the three, the interior product, so the
patterns stay easy to inspect; A.2 (Lie derivative) and A.3
(exterior derivative / Koszul formula) layer on top.

Each rule is structurally narrow: it fires only when the
:class:`~jacopy.core.multi_eval.MultiEval` head matches the operator's
shape. Slot-kind and alternating flags are preserved verbatim, the
intrinsic rewrite never converts a vector-slot evaluation into a
covector one or vice versa.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from jacopy.algebra.derivation import Act
from jacopy.algebra.lie_bracket_vf import lie_bracket_vf
from jacopy.brackets.base import BracketApply
from jacopy.calculus.exterior_d import ExteriorDerivative
from jacopy.calculus.interior import InteriorProduct
from jacopy.calculus.lie_derivative import LieDerivative
from jacopy.core.expr import Expr, Neg, Sum
from jacopy.core.multi_eval import MultiEval
from jacopy.proof.expansion import Definition

if TYPE_CHECKING:
    from jacopy.calculus.connection import AffineConnection


# --------------------------------------------------------------------- #
# (ι_X ω)(Y_1, …, Y_{p-1}) = ω(X, Y_1, …, Y_{p-1})                       #
# --------------------------------------------------------------------- #


class InteriorProductIntrinsicDefinition(Definition):
    r"""``ι_X`` intrinsic formula on a multilinear evaluation.

    Fires on ``MultiEval(Act(ι_X, ω), Y_1, …, Y_{p-1})`` and rewrites
    the head by absorbing ``X`` into the first argument slot:

    .. math::

       (\iota_X \omega)(Y_1, \dots, Y_{p-1})
           = \omega(X, Y_1, \dots, Y_{p-1}).

    Restricted to vector-slot evaluations (``slot_kind="vector"``); the
    interior product is a form-on-vectors operation, so a covector-slot
    bivector evaluation is left untouched. The alternating flag carries
    over, antisymmetry and slot-linearity are properties of the
    underlying form, not of the contracted variant.

    The operand ``ω`` is taken structurally as ``Act.arg``, it can be
    a plain :class:`~jacopy.core.expr.Symbol`, an
    :class:`Act` (e.g. ``d β``), a sum, or any expression. When ``ω``
    is itself a Sum, the auxiliary
    :class:`~jacopy.calculus.multi_eval_axioms.MultiEvalHeadLinearityDefinition`
    will subsequently distribute the head, but only after the
    surrounding ``Act(ι_X, Sum)`` is itself unfolded by an upstream
    iota-linearity rule (not part of this pass).
    """

    name = "ι_X intrinsic: (ι_X ω)(Y_1, …) = ω(X, Y_1, …)"

    def matches(self, expr: Expr) -> bool:
        return (
            isinstance(expr, MultiEval)
            and expr.slot_kind == "vector"
            and isinstance(expr.head, Act)
            and isinstance(expr.head.op, InteriorProduct)
        )

    def rewrite(self, expr: Expr) -> Expr:
        head_act = expr.head  # Act(ι_X, ω)
        iota = head_act.op  # InteriorProduct
        omega = head_act.arg  # the underlying form
        new_args = (iota.vector_field,) + expr.args
        return MultiEval(
            omega,
            *new_args,
            alternating=expr.alternating,
            slot_kind=expr.slot_kind,
        )


# --------------------------------------------------------------------- #
# (L_X ω)(Y_1, …, Y_p) = X(ω(Y_1, …, Y_p)) − Σ_i ω(…, [X, Y_i]_VF, …)    #
# --------------------------------------------------------------------- #


class LieDerivativeIntrinsicDefinition(Definition):
    r"""``L_X`` intrinsic formula on a multilinear evaluation.

    Fires on ``MultiEval(Act(L_X, ω), Y_1, …, Y_p)`` and rewrites the
    head by expanding the Lie derivative through its action on a
    p-form's value plus the bracket correction terms:

    .. math::

       (L_X \omega)(Y_1, \dots, Y_p)
           = X\bigl(\omega(Y_1, \dots, Y_p)\bigr)
             - \sum_{i=1}^{p}
                 \omega\bigl(Y_1, \dots, [X, Y_i]_{VF}, \dots, Y_p\bigr).

    The first term wraps the inner :class:`MultiEval` in an
    :class:`~jacopy.algebra.derivation.Act` along ``X``, the action
    of a vector field on the scalar function ``ω(Y_1, …, Y_p)``. The
    bracket terms use the opaque
    :class:`~jacopy.algebra.lie_bracket_vf.LieBracketVF` atom so the
    resulting expression stays inside the multilinear-evaluation
    framework and downstream rules (arg-linearity, repeat-arg-zero,
    head-linearity) can keep firing.

    Restricted to vector-slot evaluations, the Lie derivative on a
    p-form pairs against vector fields, so a covector-slot bivector
    evaluation is left untouched. The alternating flag carries over.
    """

    name = (
        "L_X intrinsic: (L_X ω)(Y_1, …) "
        "= X(ω(Y_1, …)) − Σ ω(…, [X, Y_i]_VF, …)"
    )

    def matches(self, expr: Expr) -> bool:
        return (
            isinstance(expr, MultiEval)
            and expr.slot_kind == "vector"
            and isinstance(expr.head, Act)
            and isinstance(expr.head.op, LieDerivative)
        )

    def rewrite(self, expr: Expr) -> Expr:
        head_act = expr.head  # Act(L_X, ω)
        lie = head_act.op  # LieDerivative
        omega = head_act.arg  # the underlying form
        X = lie.vector_field

        inner = MultiEval(
            omega,
            *expr.args,
            alternating=expr.alternating,
            slot_kind=expr.slot_kind,
        )
        first = Act(X, inner)

        bracket_terms = []
        for i, Y_i in enumerate(expr.args):
            bracketed = list(expr.args)
            bracketed[i] = lie_bracket_vf(X, Y_i)
            bracket_terms.append(
                Neg(
                    MultiEval(
                        omega,
                        *bracketed,
                        alternating=expr.alternating,
                        slot_kind=expr.slot_kind,
                    )
                )
            )
        return Sum.make(first, *bracket_terms)


# --------------------------------------------------------------------- #
# Exterior derivative, Koszul invariant formula                         #
# (dω)(X_0,…,X_p) = Σ_i (−1)^i X_i(ω(…,hat_i,…))                         #
#                 + Σ_{i<j} (−1)^{i+j} ω([X_i,X_j]_VF, …,hat_i,…,hat_j,…)#
# --------------------------------------------------------------------- #


class ExteriorDIntrinsicDefinition(Definition):
    r"""``d`` intrinsic Koszul formula on a multilinear evaluation.

    Fires on ``MultiEval(Act(d, ω), X_0, …, X_p)`` with ``p ≥ 1`` (so the
    inner ``ω(…,hat_i,…)`` slots stay non-empty) and rewrites the head
    by emitting the full Cartan–Koszul expansion:

    .. math::

       (d\omega)(X_0, \dots, X_p)
           &= \sum_{i=0}^{p} (-1)^i\,
                 X_i\bigl(\omega(X_0, \dots, \widehat{X_i}, \dots, X_p)\bigr)\\
           &\quad + \sum_{0 \le i < j \le p} (-1)^{i+j}\,
                 \omega\bigl([X_i, X_j]_{VF},
                              X_0, \dots, \widehat{X_i}, \dots,
                              \widehat{X_j}, \dots, X_p\bigr).

    Sign handling: an odd-parity term is wrapped in
    :class:`~jacopy.core.expr.Neg`. The bracket terms reuse the opaque
    :class:`~jacopy.algebra.lie_bracket_vf.LieBracketVF` atom so the
    output stays inside the multilinear-evaluation framework.

    Restricted to vector-slot evaluations. Both the arity-≥ 2 case (full
    Koszul expansion) and the arity-1 scalar identity ``(df)(X) = X(f)``
    fire, the latter emits a single ``Act(X, f)`` with no inner
    :class:`MultiEval` wrap and no bracket sum, since there is exactly
    one argument and therefore no pairs to enumerate. Alternating +
    slot_kind flags are propagated to every emitted inner
    :class:`MultiEval`.
    """

    name = (
        "d intrinsic (Koszul): "
        "(dω)(X_0, …, X_p) = Σ ±X_i(ω(…,hat_i,…)) + Σ ±ω([X_i,X_j]_VF, …)"
    )

    def matches(self, expr: Expr) -> bool:
        return (
            isinstance(expr, MultiEval)
            and expr.slot_kind == "vector"
            and isinstance(expr.head, Act)
            and isinstance(expr.head.op, ExteriorDerivative)
            and len(expr.args) >= 1
        )

    def rewrite(self, expr: Expr) -> Expr:
        head_act = expr.head  # Act(d, ω)
        omega = head_act.arg
        args = expr.args  # (X_0, …, X_p), len = p+1, p ≥ 0
        terms: list[Expr] = []

        # Σ_i (−1)^i X_i(ω(X_0, …, hat_i, …, X_p))
        # Arity-1 special case: remaining is empty → emit Act(X_0, ω)
        # directly, skipping the empty MultiEval wrap.
        for i, X_i in enumerate(args):
            remaining = args[:i] + args[i + 1 :]
            inner: Expr
            if remaining:
                inner = MultiEval(
                    omega,
                    *remaining,
                    alternating=expr.alternating,
                    slot_kind=expr.slot_kind,
                )
            else:
                inner = omega
            term: Expr = Act(X_i, inner)
            if i % 2 == 1:
                term = Neg(term)
            terms.append(term)

        # Σ_{i<j} (−1)^{i+j} ω([X_i, X_j]_VF, X_0, …, hat_i, …, hat_j, …)
        for i in range(len(args)):
            for j in range(i + 1, len(args)):
                bracket = lie_bracket_vf(args[i], args[j])
                rest = tuple(
                    a for k, a in enumerate(args) if k != i and k != j
                )
                inner_args = (bracket,) + rest
                inner_term: Expr = MultiEval(
                    omega,
                    *inner_args,
                    alternating=expr.alternating,
                    slot_kind=expr.slot_kind,
                )
                if (i + j) % 2 == 1:
                    inner_term = Neg(inner_term)
                terms.append(inner_term)

        return Sum.make(*terms)


class KoszulExteriorDIntrinsicDefinition(Definition):
    r"""Anchored ``d̃`` intrinsic formula for a Koszul-bracket connection.

    Connection-parametric variant of :class:`ExteriorDIntrinsicDefinition`
    for an affine connection ``∇̃`` whose vector bracket is Koszul (or any
    other graded bracket carried by the connection) and whose function
    action is anchor-routed:

    .. math::

       (d̃ω)(α_0, \dots, α_p)
           &= \sum_{i=0}^{p} (-1)^i\,
                 ρ(α_i)\bigl(ω(α_0, \dots, \widehat{α_i}, \dots, α_p)\bigr)\\
           &\quad + \sum_{0 \le i < j \le p} (-1)^{i+j}\,
                 ω\bigl([α_i, α_j]_K,
                        α_0, \dots, \widehat{α_i}, \dots,
                        \widehat{α_j}, \dots, α_p\bigr).

    The function-action lift is sourced from
    :meth:`AffineConnection.function_action` (so a Koszul connection
    emits ``Act(AnchoredVectorField(ρ, α_i), …)`` exactly as the
    Y-Leibniz rule does), and the bracket lift is sourced from the
    connection's ``bracket`` slot. Both choices are bypassed in the
    standard rule, which uses raw ``Act(X_i, …)`` and ``LieBracketVF``.

    Wiring this rule (in place of :class:`ExteriorDIntrinsicDefinition`)
    is what closes Cartan I/II on a Koszul connection: without the
    anchor pull, the LHS's ``ρ(U)(…)`` shapes (sourced from Y-Leibniz)
    don't cancel against the RHS's ``U(…)``; without the connection
    bracket, the ``[U, V]_K`` from the torsion / curvature definition
    doesn't cancel against the ``[U, V]_VF`` the standard rule emits.
    """

    def __init__(self, connection: "AffineConnection") -> None:
        from jacopy.calculus.connection import AffineConnection

        if not isinstance(connection, AffineConnection):
            raise TypeError(
                "KoszulExteriorDIntrinsicDefinition requires an AffineConnection"
            )
        if connection.bracket is None:
            raise ValueError(
                "KoszulExteriorDIntrinsicDefinition requires the connection "
                "to carry a bracket, pass a koszul_connection(...) or "
                "another bracket-equipped AffineConnection"
            )
        self._conn = connection
        self.name = (
            f"d̃ intrinsic ({connection._repr_inner()}-anchored): "
            f"(d̃ω)(…) = Σ ±ρ(α_i)(ω(…)) + Σ ±ω([α_i,α_j]_{connection.bracket.name}, …)"
        )

    def matches(self, expr: Expr) -> bool:
        return (
            isinstance(expr, MultiEval)
            and expr.slot_kind == "vector"
            and isinstance(expr.head, Act)
            and isinstance(expr.head.op, ExteriorDerivative)
            and len(expr.args) >= 1
        )

    def rewrite(self, expr: Expr) -> Expr:
        head_act = expr.head
        omega = head_act.arg
        args = expr.args
        bracket = self._conn.bracket
        terms: list[Expr] = []

        for i, X_i in enumerate(args):
            remaining = args[:i] + args[i + 1 :]
            inner: Expr
            if remaining:
                inner = MultiEval(
                    omega,
                    *remaining,
                    alternating=expr.alternating,
                    slot_kind=expr.slot_kind,
                )
            else:
                inner = omega
            term: Expr = self._conn.function_action(X_i, inner)
            if i % 2 == 1:
                term = Neg(term)
            terms.append(term)

        for i in range(len(args)):
            for j in range(i + 1, len(args)):
                pair = BracketApply(bracket, args[i], args[j])
                rest = tuple(
                    a for k, a in enumerate(args) if k != i and k != j
                )
                inner_args = (pair,) + rest
                inner_term: Expr = MultiEval(
                    omega,
                    *inner_args,
                    alternating=expr.alternating,
                    slot_kind=expr.slot_kind,
                )
                if (i + j) % 2 == 1:
                    inner_term = Neg(inner_term)
                terms.append(inner_term)

        return Sum.make(*terms)
