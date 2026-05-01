r"""
Tilde-calculus intrinsic axioms, Faz 14.E.

Three engine rewrite rules that *open up* the dual of the textbook
Cartan formulas for the tilde operators on a multivector ``V``
evaluated against 1-forms ``η_1, …, η_p``:

* ``(ι̃_ω V)(η_1, …, η_{p-1}) = V(ω, η_1, …, η_{p-1})``
* ``(L̃_ω V)(η_1, …, η_p) = π^♯(ω)·V(η_1, …, η_p)
                          − Σ_i V(η_1, …, [ω, η_i]_K, …, η_p)``
* ``(d̃ V)(η_0, …, η_p) = Σ_i (−1)^i π^♯(η_i)·V(η_0, …, η̂_i, …, η_p)
       + Σ_{i<j} (−1)^{i+j} V([η_i, η_j]_K, η_0, …, η̂_i, …, η̂_j, …, η_p)``

Each rule fires on a :class:`~jacopy.core.multi_eval.MultiEval` whose
head is a tilde-operator :class:`~jacopy.algebra.derivation.Act` and
whose ``slot_kind`` is ``"covector"``, the dual of the standard-side
``slot_kind="vector"`` discipline. A multivector ``V`` evaluated on
``p`` 1-forms is a scalar; the rules unfold the operator action into a
multilinear-evaluation residue that downstream MultiEval-linearity,
Sharp, and Koszul-bracket axioms can normalise to a canonical form.

Mirrors :mod:`jacopy.calculus.intrinsic_axioms` (Faz 12.A.1-3) on the
Koszul side. The ``d̃`` and ``L̃`` rules are π-scoped: each carries the
specific ``π`` it was constructed with and only matches operators
whose ``bivector`` attribute equals that ``π``. This keeps two
independent Poisson manifolds from aliasing under a single engine.

The ``[η_i, η_j]_K`` bracket terms are emitted as
:class:`~jacopy.brackets.base.BracketApply` over a stored
:class:`~jacopy.brackets.koszul.KoszulBracket`; downstream the
:class:`~jacopy.library.koszul_problem.KoszulBracketExpansionDefinition`
rule (already wired into KoszulProblem) unfolds them to the classical
``L_{ρα}β − L_{ρβ}α − d⟨ρα, β⟩`` form when needed. The ``π^♯(η)·…``
action terms emit ``Act(Act(Sharp(π), η), …)``, the same nested-Act
shape that :class:`~jacopy.calculus.tilde.aux_axioms.TildeLieOnZeroVectorDefinition`
already uses, so existing Sharp-axiom infrastructure picks up where
this rule leaves off.
"""

from __future__ import annotations

from typing import Optional

from jacopy.algebra.derivation import Act
from jacopy.brackets.base import BracketApply
from jacopy.brackets.koszul import KoszulBracket
from jacopy.calculus.musical import Sharp
from jacopy.calculus.tilde.operators import (
    TildeExteriorDerivative,
    TildeInteriorProduct,
    TildeLieDerivative,
)
from jacopy.core.expr import Expr, Neg, Sum
from jacopy.core.multi_eval import MultiEval
from jacopy.proof.expansion import Definition


# --------------------------------------------------------------------- #
# (ι̃_ω V)(η_1, …, η_{p-1}) = V(ω, η_1, …, η_{p-1})                       #
# --------------------------------------------------------------------- #


class TildeIotaIntrinsicDefinition(Definition):
    r"""``ι̃_ω`` intrinsic formula on a covector-slot evaluation.

    Fires on ``MultiEval(Act(ι̃_ω, V), η_1, …, η_{p-1}, slot_kind="covector")``
    and rewrites the head by absorbing the indexing form ``ω`` into the
    first argument slot:

    .. math::

       (\tilde{\iota}_\omega V)(\eta_1, \dots, \eta_{p-1})
           = V(\omega, \eta_1, \dots, \eta_{p-1}).

    The dual of :class:`~jacopy.calculus.intrinsic_axioms.InteriorProductIntrinsicDefinition`
   , there ``X`` (a vector field) was injected into a vector-slot
    multilinear evaluation; here ``ω`` (a 1-form) is injected into a
    covector-slot one. Restricted to ``slot_kind="covector"`` so a
    standard-side ``MultiEval(Act(ι_X, ω), Y_1, …)`` is left untouched.

    The alternating flag and slot kind carry over verbatim, ``ι̃_ω``
    is graded-antisymmetric in its operand-multivector slots, so the
    contracted variant inherits that symmetry. ``V`` is taken
    structurally as ``Act.arg``; if it is a :class:`Sum` or a compound
    expression, MultiEval head-linearity (Faz 12.A.0) distributes it
    after the first rewrite.
    """

    name = "ι̃_ω intrinsic: (ι̃_ω V)(η_1, …) = V(ω, η_1, …)"

    def matches(self, expr: Expr) -> bool:
        return (
            isinstance(expr, MultiEval)
            and expr.slot_kind == "covector"
            and isinstance(expr.head, Act)
            and isinstance(expr.head.op, TildeInteriorProduct)
        )

    def rewrite(self, expr: Expr) -> Expr:
        head_act = expr.head  # Act(ι̃_ω, V)
        iota = head_act.op
        assert isinstance(iota, TildeInteriorProduct)
        V = head_act.arg
        new_args = (iota.form,) + expr.args
        return MultiEval(
            V,
            *new_args,
            alternating=expr.alternating,
            slot_kind=expr.slot_kind,
        )


# --------------------------------------------------------------------- #
# (L̃_ω V)(η_1, …, η_p)                                                   #
#   = π^♯(ω)·V(η_1, …, η_p) − Σ_i V(η_1, …, [ω, η_i]_K, …, η_p)           #
# --------------------------------------------------------------------- #


class TildeLieIntrinsicDefinition(Definition):
    r"""``L̃_ω`` intrinsic formula on a covector-slot evaluation.

    Fires on ``MultiEval(Act(L̃_ω, V), η_1, …, η_p, slot_kind="covector")``
    and rewrites the head by expanding the tilde Lie derivative
    through its action on a p-vector's value plus the Koszul-bracket
    correction terms:

    .. math::

       (\tilde{L}_\omega V)(\eta_1, \dots, \eta_p)
           = \pi^\sharp(\omega)\bigl(V(\eta_1, \dots, \eta_p)\bigr)
             - \sum_{i=1}^{p}
                 V\bigl(\eta_1, \dots, [\omega, \eta_i]_K,
                        \dots, \eta_p\bigr).

    π-scoped: matches only when the head's
    :attr:`~jacopy.calculus.tilde.operators.TildeLieDerivative.bivector`
    equals ``self._pi``. The first term wraps the inner
    :class:`MultiEval` in nested :class:`Act` along ``Sharp(π)·ω``,
    the anchor's image of the indexing form, a vector field, acting
    on the scalar ``V(η_1, …, η_p)``. The bracket terms emit
    :class:`BracketApply` over the stored
    :class:`~jacopy.brackets.koszul.KoszulBracket`, leaving the
    bracket atom opaque so downstream rules
    (:class:`KoszulBracketExpansionDefinition` plus MultiEval slot
    linearity) can unfold them when proof closure requires it.

    Restricted to ``slot_kind="covector"``; the alternating flag
    carries over.
    """

    def __init__(
        self,
        pi: Expr,
        koszul: KoszulBracket,
        *,
        sharp: Optional[Sharp] = None,
    ) -> None:
        if not isinstance(pi, Expr):
            raise TypeError(
                "TildeLieIntrinsicDefinition pi must be an Expr"
            )
        if not isinstance(koszul, KoszulBracket):
            raise TypeError(
                "TildeLieIntrinsicDefinition koszul must be a KoszulBracket"
            )
        if sharp is not None and not isinstance(sharp, Sharp):
            raise TypeError(
                "TildeLieIntrinsicDefinition sharp must be a Sharp instance"
            )
        self._pi = pi
        self._koszul = koszul
        self._sharp = sharp if sharp is not None else Sharp(pi)
        self.name = (
            f"L̃_ω intrinsic [{pi._repr_inner()}]: "
            "(L̃_ω V)(η_1, …) = π^♯(ω)·V(η_1, …) − Σ V(η_1, …, [ω, η_i]_K, …)"
        )

    @property
    def pi(self) -> Expr:
        return self._pi

    @property
    def koszul(self) -> KoszulBracket:
        return self._koszul

    @property
    def sharp(self) -> Sharp:
        return self._sharp

    def matches(self, expr: Expr) -> bool:
        return (
            isinstance(expr, MultiEval)
            and expr.slot_kind == "covector"
            and isinstance(expr.head, Act)
            and isinstance(expr.head.op, TildeLieDerivative)
            and expr.head.op.bivector == self._pi
        )

    def rewrite(self, expr: Expr) -> Expr:
        head_act = expr.head  # Act(L̃_ω, V)
        lie = head_act.op
        assert isinstance(lie, TildeLieDerivative)
        V = head_act.arg
        omega = lie.form

        inner = MultiEval(
            V,
            *expr.args,
            alternating=expr.alternating,
            slot_kind=expr.slot_kind,
        )
        first: Expr = Act(Act(self._sharp, omega), inner)

        bracket_terms: list[Expr] = []
        for i, eta_i in enumerate(expr.args):
            bracketed = list(expr.args)
            bracketed[i] = BracketApply(self._koszul, omega, eta_i)
            bracket_terms.append(
                Neg(
                    MultiEval(
                        V,
                        *bracketed,
                        alternating=expr.alternating,
                        slot_kind=expr.slot_kind,
                    )
                )
            )
        return Sum.make(first, *bracket_terms)


# --------------------------------------------------------------------- #
# Lichnerowicz / Koszul invariant formula for d̃                          #
# (d̃ V)(η_0, …, η_p)                                                     #
#   = Σ_i (−1)^i π^♯(η_i)·V(η_0, …, η̂_i, …, η_p)                          #
#   + Σ_{i<j} (−1)^{i+j} V([η_i, η_j]_K, η_0, …, η̂_i, …, η̂_j, …, η_p)     #
# --------------------------------------------------------------------- #


class TildeDIntrinsicDefinition(Definition):
    r"""``d̃`` intrinsic Koszul formula on a covector-slot evaluation.

    Fires on ``MultiEval(Act(d̃, V), η_0, …, η_p, slot_kind="covector")``
    with ``p ≥ 0`` (i.e. at least one evaluation slot) and rewrites the
    head by emitting the full Lichnerowicz–Koszul expansion:

    .. math::

       (\tilde{d} V)(\eta_0, \dots, \eta_p)
           &= \sum_{i=0}^{p} (-1)^i\,
                 \pi^\sharp(\eta_i)\bigl(V(\eta_0, \dots, \widehat{\eta_i},
                                            \dots, \eta_p)\bigr) \\
           &\quad + \sum_{0 \le i < j \le p} (-1)^{i+j}\,
                 V\bigl([\eta_i, \eta_j]_K,
                        \eta_0, \dots, \widehat{\eta_i}, \dots,
                        \widehat{\eta_j}, \dots, \eta_p\bigr).

    π-scoped: matches only when the head equals
    ``TildeExteriorDerivative(self._pi)`` structurally. Sign handling
    mirrors :class:`~jacopy.calculus.intrinsic_axioms.ExteriorDIntrinsicDefinition`
   , odd-parity terms are wrapped in :class:`Neg`. The bracket terms
    use :class:`BracketApply` over the stored
    :class:`~jacopy.brackets.koszul.KoszulBracket` so the result stays
    inside the multilinear-evaluation framework.

    Arity-1 special case: when the operand multivector has degree 0
    (a function ``f``), ``(d̃ f)(η_0)`` collapses to a single
    ``π^♯(η_0)·f`` term with no inner :class:`MultiEval` wrap and no
    bracket sum, since there is exactly one argument and therefore no
    pairs to enumerate. This matches the standard-side
    :class:`ExteriorDIntrinsicDefinition` arity-1 branch shape.
    """

    def __init__(
        self,
        pi: Expr,
        koszul: KoszulBracket,
        *,
        sharp: Optional[Sharp] = None,
    ) -> None:
        if not isinstance(pi, Expr):
            raise TypeError(
                "TildeDIntrinsicDefinition pi must be an Expr"
            )
        if not isinstance(koszul, KoszulBracket):
            raise TypeError(
                "TildeDIntrinsicDefinition koszul must be a KoszulBracket"
            )
        if sharp is not None and not isinstance(sharp, Sharp):
            raise TypeError(
                "TildeDIntrinsicDefinition sharp must be a Sharp instance"
            )
        self._pi = pi
        self._head = TildeExteriorDerivative(pi)
        self._koszul = koszul
        self._sharp = sharp if sharp is not None else Sharp(pi)
        self.name = (
            f"d̃ intrinsic (Koszul) [{pi._repr_inner()}]: "
            "(d̃V)(η_0, …) = Σ ±π^♯(η_i)·V(…) + Σ ±V([η_i,η_j]_K, …)"
        )

    @property
    def pi(self) -> Expr:
        return self._pi

    @property
    def koszul(self) -> KoszulBracket:
        return self._koszul

    @property
    def sharp(self) -> Sharp:
        return self._sharp

    def matches(self, expr: Expr) -> bool:
        return (
            isinstance(expr, MultiEval)
            and expr.slot_kind == "covector"
            and isinstance(expr.head, Act)
            and isinstance(expr.head.op, TildeExteriorDerivative)
            and expr.head.op == self._head
            and len(expr.args) >= 1
        )

    def rewrite(self, expr: Expr) -> Expr:
        head_act = expr.head  # Act(d̃, V)
        V = head_act.arg
        args = expr.args  # (η_0, …, η_p), len = p+1
        terms: list[Expr] = []

        # Σ_i (−1)^i π^♯(η_i)·V(η_0, …, η̂_i, …, η_p)
        for i, eta_i in enumerate(args):
            remaining = args[:i] + args[i + 1 :]
            inner: Expr
            if remaining:
                inner = MultiEval(
                    V,
                    *remaining,
                    alternating=expr.alternating,
                    slot_kind=expr.slot_kind,
                )
            else:
                inner = V
            term: Expr = Act(Act(self._sharp, eta_i), inner)
            if i % 2 == 1:
                term = Neg(term)
            terms.append(term)

        # Σ_{i<j} (−1)^{i+j} V([η_i, η_j]_K, η_0, …, η̂_i, …, η̂_j, …, η_p)
        for i in range(len(args)):
            for j in range(i + 1, len(args)):
                bracket = BracketApply(self._koszul, args[i], args[j])
                rest = tuple(
                    a for k, a in enumerate(args) if k != i and k != j
                )
                inner_args = (bracket,) + rest
                inner_term: Expr = MultiEval(
                    V,
                    *inner_args,
                    alternating=expr.alternating,
                    slot_kind=expr.slot_kind,
                )
                if (i + j) % 2 == 1:
                    inner_term = Neg(inner_term)
                terms.append(inner_term)

        return Sum.make(*terms)
