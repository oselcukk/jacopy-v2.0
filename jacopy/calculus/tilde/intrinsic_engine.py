r"""
Public surface over the tilde intrinsic-formula axioms (Faz 14.E).

Three pieces, mirroring :mod:`jacopy.calculus.intrinsic_engine`
(Faz 12.A.5) on the Koszul side:

* :func:`tilde_intrinsic_engine`, bundles the three tilde intrinsic
  rules (Faz 14.E.1) with the standard MultiEval helpers (Faz 12.A.0),
  Sharp axioms (Faz 13.A), and the Koszul-bracket expansion rule into a
  single :class:`~jacopy.proof.expansion.ExpansionEngine`. Callers who
  want the same plumbing the tilde Cartan-relation proofs use reach for
  this instead of assembling rule lists by hand.

* :class:`TildeIntrinsicFormulaRecognizer`, pure-shape inspector
  mirroring :class:`~jacopy.calculus.intrinsic_engine.IntrinsicFormulaRecognizer`.
  Given a covector-slot :class:`~jacopy.core.multi_eval.MultiEval`, it
  answers "is the head a tilde-:math:`\iota`, tilde-:math:`L`, or
  tilde-:math:`d` applied to a multivector, and if so what's inside?".
  No engine, no rewrite, just structured field extraction for
  higher-level tactics.

* :func:`prove_tilde_cartan_relation`, wraps the
  ``MultiEval(lhs, *etas) == MultiEval(rhs, *etas)`` cycle into a
  single call. Forms both sides under the covector-slot evaluation,
  feeds the resulting equality to the tilde intrinsic engine, and
  returns a :class:`~jacopy.proof.chain.ProofChain` on success.

The three Cartan relations that close most cleanly under the bare
bundle are the iota anti-commute (relation 1), the Cartan magic
(relation 3, defining), and the bracket commutator with d̃ (relation 6).
``d̃² = 0`` (relation 2) needs the
:class:`~jacopy.core.properties.Poisson` flag plumbed through and may
require additional closure rules; ``[L̃_α, L̃_β] = L̃_{[α,β]_K}`` (4)
and ``[L̃_α, ι̃_β] = ι̃_{[α,β]_K}`` (5) need the Koszul bracket
unfolded, both are scheduled for follow-up passes that extend the
engine returned here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

from jacopy.algebra.derivation import Act
from jacopy.algorithms.product_rule import product_rule
from jacopy.algorithms.simplify import simplify
from jacopy.brackets.koszul import KoszulBracket
from jacopy.calculus.closure_axioms import (
    LieBracketVfAntiSymmetryDefinition,
    LieBracketVfJacobiDefinition,
    VfActCommutatorDefinition,
)
from jacopy.calculus.intrinsic_engine import prove_intrinsic_equivalence
from jacopy.calculus.multi_eval_axioms import (
    MultiEvalAlternatingNormalDefinition,
    MultiEvalArgLinearityDefinition,
    MultiEvalHeadLinearityDefinition,
    MultiEvalRepeatArgZeroDefinition,
    MultiEvalZeroHeadDefinition,
)
from jacopy.calculus.musical import Sharp
from jacopy.calculus.sharp_axioms import (
    SharpLinearityDefinition,
    SharpOnExactDefinition,
)
from jacopy.calculus.tilde.aux_axioms import (
    TildeIotaActAsScalarDefinition,
)
from jacopy.calculus.pairing_axioms import (
    PairingLieLeibnizDefinition,
    PairingLinearityDefinition,
)
from jacopy.calculus.tilde.closure_axioms import (
    AnchorLieHomomorphismDefinition,
    HamiltonianAnchorPairingAntisymmetryDefinition,
    TildeSnJacobiResidueDefinition,
    WrappedPairingAnchorAntisymmetryDefinition,
    LieCommutesWithDTildeDefinition,
    LieDerivativeOfAnchorBracketDefinition,
    LieDerivativeOnAnchorImageDefinition,
    MultiEvalLieCommutatorSlotDefinition,
    PairingWithExactFormDefinition,
)
from jacopy.calculus.tilde.intrinsic_axioms import (
    TildeDIntrinsicDefinition,
    TildeIotaIntrinsicDefinition,
    TildeLieIntrinsicDefinition,
)
from jacopy.calculus.tilde.operators import (
    TildeExteriorDerivative,
    TildeInteriorProduct,
    TildeLieDerivative,
)
from jacopy.core.expr import Expr
from jacopy.core.multi_eval import MultiEval
from jacopy.core.registry import PropertyRegistry
from jacopy.proof.chain import ProofChain
from jacopy.proof.expansion import ActOverSumOpDefinition, ExpansionEngine


# --------------------------------------------------------------------- #
# Engine factory                                                         #
# --------------------------------------------------------------------- #


def tilde_intrinsic_engine(
    pi: Expr,
    koszul: KoszulBracket,
    *,
    sharp: Optional[Sharp] = None,
    registry: Optional[PropertyRegistry] = None,
    d_op: Optional[Expr] = None,
) -> ExpansionEngine:
    r"""Build a fresh tilde-intrinsic :class:`ExpansionEngine`.

    Bundle (in match order):

    * :class:`TildeIotaIntrinsicDefinition`   (14.E.1)
    * :class:`TildeLieIntrinsicDefinition`    (14.E.1, π-scoped)
    * :class:`TildeDIntrinsicDefinition`      (14.E.1, π-scoped)
    * :class:`MultiEvalArgLinearityDefinition`        (12.A.0)
    * :class:`MultiEvalHeadLinearityDefinition`       (12.A.0)
    * :class:`MultiEvalRepeatArgZeroDefinition`       (12.A.0)
    * :class:`MultiEvalAlternatingNormalDefinition`   (12.A.4)
    * :class:`SharpLinearityDefinition`               (13.A)
    * :class:`SharpOnExactDefinition`                 (13.A, registry-aware)

    Parameters
    ----------
    pi
        The Poisson bivector. Used to scope the d̃ and L̃ intrinsic
        rules and to construct the default :class:`Sharp(π)`.
    koszul
        The Koszul bracket ``[·,·]_K`` on 1-forms, used by both d̃
        and L̃ intrinsic rules to label the bracket residue terms.
        Typically obtained from a
        :class:`~jacopy.library.koszul_problem.KoszulProblem` instance.
    sharp
        The :class:`Sharp(π)` derivation; constructed fresh from ``π``
        if omitted. Pass an existing instance when the surrounding
        engine already references one structurally.
    registry
        :class:`PropertyRegistry` passed through to
        :class:`SharpOnExactDefinition` so the ``π^♯(df)`` rewrite can
        consult the Hamiltonian-VF naming registry. Defaults to a
        fresh empty registry if omitted.
    d_op
        Exterior derivative singleton override. Defaults to the
        :data:`jacopy.calculus.exterior_d.d` module singleton via
        :class:`SharpOnExactDefinition`'s own default. Pass an explicit
        ``d_E`` when a Lie-algebroid exterior derivative coexists with
        the standard ``d``.

    Notes
    -----
    The Koszul-bracket expansion rule itself
    (:class:`~jacopy.library.koszul_problem.KoszulBracketExpansionDefinition`)
    is *not* included here, to keep this module's import surface
    free of the ``library`` layer; the
    :meth:`~jacopy.library.koszul_problem.KoszulProblem.tilde_intrinsic_engine`
    accessor adds it on. Callers using this factory directly should
    register their own bracket-expansion rule if their proof needs
    ``[α, β]_K`` unfolded to its Lichnerowicz form.
    """
    if not isinstance(pi, Expr):
        raise TypeError("tilde_intrinsic_engine pi must be an Expr")
    if not isinstance(koszul, KoszulBracket):
        raise TypeError("tilde_intrinsic_engine koszul must be a KoszulBracket")
    sharp_instance = sharp if sharp is not None else Sharp(pi)
    reg = registry if registry is not None else PropertyRegistry()

    sharp_on_exact_kwargs = {"registry": reg}
    if d_op is not None:
        sharp_on_exact_kwargs["d"] = d_op

    return ExpansionEngine(
        [
            TildeIotaIntrinsicDefinition(),
            TildeLieIntrinsicDefinition(pi, koszul, sharp=sharp_instance),
            TildeDIntrinsicDefinition(pi, koszul, sharp=sharp_instance),
            TildeIotaActAsScalarDefinition(registry=reg),
            MultiEvalArgLinearityDefinition(),
            MultiEvalHeadLinearityDefinition(),
            MultiEvalRepeatArgZeroDefinition(),
            MultiEvalZeroHeadDefinition(),
            MultiEvalAlternatingNormalDefinition(),
            SharpLinearityDefinition(sharp_instance),
            SharpOnExactDefinition(sharp_instance, **sharp_on_exact_kwargs),
            VfActCommutatorDefinition(),
            LieBracketVfAntiSymmetryDefinition(),
            LieBracketVfJacobiDefinition(),
            MultiEvalLieCommutatorSlotDefinition(),
            LieDerivativeOfAnchorBracketDefinition(
                pi, koszul, registry=reg, sharp=sharp_instance
            ),
            AnchorLieHomomorphismDefinition(
                pi, koszul, registry=reg, sharp=sharp_instance
            ),
            LieCommutesWithDTildeDefinition(),
            PairingLinearityDefinition(),
            PairingLieLeibnizDefinition(),
            LieDerivativeOnAnchorImageDefinition(
                pi, koszul, registry=reg, sharp=sharp_instance
            ),
            HamiltonianAnchorPairingAntisymmetryDefinition(
                pi, registry=reg, sharp=sharp_instance
            ),
            WrappedPairingAnchorAntisymmetryDefinition(
                pi, registry=reg, sharp=sharp_instance
            ),
            TildeSnJacobiResidueDefinition(
                pi, registry=reg, sharp=sharp_instance
            ),
            PairingWithExactFormDefinition(),
            ActOverSumOpDefinition(),
        ]
    )


# --------------------------------------------------------------------- #
# Recognizer                                                             #
# --------------------------------------------------------------------- #


@dataclass(frozen=True)
class TildeIntrinsicFormulaMatch:
    """Successful :class:`TildeIntrinsicFormulaRecognizer` result.

    Fields:
      * ``operator``, ``"tilde_interior"``, ``"tilde_lie"``, or
        ``"tilde_exterior_d"``.
      * ``form``, the ``ω`` in :math:`\\tilde{\\iota}_\\omega` /
        :math:`\\tilde{L}_\\omega`; ``None`` for :math:`\\tilde{d}`.
      * ``bivector``, the ``π`` carried by the operator (all three
        tilde operators carry one); preserved verbatim from the head.
      * ``multivector``, the ``V`` (``head.arg``).
      * ``args``, the multi-eval evaluation slots (1-forms).
      * ``alternating`` / ``slot_kind``, flags carried verbatim from
        the matched :class:`MultiEval`.
    """

    operator: str
    form: Optional[Expr]
    bivector: Expr
    multivector: Expr
    args: Tuple[Expr, ...]
    alternating: bool
    slot_kind: str


class TildeIntrinsicFormulaRecognizer:
    r"""Recognise ``MultiEval(Act(op, V), η_1, …, η_p)`` for tilde ``op``.

    Pure shape inspection, answers "is this a covector-slot evaluation
    whose head is a tilde-operator applied to a multivector?". No
    engine, no rewrite. Mirrors
    :class:`~jacopy.calculus.intrinsic_engine.IntrinsicFormulaRecognizer`
    on the standard side.

    Reports ``None`` on any non-match, including non-covector slot
    kinds, a vector-slot evaluation belongs to the standard-side
    recogniser. Nested operators are not unwrapped: the inner
    ``Act`` is reported verbatim as ``multivector``, so a caller can
    re-recognise on it after the outer rule has fired.
    """

    name = "tilde-intrinsic-formula"

    def recognize(self, expr: Expr) -> Optional[TildeIntrinsicFormulaMatch]:
        if not isinstance(expr, MultiEval):
            return None
        if expr.slot_kind != "covector":
            return None
        head = expr.head
        if not isinstance(head, Act):
            return None
        op = head.op
        V = head.arg
        if isinstance(op, TildeInteriorProduct):
            return TildeIntrinsicFormulaMatch(
                operator="tilde_interior",
                form=op.form,
                # ι̃_ω carries no bivector itself; surface ω as the
                # only "structural" parameter and leave bivector
                # implicit (caller must look it up elsewhere).
                bivector=op.form,
                multivector=V,
                args=expr.args,
                alternating=expr.alternating,
                slot_kind=expr.slot_kind,
            )
        if isinstance(op, TildeLieDerivative):
            return TildeIntrinsicFormulaMatch(
                operator="tilde_lie",
                form=op.form,
                bivector=op.bivector,
                multivector=V,
                args=expr.args,
                alternating=expr.alternating,
                slot_kind=expr.slot_kind,
            )
        if isinstance(op, TildeExteriorDerivative):
            return TildeIntrinsicFormulaMatch(
                operator="tilde_exterior_d",
                form=None,
                bivector=op.bivector,
                multivector=V,
                args=expr.args,
                alternating=expr.alternating,
                slot_kind=expr.slot_kind,
            )
        return None

    def classify(self, expr: Expr) -> Optional[str]:
        """Return the operator label (``"tilde_interior"`` /
        ``"tilde_lie"`` / ``"tilde_exterior_d"``) or ``None``.
        """
        match = self.recognize(expr)
        return None if match is None else match.operator


# --------------------------------------------------------------------- #
# Equivalence prover                                                     #
# --------------------------------------------------------------------- #


def prove_tilde_cartan_relation(
    lhs: Expr,
    rhs: Expr,
    *,
    etas: Tuple[Expr, ...],
    engine: ExpansionEngine,
    registry: Optional[PropertyRegistry] = None,
    alternating: bool = True,
) -> ProofChain:
    r"""Prove ``lhs == rhs`` by evaluating both on ``(η_1, …, η_p)``.

    Forms ``MultiEval(lhs, *etas, slot_kind="covector")`` and
    ``MultiEval(rhs, *etas, slot_kind="covector")``, then delegates to
    :func:`~jacopy.calculus.intrinsic_engine.prove_intrinsic_equivalence`
    with the supplied tilde engine. Returns a
    :class:`~jacopy.proof.chain.ProofChain` on success; raises
    :class:`~jacopy.proof.strategies.ProofFailure` with the surviving
    residue if the two sides do not collapse to syntactic equality.

    This is the tilde-side analogue of evaluating the textbook Cartan
    relations on a generic ``p``-tuple of vector fields and proving the
    resulting equality through the standard intrinsic engine. The
    ``slot_kind="covector"`` discipline routes the engine to the tilde
    intrinsic rules and away from the standard-side ones, so a single
    proof can carry both kinds of MultiEval nodes side-by-side without
    aliasing.

    Parameters
    ----------
    lhs, rhs
        Operator-valued :class:`Expr` whose equality is to be shown.
        Both sides typically read like ``Act(<tilde-op>, V)`` or a
        :class:`Sum` / :class:`Neg` thereof, the bare
        operator-on-multivector shape, not yet evaluated.
    etas
        Tuple of 1-forms ``(η_1, …, η_p)`` against which both sides
        are evaluated. Length determines the arity of the
        :class:`MultiEval` wrap; pick a length consistent with the
        degree of the multivector ``V`` flowing through the operator
        composition (e.g. ``len(etas) == 1`` for a 1-vector ``X``
        evaluated against a single 1-form).
    engine
        Tilde-aware :class:`ExpansionEngine`, normally constructed via
        :func:`tilde_intrinsic_engine` or
        :meth:`~jacopy.library.koszul_problem.KoszulProblem.tilde_intrinsic_engine`.
    registry
        Optional :class:`PropertyRegistry` plumbed through to
        :func:`prove_intrinsic_equivalence`'s simplify pipeline.
    alternating
        Whether the :class:`MultiEval` wrap is graded-antisymmetric in
        its argument slots. Defaults to ``True``, the convention for
        evaluating multivector-valued expressions.

    Notes
    -----
    The driver does *not* validate that ``len(etas)`` matches the
    operator-implied arity; the user is responsible for picking a
    consistent ``p``. A mismatched arity surfaces as a residual that
    fails to cancel, with the obstruction reported in the raised
    :class:`ProofFailure`.
    """
    if not isinstance(lhs, Expr):
        raise TypeError("prove_tilde_cartan_relation lhs must be an Expr")
    if not isinstance(rhs, Expr):
        raise TypeError("prove_tilde_cartan_relation rhs must be an Expr")
    if not isinstance(etas, tuple) or not etas:
        raise TypeError(
            "prove_tilde_cartan_relation etas must be a non-empty tuple of Expr"
        )
    for eta in etas:
        if not isinstance(eta, Expr):
            raise TypeError(
                "prove_tilde_cartan_relation etas entries must be Expr"
            )
    if not isinstance(engine, ExpansionEngine):
        raise TypeError(
            "prove_tilde_cartan_relation engine must be an ExpansionEngine"
        )

    lhs_eval = MultiEval(
        lhs, *etas, alternating=alternating, slot_kind="covector"
    )
    rhs_eval = MultiEval(
        rhs, *etas, alternating=alternating, slot_kind="covector"
    )
    return prove_intrinsic_equivalence(
        lhs_eval, rhs_eval, engine=engine, registry=registry
    )
