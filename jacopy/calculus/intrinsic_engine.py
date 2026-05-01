r"""
Public surface over the intrinsic-formula axioms (Faz 12.A.5).

Three pieces:

* :func:`intrinsic_engine`, bundles the seven Faz 12.A rules
  (3 intrinsic + 4 multi-eval helpers) into a single
  :class:`~jacopy.proof.expansion.ExpansionEngine`. Callers who want
  the same plumbing as the Cartan demo notebooks reach for this
  instead of assembling rule lists by hand.

* :class:`IntrinsicFormulaRecognizer`, a pure-shape inspector,
  mirroring the Faz 7 :mod:`jacopy.proof.recognizers` pattern. Given
  an :class:`~jacopy.core.multi_eval.MultiEval`, it answers "is the
  head an :math:`\iota`, :math:`L`, or :math:`d` applied to a form,
  and if so what's inside?". Doesn't rewrite, feeds higher-level
  tactics the structural fields they need.

* :func:`prove_intrinsic_equivalence`, wraps the
  ``Sum(lhs, Neg(rhs)) → simplify`` cycle into a single call that
  returns a :class:`~jacopy.proof.chain.ProofChain` on success or
  raises :class:`~jacopy.proof.strategies.ProofFailure` on residual.
  The intrinsic counterpart of
  :class:`~jacopy.proof.strategies.ExpandAndSimplify`, but driven by
  the seven Faz 12.A rules rather than the package-wide default
  engine, the latter doesn't carry multi-eval / canonicalize axioms
  and so leaves the textbook Cartan obstructions unreduced.

Faz 12.A.5 is *ergonomic*: it adds no new axioms. The four Cartan
relations that close end-to-end (ιι anti-commute, Cartan magic,
``[L_X, ι_Y] = ι_{[X,Y]}``, ι²=0) are exactly the four that already
closed manually in Faz 12.A.4. The other three (``d²=0``,
``[d, L_X]=0``, ``[L_X, L_Y] = L_{[X,Y]}``) are blocked on the
generic VF-commutator + VF-Jacobi axioms scheduled for Faz 12.A.6,
once those land, the same :func:`prove_intrinsic_equivalence` call
will close them without any API change at this layer.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

from jacopy.algebra.derivation import Act
from jacopy.algorithms.product_rule import product_rule
from jacopy.algorithms.simplify import simplify
from jacopy.calculus.closure_axioms import (
    IotaActAsScalarDefinition,
    LieBracketVfAntiSymmetryDefinition,
    LieBracketVfJacobiDefinition,
    VfActCommutatorDefinition,
)
from jacopy.calculus.exterior_d import ExteriorDerivative
from jacopy.calculus.interior import InteriorProduct
from jacopy.calculus.intrinsic_axioms import (
    ExteriorDIntrinsicDefinition,
    InteriorProductIntrinsicDefinition,
    LieDerivativeIntrinsicDefinition,
)
from jacopy.calculus.lie_derivative import LieDerivative
from jacopy.calculus.multi_eval_axioms import (
    MultiEvalAlternatingNormalDefinition,
    MultiEvalArgLinearityDefinition,
    MultiEvalHeadLinearityDefinition,
    MultiEvalRepeatArgZeroDefinition,
    MultiEvalZeroHeadDefinition,
)
from jacopy.core.expr import Expr, Integer, Neg, Sum
from jacopy.core.multi_eval import MultiEval
from jacopy.core.registry import PropertyRegistry
from jacopy.proof.chain import ProofChain
from jacopy.proof.expansion import ExpansionEngine
from jacopy.proof.step import ProofStep
from jacopy.proof.strategies import ProofFailure


# --------------------------------------------------------------------- #
# Engine factory                                                         #
# --------------------------------------------------------------------- #


def intrinsic_engine() -> ExpansionEngine:
    """Build a fresh :class:`ExpansionEngine` carrying the Faz 12.A rules.

    The bundle (in match order):

    * :class:`InteriorProductIntrinsicDefinition`     (12.A.1)
    * :class:`LieDerivativeIntrinsicDefinition`       (12.A.2)
    * :class:`ExteriorDIntrinsicDefinition`           (12.A.3)
    * :class:`MultiEvalArgLinearityDefinition`        (12.A.0)
    * :class:`MultiEvalHeadLinearityDefinition`       (12.A.0)
    * :class:`MultiEvalRepeatArgZeroDefinition`       (12.A.0)
    * :class:`MultiEvalAlternatingNormalDefinition`   (12.A.4)

    Callers are free to extend the returned engine with additional
    rules, once 12.A.6 lands its VF-commutator and VF-Jacobi axioms,
    a follow-up factory will return the closure-complete bundle.
    """
    return ExpansionEngine(
        [
            InteriorProductIntrinsicDefinition(),
            LieDerivativeIntrinsicDefinition(),
            ExteriorDIntrinsicDefinition(),
            MultiEvalArgLinearityDefinition(),
            MultiEvalHeadLinearityDefinition(),
            MultiEvalRepeatArgZeroDefinition(),
            MultiEvalZeroHeadDefinition(),
            MultiEvalAlternatingNormalDefinition(),
        ]
    )


def intrinsic_engine_with_closure() -> ExpansionEngine:
    """Like :func:`intrinsic_engine`, plus the Faz 12.A.6 closure axioms.

    Bundles the seven base rules with three Sum-level closure rules:

    * :class:`~jacopy.calculus.closure_axioms.VfActCommutatorDefinition`
     , folds ``Act(X, Act(Y, f)) − Act(Y, Act(X, f))`` into
      ``Act([X, Y]_VF, f)`` (the post-intrinsic-expansion residue
      shape; the L-specific Faz 13.C rule no longer fires because
      ``Act(L_X, ω)`` has been expanded out).
    * :class:`~jacopy.calculus.closure_axioms.LieBracketVfAntiSymmetryDefinition`
     , cancels ``[X,Y]_VF(…) + [Y,X]_VF(…)``. :class:`LieBracketVF`
      atoms are opaque, so ``[X,Y]`` and ``[Y,X]`` don't collapse on
      construction; the rule supplies that cancellation when the
      Cartan-residue distribution exposes both orientations together.
    * :class:`~jacopy.calculus.closure_axioms.LieBracketVfJacobiDefinition`
     , collapses any sign-permuted three-bracket cyclic into ``0``,
      regardless of which Cartan relation produced the residue.
    * :class:`~jacopy.calculus.closure_axioms.IotaActAsScalarDefinition`
     , bridges bare ``Act(D, Act(ι_X, ω))`` (where ``D`` is a plain
      vector field) into ``Act(D, MultiEval(ω, X))`` so the arity-1
      d-residue ``Y(ι_X ω)`` of 1-form Cartan magic eval-Y reaches
      ``Y(ω(X))``.

    Use this engine via :func:`prove_intrinsic_equivalence` to close
    the three Cartan relations that 12.A.5's bare bundle leaves open
    (``[d, L_X] = 0``, ``d² = 0``, ``[L_X, L_Y] = L_{[X,Y]}``) on
    1- and 2-forms, plus the 1-form Cartan magic eval-Y variant.
    """
    return ExpansionEngine(
        [
            InteriorProductIntrinsicDefinition(),
            LieDerivativeIntrinsicDefinition(),
            ExteriorDIntrinsicDefinition(),
            MultiEvalArgLinearityDefinition(),
            MultiEvalHeadLinearityDefinition(),
            MultiEvalRepeatArgZeroDefinition(),
            MultiEvalZeroHeadDefinition(),
            MultiEvalAlternatingNormalDefinition(),
            VfActCommutatorDefinition(),
            LieBracketVfAntiSymmetryDefinition(),
            LieBracketVfJacobiDefinition(),
            IotaActAsScalarDefinition(),
        ]
    )


# --------------------------------------------------------------------- #
# Recognizer                                                             #
# --------------------------------------------------------------------- #


@dataclass(frozen=True)
class IntrinsicFormulaMatch:
    """Successful :class:`IntrinsicFormulaRecognizer` result.

    Fields:
      * ``operator``, ``"interior"``, ``"lie"``, or ``"exterior_d"``.
      * ``vector_field``, the ``X`` in :math:`\\iota_X` / :math:`L_X`;
        ``None`` for :math:`d` (which carries no slot).
      * ``omega``, the underlying form (``head.arg``).
      * ``args``, the multi-eval evaluation slots.
      * ``alternating`` / ``slot_kind``, flags carried verbatim from
        the matched :class:`MultiEval`, so a downstream tactic that
        rebuilds doesn't have to reach back into the original node.
    """

    operator: str
    vector_field: Optional[Expr]
    omega: Expr
    args: Tuple[Expr, ...]
    alternating: bool
    slot_kind: str


class IntrinsicFormulaRecognizer:
    """Recognize ``MultiEval(Act(op, ω), Y_1, …, Y_p)`` for ``op ∈ {ι, L, d}``.

    Pure shape inspection, answers "is this a wrapped intrinsic
    operator applied to a multilinear evaluation, and if so what's
    inside?". No engine, no rewrite. Mirrors the pattern from
    :mod:`jacopy.proof.recognizers`: :meth:`recognize` returns the
    structured :class:`IntrinsicFormulaMatch`, :meth:`classify` returns
    just the operator label, and both return ``None`` on a non-match.

    The recognizer only inspects the *outer* head; nested operators
    (e.g. ``Act(L_X, Act(ι_Y, ω))``) are not unwrapped, the inner
    ``Act(ι_Y, ω)`` is reported verbatim as ``omega``. That keeps the
    recognizer composable: a caller can re-recognize on the inner
    ``Act`` after the outer rule has fired.
    """

    name = "intrinsic-formula"

    def recognize(self, expr: Expr) -> Optional[IntrinsicFormulaMatch]:
        if not isinstance(expr, MultiEval):
            return None
        head = expr.head
        if not isinstance(head, Act):
            return None
        op = head.op
        omega = head.arg
        if isinstance(op, InteriorProduct):
            return IntrinsicFormulaMatch(
                operator="interior",
                vector_field=op.vector_field,
                omega=omega,
                args=expr.args,
                alternating=expr.alternating,
                slot_kind=expr.slot_kind,
            )
        if isinstance(op, LieDerivative):
            return IntrinsicFormulaMatch(
                operator="lie",
                vector_field=op.vector_field,
                omega=omega,
                args=expr.args,
                alternating=expr.alternating,
                slot_kind=expr.slot_kind,
            )
        if isinstance(op, ExteriorDerivative):
            return IntrinsicFormulaMatch(
                operator="exterior_d",
                vector_field=None,
                omega=omega,
                args=expr.args,
                alternating=expr.alternating,
                slot_kind=expr.slot_kind,
            )
        return None

    def classify(self, expr: Expr) -> Optional[str]:
        """Return just the operator label (``"interior"`` / ``"lie"`` /
        ``"exterior_d"``) or ``None``, convenience over :meth:`recognize`.
        """
        match = self.recognize(expr)
        return None if match is None else match.operator


# --------------------------------------------------------------------- #
# Equivalence prover                                                     #
# --------------------------------------------------------------------- #


def prove_intrinsic_equivalence(
    lhs: Expr,
    rhs: Expr,
    *,
    engine: Optional[ExpansionEngine] = None,
    registry: Optional[PropertyRegistry] = None,
) -> ProofChain:
    """Prove ``lhs == rhs`` via the intrinsic engine; return the transcript.

    Forms ``Sum(lhs, Neg(rhs))``, runs the engine to fix-point, then
    a final :func:`~jacopy.algorithms.simplify.simplify` pass. Closes
    only when the residue syntactically reduces to ``0``; otherwise
    raises :class:`~jacopy.proof.strategies.ProofFailure` with the
    surviving residue in the message.

    Why a dedicated entry point rather than ``show_equal``: the
    package-wide ``default_engine()`` doesn't carry multi-eval /
    canonicalize axioms, so it leaves textbook Cartan obstructions
    (``ω(X, [Y,Z]_VF) + ω([Y,Z]_VF, X)``-style residues) unreduced.
    Callers who want the *intrinsic* picture, Cartan magic on a
    p-form, ι-tower expansion, Koszul d-formula, should reach for
    this function. Generic operator-equation work (``L²``, ``d²``,
    bracket Jacobi) still belongs to ``show_equal``.

    Override ``engine`` to extend the rule set (e.g. once 12.A.6 lands
    its VF-commutator and VF-Jacobi axioms, pass an enriched engine
    to close the remaining three Cartan relations); ``registry``
    plumbs through the simplify pipeline if the surrounding context
    has registered properties that affect canonical form.
    """
    eng = engine if engine is not None else intrinsic_engine()
    reg = registry if registry is not None else PropertyRegistry()
    chain = ProofChain()

    if lhs == rhs:
        chain.append(
            ProofStep(
                lhs,
                rhs,
                rule="reflexive",
                justification="lhs and rhs are syntactically identical",
            )
        )
        return chain

    obstruction: Expr = Sum(lhs, Neg(rhs))

    # Expand / product-rule fix-point. The closure axioms (Faz 12.A.6)
    # only see their target shapes after Act has been distributed
    # through Sum / Neg via :mod:`product_rule`; running the engine
    # alone would leave the VF-commutator pair buried under an
    # ``Act(_, Sum(...))``. Mirrors :class:`ExpandAndSimplify`'s loop,
    # cap at 64 iterations to surface non-converging axiom sets.
    current: Expr = obstruction
    for _ in range(64):
        expanded, exp_steps = eng.expand(current)
        if exp_steps:
            chain.extend(exp_steps)
        after_pr = product_rule(expanded, reg)
        if after_pr != expanded:
            chain.append(
                ProofStep(
                    expanded,
                    after_pr,
                    rule="product-rule",
                    justification="graded Leibniz + linearity",
                )
            )
        # Canonicalize intermediate Sum/Neg nesting so Sum-level rules
        # (VfActCommutator, LieBracketVfJacobi) see a single
        # children-flat Sum rather than the deeply-nested tree that
        # Act-distribution leaves behind. ``simplify`` is idempotent;
        # running it inside the loop costs little and is the cheapest
        # way to expose the closure rules to a flat residue.
        normed = simplify(after_pr, reg)
        if normed != after_pr:
            chain.append(
                ProofStep(
                    after_pr,
                    normed,
                    rule="simplify",
                    justification="canonical-form pipeline (intra-loop)",
                )
            )
        if normed == current:
            break
        current = normed
    else:
        raise ProofFailure(
            "prove_intrinsic_equivalence expand/product-rule loop did "
            f"not converge on {lhs._repr_inner()} == {rhs._repr_inner()}"
        )

    reduced = simplify(current, reg)
    if reduced != current:
        chain.append(
            ProofStep(
                current,
                reduced,
                rule="simplify",
                justification="canonical-form pipeline",
            )
        )

    if reduced != Integer(0):
        raise ProofFailure(
            f"prove_intrinsic_equivalence left residual "
            f"{reduced._repr_inner()} when proving "
            f"{lhs._repr_inner()} == {rhs._repr_inner()}"
        )

    if not chain:
        chain.append(
            ProofStep(
                obstruction,
                reduced,
                rule="simplify",
                justification="obstruction cancels under canonical form",
            )
        )

    return chain
