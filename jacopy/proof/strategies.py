"""
Proof strategies, algorithms that take two expressions and close the
gap between them with a :class:`ProofChain`.

Faz 7 A-scope ships only the base :class:`Strategy` interface and
:class:`ExpandAndSimplify`. Later fazlar add
:class:`AgreementOnGenerators` (operator-level equality via
derivation-extension), :class:`UnrollToFoundations`
(theorem-mode unrolling), and :class:`DerivedBracketStrategy`
(bracket-level Jacobi via the derived-bracket theorem).

The convention: a strategy either returns a non-empty
:class:`ProofChain` whose final step witnesses closure (typically
``after == 0``), or raises :class:`ProofFailure` with the residual. A
caller can then inspect the residual to understand what axiom or
grading declaration is missing.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Optional, Protocol, Tuple, runtime_checkable

from jacopy.algebra.derivation import Act, degree_of
from jacopy.algorithms.product_rule import product_rule
from jacopy.algorithms.simplify import simplify
from jacopy.brackets.derived import DerivedBracket
from jacopy.core.expr import Expr, Integer, Neg, Sum
from jacopy.core.registry import PropertyRegistry
from jacopy.core.symbolic_degree import Degree
from jacopy.proof.chain import ProofChain
from jacopy.proof.expansion import ExpansionEngine, default_engine
from jacopy.proof.step import ProofStep


class ProofFailure(Exception):
    """Raised when a strategy cannot close the gap between two expressions.

    The exception message includes the surviving residual, which is
    usually enough to diagnose whether the mismatch is genuine or
    whether a missing grading declaration / definition prevented the
    strategy from closing.

    When the raising strategy has a residual on hand, it may attach a
    :class:`DiagnosticReport` as the ``report`` kwarg, callers can
    read ``exc.report`` to get structural hints about which rewrite
    stalled. A ``report`` is optional; legacy call sites still raise
    with a bare message and ``report`` defaults to ``None``.
    """

    def __init__(self, message: str, *, report: Optional[Any] = None) -> None:
        super().__init__(message)
        self.report = report

    def __str__(self) -> str:
        base = super().__str__()
        if self.report is None or not self.report:
            return base
        return f"{base}\n\n{self.report.format()}"


class Strategy(ABC):
    """Abstract base for equality proof strategies."""

    name: str = "strategy"

    @abstractmethod
    def prove(
        self,
        lhs: Expr,
        rhs: Expr,
        *,
        registry: Optional[PropertyRegistry] = None,
        engine: Optional[ExpansionEngine] = None,
    ) -> ProofChain:
        """Produce a :class:`ProofChain` witnessing ``lhs == rhs``.

        Raises :class:`ProofFailure` if the strategy cannot close the
        gap. Implementations may ignore either ``registry`` or
        ``engine`` depending on how they work.
        """


class ExpandAndSimplify(Strategy):
    """Default strategy: form the obstruction, expand, simplify, check ``0``.

    The plan names this `ExpandAndSimplify`: unfold every definition
    the engine knows about, then drive the canonical-form pipeline
    (``flatten`` / ``canonicalize`` / ``distribute`` / ``sort_product``
    / ``collect_terms``) until the residual settles. If it lands on
    :class:`Integer` ``0`` the proof closes; otherwise
    :class:`ProofFailure` carries the surviving residual.

    :mod:`product_rule` is run in between, definition rewrites often
    leave :class:`Act` over :class:`Product` composition and
    :class:`Act` over :class:`Sum` shapes that :func:`simplify` alone
    doesn't unfold. Running Leibniz / linearity expansion once is the
    minimum needed to reduce everyday Cartan-calculus equalities to
    cancellation.
    """

    name = "ExpandAndSimplify"

    def prove(
        self,
        lhs: Expr,
        rhs: Expr,
        *,
        registry: Optional[PropertyRegistry] = None,
        engine: Optional[ExpansionEngine] = None,
    ) -> ProofChain:
        eng = engine if engine is not None else default_engine()
        chain = ProofChain()

        # Reflexive shortcut, syntactically identical, nothing to do.
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

        # Phases 1-2 interleave until a fix-point. Definition expansion
        # often lands shapes (``Act(d∘ι_X, arg)``, ``Act(Sum(...), arg)``)
        # that only unfold once :mod:`product_rule` distributes the
        # Leibniz / linearity through, and that distribution itself
        # exposes fresh element-level shapes (``Act(d, Act(d, x))``,
        # ``Act(ι_X, Act(d, f))``) that axioms like ``d² = 0`` and
        # ``ι_X(df) = X(f)`` only match in post-composition form. Looping
        # both passes under a single fix-point is the minimum cost to
        # make Cartan relations like ``[d, L_X] = 0`` and
        # ``[L_X, L_Y] = L_{[X,Y]}`` close via cancellation.
        current: Expr = obstruction
        for _ in range(64):
            expanded, exp_steps = eng.expand(current)
            if exp_steps:
                chain.extend(exp_steps)
            after_pr = product_rule(expanded, registry)
            if after_pr != expanded:
                chain.append(
                    ProofStep(
                        expanded,
                        after_pr,
                        rule="product-rule",
                        justification="graded Leibniz + linearity",
                    )
                )
            if after_pr == current:
                break
            current = after_pr
        else:
            raise ProofFailure(
                "ExpandAndSimplify expand/product-rule loop did not "
                f"converge on {lhs._repr_inner()} == {rhs._repr_inner()}"
            )

        # Phase 3: canonical simplification.
        reduced = simplify(current, registry)
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
            # Attach a structural diagnostic report so callers (and the
            # agent driving the proof) can see why the pipeline stalled
            # without re-deriving the residual by eye. Imported locally
            # to avoid a cycle with the diagnostics package import-time
            # rule-registration side effects.
            from jacopy.proof.diagnostics import diagnose

            report = diagnose(reduced, registry=registry, engine=eng)
            raise ProofFailure(
                f"ExpandAndSimplify left residual {reduced._repr_inner()} "
                f"when proving {lhs._repr_inner()} == {rhs._repr_inner()}",
                report=report,
            )

        # If no intermediate step recorded a change (everything cancelled
        # inside ``Sum(lhs, Neg(rhs))`` via trivial ctor folds) still
        # log a closing step so the chain is non-empty.
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


# --------------------------------------------------------------------- #
# AgreementOnGenerators                                                  #
# --------------------------------------------------------------------- #


@runtime_checkable
class _HasGenerators(Protocol):
    """Duck type for algebras AgreementOnGenerators can consume."""

    @property
    def generators(self) -> Tuple[Expr, ...]: ...  # pragma: no cover


def _operator_degree(op: Expr, registry: Optional[PropertyRegistry]) -> Degree:
    """Degree of an operator expression, handling a Sum of operators.

    :func:`degree_of` deliberately refuses to walk :class:`Sum` since a
    Sum has a single degree only when every term's degree agrees. For
    operator-level arguments to :class:`AgreementOnGenerators` that
    homogeneity is exactly the well-formedness condition we want to
    check, so this helper performs the walk and raises
    :class:`ValueError` when the summands disagree.
    """
    if isinstance(op, Sum):
        degrees = [_operator_degree(c, registry) for c in op.children]
        first = degrees[0]
        for other in degrees[1:]:
            if other != first:
                raise ValueError(
                    f"Operator sum has inhomogeneous degrees: {degrees}"
                )
        return first
    return degree_of(op, registry)


class AgreementOnGenerators(Strategy):
    """Operator equality from agreement on algebra generators.

    Two graded derivations of the same degree that agree on a
    generating set of the algebra extend uniquely to equal operators
    everywhere. This strategy discharges that uniqueness argument
    automatically:

    1. Both operators must have the same (well-defined, homogeneous)
       degree.
    2. For each generator ``g`` in ``algebra.generators``, a sub-proof
       closes the element-level equality ``lhs(g) == rhs(g)``.
    3. The parent :class:`ProofStep` records the conclusion, with each
       sub-proof's top step attached as a child so the transcript
       reads hierarchically.

    The sub-strategy defaults to :class:`ExpandAndSimplify`; callers
    who need a more specialized per-generator tactic can pass any
    :class:`Strategy`. A :class:`ProofFailure` in the sub-proof on any
    generator is re-raised, annotated with which generator failed,
    that pointer is usually the shortest path to the missing grading
    declaration or axiom.
    """

    name = "AgreementOnGenerators"

    def __init__(
        self,
        algebra: _HasGenerators,
        *,
        sub_strategy: Optional[Strategy] = None,
    ) -> None:
        if not hasattr(algebra, "generators"):
            raise TypeError(
                "AgreementOnGenerators requires an algebra with a "
                "'generators' property"
            )
        self._algebra = algebra
        self._sub = sub_strategy if sub_strategy is not None else ExpandAndSimplify()

    def prove(
        self,
        lhs: Expr,
        rhs: Expr,
        *,
        registry: Optional[PropertyRegistry] = None,
        engine: Optional[ExpansionEngine] = None,
    ) -> ProofChain:
        # Degree well-formedness. The zero operator Integer(0) is
        # polymorphic in degree, it sits in every graded slot as the
        # unique element that vanishes, so an equality against it only
        # requires the non-zero side to have a well-defined homogeneous
        # degree.
        lhs_is_zero = lhs == Integer(0)
        rhs_is_zero = rhs == Integer(0)
        try:
            deg_lhs = (
                _operator_degree(lhs, registry) if not lhs_is_zero else None
            )
            deg_rhs = (
                _operator_degree(rhs, registry) if not rhs_is_zero else None
            )
        except ValueError as exc:
            raise ProofFailure(
                f"AgreementOnGenerators needs determinable degrees: {exc}"
            )
        if deg_lhs is not None and deg_rhs is not None and deg_lhs != deg_rhs:
            raise ProofFailure(
                f"Operators have distinct degrees: "
                f"|lhs| = {deg_lhs}, |rhs| = {deg_rhs}"
            )
        deg_for_step = deg_lhs if deg_lhs is not None else deg_rhs

        generators = self._algebra.generators
        if not generators:
            raise ProofFailure(
                "Algebra has no generators, cannot apply agreement strategy"
            )

        parent = ProofStep(
            lhs,
            rhs,
            rule=self.name,
            justification=(
                f"both of degree {deg_for_step}; agree on "
                f"{len(generators)} generator(s)"
            ),
        )

        for g in generators:
            lhs_on_g = Act(lhs, g)
            rhs_on_g = Act(rhs, g)
            try:
                sub_chain = self._sub.prove(
                    lhs_on_g,
                    rhs_on_g,
                    registry=registry,
                    engine=engine,
                )
            except ProofFailure as exc:
                raise ProofFailure(
                    f"AgreementOnGenerators failed on generator "
                    f"{g._repr_inner()}: {exc}"
                )
            # Attach the sub-proof under a per-generator child step so
            # the transcript tree reads as "agreement ← on g1, on g2, …".
            gen_step = ProofStep(
                lhs_on_g,
                rhs_on_g,
                rule="check-on-generator",
                justification=f"generator g = {g._repr_inner()}",
            )
            for s in sub_chain:
                gen_step.add_child(s)
            parent.add_child(gen_step)

        chain = ProofChain([parent])
        return chain


# --------------------------------------------------------------------- #
# UnrollToFoundations                                                    #
# --------------------------------------------------------------------- #


class UnrollToFoundations(Strategy):
    """Meta-strategy: run an inner strategy against a foundational engine.

    The wrapped :class:`Strategy` sees an :class:`ExpansionEngine`
    switched into ``"foundational"`` mode, every theorem-classified
    :class:`Definition` fires as usual, but the resulting
    :class:`ProofStep` carries the theorem's own sub-proof attached as
    children rather than being taken as given. Axioms still appear as
    leaves, so the transcript bottoms out exactly at the declared
    foundations of the theory.

    This is the ispat mode the plan calls "full unroll": you pay the
    construction cost of every theorem's justification, but the chain
    is self-contained. When a caller passes an ``engine`` that is
    already foundational the strategy uses it verbatim; an efficient
    engine is rebuilt via :meth:`ExpansionEngine.with_mode`, and
    ``engine=None`` falls back to :func:`default_engine` in
    foundational mode.
    """

    name = "UnrollToFoundations"

    def __init__(self, inner: Strategy) -> None:
        if not isinstance(inner, Strategy):
            raise TypeError("UnrollToFoundations requires an inner Strategy")
        self._inner = inner

    @property
    def inner(self) -> Strategy:
        return self._inner

    def prove(
        self,
        lhs: Expr,
        rhs: Expr,
        *,
        registry: Optional[PropertyRegistry] = None,
        engine: Optional[ExpansionEngine] = None,
    ) -> ProofChain:
        if engine is None:
            eng = default_engine(registry=registry, mode="foundational")
        elif engine.mode == "foundational":
            eng = engine
        else:
            eng = engine.with_mode("foundational")
        return self._inner.prove(lhs, rhs, registry=registry, engine=eng)


# --------------------------------------------------------------------- #
# DerivedBracketStrategy                                                 #
# --------------------------------------------------------------------- #


class DerivedBracketStrategy:
    """Apply the Derived Bracket Theorem to discharge Jacobi.

    The theorem states that for any
    :class:`~jacopy.brackets.derived.DerivedBracket` ``{·, ·}_Q`` built
    over a base bracket ``[·, ·]``, the graded Jacobi identity on
    ``{·, ·}_Q`` holds *for all operands* if and only if the single
    equation

        [Q, Q]_base = 0

    holds. :meth:`prove_jacobi` uses that result to replace a triple
    cyclic Jacobi check by a universal obstruction: one reduction step
    tagged as a theorem, one expansion of the base bracket on
    ``(Q, Q)``, and a final simplification. If the obstruction
    collapses to :class:`Integer` ``0`` the proof closes, Jacobi holds
    on every triple, not just the one supplied. Otherwise the residue
    is carried in a :class:`ProofFailure`.

    The strategy isn't a :class:`Strategy` subclass: ``Strategy.prove``
    is a binary interface (``lhs == rhs``), but this tactic is
    parameterised by a bracket and a triple. It is the first entry
    point invoked by the verifier's :func:`prove_jacobi` when the
    supplied bracket is a derived bracket.
    """

    name = "DerivedBracketStrategy"

    def prove_jacobi(
        self,
        bracket: DerivedBracket,
        a: Expr,
        b: Expr,
        c: Expr,
        *,
        registry: Optional[PropertyRegistry] = None,
    ) -> ProofChain:
        if not isinstance(bracket, DerivedBracket):
            raise TypeError(
                "DerivedBracketStrategy requires a DerivedBracket"
            )

        try:
            jacobi_sum = bracket.graded_jacobi_obstruction(a, b, c, registry)
        except ValueError as exc:
            raise ProofFailure(
                f"DerivedBracketStrategy: Jacobi obstruction ill-formed "
                f"on ({a._repr_inner()}, {b._repr_inner()}, "
                f"{c._repr_inner()}): {exc}"
            )

        obstruction_raw = bracket.jacobi_obstruction_raw()
        obstruction = bracket.jacobi_obstruction(registry)
        reduced = simplify(obstruction, registry)

        chain = ProofChain()
        # Step 1: apply the derived bracket theorem, the triple Jacobi
        # check reduces to the universal obstruction [Q, Q]_base.
        chain.append(
            ProofStep(
                jacobi_sum,
                obstruction_raw,
                rule="DerivedBracketTheorem",
                justification=(
                    f"Jacobi on {bracket.name} ⟺ [Q, Q]_base = 0 "
                    f"(Derived Bracket Theorem)"
                ),
                provenance_tag="theorem",
            )
        )
        # Step 2: expand the base bracket on (Q, Q) using its own rule.
        if obstruction != obstruction_raw:
            chain.append(
                ProofStep(
                    obstruction_raw,
                    obstruction,
                    rule="base-bracket-expand",
                    justification=(
                        f"apply {bracket.base.name} definition to [Q, Q]"
                    ),
                )
            )
        # Step 3: canonical simplification of the obstruction.
        if reduced != obstruction:
            chain.append(
                ProofStep(
                    obstruction,
                    reduced,
                    rule="simplify",
                    justification="canonical-form pipeline",
                )
            )

        if reduced != Integer(0):
            raise ProofFailure(
                f"DerivedBracketStrategy: obstruction [Q, Q]_base "
                f"simplifies to {reduced._repr_inner()}, not 0; "
                f"Jacobi fails on {bracket.name}"
            )

        return chain
