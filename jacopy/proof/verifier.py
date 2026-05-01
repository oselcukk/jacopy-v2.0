"""
Public proof API.

High-level entry points that select a :class:`Strategy`, feed it a
pair of expressions, and return a :class:`ProofChain`. Faz 7 C adds
:func:`prove_jacobi` (with automatic dispatch to
:class:`DerivedBracketStrategy` for derived brackets),
:func:`prove_operator_equation` (uses
:class:`AgreementOnGenerators`), and :func:`unroll_property` (reads
a :class:`Property`'s provenance and surfaces the axioms it rests on).

The verifier module is the intended entry point for application
code, strategies and engines are plumbing that callers rarely name
directly.
"""

from __future__ import annotations

from typing import Any, Optional

from jacopy.brackets.base import BracketApply, GradedBracket
from jacopy.brackets.derived import DerivedBracket, VanishingCondition
from jacopy.core.expr import Expr, Integer
from jacopy.core.properties import Property, Provenance
from jacopy.core.registry import PropertyRegistry
from jacopy.proof.chain import ProofChain
from jacopy.proof.expansion import ExpansionEngine
from jacopy.proof.step import ProofStep
from jacopy.proof.strategies import (
    AgreementOnGenerators,
    DerivedBracketStrategy,
    ExpandAndSimplify,
    ProofFailure,
    Strategy,
)


def show_equal(
    lhs: Expr,
    rhs: Expr,
    *,
    registry: Optional[PropertyRegistry] = None,
    strategy: Optional[Strategy] = None,
    engine: Optional[ExpansionEngine] = None,
) -> ProofChain:
    """Prove ``lhs == rhs`` and return the transcript.

    Default strategy is :class:`ExpandAndSimplify` with the standard
    expansion engine. Raises
    :class:`jacopy.proof.strategies.ProofFailure` if the chosen
    strategy cannot close the gap, the exception message carries the
    surviving residual.
    """
    strat = strategy if strategy is not None else ExpandAndSimplify()
    return strat.prove(lhs, rhs, registry=registry, engine=engine)


def prove_jacobi(
    bracket: GradedBracket,
    a: Expr,
    b: Expr,
    c: Expr,
    *,
    registry: Optional[PropertyRegistry] = None,
    engine: Optional[ExpansionEngine] = None,
    strategy: Optional[Strategy] = None,
) -> ProofChain:
    """Prove the graded Jacobi identity on ``(a, b, c)`` for ``bracket``.

    Dispatches on the bracket type:

    * a :class:`DerivedBracket` is handled by
      :class:`DerivedBracketStrategy`, one theorem step replaces the
      triple Jacobi sum with the universal obstruction ``[Q, Q]_base``,
      which is then simplified to zero.
    * any other :class:`GradedBracket` falls back to
      :meth:`GradedBracket.graded_jacobi_obstruction` and runs
      :class:`ExpandAndSimplify` (or a caller-supplied ``strategy``)
      against :class:`Integer` ``0``.

    ``strategy`` is only consulted on the generic path; passing one
    together with a :class:`DerivedBracket` has no effect, since the
    derived-bracket theorem is preferred whenever it applies. Raises
    :class:`ProofFailure` when neither path closes.
    """
    if isinstance(bracket, DerivedBracket):
        return DerivedBracketStrategy().prove_jacobi(
            bracket, a, b, c, registry=registry,
        )
    obstruction = bracket.graded_jacobi_obstruction(a, b, c, registry)
    # `product_rule` can't reconstruct a :class:`BracketApply` from its
    # children alone (the bracket reference isn't one of them), so any
    # residual bracket node downstream breaks the ExpandAndSimplify
    # pipeline. Unfold every bracket node to its definitional form here
    #, the pipeline then only sees ordinary :class:`Sum` / :class:`Neg`
    # / :class:`Product` / atom shapes. Once Faz 7 adds a bracket-aware
    # definition to :class:`ExpansionEngine`, this pre-pass can move
    # into the engine and drop from the verifier layer.
    flattened = _expand_all_brackets(obstruction, registry)
    chain = ProofChain()
    if flattened != obstruction:
        chain.append(
            ProofStep(
                obstruction,
                flattened,
                rule="bracket-expand",
                justification="expand bracket nodes by definition",
            )
        )
    strat = strategy if strategy is not None else ExpandAndSimplify()
    inner = strat.prove(flattened, Integer(0), registry=registry, engine=engine)
    chain.extend(inner)
    return chain


def prove_equivalence(
    stmt1: Any,
    stmt2: Any,
    *operands: Expr,
    registry: Optional[PropertyRegistry] = None,
    engine: Optional[ExpansionEngine] = None,
    strategy: Optional[Strategy] = None,
) -> ProofChain:
    """Prove two definitions (or statements) are equivalent.

    The helper dispatches on the type pair so that callers can use a
    single verb regardless of whether they're comparing brackets,
    vanishing conditions, or raw expressions:

    * ``GradedBracket, GradedBracket`` with two trailing :class:`Expr`
      operands ``a, b``, expands each bracket on ``(a, b)`` and proves
      the two expansions equal. This is the classical-vs-derived
      bracket equivalence the package uses for Koszul, Poisson, and
      Courant identifications.
    * ``VanishingCondition, VanishingCondition``, proves the two
      obstructions are equal as expressions. Two conditions with equal
      obstructions describe the same constraint even when their display
      names differ; callers typically use this to relate e.g.
      ``[Θ + H, Θ + H]_SN = 0`` to ``dH = 0``.
    * ``Expr, Expr``, straight aliasing of :func:`show_equal`, provided
      so a caller can write ``prove_equivalence`` without switching
      verbs mid-file.

    The sub-proof uses :class:`ExpandAndSimplify` by default;
    ``strategy`` overrides it. Raises :class:`ProofFailure` when the
    underlying strategy leaves a residual, or :class:`TypeError` on
    unsupported type combinations.
    """
    if isinstance(stmt1, GradedBracket) and isinstance(stmt2, GradedBracket):
        if len(operands) != 2:
            raise TypeError(
                "prove_equivalence on two GradedBrackets requires exactly "
                "two trailing Expr operands (a, b)"
            )
        a, b = operands
        if not isinstance(a, Expr) or not isinstance(b, Expr):
            raise TypeError(
                "prove_equivalence bracket operands must be Expr instances"
            )
        lhs = stmt1.expand(a, b, registry)
        rhs = stmt2.expand(a, b, registry)
        # Both sides may still contain nested BracketApply nodes that
        # product_rule can't rebuild (same limitation as prove_jacobi),
        # so flatten before handing off.
        lhs_flat = _expand_all_brackets(lhs, registry)
        rhs_flat = _expand_all_brackets(rhs, registry)
        strat = strategy if strategy is not None else ExpandAndSimplify()
        return strat.prove(lhs_flat, rhs_flat, registry=registry, engine=engine)

    if isinstance(stmt1, VanishingCondition) and isinstance(stmt2, VanishingCondition):
        if operands:
            raise TypeError(
                "prove_equivalence on VanishingConditions takes no operands"
            )
        strat = strategy if strategy is not None else ExpandAndSimplify()
        return strat.prove(
            stmt1.obstruction,
            stmt2.obstruction,
            registry=registry,
            engine=engine,
        )

    if isinstance(stmt1, Expr) and isinstance(stmt2, Expr):
        if operands:
            raise TypeError(
                "prove_equivalence on Expr pair takes no additional operands"
            )
        return show_equal(
            stmt1, stmt2,
            registry=registry,
            strategy=strategy,
            engine=engine,
        )

    raise TypeError(
        f"prove_equivalence: unsupported pair "
        f"({type(stmt1).__name__}, {type(stmt2).__name__}); "
        f"expected two GradedBrackets, two VanishingConditions, or two Exprs"
    )


def _expand_all_brackets(
    expr: Expr, registry: Optional[PropertyRegistry]
) -> Expr:
    """Recursively replace every :class:`BracketApply` with its expansion.

    Walks the tree bottom-up; on a :class:`BracketApply` node the
    bracket's own :meth:`~jacopy.brackets.base.GradedBracket.expand`
    produces the defining formula, which is then re-walked so nested
    bracket nodes inside the expansion get unfolded too. Non-bracket
    nodes rebuild structurally via :meth:`Expr._rebuild`.
    """
    if isinstance(expr, BracketApply):
        expanded = expr.expand(registry)
        return _expand_all_brackets(expanded, registry)
    if expr.is_atom:
        return expr
    new_children = tuple(_expand_all_brackets(c, registry) for c in expr.children)
    if new_children == tuple(expr.children):
        return expr
    return expr._rebuild(new_children)


def prove_operator_equation(
    op1: Expr,
    op2: Expr,
    algebra: Any,
    *,
    registry: Optional[PropertyRegistry] = None,
    engine: Optional[ExpansionEngine] = None,
    sub_strategy: Optional[Strategy] = None,
) -> ProofChain:
    """Prove the operator equation ``op1 == op2`` on ``algebra``.

    A thin wrapper around :class:`AgreementOnGenerators`: two graded
    derivations that agree on a generating set extend uniquely to
    equal operators. ``algebra`` must expose a ``generators``
    property, :class:`~jacopy.calculus.exterior_algebra.ExteriorAlgebra`
    is the canonical example. ``sub_strategy`` is forwarded to
    :class:`AgreementOnGenerators` and controls the per-generator
    element-level proof (defaults to :class:`ExpandAndSimplify`).
    Raises :class:`ProofFailure` when the degree well-formedness fails
    or any generator's sub-proof fails.
    """
    strat = AgreementOnGenerators(algebra, sub_strategy=sub_strategy)
    return strat.prove(op1, op2, registry=registry, engine=engine)


def unroll_property(prop: Property) -> ProofChain:
    """Surface the axioms / rule a :class:`Property` depends on.

    Behaviour:

    * :attr:`Provenance.AXIOM`, returns a one-step chain whose single
      :class:`ProofStep` is tagged ``"axiom"`` with a justification
      recording that the property is declared, not derived.
    * :attr:`Provenance.DERIVED`, returns a one-step chain tagged
      ``"theorem"`` whose justification names the
      :class:`~jacopy.core.properties.ProofRef` rule and lists the
      sources the property depends on.

    The chain is intentionally shallow. A :class:`ProofRef` today only
    carries a rule name and a tuple of source specifiers, there is
    no rehydratable proof tree to expand. Once the Theorem Book (Faz
    9) lets a :class:`ProofRef` resolve to a concrete proof builder,
    this helper will delegate to it and return the full unrolled
    chain; until then the one-step summary is the contract.
    """
    if not isinstance(prop, Property):
        raise TypeError("unroll_property requires a Property")

    # Build a marker Expr pair purely for the transcript's sake. The
    # ``before``/``after`` positions on a property-level step are a
    # bit artificial, the step isn't a rewrite, so we reuse the
    # property's repr on both sides to make the chain displayable.
    marker = _PropertyMarker(prop)

    chain = ProofChain()
    if prop.provenance is Provenance.AXIOM:
        chain.append(
            ProofStep(
                marker,
                marker,
                rule="axiom",
                justification=f"{type(prop).__name__} declared axiomatically",
                provenance_tag="axiom",
            )
        )
        return chain

    if prop.provenance is Provenance.DERIVED:
        ref = prop.proof
        assert ref is not None, "DERIVED property must carry a ProofRef"
        sources = ", ".join(ref.sources) if ref.sources else "no recorded sources"
        chain.append(
            ProofStep(
                marker,
                marker,
                rule=ref.rule,
                justification=(
                    f"{type(prop).__name__} derived by {ref.rule}; "
                    f"sources: {sources}"
                ),
                provenance_tag="theorem",
            )
        )
        return chain

    raise ProofFailure(
        f"unknown provenance on property {prop!r}"
    )


class _PropertyMarker(Expr):
    """Placeholder :class:`Expr` that labels a :class:`ProofStep` whose
    subject is a :class:`Property`, not an expression rewrite.

    :func:`unroll_property` records a property-level fact as a single
    :class:`ProofStep`. Step construction requires both ``before`` and
    ``after`` to be :class:`Expr`, so we wrap the property in this
    minimal atom-like node. Its only job is to render the property
    name when :meth:`~jacopy.proof.step.ProofStep.format` asks.
    """

    __slots__ = ("_prop",)

    def __init__(self, prop: Property) -> None:
        self._prop = prop

    @property
    def prop(self) -> Property:
        return self._prop

    @property
    def children(self) -> tuple:
        return ()

    def _key(self) -> Any:
        return (type(self._prop).__name__, self._prop)

    def _repr_inner(self) -> str:
        return type(self._prop).__name__
