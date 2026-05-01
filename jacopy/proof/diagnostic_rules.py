"""
Built-in diagnostic rules for :func:`jacopy.proof.diagnostics.diagnose`.

Each rule is a pure tree-walk on the residual :class:`Expr` plus the
context (registry, engine). When it recognises a *stalled shape*, a
subtree that a well-wired proof pipeline should have rewritten but
didn't, it emits one or more :class:`DiagnosticHint`. Rules are
intentionally independent: catching the same stall via two rules is
fine, the dispatcher de-dupes by ``(category, location)``.

The catalogue below is seeded from the five modelling gaps closed in
the Cartan ``verify()`` closure pass, every gap there corresponds to
a rule that would have flagged the residual before the fix landed:

* ``stalled-d-squared`` / ``stalled-iota-squared``,
  ``Act(d, Act(d, x))`` survived despite ``d² = 0``.
* ``stalled-act-over-zero``, ``Act(op, 0)`` should annihilate.
* ``stalled-act-over-neg-op``, ``Act(Neg(op), x)`` should peel.
* ``unreduced-iota-on-df``, ``ι_V(d(f))`` where ``V`` is a sum or
  product of derivations the narrow
  :class:`IotaOnExactOneFormDefinition` pre-B.2 wouldn't match.
* ``unclassified-factor``, a :class:`Product` factor with neither a
  registered grading nor an intrinsic :class:`Derivation.degree`.

Registration happens at import time, so simply importing this module
(done from :mod:`jacopy.proof.__init__`) is enough to make the rules
active for any call to :func:`diagnose`.
"""

from __future__ import annotations

from typing import Iterable, Optional

from jacopy.algebra.derivation import Act, Derivation
from jacopy.core.expr import Expr, Integer, Neg, Product, Rational, Sum
from jacopy.core.properties import Graded, Scalar
from jacopy.core.registry import PropertyRegistry
from jacopy.proof.diagnostics import (
    DiagnosticHint,
    register_rule,
)
from jacopy.proof.expansion import ExpansionEngine


# --------------------------------------------------------------------- #
# Helpers                                                                #
# --------------------------------------------------------------------- #


def _is_derivation_combination(expr: Expr) -> bool:
    """True when ``expr`` is a Derivation or a Sum/Product/Neg of Derivations.

    This is the same predicate that ``IotaOnExactOneFormDefinition``
    uses post-B.2 to decide whether a vector-field argument is eligible
    for ``ι_V(df) = V(f)``. Keeping a local copy lets the diagnostic
    layer stay independent of the expansion-engine definitions, the
    hint can still fire even when the caller swapped out the iota
    definition for a custom one.
    """
    if isinstance(expr, Derivation):
        return True
    if isinstance(expr, (Sum, Product, Neg)):
        return all(_is_derivation_combination(c) for c in expr.children)
    return False


def _classifies(
    factor: Expr, registry: Optional[PropertyRegistry]
) -> bool:
    """True when ``sort_product`` would accept this factor.

    Mirrors the check in :func:`jacopy.algorithms.sort_product._degree_of`:
    scalars, numeric literals, :class:`Derivation` self-describers,
    registry-declared :class:`Graded` factors, and compound shapes
    whose degree :func:`jacopy.algebra.derivation.degree_of` can
    resolve (``Act``, nested ``Product``, ``Pairing``, ...).
    """
    if isinstance(factor, (Integer, Rational)):
        return True
    if isinstance(factor, Derivation):
        return True
    if registry is not None:
        if registry.has(factor, Scalar):
            return True
        if registry.get(factor, Graded) is not None:
            return True
    from jacopy.algebra.derivation import degree_of
    try:
        degree_of(factor, registry)
        return True
    except ValueError:
        return False


# --------------------------------------------------------------------- #
# Rules                                                                  #
# --------------------------------------------------------------------- #


@register_rule
def stalled_d_squared(
    residual: Expr,
    registry: Optional[PropertyRegistry],
    engine: Optional[ExpansionEngine],
) -> Iterable[DiagnosticHint]:
    """``Act(D, Act(D, x))`` with the same ``D`` and odd degree, ``D² = 0``.

    Fires for any graded derivation of odd degree applied twice, so
    exterior ``d`` is the common case but an algebroid ``d_ρ`` or a
    BRST ``Q`` operator surface the same hint.
    """
    for node in residual.walk():
        if not isinstance(node, Act):
            continue
        outer_op = node.op
        if not isinstance(outer_op, Derivation):
            continue
        deg = outer_op.degree
        parity = deg.parity()
        if parity != 1:
            continue
        inner = node.arg
        if not isinstance(inner, Act):
            continue
        if inner.op != outer_op:
            continue
        yield DiagnosticHint(
            category="stalled-d-squared",
            message=(
                f"{outer_op._repr_inner()} applied twice, an odd "
                "derivation should square to zero"
            ),
            location=node,
            suggestion=(
                "enable d_squared_mode=\"axiom\" on default_engine or "
                "register a DSquaredZeroDefinition for this operator"
            ),
        )


@register_rule
def stalled_act_over_zero(
    residual: Expr,
    registry: Optional[PropertyRegistry],
    engine: Optional[ExpansionEngine],
) -> Iterable[DiagnosticHint]:
    """``Act(op, 0)``, every graded derivation kills zero."""
    for node in residual.walk():
        if not isinstance(node, Act):
            continue
        if isinstance(node.arg, Integer) and node.arg == Integer(0):
            yield DiagnosticHint(
                category="stalled-act-over-zero",
                message=(
                    f"{node.op._repr_inner()} applied to 0, linearity "
                    "gives 0 regardless of the operator"
                ),
                location=node,
                suggestion=(
                    "product_rule._expand_act should recognise "
                    "Act(op, Integer(0))"
                ),
            )


@register_rule
def stalled_act_over_neg_op(
    residual: Expr,
    registry: Optional[PropertyRegistry],
    engine: Optional[ExpansionEngine],
) -> Iterable[DiagnosticHint]:
    """``Act(Neg(op), x) = -Act(op, x)``, sign should peel outward."""
    for node in residual.walk():
        if not isinstance(node, Act):
            continue
        if isinstance(node.op, Neg):
            yield DiagnosticHint(
                category="stalled-act-over-neg-op",
                message=(
                    f"negation on the operator side of "
                    f"{node._repr_inner()} should peel to an outer Neg"
                ),
                location=node,
                suggestion=(
                    "product_rule._expand_act should recognise "
                    "Act(Neg(op), x)"
                ),
            )


@register_rule
def unreduced_iota_on_df(
    residual: Expr,
    registry: Optional[PropertyRegistry],
    engine: Optional[ExpansionEngine],
) -> Iterable[DiagnosticHint]:
    """``Act(ι_V, Act(d, f))``, should fire ``ι_V(df) = V(f)`` when V is
    a derivation-combination.

    Detection is structural: the outer operator must itself be a
    degree-(-1) derivation, its argument must be an ``Act(d, _)`` where
    ``d`` has degree ``+1``, and the ι's vector-field child must be a
    derivation-combination. The rule deliberately avoids hard-coding
    ``InteriorProduct``/``ExteriorDerivative`` so it catches algebroid
    variants too.
    """
    for node in residual.walk():
        if not isinstance(node, Act):
            continue
        iota = node.op
        # Only operators with a ``vector_field`` slot are iota-like;
        # this naturally filters out ExteriorDerivative / LieDerivative
        # without hard-coding class names.
        vf = getattr(iota, "vector_field", None)
        if vf is None:
            continue
        inner = node.arg
        if not isinstance(inner, Act):
            continue
        d_op = inner.op
        if not isinstance(d_op, Derivation):
            continue
        if d_op.degree.parity() != 1:
            continue
        if isinstance(vf, Derivation):
            # Plain-Derivation case is already handled by the default
            # ``IotaOnExactOneFormDefinition``; no hint needed.
            continue
        if not _is_derivation_combination(vf):
            continue
        yield DiagnosticHint(
            category="unreduced-iota-on-df",
            message=(
                f"{iota._repr_inner()} has a compound vector field "
                f"({vf._repr_inner()}); ι_V(df) = V(f) should still apply"
            ),
            location=node,
            suggestion=(
                "IotaOnExactOneFormDefinition.matches should accept "
                "Sum/Product/Neg of Derivations as the vector field"
            ),
        )


def _symbol_leaves_in_vector_field(expr: Expr) -> list:
    """Return the list of non-Derivation leaves in a vector-field expression.

    ``_is_derivation_combination`` only accepts :class:`Derivation` atoms
    and Sum/Product/Neg composites of them. Anything else at a leaf
    position, a bare :class:`Symbol`, most commonly, breaks pairing
    rules like ``ι_V(df) = V(f)`` and ``L_V(f) = V(f)``. This helper
    surfaces those offending leaves so the diagnostic can name them.
    """
    if isinstance(expr, Derivation):
        return []
    if isinstance(expr, (Sum, Product, Neg)):
        out: list = []
        for child in expr.children:
            out.extend(_symbol_leaves_in_vector_field(child))
        return out
    return [expr]


@register_rule
def symbol_vector_field(
    residual: Expr,
    registry: Optional[PropertyRegistry],
    engine: Optional[ExpansionEngine],
) -> Iterable[DiagnosticHint]:
    """Vector-field slot holds a plain :class:`Symbol` (or composite thereof).

    The pairing rules ``ι_V(df) = V(f)`` and ``L_V(f) = V(f)`` only fire
    when the vector field is a :class:`Derivation` (or Sum/Product/Neg
    composite of Derivations), bare Symbols don't carry the action
    semantics that make ``V(f)`` meaningful. This is easy to trip on
    when prototyping: ``Symbol("X")`` looks indistinguishable from
    ``Derivation("X", degree=0)`` in a printed residual, but only the
    latter triggers the pairing fire-path.

    The rule walks every :class:`Act` whose operator has a
    ``vector_field`` slot (ι_X, L_X, and any future operator keyed on a
    vector field) and flags when that slot contains non-Derivation
    leaves. It is strictly complementary to
    :func:`unreduced_iota_on_df`: that rule fires on compound
    *Derivation* combinations the matcher would accept; this one fires
    on *Symbol* leaves the matcher never will.
    """
    for node in residual.walk():
        if not isinstance(node, Act):
            continue
        vf = getattr(node.op, "vector_field", None)
        if vf is None:
            continue
        offenders = _symbol_leaves_in_vector_field(vf)
        if not offenders:
            continue
        # De-duplicate preserving first-seen order, the bracket
        # ``X*Y − Y*X`` visits each Symbol twice, and the hint should
        # name each offender once.
        seen_offenders: set = set()
        unique_offenders: list = []
        for o in offenders:
            key = o._repr_inner()
            if key in seen_offenders:
                continue
            seen_offenders.add(key)
            unique_offenders.append(o)
        names = ", ".join(o._repr_inner() for o in unique_offenders)
        yield DiagnosticHint(
            category="symbol-vector-field",
            message=(
                f"{node.op._repr_inner()} carries a non-Derivation vector "
                f"field ({vf._repr_inner()}); pairing rules like "
                f"ι_V(df) = V(f) and L_V(f) = V(f) stay inert because "
                f"{names} has no derivation semantics"
            ),
            location=node,
            suggestion=(
                f"rebuild {names} as Derivation(\"...\", degree=0), "
                "Symbol is a bare name; Derivation carries the action "
                "semantics the pairing rules key on"
            ),
        )


@register_rule
def unclassified_factor(
    residual: Expr,
    registry: Optional[PropertyRegistry],
    engine: Optional[ExpansionEngine],
) -> Iterable[DiagnosticHint]:
    """Factors inside a :class:`Product` with no grading evidence.

    ``sort_product`` refuses to sort a Product whose factors it can't
    grade (Scalar / Graded / Derivation self-degree / numeric literal).
    A residual that still contains such a Product typically means the
    sort layer aborted silently earlier in the pipeline and the
    obstruction never reached canonical form. Flagging the offender is
    the shortest path to the missing declaration.
    """
    for node in residual.walk():
        if not isinstance(node, Product):
            continue
        for factor in node.children:
            if _classifies(factor, registry):
                continue
            yield DiagnosticHint(
                category="unclassified-factor",
                message=(
                    f"factor {factor._repr_inner()} has no grading "
                    "evidence (neither Scalar/Graded in registry nor a "
                    "Derivation self-degree)"
                ),
                location=factor,
                suggestion=(
                    "declare registry.declare(factor, Graded(degree=...)) "
                    "or registry.declare(factor, Scalar())"
                ),
            )
