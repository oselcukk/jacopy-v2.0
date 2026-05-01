"""
Derivator of an operator on a bracket.

Given a degree-``d`` operator ``φ`` and a bracket ``[·, ·]_E`` the
*derivator*

    D^E_φ(u, v) := φ[u, v]_E − [φu, v]_E − (−1)^{d|u|} [u, φv]_E

measures the failure of ``φ`` to act as a graded derivation of the
bracket. ``φ`` is a derivation of ``[·, ·]_E`` iff ``D^E_φ ≡ 0``; if it
does not, the derivator is the obstruction to that property.

Section 3.1.5 of the question text specialises this to the (Poisson)
manifold setting where ``φ`` is a Cartan-style operator (``L_V``,
``ι_V``, ``d``, or their tildes) and ``[·, ·]_E`` is the Koszul or Lie
bracket. The six identities (1)/(2)/(3) and their tilde duals
(1')/(2')/(3') are *symbolic* statements about specific derivators,
this helper produces those as plain Sum/Neg/BracketApply trees so the
proof engine can rewrite them.

The current implementation assumes ``φ`` has degree ``0`` (the case
covered by every identity in 3.1.5: ``L_V``, ``L̃_η``, and the
``d̃ ι̃_η`` / ``L_{…} ∘ …`` compositions all have grading 0). A graded
generalisation would add a ``(−1)^{d|u|}`` sign on the third term; for
now we keep the unsigned form because

* it matches the textbook display of (1) / (1');
* introducing a generic sign would force callers to thread degrees of
  ``u``, which Section 3.1.5's operands (1-forms and 1-vectors) don't
  carry as Degree polynomials at the call site.

The signed variant is a one-line follow-up if a non-degree-0 ``φ``
operand ever appears in the proof corpus.
"""

from __future__ import annotations

from typing import Optional, Tuple

from jacopy.algebra.derivation import Act
from jacopy.brackets.base import BracketApply, GradedBracket
from jacopy.calculus.intrinsic_engine import prove_intrinsic_equivalence
from jacopy.core.expr import Expr, Neg, Sum
from jacopy.core.multi_eval import MultiEval
from jacopy.core.registry import PropertyRegistry
from jacopy.proof.chain import ProofChain
from jacopy.proof.expansion import ExpansionEngine


def derivator(
    phi: Expr,
    bracket: GradedBracket,
    u: Expr,
    v: Expr,
    *,
    registry: Optional[PropertyRegistry] = None,
) -> Expr:
    """Return ``D^E_φ(u, v) = φ[u,v]_E − [φu, v]_E − [u, φv]_E``.

    Parameters
    ----------
    phi
        Operator whose derivator on ``bracket`` is being formed,
        typically a :class:`~jacopy.algebra.derivation.Derivation`
        atom, or a composition Expr that the engine can apply through
        :class:`~jacopy.algebra.derivation.Act`.
    bracket
        The :class:`GradedBracket` whose Leibniz failure is being
        measured.
    u, v
        Bracket operands.
    registry
        Currently unused, accepted for API parity with the other
        Section 3.1.5 helpers (``derivator``, ``K``, ``K̃``) so callers
        can pass a registry uniformly. May be threaded through future
        signed variants.

    Returns
    -------
    Expr
        A :class:`Sum` of three :class:`BracketApply` /
        :class:`Act` terms, the inert derivator expression. The proof
        engine evaluates it by expanding each term against its operator
        and bracket axioms.
    """
    if not isinstance(phi, Expr):
        raise TypeError("derivator: phi must be an Expr (typically a Derivation)")
    if not isinstance(bracket, GradedBracket):
        raise TypeError("derivator: bracket must be a GradedBracket")
    if not isinstance(u, Expr):
        raise TypeError("derivator: u must be an Expr")
    if not isinstance(v, Expr):
        raise TypeError("derivator: v must be an Expr")
    return Sum(
        Act(phi, BracketApply(bracket, u, v)),
        Neg(BracketApply(bracket, Act(phi, u), v)),
        Neg(BracketApply(bracket, u, Act(phi, v))),
    )


# --------------------------------------------------------------------- #
# Equivalence prover for derivator-shaped identities                     #
# --------------------------------------------------------------------- #


def prove_derivator_identity(
    lhs: Expr,
    rhs: Expr,
    *,
    engine: ExpansionEngine,
    eval_args: Tuple[Expr, ...],
    slot_kind: str = "vector",
    alternating: bool = True,
    registry: Optional[PropertyRegistry] = None,
) -> ProofChain:
    """Prove ``lhs == rhs`` by evaluating both on a ``p``-tuple.

    The Section 3.1.5 derivator identities equate two operator-valued
    expressions on either the form side (``slot_kind="vector"``,
    evaluate against vector fields) or the multivector side
    (``slot_kind="covector"``, evaluate against 1-forms). This driver
    wraps both sides under ``MultiEval(_, *eval_args, slot_kind=...)``
    and delegates to
    :func:`~jacopy.calculus.intrinsic_engine.prove_intrinsic_equivalence`
    with the supplied engine, returning a
    :class:`~jacopy.proof.chain.ProofChain` on success or raising
    :class:`~jacopy.proof.strategies.ProofFailure` on residue.

    Mirrors :func:`~jacopy.calculus.tilde.intrinsic_engine.prove_tilde_cartan_relation`
    on the form side: the two functions differ only in the default
    ``slot_kind`` and the audience of axioms each engine is expected to
    carry. A single 3.1.5 proof typically threads both, the
    multivector-side identities (1')/(2')/(3') reach for the tilde
    engine, the form-side (1)/(2)/(3) for the standard one.

    Parameters
    ----------
    lhs, rhs
        Operator-valued :class:`Expr` whose equality is to be shown.
        Typically Sum/Neg/Act trees over the Cartan operators
        (standard or tilde) and Cartan remainders.
    engine
        :class:`ExpansionEngine` carrying the rules needed to expand
        both sides. The expected mix for 3.1.5: standard intrinsic +
        closure axioms (Faz 12.A.5/12.A.6) + KoszulBracket expansion +
        :class:`~jacopy.calculus.cartan_remainder_axioms.CartanRemainderDefinition` +
        :class:`~jacopy.calculus.cartan_remainder_axioms.TildeCartanRemainderDefinition` +
        the tilde defining axioms (so a ``K̃_η V`` chain reaches the
        standard side via :class:`TildeIotaSwapDefinition`).
    eval_args
        Tuple of expressions against which both sides are evaluated,
        vector fields when ``slot_kind="vector"``, 1-forms when
        ``slot_kind="covector"``.
    slot_kind
        ``"vector"`` (default) for form-side identities, ``"covector"``
        for multivector-side. Threads through to the :class:`MultiEval`
        wrap and routes the engine to the appropriate intrinsic rules.
    alternating
        Whether the :class:`MultiEval` wrap is graded-antisymmetric in
        its argument slots. Defaults to ``True``, the convention every
        Cartan / Koszul evaluation in this codebase uses.
    registry
        Optional :class:`PropertyRegistry` plumbed through to the
        simplify pipeline.

    Notes
    -----
    The driver does not validate that ``len(eval_args)`` matches the
    operator-implied arity; pick a length consistent with the form
    /multivector degrees flowing through the operator composition. A
    mismatched arity surfaces as a residue that does not cancel, with
    the obstruction reported by the raised :class:`ProofFailure`.
    """
    if not isinstance(lhs, Expr):
        raise TypeError("prove_derivator_identity lhs must be an Expr")
    if not isinstance(rhs, Expr):
        raise TypeError("prove_derivator_identity rhs must be an Expr")
    if not isinstance(engine, ExpansionEngine):
        raise TypeError(
            "prove_derivator_identity engine must be an ExpansionEngine"
        )
    if not isinstance(eval_args, tuple) or not eval_args:
        raise TypeError(
            "prove_derivator_identity eval_args must be a non-empty tuple of Expr"
        )
    for arg in eval_args:
        if not isinstance(arg, Expr):
            raise TypeError(
                "prove_derivator_identity eval_args entries must be Expr"
            )
    if slot_kind not in ("vector", "covector"):
        raise ValueError(
            "prove_derivator_identity slot_kind must be 'vector' or 'covector'"
        )

    lhs_eval = MultiEval(
        lhs, *eval_args, alternating=alternating, slot_kind=slot_kind
    )
    rhs_eval = MultiEval(
        rhs, *eval_args, alternating=alternating, slot_kind=slot_kind
    )
    return prove_intrinsic_equivalence(
        lhs_eval, rhs_eval, engine=engine, registry=registry
    )
