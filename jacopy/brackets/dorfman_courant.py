"""
Dorfman-Courant bridge, pedagogical helper.

The Dorfman bracket :math:`[\\cdot,\\cdot]_D` and the (untwisted) Courant
bracket :math:`[\\cdot,\\cdot]_C` agree on the vector half and differ on
the form half by an exact, *symmetric* correction:

    [(X, α), (Y, β)]_D − [(X, α), (Y, β)]_C  =  ( 0,  ½ d(ι_X β + ι_Y α) ).

Equivalently, and this is the form most useful for the proof layer,
the Dorfman form part ``L_X β − ι_Y dα`` minus the Courant form part
``L_X β − L_Y α − ½ d(ι_X β − ι_Y α)`` collapses to ``½ d(ι_X β + ι_Y α)``
once Cartan's magic formula ``L_Y α = ι_Y dα + d(ι_Y α)`` is applied.

This module is *pedagogical*: ``library/courant_algebroid.py`` and
``library/dirac.py`` consume :class:`CourantBracket` directly, so the
bridge is not a runtime dependency. Its job is to package the textbook
identity into a single :class:`ProofChain` callers can cite when
discussing the relationship between the two brackets.

Public surface
--------------

* :func:`dorfman_courant_correction`, build the canonical correction
  :class:`SectionPair` ``(0, ½ d(ι_X β + ι_Y α))`` for a given pair of
  operands.
* :func:`prove_dorfman_courant_bridge`, prove
  ``D.expand(a, b) − C.expand(a, b) == correction`` componentwise and
  return a single concatenated :class:`ProofChain`.

Both helpers require the Courant bracket to be untwisted; the
H-twisted variant adds a ``ι_Y ι_X H`` term to the form part that the
canonical bridge identity does not absorb.
"""

from __future__ import annotations

from typing import Optional

from jacopy.algebra.derivation import Act, Derivation
from jacopy.brackets.courant import CourantBracket
from jacopy.brackets.dorfman import DorfmanBracket, SectionPair
from jacopy.calculus.exterior_d import ExteriorDerivative, d as default_d
from jacopy.calculus.interior import interior as default_interior
from jacopy.core.expr import Expr, Integer, Neg, Product, Rational, Sum, Zero
from jacopy.core.registry import PropertyRegistry
from jacopy.proof.chain import ProofChain
from jacopy.proof.strategies import ExpandAndSimplify


def dorfman_courant_correction(
    a: SectionPair,
    b: SectionPair,
    *,
    d: Optional[ExteriorDerivative] = None,
    interior=None,
) -> SectionPair:
    """Return the canonical Dorfman-Courant gap ``(0, ½ d(ι_X β + ι_Y α))``.

    The correction is the SectionPair difference
    ``[·,·]_D − [·,·]_C`` for the untwisted Courant bracket. Vector
    component is :data:`Zero` (both brackets agree on the vector half);
    form component is the symmetric exact term ``½ d(ι_X β + ι_Y α)``.

    ``d`` and ``interior`` default to the smooth-manifold singletons,
    pass overrides when working in a Lie-algebroid bundle whose Cartan
    operators differ from the default.
    """
    if not isinstance(a, SectionPair):
        raise TypeError(
            f"dorfman_courant_correction left operand must be a SectionPair, "
            f"got {type(a).__name__}"
        )
    if not isinstance(b, SectionPair):
        raise TypeError(
            f"dorfman_courant_correction right operand must be a SectionPair, "
            f"got {type(b).__name__}"
        )
    dop = default_d if d is None else d
    iop = default_interior if interior is None else interior
    X, alpha = a.vector, a.form
    Y, beta = b.vector, b.form
    half = Rational(1, 2)
    inner = Sum(Act(iop(X), beta), Act(iop(Y), alpha))
    form_part = Product(half, Act(dop, inner))
    return SectionPair(Zero, form_part)


def prove_dorfman_courant_bridge(
    D: DorfmanBracket,
    C: CourantBracket,
    a: SectionPair,
    b: SectionPair,
    *,
    registry: Optional[PropertyRegistry] = None,
) -> ProofChain:
    """Prove ``D.expand(a, b) − C.expand(a, b) == (0, ½ d(ι_X β + ι_Y α))``.

    Splits the SectionPair difference into its vector and form halves,
    runs :class:`~jacopy.proof.strategies.ExpandAndSimplify` on each,
    and concatenates the two transcripts into a single
    :class:`ProofChain`. The form half is where Cartan's magic formula
    fires (the default engine carries it as a rewrite), reducing
    ``L_Y α − ι_Y dα`` to ``d(ι_Y α)`` and exposing the symmetric
    correction.

    The Courant bracket must be untwisted; an H-twisted Courant bracket
    introduces a ``ι_Y ι_X H`` term on the form side that the canonical
    bridge identity doesn't absorb. Raises :class:`TypeError` on a
    twisted ``C`` or wrong operand types, and
    :class:`~jacopy.proof.strategies.ProofFailure` if either half
    leaves a residual.
    """
    if not isinstance(D, DorfmanBracket):
        raise TypeError(
            f"prove_dorfman_courant_bridge expected a DorfmanBracket, "
            f"got {type(D).__name__}"
        )
    if not isinstance(C, CourantBracket):
        raise TypeError(
            f"prove_dorfman_courant_bridge expected a CourantBracket, "
            f"got {type(C).__name__}"
        )
    if C.is_twisted:
        raise TypeError(
            "prove_dorfman_courant_bridge requires an untwisted CourantBracket "
            ", the H-twisted variant carries a ι_Y ι_X H term that the "
            "canonical correction does not absorb"
        )
    if not isinstance(a, SectionPair):
        raise TypeError(
            f"prove_dorfman_courant_bridge left operand must be a SectionPair, "
            f"got {type(a).__name__}"
        )
    if not isinstance(b, SectionPair):
        raise TypeError(
            f"prove_dorfman_courant_bridge right operand must be a SectionPair, "
            f"got {type(b).__name__}"
        )

    D_pair = D.expand(a, b, registry)
    C_pair = C.expand(a, b, registry)
    correction = dorfman_courant_correction(
        a, b, d=C._d, interior=C._interior,
    )

    strat = ExpandAndSimplify()
    chain = ProofChain()

    # Vector half: D_vec − C_vec == 0.
    vector_lhs = Sum(D_pair.vector, Neg(C_pair.vector))
    vector_chain = strat.prove(
        vector_lhs, Integer(0), registry=registry,
    )
    chain.extend(vector_chain)

    # Form half: D_form − C_form == ½ d(ι_X β + ι_Y α).
    form_lhs = Sum(D_pair.form, Neg(C_pair.form))
    form_chain = strat.prove(
        form_lhs, correction.form, registry=registry,
    )
    chain.extend(form_chain)

    return chain
