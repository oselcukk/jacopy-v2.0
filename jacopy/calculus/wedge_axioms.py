r"""
Engine rules for :class:`~jacopy.core.wedge.Wedge`, Faz 17.F.1.5.

A :class:`Wedge` node is purely structural; the alternating-sum
expansion that gives a wedge its meaning lives here as a
:class:`~jacopy.proof.expansion.Definition`.

* :class:`WedgeMultiEvalAlternatingDefinition`,

  .. math::

      (\alpha_1 \wedge \cdots \wedge \alpha_p)(X_1, \dots, X_p)
          = \sum_{\sigma \in S_p} \mathrm{sign}(\sigma)\,
              \alpha_1(X_{\sigma(1)}) \cdots \alpha_p(X_{\sigma(p)}).

  Fires on ``MultiEval(Wedge(α_1, …, α_p), X_1, …, X_p)`` when

    - the :class:`MultiEval` is alternating (``alternating=True``);
    - every factor ``α_i`` has registry-determinable degree ``1``;
    - the arity matches the number of wedge factors.

  Output is a :class:`~jacopy.core.expr.Sum` of signed
  :class:`~jacopy.core.expr.Product`'s of arity-1
  :class:`MultiEval` evaluations ``α_i(X_{σ(j)})``. Those arity-1
  one-form-on-vector evaluations are then turned into the canonical
  :class:`~jacopy.calculus.pairing.Pairing` shape by the bridge rule
  in :mod:`jacopy.calculus.pairing_axioms` (Faz 17.F.1.6).

The matcher is conservative on purpose: it only fires when the wedge's
factors are *all* one-forms and the arity matches. Mixed-degree wedges
(e.g. a 0-form prefactor) need either pre-processing (canonicalise the
0-form out of the wedge) or a separate rule. Cartan I/II only ever see
the all-one-form shape, so the conservative matcher suffices for Faz 17.
"""

from __future__ import annotations

from itertools import permutations
from typing import Optional, Sequence

from jacopy.core.expr import Expr, Neg, Product, Sum
from jacopy.core.multi_eval import MultiEval
from jacopy.core.registry import PropertyRegistry
from jacopy.core.symbolic_degree import Degree
from jacopy.core.wedge import Wedge
from jacopy.proof.expansion import Definition


def _permutation_sign(perm: Sequence[int]) -> int:
    """Sign of a permutation expressed as the image tuple ``(σ(0), …)``.

    Counts inversions in ``perm``; returns ``+1`` for an even count,
    ``−1`` for an odd count. Used by the alternating expansion to label
    each term in the wedge's full ``S_p`` sum.
    """
    inversions = 0
    n = len(perm)
    for i in range(n):
        for j in range(i + 1, n):
            if perm[i] > perm[j]:
                inversions += 1
    return 1 if inversions % 2 == 0 else -1


def _is_degree_one(
    expr: Expr, registry: Optional[PropertyRegistry]
) -> bool:
    """Safe ``|α| = 1`` check that returns ``False`` on any undecidable case."""
    from jacopy.algebra.derivation import degree_of

    try:
        return degree_of(expr, registry) == Degree.const(1)
    except ValueError:
        return False


class WedgeMultiEvalAlternatingDefinition(Definition):
    r"""Alternating-sum expansion of a wedge of one-forms.

    .. math::

        (\alpha_1 \wedge \cdots \wedge \alpha_p)(X_1, \dots, X_p)
            \to \sum_{\sigma \in S_p} \mathrm{sign}(\sigma)\,
                \alpha_1(X_{\sigma(1)}) \cdots \alpha_p(X_{\sigma(p)}).

    Match guards:

    * the head is a :class:`Wedge`;
    * every wedge factor has registry-determinable degree ``1``;
    * the :class:`MultiEval` is alternating (``alternating=True``),
      a non-alternating evaluation does not get the antisymmetric
      expansion, and pretending otherwise would be incorrect;
    * the arity equals the number of wedge factors.

    The output's ``α_i(X_j)`` evaluations are emitted as arity-1
    :class:`MultiEval` nodes; the
    :class:`~jacopy.calculus.pairing_axioms.MultiEvalOneFormPairingBridgeDefinition`
    rule converts those to canonical
    :class:`~jacopy.calculus.pairing.Pairing` shapes downstream.
    """

    name = (
        "wedge alternating expansion: "
        "(α_1 ∧ … ∧ α_p)(X_1, …, X_p) → Σ_σ sign(σ) ∏_i α_i(X_{σ(i)})"
    )

    def __init__(
        self, *, registry: Optional[PropertyRegistry] = None
    ) -> None:
        self._registry = registry

    def matches(self, expr: Expr) -> bool:
        if not isinstance(expr, MultiEval):
            return False
        if not expr.alternating:
            return False
        head = expr.head
        if not isinstance(head, Wedge):
            return False
        factors = head.children
        if len(factors) != len(expr.args):
            return False
        return all(_is_degree_one(f, self._registry) for f in factors)

    def rewrite(self, expr: Expr) -> Expr:
        assert isinstance(expr, MultiEval)
        head = expr.head
        assert isinstance(head, Wedge)
        factors = head.children
        args = expr.args
        terms: list[Expr] = []
        for sigma in permutations(range(len(factors))):
            sign = _permutation_sign(sigma)
            piece_factors = [
                MultiEval(
                    factors[i],
                    args[sigma[i]],
                    alternating=expr.alternating,
                    slot_kind=expr.slot_kind,
                )
                for i in range(len(factors))
            ]
            piece: Expr = Product.make(*piece_factors)
            if sign < 0:
                piece = Neg(piece)
            terms.append(piece)
        return Sum.make(*terms)
