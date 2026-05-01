r"""
Multilinear C∞-linearity axiom, Faz 12.B #6.

Engine rewrite that pulls a scalar factor out of any argument slot of a
:class:`~jacopy.core.multi_eval.MultiEval`:

.. math::

    \omega(\dots,\, f\,X,\, \dots) \;\longrightarrow\;
        f \,\cdot\, \omega(\dots,\, X,\, \dots).

The rule is the rank-``p`` generalisation of
:class:`~jacopy.calculus.pairing_linearity_axioms.PairingScalarPullDefinition`.
A ``MultiEval`` represents a graded multilinear contract, either a
``p``-form evaluated on ``p`` vector fields or a multivector evaluated
on ``p`` covectors, and both are C∞(M)-multilinear in every slot. The
existing :class:`~jacopy.calculus.multi_eval_axioms.MultiEvalArgLinearityDefinition`
handles ``Sum``/``Neg`` distribution; this rule adds the missing
``Product``-slot scalar-pull so that a Poisson-style evaluation
``π(f · dg, dh)`` opens to ``f · π(dg, dh)`` without an inline axiom.

By convention the **last** factor in a slot ``Product`` is the
underlying covector / vector and the leading factors are scalars,
the same split used by
:class:`~jacopy.calculus.lie_rescaling_axioms.LieRescalingDefinition`
and :class:`PairingScalarPullDefinition`. The rule fires on the
leftmost slot whose arg is a ``Product`` of two or more factors;
subsequent slots clear in successive engine passes.
"""

from __future__ import annotations

from jacopy.core.expr import Expr, Product
from jacopy.core.multi_eval import MultiEval
from jacopy.proof.expansion import Definition


class MultiEvalScalarPullDefinition(Definition):
    r"""``ω(…, f·X, …) → f·ω(…, X, …)``, C∞-linearity in any arg slot.

    Fires on a :class:`MultiEval` whenever any argument is a
    :class:`Product` of two or more factors. The leading factors of
    that argument are folded into the scalar coefficient ``f`` and the
    last factor is taken as the underlying covector / vector. The
    resulting expression is ``Product(f, MultiEval(head, …, X, …))``
    with the original ``alternating`` and ``slot_kind`` flags
    preserved on the inner node.

    Multiple ``Product`` slots are cleared one at a time, the rule
    walks the args left-to-right and pulls the first eligible slot;
    the engine fires it again on the resulting node to clear the next
    slot, and so on.
    """

    name = "MultiEval C∞-linearity: ω(…, f·X, …) = f·ω(…, X, …)"

    def matches(self, expr: Expr) -> bool:
        if not isinstance(expr, MultiEval):
            return False
        return self._slot_to_pull(expr) >= 0

    def rewrite(self, expr: Expr) -> Expr:
        i = self._slot_to_pull(expr)
        prod: Product = expr.args[i]  # type: ignore[assignment]
        *scalars, head = prod.children
        f = Product.make(*scalars)
        new_args = list(expr.args)
        new_args[i] = head
        inner = MultiEval(
            expr.head,
            *new_args,
            alternating=expr.alternating,
            slot_kind=expr.slot_kind,
        )
        return Product(f, inner)

    @staticmethod
    def _slot_to_pull(expr: MultiEval) -> int:
        for i, a in enumerate(expr.args):
            if isinstance(a, Product) and len(a.children) >= 2:
                return i
        return -1
