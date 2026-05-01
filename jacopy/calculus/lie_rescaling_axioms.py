r"""
Lie-derivative rescaling axiom, Faz 12.B #10.

Engine rewrite that turns the action of a Lie derivative whose vector
field is itself a scaled product into the canonical Cartan-style
expansion:

.. math::

    \mathcal{L}_{f\,X}(\omega) \;\longrightarrow\;
        f\,\mathcal{L}_X(\omega) \;+\; df \wedge \iota_X(\omega).

The rule fires on ``Act(L, ω)`` whenever ``L.vector_field`` is a
:class:`~jacopy.core.expr.Product` with two or more factors. The
**last** child is treated as the underlying vector field; the leading
factors are folded into the scalar coefficient ``f`` (so
``L_{g·f·X}`` rewrites with ``f := g·f``). Using ``Product`` for the
``f``-coefficient and the ``df`` wedge mirrors the convention already
used by ad-hoc 2d-notebook rules; downstream
:func:`~jacopy.algorithms.simplify.simplify` flattens nested products.

The rewrite carries the original Lie derivative's bundle hooks
(:attr:`LieDerivative.d`, :attr:`LieDerivative.iota_factory`,
:attr:`LieDerivative.definition`) into both the rebuilt ``L_X`` and
the freshly-built ``ι_X`` and ``d``, so an algebroid ``L_{E,fX}``
expands to ``f·L_{E,X}(ω) + d_E(f)∧ι_{E,X}(ω)``, the bundle never
leaks back to the ambient ``TM``.
"""

from __future__ import annotations

from jacopy.algebra.derivation import Act
from jacopy.calculus.exterior_d import d as default_d
from jacopy.calculus.interior import interior
from jacopy.calculus.lie_derivative import LieDerivative
from jacopy.core.expr import Expr, Product, Sum
from jacopy.proof.expansion import Definition


class LieRescalingDefinition(Definition):
    r"""``L_{f·X}(ω) → f·L_X(ω) + df∧ι_X(ω)``, Lie-derivative rescaling.

    Fires on ``Act(LieDerivative, ω)`` when the underlying vector
    field is a :class:`Product` of two or more factors. The leading
    factors are folded into ``f`` (a scalar coefficient) and the last
    factor is taken as ``X``; the rule emits the standard rescaling
    expansion. Bundle hooks (``d`` override and ``iota_factory``) ride
    along so an algebroid Lie derivative stays inside its bundle.

    For arity 1 the rule does nothing, that case is just plain
    ``L_X(ω)`` and the existing intrinsic / Cartan-magic rules cover
    it.
    """

    name = "L_{fX}: L_{f·X}(ω) = f·L_X(ω) + df∧ι_X(ω)"

    def matches(self, expr: Expr) -> bool:
        if not isinstance(expr, Act):
            return False
        if not isinstance(expr.op, LieDerivative):
            return False
        vf = expr.op.vector_field
        if not isinstance(vf, Product):
            return False
        return len(vf.children) >= 2

    def rewrite(self, expr: Expr) -> Expr:
        L = expr.op
        omega = expr.arg
        vf = L.vector_field
        *scalar_factors, X = vf.children
        f = Product.make(*scalar_factors)

        d_op = L.d if L.d is not None else default_d
        if L.iota_factory is not None:
            iota_X = L.iota_factory(X)
        else:
            iota_X = interior(X)

        L_X = LieDerivative(
            X,
            definition=L.definition,
            d=L.d,
            iota_factory=L.iota_factory,
        )
        return Sum(
            Product(f, Act(L_X, omega)),
            Product(Act(d_op, f), Act(iota_X, omega)),
        )
