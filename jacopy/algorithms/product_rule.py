"""
Graded Leibniz expansion.

Rewrites every :class:`Act` node ``D(x)`` by pushing the operator
through sums (R-linearity) and products (graded Leibniz):

* ``D(a + b) → D(a) + D(b)``
* ``D(−a) → −D(a)``
* ``D(a*b*c) → D(a)*b*c + (−1)^{|D||a|} a*D(b)*c
               + (−1)^{|D|(|a|+|b|)} a*b*D(c)``

For the Leibniz expansion the degrees of the factors must be
determinable, either because the factor is a :class:`Derivation`
(self-describing), a numeric literal (degree 0), or registered as
:class:`Scalar`/:class:`Graded` in the supplied registry. An
undetermined factor raises :class:`ValueError`: the modelling gap is
something the user needs to fix, not something we should silently paper
over with a zero-degree assumption.

The pass is bottom-up and inert on nodes that aren't :class:`Act` at
their head: inner structure inside the argument is expanded first, so
``D(E(a*b))`` first expands the inner ``E`` Leibniz then the outer
``D``. Outer derivation degrees see the *expanded* operand structure.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

from jacopy.algebra.derivation import Act, Derivation, degree_of
from jacopy.algorithms.base import Algorithm
from jacopy.core.expr import Expr, Integer, Neg, Product, Sum
from jacopy.core.registry import PropertyRegistry
from jacopy.core.symbolic_degree import Degree


# --------------------------------------------------------------------- #
# Public entry                                                           #
# --------------------------------------------------------------------- #


def product_rule(
    expr: Expr, registry: Optional[PropertyRegistry] = None
) -> Expr:
    """Recursively expand every ``Act`` via graded Leibniz and linearity.

    Bottom-up: children are expanded first, so inner ``Act`` nodes are
    simplified before their outer enclosing ``Act`` looks at them.
    Atoms pass through untouched.
    """
    if expr.is_atom:
        return expr

    new_children = tuple(product_rule(c, registry) for c in expr.children)

    if isinstance(expr, Act):
        op, arg = new_children
        return _expand_act(op, arg, registry)

    return expr._rebuild(new_children)


# --------------------------------------------------------------------- #
# Act-specific expansion                                                 #
# --------------------------------------------------------------------- #


def _expand_act(
    op: Expr, arg: Expr, registry: Optional[PropertyRegistry]
) -> Expr:
    """Apply linearity + graded Leibniz to a single ``Act`` node.

    When ``op`` is itself a :class:`Product` of operators, i.e. an
    explicit composition ``D1 ∘ D2 ∘ … ∘ Dn``, the application is
    unfolded right-to-left into nested ``Act`` nodes and each layer is
    expanded in turn. That is the only place in the algorithm where
    derivation composition actually acts on an operand; at
    :class:`Act` construction it stays inert.

    Scalar-level linearity on the operator side, ``Neg(op)`` and the
    zero operator ``Integer(0)``, is peeled here too. Commutator
    expansion routinely generates ``Act(Neg(compose(D1, D2)), x)``
    shapes; pulling the sign out and recursing lets the composition
    unfold through the negation instead of stalling.
    """
    if isinstance(op, Neg):
        return Neg(_expand_act(op.arg, arg, registry))
    if isinstance(op, Integer) and op == Integer(0):
        return Integer(0)
    if isinstance(arg, Integer) and arg == Integer(0):
        # Every graded derivation is linear, so D(0) = 0 regardless of
        # the specific operator. Dropping this eagerly keeps axiom
        # residues like Act(d, Act(d², x))=Act(d, 0) from surviving into
        # later passes that would otherwise stall waiting for an engine
        # rewrite that never fires on this surviving shape.
        return Integer(0)
    if isinstance(op, Product) and op.children:
        result = arg
        for d in reversed(op.children):
            result = _expand_act(d, result, registry)
        return result
    if isinstance(arg, Sum):
        return Sum.make(
            *(_expand_act(op, c, registry) for c in arg.children)
        )
    if isinstance(arg, Neg):
        inner = _expand_act(op, arg.arg, registry)
        return Neg(inner)
    if isinstance(arg, Product):
        return _expand_leibniz(op, arg, registry)
    # Atom or other compound (Power, Act, Commutator, ...): leave inert.
    return Act(op, arg)


def _expand_leibniz(
    op: Expr, prod: Product, registry: Optional[PropertyRegistry]
) -> Expr:
    """``D(a1*...*an) → Σ_i sign_i * a1*...*D(a_i)*...*an``.

    ``sign_i = (−1)^{|D| * (|a1|+...+|a_{i−1}|)}``. Undecidable sign
    parity raises, the caller needs to narrow down the degrees. Scalar
    factors (degree 0) contribute nothing to the running sign, so
    symbolic-degree factors only cause trouble when they actually
    precede a splitting point.
    """
    factors: Tuple[Expr, ...] = prod.children
    if not factors:
        # Empty product, D(1) = 0. Shouldn't normally arise but
        # handle defensively; Sum.make([]) returns Zero.
        return Integer(0)

    deg_op = degree_of(op, registry)
    # When the derivation has degree 0 the sign vanishes identically,
    # so we don't need factor degrees at all. This is the common case
    # for ordinary (ungraded) derivations and spares the caller from
    # declaring a grading for operand factors that don't have one.
    deg_op_is_zero = deg_op == Degree.const(0)

    running = Degree.const(0)  # |a_1| + ... + |a_{i-1}|
    terms: List[Expr] = []
    for i, fac in enumerate(factors):
        if deg_op_is_zero:
            parity = 0
        else:
            sign_exp = deg_op * running
            parity = sign_exp.parity()
            if parity is None:
                raise ValueError(
                    f"Cannot expand Leibniz at factor index {i} of "
                    f"{prod!r}: sign parity is symbolic. Narrow the "
                    "degrees of the left-of-split factors."
                )
        new_factors = list(factors)
        new_factors[i] = Act(op, fac)
        term: Expr = Product(*new_factors) if len(new_factors) > 1 else new_factors[0]
        if parity == 1:
            term = Neg(term)
        terms.append(term)
        if not deg_op_is_zero and i + 1 < len(factors):
            running = running + degree_of(fac, registry)

    return Sum.make(*terms)


# --------------------------------------------------------------------- #
# Algorithm wrapper                                                      #
# --------------------------------------------------------------------- #


class ProductRule(Algorithm):
    """:class:`Algorithm` wrapper around :func:`product_rule`.

    ``registry`` is stored on the instance so the generic
    :meth:`Algorithm.run` plumbing stays registry-free.
    """

    def __init__(self, registry: Optional[PropertyRegistry] = None) -> None:
        self._registry = registry

    def can_apply(self, expr: Expr) -> bool:
        for node in expr.walk():
            if isinstance(node, Act):
                return True
        return False

    def apply(self, expr: Expr) -> Expr:
        return product_rule(expr, self._registry)
