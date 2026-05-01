"""
Exterior derivative ``d``.

``d`` is the unique graded anti-derivation on the exterior algebra
``Ω*(M)`` with

* degree ``+1``,
* graded Leibniz: ``d(α ∧ β) = dα ∧ β + (−1)^{|α|} α ∧ dβ``, inherited
  automatically from :class:`Derivation` + :mod:`product_rule` because
  this module's ``d`` has degree 1,
* idempotent on squares: ``d² = 0``.

The axiom ``d(f) = df`` on functions is *not* a rewrite in this
module: there is no separate ``OneForm`` / ``df`` Expr node. Instead,
``Act(d, f)`` is already the canonical syntactic form of ``df``,
the :class:`Act` node stays inert when its argument is a 0-form atom
(:func:`product_rule` only expands Leibniz on multi-factor
:class:`Product` operands), so ``d(f)`` *is* ``df`` under the
package's expression-tree convention. Anything a caller might do
with ``df``, apply ``ι_X``, wedge with another form, feed into a
Cartan relation, lands on the same :class:`Act` node.

The ``d² = 0`` axiom is encoded here as a bottom-up rewrite,
:func:`apply_d_squared_zero`, that zeros out every subtree whose head
is a double application of ``d``, either the element-level
``d(d(x))`` shape or the operator-level composition ``(d ∘ d)(x)``
sitting as a ``Product`` on the left of an :class:`Act`. The plan
calls for *both* an axiomatic form and a theorem form (the latter
derived from the graded Jacobi identity on the Lie bracket of
derivations); only the axiom form lives here. The theorem form
belongs in the Faz 7 proof layer.

The module exposes a shared singleton :data:`d`. Users who need a
distinct-named exterior derivative, e.g. a Lie-algebroid ``d_E``,
construct their own :class:`ExteriorDerivative` instance.
"""

from __future__ import annotations

from typing import Optional

from jacopy.algebra.derivation import Act, Derivation
from jacopy.core.expr import Expr, Integer, Product
from jacopy.core.registry import PropertyRegistry


class ExteriorDerivative(Derivation):
    """Exterior derivative, degree ``+1`` graded anti-derivation.

    Structurally a :class:`Derivation` with a fixed degree of 1; the
    Koszul sign machinery in :mod:`product_rule` is what delivers the
    anti-derivation behaviour. The only operator-specific axiom beyond
    that is ``d² = 0``, which is applied by :func:`apply_d_squared_zero`
    rather than baked into Leibniz expansion, keeping the two
    concerns separable means a proof can unfold Leibniz first and
    close with ``d² = 0`` afterwards, matching how the identity is
    typically stated.
    """

    def __init__(self, name: str = "d") -> None:
        super().__init__(name, degree=1)


# Shared singleton. Distinct instances (e.g. for Lie-algebroid or
# twisted variants) are constructed on demand by the caller.
d = ExteriorDerivative()


# --------------------------------------------------------------------- #
# d² = 0 rewrite                                                         #
# --------------------------------------------------------------------- #


def _is_d(op: Expr, target: ExteriorDerivative) -> bool:
    """Is ``op`` the given exterior derivative?"""
    return isinstance(op, ExteriorDerivative) and op == target


def _product_starts_with_dd(
    op: Expr, target: ExteriorDerivative
) -> bool:
    """True if ``op`` is a Product whose first two factors are ``d``.

    Composition is represented as a :class:`Product` of operators, so
    the tree for ``(d ∘ d ∘ E)(x)`` reads
    ``Act(Product(d, d, E), x)``. Any such arrangement is zero because
    the leftmost two factors already compose to ``d² = 0``.
    """
    if not isinstance(op, Product):
        return False
    children = op.children
    return (
        len(children) >= 2
        and _is_d(children[0], target)
        and _is_d(children[1], target)
    )


def apply_d_squared_zero(
    expr: Expr,
    target: Optional[ExteriorDerivative] = None,
    registry: Optional[PropertyRegistry] = None,  # noqa: ARG001
) -> Expr:
    """Rewrite every ``d²`` occurrence in ``expr`` to ``0``.

    Recognised shapes:

    * ``Act(d, Act(d, x))``, the element-level statement ``d(d(x)) = 0``.
    * ``Act(Product(d, d, ...), x)``, the operator-level statement
      ``(d ∘ d ∘ ...)(x) = 0``. Trailing operators in the composition
      are irrelevant once the leading ``d²`` is known to vanish.

    Both shapes collapse to :class:`Integer` ``0``. Downstream
    ``collect_terms`` / ``simplify`` will drop the resulting zero out of
    any enclosing sum.

    When ``target`` is omitted, the module singleton :data:`d` is used.
    Passing an explicit instance lets the caller target a specific
    variant, e.g. apply ``d_E² = 0`` while leaving a plain ``d²`` in
    place if the two coexist in the same expression.

    ``registry`` is accepted for API symmetry with other passes; the
    rewrite itself is registry-free.
    """
    tgt = d if target is None else target
    if expr.is_atom:
        return expr
    new_children = tuple(
        apply_d_squared_zero(c, tgt) for c in expr.children
    )
    if any(a is not b for a, b in zip(new_children, expr.children)):
        rebuilt: Expr = expr._rebuild(new_children)
    else:
        rebuilt = expr
    if isinstance(rebuilt, Act):
        op, arg = rebuilt.op, rebuilt.arg
        if _is_d(op, tgt) and isinstance(arg, Act) and _is_d(arg.op, tgt):
            return Integer(0)
        if _product_starts_with_dd(op, tgt):
            return Integer(0)
    return rebuilt
