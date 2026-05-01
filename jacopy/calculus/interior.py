"""
Interior product ``ι_X``.

The interior product with a vector field ``X`` is the degree ``−1``
graded anti-derivation on the exterior algebra characterised by

* ``ι_X(f) = 0`` on functions (degree-0 forms),
* ``ι_X(α)(Y_1, …, Y_{p-1}) = α(X, Y_1, …, Y_{p-1})`` on ``p``-forms,
  i.e. contraction against ``X`` in the first slot,
* graded Leibniz ``ι_X(α ∧ β) = ι_X(α) ∧ β + (−1)^{|α|} α ∧ ι_X(β)``,
* ``ι_X ∘ ι_X = 0``, this is a corollary of form antisymmetry (the
  same slot cannot receive ``X`` twice without the form vanishing),
  not an independent axiom.

As with :mod:`exterior_d`, the graded Leibniz is carried by the
degree and :mod:`product_rule`; this module adds the ``ι_X² = 0``
rewrite and a thin parametrized wrapper for constructing a
:class:`Derivation` keyed to a specific vector field ``X``.

Unlike :data:`d`, there is no module-level singleton. ``ι_X`` is a
*family* indexed by the vector field, naming is keyed to ``X``, and
two interior products for distinct fields are distinct operators.
Use :func:`interior` to build the operator for a given field.
"""

from __future__ import annotations

from typing import Optional

from jacopy.algebra.derivation import Act, Derivation, degree_of
from jacopy.calculus.exterior_d import ExteriorDerivative, d as default_d
from jacopy.core.expr import Expr, Integer, Product
from jacopy.core.registry import PropertyRegistry
from jacopy.core.symbolic_degree import Degree


class InteriorProduct(Derivation):
    """Interior product ``ι_X``, degree ``−1`` graded anti-derivation.

    Carries a reference to the vector field ``X`` (an arbitrary
    :class:`Expr`, typically a :class:`Symbol` declared as ``Graded``
    degree 0 or higher). Name defaults to ``"ι_{name(X)}"``, two
    interior products are equal iff they share both name and degree,
    and the default naming scheme makes that line up with equality of
    ``X``.
    """

    __slots__ = ("_vector_field",)

    def __init__(
        self,
        X: Expr,
        *,
        name: Optional[str] = None,
    ) -> None:
        if not isinstance(X, Expr):
            raise TypeError("Interior product requires an Expr vector field")
        display_name = name if name is not None else f"ι_{X._repr_inner()}"
        super().__init__(display_name, degree=-1)
        self._vector_field = X

    @property
    def vector_field(self) -> Expr:
        return self._vector_field


def interior(X: Expr, *, name: Optional[str] = None) -> InteriorProduct:
    """Build the interior-product operator for vector field ``X``."""
    return InteriorProduct(X, name=name)


# --------------------------------------------------------------------- #
# ι_X ∘ ι_X = 0 rewrite                                                  #
# --------------------------------------------------------------------- #


def _is_iota_on(op: Expr, target: InteriorProduct) -> bool:
    """Is ``op`` the same interior product as ``target``?"""
    return isinstance(op, InteriorProduct) and op == target


def _product_starts_with_iota_squared(
    op: Expr, target: InteriorProduct
) -> bool:
    """True if ``op`` is a Product whose first two factors both equal ``target``."""
    if not isinstance(op, Product):
        return False
    children = op.children
    return (
        len(children) >= 2
        and _is_iota_on(children[0], target)
        and _is_iota_on(children[1], target)
    )


def apply_iota_squared_zero(
    expr: Expr,
    target: InteriorProduct,
    registry: Optional[PropertyRegistry] = None,  # noqa: ARG001
) -> Expr:
    """Rewrite every ``ι_X ∘ ι_X`` occurrence in ``expr`` to ``0``.

    Mirrors :func:`jacopy.calculus.exterior_d.apply_d_squared_zero`
    for the interior product: both the element-level
    ``Act(ι_X, Act(ι_X, x))`` and the operator-level
    ``Act(Product(ι_X, ι_X, …), x)`` collapse. The ``target`` argument
    is required here, interior products are a family, so there is no
    ambient default to apply the rewrite against.
    """
    if expr.is_atom:
        return expr
    new_children = tuple(
        apply_iota_squared_zero(c, target) for c in expr.children
    )
    if any(a is not b for a, b in zip(new_children, expr.children)):
        rebuilt: Expr = expr._rebuild(new_children)
    else:
        rebuilt = expr
    if isinstance(rebuilt, Act):
        op, arg = rebuilt.op, rebuilt.arg
        if (
            _is_iota_on(op, target)
            and isinstance(arg, Act)
            and _is_iota_on(arg.op, target)
        ):
            return Integer(0)
        if _product_starts_with_iota_squared(op, target):
            return Integer(0)
    return rebuilt


# --------------------------------------------------------------------- #
# ι_X on functions and exact 1-forms                                     #
# --------------------------------------------------------------------- #


def _degree_is_zero(expr: Expr, registry: Optional[PropertyRegistry]) -> bool:
    """True if ``expr`` is known to be degree 0.

    Uses :func:`degree_of` and falls back to ``False`` on any
    undecidable case, it's safer to leave a rewrite pending than
    apply it on the wrong shape.
    """
    try:
        return degree_of(expr, registry) == Degree.const(0)
    except ValueError:
        return False


def apply_iota_axioms(
    expr: Expr,
    target: InteriorProduct,
    registry: Optional[PropertyRegistry] = None,
    *,
    d: Optional[ExteriorDerivative] = None,
    X: Optional[Derivation] = None,
) -> Expr:
    """Apply the element-level ι_X axioms in one bottom-up pass.

    Rewrites recognised:

    * ``Act(ι_X, f) → 0`` whenever ``f`` resolves to degree 0 under
      ``registry`` (ι_X lowers degree by one and Ω^{−1}(M) = 0).
    * ``Act(ι_X, Act(d, f)) → Act(X, f)`` when ``X`` is supplied as a
      :class:`Derivation` on functions and ``f`` is degree 0, this
      is the pairing axiom ``ι_X(df) = X(f)``.

    Degrees are checked via :func:`degree_of` against ``registry``.
    Unknown-degree operands are left alone rather than assumed zero,
    silently rewriting an ambiguous shape would hide a missing
    grading declaration.

    ``d`` defaults to the module-level exterior derivative from
    :mod:`exterior_d`. Passing a distinct instance (e.g. ``d_E``)
    restricts the exact-1-form rewrite to that specific
    :class:`ExteriorDerivative`.

    ``X`` is optional. Omit it to apply only the ``ι_X(f) = 0``
    clause, useful when the user's vector field isn't represented as
    a :class:`Derivation` on functions yet but the function-vanishing
    rewrite is still wanted.

    The ``ι_X ∘ ι_X = 0`` rewrite is *not* bundled here, use
    :func:`apply_iota_squared_zero` when that is wanted, either before
    or after this pass.
    """
    dop = default_d if d is None else d
    if expr.is_atom:
        return expr
    new_children = tuple(
        apply_iota_axioms(c, target, registry, d=dop, X=X)
        for c in expr.children
    )
    if any(a is not b for a, b in zip(new_children, expr.children)):
        rebuilt: Expr = expr._rebuild(new_children)
    else:
        rebuilt = expr
    if isinstance(rebuilt, Act) and _is_iota_on(rebuilt.op, target):
        inner = rebuilt.arg
        # ι_X(df) = X(f), exact 1-form pairing.
        if (
            X is not None
            and isinstance(inner, Act)
            and isinstance(inner.op, ExteriorDerivative)
            and inner.op == dop
            and _degree_is_zero(inner.arg, registry)
        ):
            return Act(X, inner.arg)
        # ι_X(f) = 0 on functions.
        if _degree_is_zero(inner, registry):
            return Integer(0)
    return rebuilt
