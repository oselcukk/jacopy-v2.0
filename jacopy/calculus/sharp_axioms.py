"""
Sharp axioms, Faz 13.A.

Two engine-level rewrite rules that promote ``π^♯`` from a tensorial
:class:`~jacopy.calculus.musical.Sharp` atom to a fully usable building
block in derived-bracket / Schouten-Nijenhuis proofs:

* :class:`SharpLinearityDefinition`, ``π^♯(A + B + …) → π^♯(A) + π^♯(B) + …``.
  Sharp is :math:`\\mathbb{R}`-linear; the rewrite makes that linearity a
  named axiom step in the transcript instead of a silent product-rule
  side effect.
* :class:`SharpOnExactDefinition`, ``π^♯(df) → X_f``. Names the
  composition ``π^♯ ∘ d`` on a 0-form as a Hamiltonian vector field, so
  later steps can reason about ``X_f`` as an opaque
  :class:`~jacopy.algebra.derivation.Derivation` atom rather than the
  nested ``Act(Sharp, Act(d, f))`` shape.

The first rule has structural overlap with
:func:`~jacopy.algorithms.product_rule.product_rule` (which already
distributes any ``Act(D, Sum(...))``), but registering it as an explicit
:class:`Definition` is what gives the resulting transcript a labelled
"Sharp R-linearity" step instead of a generic "product-rule"
justification, important for the 2f-deep / 2g-deep notebooks where
each axiom application must be visible.
"""

from __future__ import annotations

from typing import Optional

from jacopy.algebra.derivation import Act, degree_of
from jacopy.calculus.exterior_d import ExteriorDerivative, d as default_d
from jacopy.calculus.musical import Sharp
from jacopy.core.expr import Expr, Neg, Sum
from jacopy.core.registry import PropertyRegistry
from jacopy.core.symbolic_degree import Degree
from jacopy.proof.expansion import Definition


def _is_degree_zero(
    expr: Expr, registry: Optional[PropertyRegistry]
) -> bool:
    try:
        return degree_of(expr, registry) == Degree.const(0)
    except ValueError:
        return False


# --------------------------------------------------------------------- #
# Axiom 1, π^♯ R-linearity over Sum                                     #
# --------------------------------------------------------------------- #


class SharpLinearityDefinition(Definition):
    """``π^♯(A + B + …) → π^♯(A) + π^♯(B) + …``, Sharp is R-linear.

    Scoped to a specific :class:`~jacopy.calculus.musical.Sharp`
    instance so that two distinct bivectors' sharps coexisting in a
    proof don't get conflated; only the targeted operator distributes.
    """

    def __init__(self, sharp: Sharp) -> None:
        if not isinstance(sharp, Sharp):
            raise TypeError("SharpLinearityDefinition requires a Sharp")
        self._sharp = sharp
        self.name = f"π♯ R-linearity [{sharp.bivector._repr_inner()}]"

    @property
    def sharp(self) -> Sharp:
        return self._sharp

    def matches(self, expr: Expr) -> bool:
        return (
            isinstance(expr, Act)
            and expr.op == self._sharp
            and isinstance(expr.arg, Sum)
        )

    def rewrite(self, expr: Expr) -> Expr:
        return Sum.make(
            *(Act(self._sharp, c) for c in expr.arg.children)
        )


# --------------------------------------------------------------------- #
# Axiom 1b, π^♯ Neg-linearity                                           #
# --------------------------------------------------------------------- #


class SharpNegLinearityDefinition(Definition):
    """``π^♯(Neg(A)) → Neg(π^♯(A))``, Sharp commutes with Neg.

    Sibling of :class:`SharpLinearityDefinition`: the Sum-distribution
    rule does not fire on a bare ``Neg`` argument, but the residue from
    Lichnerowicz / Cartan-magic emits expressions of the form
    ``π^♯(-L_V η)`` whose Neg must surface to the outside before
    LBVF Neg-linearity and ``collect_terms`` can pair-cancel.
    """

    def __init__(self, sharp: Sharp) -> None:
        if not isinstance(sharp, Sharp):
            raise TypeError("SharpNegLinearityDefinition requires a Sharp")
        self._sharp = sharp
        self.name = f"π♯ Neg-linearity [{sharp.bivector._repr_inner()}]"

    @property
    def sharp(self) -> Sharp:
        return self._sharp

    def matches(self, expr: Expr) -> bool:
        return (
            isinstance(expr, Act)
            and expr.op == self._sharp
            and isinstance(expr.arg, Neg)
        )

    def rewrite(self, expr: Expr) -> Expr:
        return Neg(Act(self._sharp, expr.arg.arg))


# --------------------------------------------------------------------- #
# Axiom 2, π^♯(df) → X_f                                                #
# --------------------------------------------------------------------- #


class SharpOnExactDefinition(Definition):
    """``π^♯(df) → X_f``, Hamiltonian vector field naming.

    Fires on the shape ``Act(π^♯, Act(d, f))`` whenever ``f`` resolves
    to degree 0 in the registry. The rewrite produces a fresh
    :class:`~jacopy.calculus.hamiltonian_vf.HamiltonianVectorField`
    atom carrying ``f`` and the bivector inherited from this rule's
    target Sharp, downstream rules see a clean degree-0 operator and
    can reason about ``X_f`` without descending through the Sharp/d
    composition.

    ``d`` defaults to the :mod:`exterior_d` module singleton; pass an
    explicit instance when a Lie-algebroid ``d_E`` coexists with the
    standard ``d``.
    """

    def __init__(
        self,
        sharp: Sharp,
        *,
        d: Optional[ExteriorDerivative] = None,
        registry: Optional[PropertyRegistry] = None,
    ) -> None:
        if not isinstance(sharp, Sharp):
            raise TypeError("SharpOnExactDefinition requires a Sharp")
        self._sharp = sharp
        self._d = default_d if d is None else d
        self._registry = registry
        self.name = (
            f"π♯(df) = X_f [{sharp.bivector._repr_inner()}]"
        )

    @property
    def sharp(self) -> Sharp:
        return self._sharp

    @property
    def d(self) -> ExteriorDerivative:
        return self._d

    def matches(self, expr: Expr) -> bool:
        if not (isinstance(expr, Act) and expr.op == self._sharp):
            return False
        inner = expr.arg
        if not (
            isinstance(inner, Act)
            and isinstance(inner.op, ExteriorDerivative)
            and inner.op == self._d
        ):
            return False
        return _is_degree_zero(inner.arg, self._registry)

    def rewrite(self, expr: Expr) -> Expr:
        # Lazy import: hamiltonian_vf imports musical (and indirectly
        # this module via the package re-exports), so resolving the
        # class at call time avoids a top-level cycle.
        from jacopy.calculus.hamiltonian_vf import HamiltonianVectorField

        f = expr.arg.arg  # type: ignore[union-attr]
        return HamiltonianVectorField(f, bivector=self._sharp.bivector)
