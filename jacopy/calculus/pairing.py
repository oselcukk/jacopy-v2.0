"""
Canonical pairing ``⟨·, ·⟩: T^*M ⊗ TM → ℝ``.

The evaluation map between 1-forms and vector fields is a basic
building block for several downstream constructions:

* The classical Koszul bracket on 1-forms uses ``d⟨ρα, β⟩`` as its
  third term (Stage 2 of :doc:`schouten`).
* The musical compatibility between a symplectic form ``ω`` and a
  Poisson bivector ``π`` surfaces here whenever a proof needs to
  reduce ``ω(X, Y)`` or ``π(α, β)`` to a pairing.
* Anchor maps on Lie algebroids consume a pairing to close
  Leibniz-style identities.

:class:`Pairing` is intentionally inert, it stores its two arguments
as children in order ``(alpha, X)`` and defers bilinearity to the
algorithms layer (``distribute`` handles :class:`Sum` slots, and the
canonical-form pipeline threads scalar factors through). The pairing's
degree is ``0``: it produces a scalar regardless of its inputs, and
:func:`jacopy.algebra.derivation.degree_of` special-cases it below.

A user-facing factory :func:`pairing` is provided so call sites read
as ``pairing(alpha, X)`` rather than the capitalised class name.
"""

from __future__ import annotations

from typing import Any, Tuple

from jacopy.core.expr import Expr


class Pairing(Expr):
    """The canonical pairing ``⟨α, X⟩``.

    Treats its two arguments symmetrically at the structural level,
    children are ``(alpha, X)`` in the order given and equality is
    structural. The *semantic* convention (first slot: 1-form; second
    slot: vector field) is the user's to enforce; the Expr tree stores
    either order the caller supplies.

    The pairing is bilinear, so algorithms that distribute through
    :class:`Sum` in its children or pull scalar factors outward are
    expected to operate on this node by visiting its children like any
    other Expr. No special Leibniz is attached, the node is a leaf
    from the product-rule layer's perspective.
    """

    __slots__ = ("_alpha", "_X")

    def __init__(self, alpha: Expr, X: Expr) -> None:
        if not isinstance(alpha, Expr):
            raise TypeError("Pairing first argument must be an Expr")
        if not isinstance(X, Expr):
            raise TypeError("Pairing second argument must be an Expr")
        self._alpha = alpha
        self._X = X

    @property
    def alpha(self) -> Expr:
        return self._alpha

    @property
    def X(self) -> Expr:
        return self._X

    @property
    def children(self) -> Tuple[Expr, ...]:
        return (self._alpha, self._X)

    def _key(self) -> Any:
        return (self._alpha, self._X)

    def _repr_inner(self) -> str:
        return f"⟨{self._alpha._repr_inner()}, {self._X._repr_inner()}⟩"


def pairing(alpha: Expr, X: Expr) -> Pairing:
    """Build the pairing ``⟨α, X⟩``.

    Preferred call site helper, mirrors the :class:`Pairing`
    constructor with a functional name that matches how callers write
    the pairing in prose.
    """
    return Pairing(alpha, X)
