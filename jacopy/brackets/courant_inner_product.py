"""
Canonical (Vaisman) inner product on ``TM ⊕ T*M``.

The standard exact Courant algebroid carries a non-degenerate symmetric
bilinear form

    ⟨(X, α), (Y, β)⟩ = ½ (α(Y) + β(X))

with signature ``(n, n)``. It is the data that the Courant bracket
preserves under anchor action, and the inner-product compatibility
axiom (Vaisman 2005, Roytenberg 1999) is one of the five Courant
algebroid axioms.

This module provides a dedicated :class:`CourantInnerProduct` Expr node
plus the engine-level rewrite rule that unfolds it to two
:class:`~jacopy.calculus.pairing.Pairing` summands. Keeping the inner
product as its own shape (rather than overloading
:class:`~jacopy.calculus.pairing.Pairing` directly) lets the Stage E
prove suite emit chain steps with the readable identifier
``⟨a, b⟩`` and tag the unfold step with a Courant-specific rule name,
without polluting the global pairing axiom space with TM⊕T*M-only
semantics.
"""

from __future__ import annotations

from typing import Any, Tuple

from jacopy.brackets.dorfman import SectionPair
from jacopy.calculus.pairing import Pairing
from jacopy.core.expr import Expr, Product, Rational, Sum
from jacopy.proof.expansion import Definition


# --------------------------------------------------------------------- #
# Expr node                                                              #
# --------------------------------------------------------------------- #


class CourantInnerProduct(Expr):
    """The symmetric inner product ``⟨a, b⟩`` on ``TM ⊕ T*M`` sections.

    Parameters
    ----------
    a, b
        Section operands. The unfold rule
        :class:`CourantInnerProductDefinition` expects both to be
        :class:`SectionPair` instances; the constructor accepts any
        :class:`Expr` so that intermediate proof steps can carry
        unevaluated forms (e.g. a literal symbolic ``a`` standing in
        for an arbitrary section).

    Notes
    -----
    * Degree ``0``: the inner product produces a scalar.
    * Symmetric in its two slots; by the Vaisman convention the
      definition unfolds to ``½ (⟨α, Y⟩ + ⟨β, X⟩)``, which is
      manifestly invariant under ``a ↔ b``.
    * Bilinearity follows from the unfold + Pairing R-linearity, no
      separate distribution rule is needed at this layer.
    """

    __slots__ = ("_a", "_b")

    def __init__(self, a: Expr, b: Expr) -> None:
        if not isinstance(a, Expr):
            raise TypeError("CourantInnerProduct first operand must be an Expr")
        if not isinstance(b, Expr):
            raise TypeError("CourantInnerProduct second operand must be an Expr")
        self._a = a
        self._b = b

    @property
    def left(self) -> Expr:
        return self._a

    @property
    def right(self) -> Expr:
        return self._b

    @property
    def children(self) -> Tuple[Expr, ...]:
        return (self._a, self._b)

    def _key(self) -> Any:
        return (self._a, self._b)

    def _repr_inner(self) -> str:
        return f"⟨{self._a._repr_inner()}, {self._b._repr_inner()}⟩"


def courant_inner_product(a: Expr, b: Expr) -> CourantInnerProduct:
    """Build the inner product ``⟨a, b⟩`` on ``TM ⊕ T*M`` sections."""
    return CourantInnerProduct(a, b)


# --------------------------------------------------------------------- #
# Unfold definition                                                      #
# --------------------------------------------------------------------- #


class CourantInnerProductDefinition(Definition):
    """``⟨(X, α), (Y, β)⟩ → ½ (⟨α, Y⟩ + ⟨β, X⟩)``.

    The Vaisman normalisation. Fires only when both slots are explicit
    :class:`SectionPair` instances, so that proofs working with
    symbolic sections (``a``, ``b`` opaque) keep the unevaluated
    inner-product shape until the caller substitutes pair operands.

    The output is a :class:`Sum` of two :class:`Pairing` nodes scaled
    by ``½``. Once emitted, downstream
    :class:`~jacopy.calculus.pairing_axioms.PairingLinearityDefinition`
    and
    :class:`~jacopy.calculus.pairing_axioms.PairingLieLeibnizDefinition`
    take over for any further unfolding.
    """

    name = "CourantInnerProduct definition"

    def matches(self, expr: Expr) -> bool:
        return (
            isinstance(expr, CourantInnerProduct)
            and isinstance(expr.left, SectionPair)
            and isinstance(expr.right, SectionPair)
        )

    def rewrite(self, expr: Expr) -> Expr:
        assert isinstance(expr, CourantInnerProduct)
        a, b = expr.left, expr.right
        assert isinstance(a, SectionPair)
        assert isinstance(b, SectionPair)
        X, alpha = a.vector, a.form
        Y, beta = b.vector, b.form
        return Product(
            Rational(1, 2),
            Sum(Pairing(alpha, Y), Pairing(beta, X)),
        )
