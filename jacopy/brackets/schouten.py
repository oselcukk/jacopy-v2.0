"""
Schouten-Nijenhuis bracket on multivector fields.

The Schouten-Nijenhuis (SN) bracket extends the Lie bracket on vector
fields to the full algebra of multivector fields ``⊕_k Γ(Λ^k TM)``,
including functions (0-vectors). In the shifted grading ``|X| = k − 1``
for ``X ∈ Λ^k TM`` it is a graded Lie bracket of degree 0 that acts as
a graded derivation in each slot with respect to the wedge product.

This module implements the bracket via its characterizing rules:

* **Base cases** on atomic operands of SN-degree ``0`` (1-vectors) and
  ``−1`` (functions):

  1. ``[X, Y]_SN = [X, Y]_Lie`` when both are 1-vectors
  2. ``[f, X]_SN = −X(f)`` when ``f`` is a function and ``X`` a vector
  3. ``[X, f]_SN = X(f)``          (graded-antisymmetric partner of 2)
  4. ``[f, g]_SN = 0``              on two functions

* **Wedge Leibniz** recursion, letting the bracket climb into
  higher-order multivectors:

  .. code::

     [X ∧ Y, Z]_SN = X ∧ [Y, Z]_SN + (−1)^{|Y||Z|} [X, Z]_SN ∧ Y
     [Z, X ∧ Y]_SN = [Z, X]_SN ∧ Y + (−1)^{|X||Z|} X ∧ [Z, Y]_SN

Wedges reuse :class:`Product`, a dedicated ``Wedge`` Expr type is
deferred polish, not needed here. Degrees of wedge products follow the
SN convention ``|X ∧ Y| = |X| + |Y| + 1`` (computed by
:func:`_sn_degree`), so that a wedge of two 1-vectors has SN-degree 1
(2-vector = bivector), while plain :func:`degree_of` would give 0.

Atomic higher-order multivectors (e.g. a bare :class:`Symbol` declared
``Graded(degree=1)`` standing in for a bivector ``π``) cannot be
further decomposed. In that case :meth:`expand` returns the inert
:class:`BracketApply` node unchanged, the obstruction
``[π, π]_SN`` surfaces as a typed handle that the proof layer and
derived-bracket machinery consume as the Poisson condition.

**No literal lift onto forms.** ``sn.expand(α, π)`` for a 1-form
``α`` and a multivector ``π`` is not defined here, SN is the
multivector-only bracket and ``DerivedBracket(sn, π).expand(α, β)``
without ``acting_on`` deliberately returns the opaque BracketApply.
Form-level Poisson/Koszul work goes through ``acting_on=ρ`` on the
:class:`~jacopy.brackets.derived.DerivedBracket` (i.e.
``DerivedBracket(sn, π, acting_on=Sharp(π))`` for Poisson,
``acting_on=ρ_E`` on a Lie algebroid). That route emits the Koszul
3-term formula directly and supports arbitrary anchors; a literal
SN-on-forms convention would only handle ``ρ = Sharp(π)`` and adds
no expressive power.
"""

from __future__ import annotations

from typing import Optional

from jacopy.algebra.derivation import Act, degree_of
from jacopy.brackets.base import BracketApply, GradedBracket
from jacopy.core.expr import Expr, Neg, Product, Sum, Zero
from jacopy.core.registry import PropertyRegistry
from jacopy.core.symbolic_degree import Degree


# --------------------------------------------------------------------- #
# SN-aware degree helper                                                 #
# --------------------------------------------------------------------- #


def _sn_degree(
    expr: Expr, registry: Optional[PropertyRegistry]
) -> Degree:
    """Return the SN-grading degree of ``expr``.

    For a wedge :class:`Product` of ``n`` multivector factors the SN
    degree is ``sum(|X_i|) + (n − 1)``, the ``+1`` per extra factor is
    what distinguishes the ``|X| = k − 1`` convention from the plain
    sum-of-degrees that :func:`degree_of` would give.

    Non-Product expressions fall through to :func:`degree_of`, whose
    error on undetermined operands is the right failure mode here: the
    SN algorithm cannot sign-check a Leibniz step whose factor grading
    isn't known.
    """
    if isinstance(expr, Product):
        total = Degree.const(len(expr.children) - 1)
        for c in expr.children:
            total = total + _sn_degree(c, registry)
        return total
    return degree_of(expr, registry)


def _safe_sn_degree(
    expr: Expr, registry: Optional[PropertyRegistry]
) -> Optional[Degree]:
    """:func:`_sn_degree` wrapper that swallows ``ValueError`` to ``None``.

    Used at the top of :meth:`SchoutenBracket.expand`, where the four
    base cases should only fire on operands whose degrees are concrete
    integers, any unresolved grading silently skips to the wedge /
    opaque fallback rather than raising at the user.
    """
    try:
        return _sn_degree(expr, registry)
    except ValueError:
        return None


# --------------------------------------------------------------------- #
# Bracket                                                                #
# --------------------------------------------------------------------- #


class SchoutenBracket(GradedBracket):
    """``[·, ·]_SN``, the Schouten-Nijenhuis bracket.

    Degree 0 in the shifted (``|X| = k − 1``) grading, graded
    antisymmetric, graded Leibniz in each slot, graded Jacobi. Expansion
    dispatches on operand structure: atomic operands of SN-degree 0 / −1
    hit the four base cases, :class:`Product` operands trigger wedge
    Leibniz, and anything else (atomic higher multivectors like a bare
    bivector symbol) is returned as an opaque :class:`BracketApply` so
    the proof layer can still reason about it symbolically.
    """

    def __init__(self, name: str = "[·,·]_SN") -> None:
        super().__init__(
            name,
            degree=0,
            is_graded_antisymmetric=True,
            satisfies_leibniz=True,
            satisfies_graded_jacobi=True,
        )

    # ---- expansion -------------------------------------------------- #

    def expand(
        self,
        a: Expr,
        b: Expr,
        registry: Optional[PropertyRegistry] = None,
    ) -> Expr:
        deg_a = _safe_sn_degree(a, registry)
        deg_b = _safe_sn_degree(b, registry)

        # Base cases, only fire on atomic operands whose SN degrees are
        # concrete integers. Wedge products are handled below; anything
        # with symbolic degree falls through to the opaque return.
        if not isinstance(a, Product) and not isinstance(b, Product):
            a_int = deg_a.as_int() if deg_a is not None else None
            b_int = deg_b.as_int() if deg_b is not None else None
            base = self._try_base_cases(a, b, a_int, b_int)
            if base is not None:
                return base

        # Wedge Leibniz, first slot.
        if isinstance(a, Product) and len(a.children) >= 2:
            wedge = self._wedge_leibniz_slot1(a, b, registry)
            if wedge is not None:
                return wedge

        # Wedge Leibniz, second slot.
        if isinstance(b, Product) and len(b.children) >= 2:
            wedge = self._wedge_leibniz_slot2(a, b, registry)
            if wedge is not None:
                return wedge

        # Opaque, atomic higher-order multivector, or sign parity
        # couldn't be decided. Leave the BracketApply intact so the
        # proof layer can still consume it as a symbolic condition.
        return BracketApply(self, a, b)

    # ---- base cases ------------------------------------------------- #

    @staticmethod
    def _try_base_cases(
        a: Expr, b: Expr, a_int: Optional[int], b_int: Optional[int]
    ) -> Optional[Expr]:
        """Return the base-case expansion for a 1-vector / function pair,
        or ``None`` if the pair is not one of the four closed forms."""
        if a_int == 0 and b_int == 0:
            # [X, Y]_SN = [X, Y]_Lie = X*Y − Y*X
            return Sum(Product(a, b), Neg(Product(b, a)))
        if a_int == -1 and b_int == -1:
            # [f, g]_SN = 0
            return Zero
        if a_int == -1 and b_int == 0:
            # [f, X]_SN = −X(f)
            return Neg(Act(b, a))
        if a_int == 0 and b_int == -1:
            # [X, f]_SN = X(f)
            return Act(a, b)
        return None

    # ---- wedge Leibniz ---------------------------------------------- #

    def _wedge_leibniz_slot1(
        self,
        a: Product,
        b: Expr,
        registry: Optional[PropertyRegistry],
    ) -> Optional[Expr]:
        """``[X ∧ Y, Z]_SN = X ∧ [Y, Z] + (−1)^{|Y||Z|} [X, Z] ∧ Y``.

        Splits ``a`` at its first factor, the tail of length ``n − 1``
        becomes ``Y``, so a three-factor wedge ``X1 ∧ X2 ∧ X3`` recurses
        as ``X1 ∧ [X2 ∧ X3, Z]``, peeling one factor per level. Returns
        ``None`` when the sign parity ``|Y||Z|`` is symbolic, letting
        the caller fall back to an opaque :class:`BracketApply`.
        """
        X = a.children[0]
        Y_children = a.children[1:]
        Y = Y_children[0] if len(Y_children) == 1 else Product(*Y_children)
        deg_Y = _safe_sn_degree(Y, registry)
        deg_Z = _safe_sn_degree(b, registry)
        if deg_Y is None or deg_Z is None:
            return None
        parity = (deg_Y * deg_Z).parity()
        if parity is None:
            return None
        term1 = Product(X, self.expand(Y, b, registry))
        term2 = Product(self.expand(X, b, registry), Y)
        return Sum(term1, Neg(term2) if parity == 1 else term2)

    def _wedge_leibniz_slot2(
        self,
        a: Expr,
        b: Product,
        registry: Optional[PropertyRegistry],
    ) -> Optional[Expr]:
        """``[Z, X ∧ Y]_SN = [Z, X] ∧ Y + (−1)^{|X||Z|} X ∧ [Z, Y]``.

        Symmetric to :meth:`_wedge_leibniz_slot1` but peels the first
        factor of the right operand. Returns ``None`` on symbolic parity
        for the same reason.
        """
        X = b.children[0]
        Y_children = b.children[1:]
        Y = Y_children[0] if len(Y_children) == 1 else Product(*Y_children)
        deg_X = _safe_sn_degree(X, registry)
        deg_Z = _safe_sn_degree(a, registry)
        if deg_X is None or deg_Z is None:
            return None
        parity = (deg_X * deg_Z).parity()
        if parity is None:
            return None
        term1 = Product(self.expand(a, X, registry), Y)
        term2 = Product(X, self.expand(a, Y, registry))
        return Sum(term1, Neg(term2) if parity == 1 else term2)

    # ---- obstruction helpers --------------------------------------- #

    def self_bracket(
        self, Q: Expr, registry: Optional[PropertyRegistry] = None
    ) -> Expr:
        """Return ``[Q, Q]_SN``, the universal obstruction of ``Q``.

        For a Poisson bivector ``π`` this is the 3-vector whose
        vanishing is the Poisson condition. For a Courant generator
        ``Θ`` (once higher algebras land) it plays the same role with
        the Courant compatibility condition. The helper is a thin wrap
        around :meth:`expand` so the caller doesn't have to spell out
        the self-pairing, and reads cleanly at call sites.
        """
        return self.expand(Q, Q, registry)


# --------------------------------------------------------------------- #
# Convenience singleton                                                  #
# --------------------------------------------------------------------- #

sn = SchoutenBracket()
