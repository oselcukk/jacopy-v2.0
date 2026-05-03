"""
Anchor projection and ``D`` operator for the exact Courant algebroid.

Two structural operators on ``TM ⊕ T*M`` that complement the Courant
bracket and inner product:

* :class:`CourantAnchor`, the canonical projection
  ``π: TM ⊕ T*M → TM, (X, α) ↦ X``. The Faz 9 :class:`CourantAlgebroid`
  wrapper documents the anchor as "implicit, since
  :class:`~jacopy.brackets.dorfman.SectionPair` extraction handles
  it"; that remains true at the bracket-expansion layer, but the
  Stage E prove suite needs a *named* anchor shape so chains can read
  ``anchor([e1, e2]_C) = [anchor(e1), anchor(e2)]_VF`` literally. This
  module surfaces that name and its single-step unfold rule.

* :class:`DOperator`, the Vaisman ``D: C∞(M) → Γ(TM ⊕ T*M)`` defined
  by ``D f = (0, d f)``. Equivalent characterisation via the inner
  product: ``⟨D f, e⟩ = ½ ρ(e)(f)``. The direct form is what
  :class:`DOperatorDefinition` rewrites to; the inner-product form
  is then derivable through
  :class:`~jacopy.brackets.courant_inner_product.CourantInnerProductDefinition`
  + :class:`~jacopy.calculus.pairing_axioms.PairingLinearityDefinition`,
  no separate axiom is needed.

Both unfold rules tag with ``provenance_tag="axiom"`` at the call
site; the Definitions themselves are pure rewrite rules and the
provenance tagging happens in the prove method emitting the step.
"""

from __future__ import annotations

from typing import Any, Callable, Optional, Tuple

from jacopy.algebra.derivation import Act, Derivation
from jacopy.brackets.dorfman import SectionPair
from jacopy.calculus.exterior_d import d as default_d
from jacopy.core.expr import Expr, Integer
from jacopy.proof.expansion import Definition


# --------------------------------------------------------------------- #
# Anchor projection                                                      #
# --------------------------------------------------------------------- #


class CourantAnchor(Expr):
    """The projection ``anchor(e) = π_TM(e)`` for ``e ∈ Γ(TM ⊕ T*M)``.

    Wraps a single :class:`Expr` argument representing a section. When
    the argument is a :class:`SectionPair`, the unfold rule
    :class:`CourantAnchorDefinition` collapses ``anchor((X, α)) → X``;
    otherwise the node remains opaque so symbolic sections survive in
    proofs until a SectionPair witness is supplied.

    Notes
    -----
    The Faz 9 wrapper consumed the anchor implicitly through
    SectionPair extraction. Stage E re-surfaces it because the Courant
    algebroid axioms refer to anchor compatibility *at the section
    level*: writing ``anchor([e1, e2]_C)`` instead of accessing
    ``.vector`` keeps the chain readable as a textbook proof.
    """

    __slots__ = ("_section",)

    def __init__(self, section: Expr) -> None:
        if not isinstance(section, Expr):
            raise TypeError("CourantAnchor argument must be an Expr")
        self._section = section

    @property
    def section(self) -> Expr:
        return self._section

    @property
    def children(self) -> Tuple[Expr, ...]:
        return (self._section,)

    def _key(self) -> Any:
        return (self._section,)

    def _repr_inner(self) -> str:
        return f"anchor({self._section._repr_inner()})"


def anchor(section: Expr) -> CourantAnchor:
    """Build the anchor projection ``anchor(section)``."""
    return CourantAnchor(section)


class CourantAnchorDefinition(Definition):
    """``anchor((X, α)) → X``, projection onto the vector half.

    Fires only when the argument is a literal :class:`SectionPair`, so
    that proofs working with symbolic sections retain the
    ``anchor(·)`` shape until they supply a pair operand.
    """

    name = "Courant anchor projection"

    def matches(self, expr: Expr) -> bool:
        return (
            isinstance(expr, CourantAnchor)
            and isinstance(expr.section, SectionPair)
        )

    def rewrite(self, expr: Expr) -> Expr:
        assert isinstance(expr, CourantAnchor)
        section = expr.section
        assert isinstance(section, SectionPair)
        return section.vector


# --------------------------------------------------------------------- #
# D operator                                                             #
# --------------------------------------------------------------------- #


class DOperator(Expr):
    """The Vaisman ``D: C∞(M) → Γ(TM ⊕ T*M)``.

    ``D f`` is the section ``(0, d f)``. Wraps the function argument
    ``f`` symbolically; the unfold rule
    :class:`DOperatorDefinition` rewrites it to a literal
    :class:`SectionPair` whose vector half is :class:`Integer` zero
    and form half is ``Act(d, f)``.

    Parameters
    ----------
    f
        The smooth function argument. Stored as-is, no degree
        introspection happens at this layer; downstream rules read the
        argument back through :attr:`f`.
    d
        The exterior derivative operator to use when unfolding.
        Defaults to the smooth-manifold singleton
        :data:`jacopy.calculus.exterior_d.d`. Kept as instance state so
        a non-default Cartan family (e.g. an algebroid's ``d_ρ``)
        propagates faithfully into the prove suite.
    """

    __slots__ = ("_f", "_d")

    def __init__(
        self,
        f: Expr,
        *,
        d: Optional[Derivation] = None,
    ) -> None:
        if not isinstance(f, Expr):
            raise TypeError("DOperator argument must be an Expr")
        self._f = f
        self._d = d if d is not None else default_d

    @property
    def f(self) -> Expr:
        return self._f

    @property
    def d_op(self) -> Derivation:
        return self._d

    @property
    def children(self) -> Tuple[Expr, ...]:
        return (self._f,)

    def _key(self) -> Any:
        return (self._f, self._d)

    def _repr_inner(self) -> str:
        return f"D({self._f._repr_inner()})"


def d_operator(f: Expr, *, d: Optional[Derivation] = None) -> DOperator:
    """Build ``D f`` for ``f ∈ C∞(M)``."""
    return DOperator(f, d=d)


class DOperatorDefinition(Definition):
    """``D f → (0, d f)``, the Vaisman direct definition.

    The image is a :class:`SectionPair` with vector half
    :class:`Integer` zero and form half ``Act(d, f)``, where ``d`` is
    the operator carried on the source :class:`DOperator` instance.
    Always matches a :class:`DOperator`; idempotent because the result
    is a :class:`SectionPair`, not a :class:`DOperator`.
    """

    name = "D operator definition"

    def matches(self, expr: Expr) -> bool:
        return isinstance(expr, DOperator)

    def rewrite(self, expr: Expr) -> Expr:
        assert isinstance(expr, DOperator)
        return SectionPair(Integer(0), Act(expr.d_op, expr.f))
