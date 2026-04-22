"""
Derived bracket construction — the mathematical heart of the package.

Given a graded bracket ``[·, ·]`` on a graded module and a generator
``Q`` of odd (or appropriate) degree, the *derived bracket* is

    {a, b}_Q := [[a, Q], b]

which is automatically a graded Leibniz bracket regardless of any
conditions on ``Q``. The derived bracket is *graded antisymmetric* and
satisfies the graded Jacobi identity **if and only if** the
compatibility condition

    [Q, Q] = 0

holds in the base bracket. This conditionality is the whole point of
the construction — structures like Poisson, Courant, and Koszul
brackets arise as derived brackets of a Schouten-type base bracket,
and each of their Jacobi identities reduces to a single equation on
the generator ``Q``.

The :class:`DerivedBracket` class captures this construction at the
algorithmic level:

* :meth:`expand` produces ``[[a, Q], b]`` using the base bracket's own
  :meth:`~gradalg.brackets.base.GradedBracket.expand` on each layer.
* :meth:`jacobi_obstruction` returns ``[Q, Q]_base`` — the Expr whose
  vanishing is equivalent to the derived bracket satisfying Jacobi.
* :func:`derived_bracket` is a light factory helper.

The *Derived Bracket Theorem* itself — that Jacobi holds ⟺ the
obstruction vanishes — is registered as a package-level theorem in
Faz 9 and consulted from the proof layer. This module only provides
the object; the theorem lives elsewhere so that each new
``DerivedBracket`` instance automatically inherits the result rather
than re-proving it.
"""

from __future__ import annotations

from typing import Any, Optional

from gradalg.brackets.base import BracketApply, GradedBracket
from gradalg.core.expr import Expr
from gradalg.core.registry import PropertyRegistry
from gradalg.core.symbolic_degree import Degree, DegreeLike, as_degree


class DerivedBracket(GradedBracket):
    """``{a, b}_Q := [[a, Q]_base, b]_base``.

    Parameters
    ----------
    base
        The underlying graded bracket.
    Q
        The generator — an :class:`Expr` of degree ``degree_Q``.
    degree_Q
        Explicit generator degree. Defaults to ``0`` when unspecified.
        The derived bracket's own degree is ``degree_Q − 2`` per the
        derived-bracket degree formula.
    name
        Optional display name. Defaults to a structured tag derived
        from the base bracket and generator.
    """

    def __init__(
        self,
        base: GradedBracket,
        Q: Expr,
        *,
        degree_Q: DegreeLike = 0,
        name: Optional[str] = None,
    ) -> None:
        if not isinstance(base, GradedBracket):
            raise TypeError("DerivedBracket 'base' must be a GradedBracket")
        if not isinstance(Q, Expr):
            raise TypeError("DerivedBracket generator 'Q' must be an Expr")
        self._base = base
        self._Q = Q
        self._degree_Q = as_degree(degree_Q)
        display = name or f"{{·,·}}_{Q._repr_inner()}"
        # Derived-bracket degree formula: |{·,·}_Q| = |Q| - 2. Graded
        # Leibniz always holds; antisymmetry and Jacobi are conditional
        # on [Q, Q]_base = 0. We surface that by reporting Jacobi as
        # None ("conditional") rather than True or False — the proof
        # layer is the one that discharges it against the obstruction.
        super().__init__(
            display,
            degree=self._degree_Q + Degree.const(-2),
            is_graded_antisymmetric=True,
            satisfies_leibniz=True,
            satisfies_graded_jacobi=None,
        )

    @property
    def base(self) -> GradedBracket:
        return self._base

    @property
    def Q(self) -> Expr:
        return self._Q

    @property
    def degree_Q(self) -> Degree:
        return self._degree_Q

    # ---- core expansion -------------------------------------------- #

    def expand(
        self,
        a: Expr,
        b: Expr,
        registry: Optional[PropertyRegistry] = None,
    ) -> Expr:
        """``{a, b}_Q = [[a, Q]_base, b]_base``.

        Two base-bracket applications are produced and each is
        immediately expanded by the base bracket. The result is the
        fully-unfolded Expr — the caller is free to pipe it through
        ``simplify`` / ``canonicalize`` for a readable form.
        """
        inner = self._base.expand(a, self._Q, registry)
        return self._base.expand(inner, b, registry)

    def expand_definition(
        self,
        a: Expr,
        b: Expr,
        registry: Optional[PropertyRegistry] = None,
    ) -> Expr:
        """Return ``[[a, Q]_base, b]_base`` *before* base-expansion.

        Useful for proofs that want to show the derived definition
        literally, without collapsing the inner base-bracket nodes. The
        result is a :class:`BracketApply` of the base bracket applied
        to a base-bracket node.
        """
        inner = BracketApply(self._base, a, self._Q)
        return BracketApply(self._base, inner, b)

    # ---- Jacobi obstruction ---------------------------------------- #

    def jacobi_obstruction(
        self, registry: Optional[PropertyRegistry] = None
    ) -> Expr:
        """Return ``[Q, Q]_base`` — the expression whose vanishing is
        equivalent to the derived bracket satisfying graded Jacobi.

        This is the *universal* obstruction: the Derived Bracket Theorem
        says that for any derived bracket, Jacobi on ``{·, ·}_Q`` holds
        on all operands ⟺ this single expression vanishes.
        """
        return self._base.expand(self._Q, self._Q, registry)

    def jacobi_obstruction_raw(self) -> BracketApply:
        """Return the unexpanded ``[Q, Q]_base`` :class:`BracketApply`
        node — leaving the base-bracket shape intact for display or
        further pattern matching."""
        return BracketApply(self._base, self._Q, self._Q)

    # ---- identity -------------------------------------------------- #

    def _identity_key(self) -> Any:
        # Extend the base key with the base-bracket reference, the
        # generator, and the generator's degree so that two derived
        # brackets with identical parameters compare equal and hash
        # alike.
        return super()._identity_key() + (self._base, self._Q, self._degree_Q)


# --------------------------------------------------------------------- #
# Factory                                                                #
# --------------------------------------------------------------------- #


def derived_bracket(
    base: GradedBracket,
    Q: Expr,
    *,
    degree_Q: DegreeLike = 0,
    name: Optional[str] = None,
) -> DerivedBracket:
    """Construct a :class:`DerivedBracket` — mirror of the class
    constructor with a friendlier functional name."""
    return DerivedBracket(base, Q, degree_Q=degree_Q, name=name)
