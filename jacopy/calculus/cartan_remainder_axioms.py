r"""
Defining axioms for the Cartan-remainder operators (Faz 15.B).

Two engine rewrite rules realise the defining identities of the
Cartan-remainder atoms introduced in
:mod:`jacopy.calculus.cartan_remainder` (standard) and
:mod:`jacopy.calculus.tilde.cartan_remainder` (tilde):

* :class:`CartanRemainderDefinition`,
  ``K_V → −L_V + d ∘ ι_V``. Acts on ``Act(K_V, ω)`` and emits the inert
  Sum ``Sum(Neg(Act(L_V, ω)), Act(d, Act(ι_V, ω)))``. Configurable
  ``d`` / ``lie_derivative`` / ``interior`` factories let Lie-algebroid
  callers swap the standard Cartan operators for their algebroid
  counterparts.

* :class:`TildeCartanRemainderDefinition`,
  ``K̃_η → −L̃_η + d̃ ∘ ι̃_η``. Reads the indexing form ``η`` and Poisson
  bivector ``π`` off the matched
  :class:`~jacopy.calculus.tilde.cartan_remainder.TildeCartanRemainder`
  atom and constructs fresh ``L̃_η``, ``ι̃_η``, ``d̃`` heads on the same
  ``π``. Two ``K̃`` rules with distinct ``π``'s coexist on one engine
  without aliasing because :class:`TildeCartanRemainder.__eq__` keys on
  the bivector.

Neither rule is registry-aware: the Cartan remainder is a *symbolic*
shorthand whose definition holds unconditionally on the corresponding
exterior algebra. Pair them with the rest of the standard / tilde
calculus rules to expand a derivator-identity LHS or RHS down to a
shape that the closure axioms can collapse.
"""

from __future__ import annotations

from typing import Callable, Optional

from jacopy.algebra.derivation import Act, Derivation
from jacopy.calculus.cartan_remainder import CartanRemainder
from jacopy.calculus.exterior_d import ExteriorDerivative, d as default_d
from jacopy.calculus.interior import InteriorProduct, interior as default_interior
from jacopy.calculus.lie_derivative import (
    lie_derivative as default_lie_derivative,
)
from jacopy.calculus.tilde.cartan_remainder import TildeCartanRemainder
from jacopy.calculus.tilde.operators import (
    TildeExteriorDerivative,
    TildeInteriorProduct,
    TildeLieDerivative,
)
from jacopy.core.expr import Expr, Neg, Sum
from jacopy.proof.expansion import Definition


LieDerivativeFactory = Callable[[Expr], Derivation]
InteriorFactory = Callable[[Expr], Derivation]


# --------------------------------------------------------------------- #
# K_V ω = −L_V ω + d(ι_V ω)                                              #
# --------------------------------------------------------------------- #


class CartanRemainderDefinition(Definition):
    r"""``Act(K_V, ω) → −Act(L_V, ω) + Act(d, Act(ι_V, ω))``.

    Matches on the outer head being a
    :class:`~jacopy.calculus.cartan_remainder.CartanRemainder` atom.
    The vector field ``V`` is read off the matched atom; the rewrite
    constructs fresh :class:`LieDerivative` and :class:`InteriorProduct`
    heads on it (structural equality on those classes makes them
    compare equal to any pre-existing instances downstream rules may
    have constructed).

    Optional ``d`` / ``lie_derivative`` / ``interior`` overrides let
    Lie-algebroid callers substitute their Cartan operators (different
    exterior derivative ``d_E``, algebroid Lie derivative
    ``L_X^E``, algebroid interior product), the rule itself is shape-
    agnostic, only requiring the three pieces to compose into a
    degree-0 form-side operator.
    """

    name = "K_V ω = −L_V ω + d(ι_V ω)"

    def __init__(
        self,
        *,
        d: Optional[ExteriorDerivative] = None,
        lie_derivative: Optional[LieDerivativeFactory] = None,
        interior: Optional[InteriorFactory] = None,
    ) -> None:
        self._d = d if d is not None else default_d
        self._lie = (
            lie_derivative if lie_derivative is not None else default_lie_derivative
        )
        self._iota = interior if interior is not None else default_interior

    def matches(self, expr: Expr) -> bool:
        return (
            isinstance(expr, Act)
            and isinstance(expr.op, CartanRemainder)
        )

    def rewrite(self, expr: Expr) -> Expr:
        head = expr.op
        assert isinstance(head, CartanRemainder)
        V = head.vector_field
        omega = expr.arg
        L_V = self._lie(V)
        iota_V = self._iota(V)
        return Sum(
            Neg(Act(L_V, omega)),
            Act(self._d, Act(iota_V, omega)),
        )


# --------------------------------------------------------------------- #
# K̃_η V = −L̃_η V + d̃(ι̃_η V)                                            #
# --------------------------------------------------------------------- #


class TildeCartanRemainderDefinition(Definition):
    r"""``Act(K̃_η, V) → −Act(L̃_η, V) + Act(d̃, Act(ι̃_η, V))``.

    Matches on the outer head being a
    :class:`~jacopy.calculus.tilde.cartan_remainder.TildeCartanRemainder`
    atom. The form ``η`` and bivector ``π`` are read off the matched
    atom, no constructor parameter needed because each ``K̃_η`` carries
    its own ``π``. The rewrite constructs fresh ``L̃_η``, ``ι̃_η``, and
    ``d̃`` heads on that ``π``.

    Unconditional: the rule fires on any tilde-Cartan-remainder
    application. The Poisson-flag-gated downstream rewrites (e.g.
    ``d̃² V → 0``) read the registry themselves; this rule only unfolds
    the symbolic ``K̃_η`` shorthand into its three constituent operators.
    """

    name = "K̃_η V = −L̃_η V + d̃(ι̃_η V)"

    def matches(self, expr: Expr) -> bool:
        return (
            isinstance(expr, Act)
            and isinstance(expr.op, TildeCartanRemainder)
        )

    def rewrite(self, expr: Expr) -> Expr:
        head = expr.op
        assert isinstance(head, TildeCartanRemainder)
        eta = head.form
        pi = head.bivector
        V = expr.arg
        L_tilde = TildeLieDerivative(eta, pi)
        iota_tilde = TildeInteriorProduct(eta)
        d_tilde = TildeExteriorDerivative(pi)
        return Sum(
            Neg(Act(L_tilde, V)),
            Act(d_tilde, Act(iota_tilde, V)),
        )
