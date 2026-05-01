"""
Tilde Cartan remainder operator ``K̃_η``.

For a 1-form ``η`` and a Poisson bivector ``π`` the *tilde Cartan
remainder* is the degree-0 operator on multivectors

    K̃_η := −L̃_η + d̃ ∘ ι̃_η.

It is the Koszul-side analogue of :class:`~jacopy.calculus.cartan_remainder.CartanRemainder`,
mirroring the relation between the standard and tilde Cartan magic
formulas:

* standard: ``L_V = d ι_V + ι_V d``
* tilde:    ``L̃_η = d̃ ι̃_η + ι̃_η d̃``

so each remainder collapses on the magic-formula expansion and the
"definition vs. magic" gap is exactly the negative of the missing term.

Like the standard remainder this class is the inert :class:`Derivation`
atom. The defining-axiom rewrite
``Act(K̃_η, V) → −Act(L̃_η, V) + Act(d̃, Act(ι̃_η, V))`` lives as an
engine rule in :mod:`jacopy.calculus.tilde.cartan_remainder_axioms`
(Faz 15.B). The two parameters are stored separately so independent
``K̃_η``, ``K̃_μ`` operators can co-exist in a single proof without
aliasing; equality is structural over ``(name, degree, form, bivector)``.
"""

from __future__ import annotations

from typing import Any, Optional

from jacopy.algebra.derivation import Derivation
from jacopy.core.expr import Expr


class TildeCartanRemainder(Derivation):
    """``K̃_η``, degree-0 tilde Cartan-remainder operator on multivectors.

    Carries the indexing 1-form ``η`` and Poisson bivector ``π`` on
    :attr:`form` and :attr:`bivector`. The defining identity
    ``K̃_η := −L̃_η + d̃ ∘ ι̃_η`` is realised as an engine rewrite
    (Faz 15.B); this class is the inert atom.
    """

    __slots__ = ("_form", "_bivector")

    def __init__(
        self,
        eta: Expr,
        pi: Expr,
        *,
        name: Optional[str] = None,
    ) -> None:
        if not isinstance(eta, Expr):
            raise TypeError("TildeCartanRemainder requires an Expr form")
        if not isinstance(pi, Expr):
            raise TypeError("TildeCartanRemainder requires an Expr bivector")
        display = name if name is not None else f"K̃_{eta._repr_inner()}"
        super().__init__(display, degree=0)
        self._form = eta
        self._bivector = pi

    @property
    def form(self) -> Expr:
        return self._form

    @property
    def bivector(self) -> Expr:
        return self._bivector

    def _key(self) -> Any:
        return (self._name, self._degree, self._form, self._bivector)


def K_tilde(
    eta: Expr,
    pi: Expr,
    *,
    name: Optional[str] = None,
) -> TildeCartanRemainder:
    """Build the tilde Cartan remainder ``K̃_η`` for form ``η`` and bivector ``π``."""
    return TildeCartanRemainder(eta, pi, name=name)
