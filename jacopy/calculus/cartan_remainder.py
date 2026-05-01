"""
Cartan remainder operator ``K_V``.

For a vector field ``V`` the *Cartan remainder* is the degree-0 operator
on forms

    K_V := −L_V + d ∘ ι_V.

Cartan magic ``L_V = d ι_V + ι_V d`` collapses this to the equivalent
identity ``K_V = −ι_V ∘ d``, but expressing it as the explicit
two-term combination is the form Section 3.1.5's derivator identities
use as their right-hand side, so we keep the longer signature primary
and the shorter ``−ι_V d`` shape derivable.

This module provides only the inert :class:`Derivation` atom plus the
``K(V)`` factory. The defining-axiom rewrite
``Act(K_V, ω) → −Act(L_V, ω) + Act(d, Act(ι_V, ω))`` lands as a
separate engine rule in :mod:`jacopy.calculus.cartan_remainder_axioms`
(Faz 15.B), so a proof can step through the expansion explicitly.

Like :class:`~jacopy.calculus.interior.InteriorProduct` and
:class:`~jacopy.calculus.lie_derivative.LieDerivative` there is no
module-level singleton, Cartan remainders are a family indexed by the
underlying vector field. Use :func:`K` to construct them.
"""

from __future__ import annotations

from typing import Any, Optional

from jacopy.algebra.derivation import Derivation
from jacopy.core.expr import Expr


class CartanRemainder(Derivation):
    """``K_V``, degree-0 Cartan-remainder operator on forms.

    Carries the indexing vector field ``V`` on :attr:`vector_field`.
    The defining identity ``K_V := −L_V + d ∘ ι_V`` is realised as an
    engine rewrite (Faz 15.B); this class is the inert atom that the
    rewrite recognises.

    Equality is structural over ``(name, degree, vector_field)``, two
    Cartan remainders with the same ``V`` and the default name compare
    equal. Custom ``name`` overrides participate in the equality key.
    """

    __slots__ = ("_vector_field",)

    def __init__(self, V: Expr, *, name: Optional[str] = None) -> None:
        if not isinstance(V, Expr):
            raise TypeError("CartanRemainder requires an Expr vector field")
        display = name if name is not None else f"K_{V._repr_inner()}"
        super().__init__(display, degree=0)
        self._vector_field = V

    @property
    def vector_field(self) -> Expr:
        return self._vector_field

    def _key(self) -> Any:
        return (self._name, self._degree, self._vector_field)


def K(V: Expr, *, name: Optional[str] = None) -> CartanRemainder:
    """Build the Cartan remainder ``K_V`` for vector field ``V``."""
    return CartanRemainder(V, name=name)
