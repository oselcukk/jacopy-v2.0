"""
Vector-field Lie bracket as a :class:`Derivation` atom, Faz 13.C.

Two vector fields' Lie bracket ``[X, Y]_VF`` is itself a vector field
(degree-0 derivation on functions). This module provides
:class:`LieBracketVF`: an opaque Derivation parametric in the two
vector fields, equipped with structural ``(X, Y)`` identity so two
brackets with the same arguments compare equal regardless of display
name choice.

Why a dedicated atom (instead of expanding to ``X*Y − Y*X``)
-----------------------------------------------------------

The 2f-deep / 2g-deep proof chain folds an *operator-commutator*
shape ``L_X ∘ L_Y − L_Y ∘ L_X`` into a single ``L_{[X,Y]_VF}`` step
(Faz 13.C axiom 5). After that fold the resulting Lie derivative
must be applicable to a form just like any ordinary ``L_W``, in
particular, downstream Cartan-style rewrites need to see a single
:class:`~jacopy.algebra.derivation.Derivation` operand, not a
2-term commutator expansion. Keeping ``[X, Y]_VF`` opaque is what
preserves that uniformity. The literal expansion is available
through :func:`~jacopy.brackets.lie.LieBracket.expand` on the
existing :class:`~jacopy.brackets.lie.LieBracket` for callers that
need the algebraic ``X*Y − Y*X`` shape.
"""

from __future__ import annotations

from typing import Any, Optional

from jacopy.algebra.derivation import Derivation
from jacopy.core.expr import Expr


class LieBracketVF(Derivation):
    """``[X, Y]_VF``, vector-field Lie bracket as a Derivation atom.

    A degree-0 named derivation parametric in the two vector fields.
    Equality / hash are structural over ``(name, degree, X, Y)`` so
    two brackets built from the same arguments compare equal even
    when separately constructed; brackets over different fields
    remain distinct.
    """

    __slots__ = ("_X", "_Y")

    def __init__(
        self,
        X: Expr,
        Y: Expr,
        *,
        name: Optional[str] = None,
    ) -> None:
        if not isinstance(X, Expr):
            raise TypeError("LieBracketVF first argument must be an Expr")
        if not isinstance(Y, Expr):
            raise TypeError("LieBracketVF second argument must be an Expr")
        display = (
            name
            if name is not None
            else f"[{X._repr_inner()},{Y._repr_inner()}]_VF"
        )
        super().__init__(display, degree=0)
        self._X = X
        self._Y = Y

    @property
    def X(self) -> Expr:
        return self._X

    @property
    def Y(self) -> Expr:
        return self._Y

    def _key(self) -> Any:
        return (self._name, self._degree, self._X, self._Y)


def lie_bracket_vf(
    X: Expr, Y: Expr, *, name: Optional[str] = None
) -> LieBracketVF:
    """Build the vector-field Lie bracket atom ``[X, Y]_VF``."""
    return LieBracketVF(X, Y, name=name)
