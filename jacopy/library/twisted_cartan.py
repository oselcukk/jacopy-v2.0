"""
Twisted Cartan calculus bundle.

A :class:`TwistedCartanBundle` packages ``(d_H, L_{H,·}, ι_{·}, [·,·])``
for an H-twisted exterior derivative

    d_H = d + H∧,

where ``H`` is a closed 3-form. Treating ``d_H`` as a fresh degree-+1
derivation lets the Cartan bundle hold the same five relations
(``d_H² = 0``, Cartan magic, ``[d_H, L_{H,X}] = 0``, ``[L, L]``,
``[L, ι]``) by the same agreement-on-generators closure used for the
untwisted TM bundle. The package does *not* decompose ``d_H`` as
``d + H∧`` inside the expansion engine, closure of ``d_H² = 0``
relies on the classical identity ``d_H² = dH ∧ (−)``, which vanishes
precisely when ``H`` is closed. Constructing a
:class:`TwistedCartanBundle` is the act of asserting that closure.

The pattern mirrors :class:`~jacopy.library.lie_algebroid.LieAlgebroid`
, a fresh :class:`~jacopy.calculus.exterior_d.ExteriorDerivative` for
``d_H``, a Lie-derivative factory that threads ``d_H`` into each
``L_{H,X}``'s bundle slots, and the standard ``ι_X`` factory (the
interior product carries no dependency on the twist). The expansion
engine's ``d² = 0`` rule is pinned to ``d_H`` via the
:meth:`~jacopy.calculus.cartan.CartanCalculus.verify` threading that
landed in the deferral-2 pass, so the caller does not need to
register anything manually.
"""

from __future__ import annotations

from typing import Optional

from jacopy.brackets.base import GradedBracket
from jacopy.brackets.lie import LieBracket, lie as _lie_TM
from jacopy.calculus.cartan import CartanCalculus
from jacopy.calculus.exterior_d import ExteriorDerivative
from jacopy.calculus.interior import InteriorProduct, interior
from jacopy.calculus.lie_derivative import LieDerivative, lie_derivative
from jacopy.core.expr import Expr


class TwistedCartanBundle:
    """``(d_H, L_{H,·}, ι_{·}, [·,·])``, Cartan bundle for ``d_H = d + H∧``.

    Parameters
    ----------
    twist_form
        The twist 3-form ``H``. Used only as a display handle, the
        bundle does *not* expand ``d_H`` as ``d + H∧`` anywhere; it
        treats ``d_H`` as a formal degree-+1 derivation whose square
        vanishes. Caller asserts ``dH = 0`` by instantiating this
        bundle.
    vector_bracket
        Vector-field bracket used by ``L_{[X, Y]}`` / ``ι_{[X, Y]}``.
        Defaults to :data:`jacopy.brackets.lie.lie`.
    name
        Optional display name; defaults to ``f"d_{twist_form_repr}"``.

    Notes
    -----
    The Leibniz decomposition ``d_H(α ∧ β)`` is *not* equal to the
    formal graded-Leibniz application of ``d_H``: the exact
    identification ``d_H = d + H∧`` forces a twisted correction term
    that the untwisted Leibniz machinery doesn't produce. This bundle
    works at one level of abstraction above that, ``d_H`` is a
    symbolic degree-+1 derivation with ``d_H² = 0``, and the five
    Cartan relations are verified against that abstract data. Use
    :meth:`verify` when you want to close a relation on a concrete
    exterior algebra; users who need the full ``d + H∧`` expansion can
    layer a rewrite on top with the twist form in hand.
    """

    __slots__ = ("_twist_form", "_vector_bracket", "_d_H", "_cartan", "_name")

    def __init__(
        self,
        twist_form: Expr,
        *,
        vector_bracket: Optional[GradedBracket] = None,
        name: Optional[str] = None,
    ) -> None:
        if not isinstance(twist_form, Expr):
            raise TypeError(
                "TwistedCartanBundle twist_form must be an Expr"
            )
        if vector_bracket is None:
            vector_bracket = _lie_TM
        elif not isinstance(vector_bracket, GradedBracket):
            raise TypeError(
                "TwistedCartanBundle vector_bracket must be a GradedBracket"
            )
        self._twist_form = twist_form
        self._vector_bracket = vector_bracket
        twist_tag = twist_form._repr_inner()
        self._d_H = ExteriorDerivative(name=f"d_{twist_tag}")
        self._cartan = CartanCalculus(
            d=self._d_H,
            lie_derivative=self._make_lie_derivative_factory(),
            interior=self._make_interior_factory(),
            vector_bracket=vector_bracket,
        )
        self._name = name if name is not None else f"TwistedCartan({twist_tag})"

    # ---- factories -------------------------------------------------- #

    def _make_lie_derivative_factory(self):
        twist_tag = self._twist_form._repr_inner()
        iota_factory = self._make_interior_factory()
        d_H = self._d_H

        def factory(X: Expr) -> LieDerivative:
            if not isinstance(X, Expr):
                raise TypeError("twisted L factory requires an Expr section")
            return lie_derivative(
                X,
                name=f"L_{twist_tag},{X._repr_inner()}",
                d=d_H,
                iota_factory=iota_factory,
            )

        return factory

    def _make_interior_factory(self):
        def factory(X: Expr) -> InteriorProduct:
            if not isinstance(X, Expr):
                raise TypeError("twisted ι factory requires an Expr section")
            return interior(X)

        return factory

    # ---- accessors -------------------------------------------------- #

    @property
    def twist_form(self) -> Expr:
        return self._twist_form

    @property
    def vector_bracket(self) -> GradedBracket:
        return self._vector_bracket

    @property
    def d(self) -> ExteriorDerivative:
        """The twisted exterior derivative ``d_H``."""
        return self._d_H

    @property
    def cartan(self) -> CartanCalculus:
        """Twisted Cartan bundle, ``(d_H, L_{H,·}, ι_·, [·,·])``."""
        return self._cartan

    @property
    def name(self) -> str:
        return self._name

    # ---- identity --------------------------------------------------- #

    def __repr__(self) -> str:
        return (
            f"TwistedCartanBundle(H={self._twist_form._repr_inner()}, "
            f"d_H={self._d_H._repr_inner()})"
        )


def twisted_cartan_bundle(
    twist_form: Expr,
    *,
    vector_bracket: Optional[GradedBracket] = None,
    name: Optional[str] = None,
) -> TwistedCartanBundle:
    """Construct a :class:`TwistedCartanBundle`. Thin wrapper for symmetry
    with :func:`jacopy.library.lie_algebroid.lie_algebroid`."""
    return TwistedCartanBundle(
        twist_form, vector_bracket=vector_bracket, name=name
    )
