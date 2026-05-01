"""
Tilde calculus operator atoms.

Three :class:`~jacopy.algebra.derivation.Derivation` subclasses
parameterise the Cartan operators on the Koszul (multivector) side of
the calculus on a Poisson manifold ``(M, π)``:

* :class:`TildeInteriorProduct` ``ι̃_ω``, degree ``-1`` derivation
  indexed by a form ``ω``. The defining identity ``ι̃_ω V := ι_V ω``
  swaps the standard interior-product roles: the multivector becomes
  the "vector to contract with" and the form becomes the parameter.
* :class:`TildeExteriorDerivative` ``d̃``, degree ``+1`` derivation
  indexed by a Poisson bivector ``π``. The defining identity is the
  Lichnerowicz formula ``d̃ V := [π, V]_SN``.
* :class:`TildeLieDerivative` ``L̃_ω``, degree ``0`` derivation indexed
  by a form ``ω`` and a bivector ``π``. The defining identity is the
  tilde Cartan magic formula ``L̃_ω := d̃ ∘ ι̃_ω + ι̃_ω ∘ d̃``.

The classes carry the indexing form/bivector as an attribute on the
atom and fold it into ``_key()`` so two operators are equal iff they
share name, degree, and parameters. None of these defining identities
are applied at construction; they live as engine rewrite rules in
:mod:`jacopy.calculus.tilde.axioms` (Faz 14.B), so a proof can step
through the rewrite explicitly rather than seeing the formula only as
collapsed output.

Like :class:`~jacopy.calculus.interior.InteriorProduct` and
:class:`~jacopy.calculus.lie_derivative.LieDerivative` there is no
module-level singleton, tilde operators are families indexed by their
parameters. Use :func:`tilde_interior`, :func:`tilde_d`, and
:func:`tilde_lie` to construct them.
"""

from __future__ import annotations

from typing import Any, Optional

from jacopy.algebra.derivation import Derivation
from jacopy.core.expr import Expr


# --------------------------------------------------------------------- #
# ι̃_ω, tilde interior product                                         #
# --------------------------------------------------------------------- #


class TildeInteriorProduct(Derivation):
    """``ι̃_ω``, degree ``-1`` form-indexed contraction on multivectors.

    Carries the indexing form ``ω`` on :attr:`form`. The defining
    identity ``ι̃_ω V := ι_V ω`` is realised as an engine rewrite in
    :mod:`jacopy.calculus.tilde.axioms`; this class is the inert atom
    that the rewrite recognises.

    Equality is structural over ``(name, degree, form)``, two tilde
    interior products with the same form and the default name compare
    equal. Custom ``name`` overrides participate in the equality key.
    """

    __slots__ = ("_form",)

    def __init__(self, omega: Expr, *, name: Optional[str] = None) -> None:
        if not isinstance(omega, Expr):
            raise TypeError("TildeInteriorProduct requires an Expr form")
        display = name if name is not None else f"ι̃_{omega._repr_inner()}"
        super().__init__(display, degree=-1)
        self._form = omega

    @property
    def form(self) -> Expr:
        return self._form

    def _key(self) -> Any:
        return (self._name, self._degree, self._form)


def tilde_interior(omega: Expr, *, name: Optional[str] = None) -> TildeInteriorProduct:
    """Build ``ι̃_ω`` for the form ``omega``."""
    return TildeInteriorProduct(omega, name=name)


# --------------------------------------------------------------------- #
# d̃, tilde exterior derivative (Lichnerowicz)                         #
# --------------------------------------------------------------------- #


class TildeExteriorDerivative(Derivation):
    """``d̃``, degree ``+1`` Lichnerowicz differential on multivectors.

    Indexed by a Poisson bivector ``π`` (carried on :attr:`bivector`).
    The defining identity ``d̃ V := [π, V]_SN`` is realised as an engine
    rewrite in :mod:`jacopy.calculus.tilde.axioms`. Because each
    Poisson manifold gives its own ``d̃``, a separate instance is
    constructed per ``π``; two ``TildeExteriorDerivative`` instances
    with the same bivector and the default name compare equal.
    """

    __slots__ = ("_bivector",)

    def __init__(self, pi: Expr, *, name: Optional[str] = None) -> None:
        if not isinstance(pi, Expr):
            raise TypeError("TildeExteriorDerivative requires an Expr bivector")
        display = name if name is not None else f"d̃_{pi._repr_inner()}"
        super().__init__(display, degree=1)
        self._bivector = pi

    @property
    def bivector(self) -> Expr:
        return self._bivector

    def _key(self) -> Any:
        return (self._name, self._degree, self._bivector)


def tilde_d(pi: Expr, *, name: Optional[str] = None) -> TildeExteriorDerivative:
    """Build ``d̃`` for the Poisson bivector ``pi``."""
    return TildeExteriorDerivative(pi, name=name)


# --------------------------------------------------------------------- #
# L̃_ω, tilde Lie derivative                                           #
# --------------------------------------------------------------------- #


class TildeLieDerivative(Derivation):
    """``L̃_ω``, degree ``0`` Lie-style derivation on multivectors.

    Indexed by both a form ``ω`` (the "direction") and a Poisson
    bivector ``π`` (the ambient structure). The defining identity
    ``L̃_ω := d̃ ∘ ι̃_ω + ι̃_ω ∘ d̃`` (tilde Cartan magic) is realised as
    an engine rewrite in :mod:`jacopy.calculus.tilde.axioms`.

    Both parameters participate in equality so a single proof can carry
    independent ``L̃_ω``, ``L̃_η`` operators side-by-side without
    aliasing. Two-form / two-bivector equality is structural.
    """

    __slots__ = ("_form", "_bivector")

    def __init__(
        self,
        omega: Expr,
        pi: Expr,
        *,
        name: Optional[str] = None,
    ) -> None:
        if not isinstance(omega, Expr):
            raise TypeError("TildeLieDerivative requires an Expr form")
        if not isinstance(pi, Expr):
            raise TypeError("TildeLieDerivative requires an Expr bivector")
        display = name if name is not None else f"L̃_{omega._repr_inner()}"
        super().__init__(display, degree=0)
        self._form = omega
        self._bivector = pi

    @property
    def form(self) -> Expr:
        return self._form

    @property
    def bivector(self) -> Expr:
        return self._bivector

    def _key(self) -> Any:
        return (self._name, self._degree, self._form, self._bivector)


def tilde_lie(
    omega: Expr,
    pi: Expr,
    *,
    name: Optional[str] = None,
) -> TildeLieDerivative:
    """Build ``L̃_ω`` for the form ``omega`` and bivector ``pi``."""
    return TildeLieDerivative(omega, pi, name=name)
