"""
Symplectic manifold bundle.

A :class:`SymplecticManifold` pairs a non-degenerate 2-form ``ω`` with
an optional compatible bivector ``π``. When the bivector is supplied
the wrapper derives the musical operators
:class:`~gradalg.calculus.musical.Flat` (``ω^♭``) and
:class:`~gradalg.calculus.musical.Sharp` (``π^♯``) from them, together
with the :class:`~gradalg.calculus.musical.MusicalCompatibility` axiom
that declares the two maps mutually inverse.

The wrapper is thin — it bundles data that already exist in
:mod:`gradalg.calculus.musical` and :mod:`gradalg.calculus.hamiltonian_vf`
so application code can name a symplectic manifold once and obtain the
Hamiltonian vector field, the symplectic obstruction, and the musical
bridge proof from the same object. Nothing here is mathematically new;
the value is keeping the ``(ω, π, ♭, ♯, compat)`` quintuple consistent
by construction.
"""

from __future__ import annotations

from typing import Optional

from gradalg.calculus.hamiltonian_vf import (
    HamiltonianVectorField,
    hamiltonian_vf as _hamiltonian_vf,
)
from gradalg.calculus.musical import Flat, MusicalCompatibility, Sharp
from gradalg.core.expr import Expr
from gradalg.core.registry import PropertyRegistry
from gradalg.proof.chain import ProofChain


class SymplecticManifold:
    """``(M, ω)`` — a symplectic manifold, optionally carrying ``π = ω^{-1}``.

    Parameters
    ----------
    omega
        The symplectic 2-form ``ω`` on ``M``. Required.
    bivector
        Optional Poisson bivector ``π`` assumed to be the musical
        inverse of ``ω`` (i.e. ``ω^♭ ∘ π^♯ = id``). Supplying it
        unlocks :attr:`sharp`, :attr:`compatibility`, and the
        :meth:`prove_hamiltonian_equivalence` bridge.
    name
        Optional display name; defaults to ``f"Symp({ω})"``.

    Notes
    -----
    * Without ``bivector``, :attr:`flat` is still built — the flat map
      ``ω^♭`` is intrinsic to ``ω`` alone — but :attr:`sharp` and
      :attr:`compatibility` are ``None``.
    * The wrapper does *not* verify ``dω = 0`` or non-degeneracy; those
      belong to a higher-level verification pass, not to the data
      bundle. The invariants are recorded in :attr:`flat` / :attr:`sharp`
      identity so callers don't accidentally mix a ``Flat(ω₁)`` with a
      ``Sharp(π₂)`` belonging to a different manifold.
    """

    __slots__ = ("_omega", "_bivector", "_flat", "_sharp", "_compat", "_name")

    def __init__(
        self,
        omega: Expr,
        *,
        bivector: Optional[Expr] = None,
        name: Optional[str] = None,
    ) -> None:
        if not isinstance(omega, Expr):
            raise TypeError("SymplecticManifold requires an Expr 2-form 'omega'")
        if bivector is not None and not isinstance(bivector, Expr):
            raise TypeError("SymplecticManifold bivector must be an Expr when provided")
        self._omega = omega
        self._bivector = bivector
        self._flat = Flat(omega)
        if bivector is not None:
            self._sharp = Sharp(bivector)
            self._compat = MusicalCompatibility.between(
                omega,
                bivector,
                flat_instance=self._flat,
                sharp_instance=self._sharp,
            )
        else:
            self._sharp = None
            self._compat = None
        self._name = name if name is not None else f"Symp({omega._repr_inner()})"

    # ---- accessors -------------------------------------------------- #

    @property
    def omega(self) -> Expr:
        return self._omega

    @property
    def bivector(self) -> Optional[Expr]:
        return self._bivector

    @property
    def flat(self) -> Flat:
        return self._flat

    @property
    def sharp(self) -> Optional[Sharp]:
        return self._sharp

    @property
    def compatibility(self) -> Optional[MusicalCompatibility]:
        return self._compat

    @property
    def name(self) -> str:
        return self._name

    # ---- Hamiltonian helpers --------------------------------------- #

    def hamiltonian_vf(self, f: Expr) -> HamiltonianVectorField:
        """Return ``X_f`` on this manifold.

        The returned :class:`HamiltonianVectorField` carries whichever
        of ``(bivector, symplectic_form)`` are attached to the manifold —
        both if ``π`` was supplied, ``ω`` alone otherwise. The caller
        then has access to the matching subset of ``X_f`` methods.
        """
        if not isinstance(f, Expr):
            raise TypeError("hamiltonian_vf requires an Expr function")
        return _hamiltonian_vf(
            f,
            bivector=self._bivector,
            symplectic_form=self._omega,
        )

    def prove_hamiltonian_equivalence(
        self,
        f: Expr,
        *,
        registry: Optional[PropertyRegistry] = None,
    ) -> ProofChain:
        """Close ``ι_{X_f} ω + df = 0`` using this manifold's compatibility.

        Requires the manifold to have been built with a ``bivector`` so
        the :class:`MusicalCompatibility` axiom is available. Delegates
        to :meth:`HamiltonianVectorField.prove_equivalence` with the
        manifold's own compatibility, guaranteeing the Hamiltonian and
        the axiom agree on ``(ω, π)``.
        """
        if self._compat is None:
            raise ValueError(
                "prove_hamiltonian_equivalence requires a compatible "
                "bivector; construct SymplecticManifold with bivector=..."
            )
        Xf = self.hamiltonian_vf(f)
        return Xf.prove_equivalence(self._compat, registry=registry)

    # ---- dunder ---------------------------------------------------- #

    def __repr__(self) -> str:
        if self._bivector is None:
            return f"SymplecticManifold({self._omega._repr_inner()})"
        return (
            f"SymplecticManifold({self._omega._repr_inner()}, "
            f"π={self._bivector._repr_inner()})"
        )
