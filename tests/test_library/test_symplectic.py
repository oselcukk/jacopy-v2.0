"""Tests for ``gradalg.library.symplectic.SymplecticManifold``."""

from __future__ import annotations

import pytest

from gradalg.calculus.hamiltonian_vf import HamiltonianVectorField
from gradalg.calculus.musical import Flat, MusicalCompatibility, Sharp
from gradalg.core.expr import Integer, Symbol
from gradalg.core.properties import Graded
from gradalg.core.registry import PropertyRegistry
from gradalg.library.symplectic import SymplecticManifold
from gradalg.proof.chain import ProofChain


# --------------------------------------------------------------------- #
# Fixtures                                                              #
# --------------------------------------------------------------------- #


@pytest.fixture
def omega():
    return Symbol("ω")


@pytest.fixture
def pi():
    return Symbol("π")


@pytest.fixture
def registry(omega, pi):
    r = PropertyRegistry()
    r.declare(omega, Graded(degree=2))
    r.declare(pi, Graded(degree=1))
    return r


# --------------------------------------------------------------------- #
# Construction                                                          #
# --------------------------------------------------------------------- #


class TestConstruction:
    def test_omega_only_builds_flat_no_sharp(self, omega):
        M = SymplecticManifold(omega)
        assert M.omega is omega
        assert M.bivector is None
        assert isinstance(M.flat, Flat)
        assert M.flat.form is omega
        assert M.sharp is None
        assert M.compatibility is None

    def test_with_bivector_builds_full_musical_package(self, omega, pi):
        M = SymplecticManifold(omega, bivector=pi)
        assert M.bivector is pi
        assert isinstance(M.flat, Flat)
        assert isinstance(M.sharp, Sharp)
        assert M.sharp.bivector is pi
        assert isinstance(M.compatibility, MusicalCompatibility)

    def test_compatibility_reuses_manifold_flat_sharp(self, omega, pi):
        """The compatibility axiom must carry the *same* Flat/Sharp
        instances the manifold exposes, not freshly built ones — else
        downstream proof engines see two `Flat(ω)` operators that compare
        structurally equal but fail identity checks callers rely on."""
        M = SymplecticManifold(omega, bivector=pi)
        assert M.compatibility.flat is M.flat
        assert M.compatibility.sharp is M.sharp
        assert M.compatibility.omega is omega
        assert M.compatibility.pi is pi

    def test_default_name(self, omega):
        M = SymplecticManifold(omega)
        assert M.name == "Symp(ω)"

    def test_custom_name(self, omega):
        M = SymplecticManifold(omega, name="StdSymp")
        assert M.name == "StdSymp"

    def test_rejects_non_expr_omega(self):
        with pytest.raises(TypeError, match="omega"):
            SymplecticManifold("not an expr")  # type: ignore[arg-type]

    def test_rejects_non_expr_bivector(self, omega):
        with pytest.raises(TypeError, match="bivector"):
            SymplecticManifold(omega, bivector="π")  # type: ignore[arg-type]


# --------------------------------------------------------------------- #
# hamiltonian_vf                                                        #
# --------------------------------------------------------------------- #


class TestHamiltonianVf:
    def test_with_bivector_carries_both_data(self, omega, pi):
        M = SymplecticManifold(omega, bivector=pi)
        f = Symbol("f")
        Xf = M.hamiltonian_vf(f)
        assert isinstance(Xf, HamiltonianVectorField)
        assert Xf.function is f
        assert Xf.bivector is pi
        assert Xf.symplectic_form is omega

    def test_without_bivector_carries_only_omega(self, omega):
        M = SymplecticManifold(omega)
        f = Symbol("f")
        Xf = M.hamiltonian_vf(f)
        assert Xf.symplectic_form is omega
        assert Xf.bivector is None

    def test_rejects_non_expr_function(self, omega):
        M = SymplecticManifold(omega)
        with pytest.raises(TypeError, match="Expr"):
            M.hamiltonian_vf("f")  # type: ignore[arg-type]


# --------------------------------------------------------------------- #
# prove_hamiltonian_equivalence                                         #
# --------------------------------------------------------------------- #


class TestProveEquivalence:
    def test_requires_bivector(self, omega):
        M = SymplecticManifold(omega)
        f = Symbol("f")
        with pytest.raises(ValueError, match="bivector"):
            M.prove_hamiltonian_equivalence(f)

    def test_closes_to_proof_chain(self, omega, pi, registry):
        """The manifold's compatibility axiom discharges the symplectic
        obstruction to zero — the same close used in
        :meth:`HamiltonianVectorField.prove_equivalence` directly, just
        driven off the manifold's canonical compatibility object."""
        M = SymplecticManifold(omega, bivector=pi)
        f = Symbol("f")
        registry.declare(f, Graded(degree=-1))
        chain = M.prove_hamiltonian_equivalence(f, registry=registry)
        assert isinstance(chain, ProofChain)
        assert len(chain) > 0
        assert chain.final == Integer(0)


# --------------------------------------------------------------------- #
# Repr                                                                  #
# --------------------------------------------------------------------- #


class TestRepr:
    def test_repr_omega_only(self, omega):
        M = SymplecticManifold(omega)
        assert repr(M) == "SymplecticManifold(ω)"

    def test_repr_with_bivector(self, omega, pi):
        M = SymplecticManifold(omega, bivector=pi)
        assert "ω" in repr(M) and "π" in repr(M)
