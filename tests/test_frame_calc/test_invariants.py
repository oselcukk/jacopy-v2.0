"""Curvature invariant tests (Faz 19 Chunk A).

Coordinate-independent scalars: Kretschmann ``K = R_{abcd} R^{abcd}``,
Ricci² ``R_{ab}R^{ab}``, and 3D Cotton tensor.
"""

import pytest
import sympy as sp

from jacopy.frame_calc import (
    ComponentMetric,
    ComponentTensor,
    CoordinateFrame,
    cotton,
    curvature,
    kretschmann,
    levi_civita,
    ricci,
    ricci_squared,
)
from jacopy.frame_calc.library import (
    de_sitter,
    minkowski,
    schwarzschild,
)


# --------------------------------------------------------------------- #
# Kretschmann                                                            #
# --------------------------------------------------------------------- #


class TestKretschmann:
    def test_minkowski_zero(self) -> None:
        """Flat space: K = 0."""
        F, g = minkowski()
        R = curvature(levi_civita(g))
        K = kretschmann(R, g)
        assert sp.simplify(K) == 0

    def test_schwarzschild_textbook(self) -> None:
        """Schwarzschild: K = 48 M² / r⁶, the textbook closed form.

        Note: K is FINITE at the horizon r=2M (no real singularity),
        but DIVERGES as r → 0 (genuine curvature singularity).
        """
        F, g = schwarzschild()
        R = curvature(levi_civita(g))
        K = kretschmann(R, g)
        M = sp.Symbol("M", positive=True)
        r = sp.Symbol("r", positive=True)
        expected = 48 * M ** 2 / r ** 6
        assert sp.simplify(K - expected) == 0

    def test_de_sitter_constant(self) -> None:
        """de Sitter: K = 8 Λ² / 3, constant, max-symmetric value."""
        F, g = de_sitter()
        R = curvature(levi_civita(g))
        K = kretschmann(R, g)
        Lam = sp.Symbol("Lambda", positive=True)
        expected = sp.Rational(8, 3) * Lam ** 2
        assert sp.simplify(K - expected) == 0

    def test_signature_validation(self) -> None:
        """Kretschmann rejects non-(1,3) tensors."""
        F, g = minkowski()
        Ric = ricci(levi_civita(g))  # (0, 2)
        with pytest.raises(ValueError, match="signature"):
            kretschmann(Ric, g)


# --------------------------------------------------------------------- #
# Ricci-squared                                                          #
# --------------------------------------------------------------------- #


class TestRicciSquared:
    def test_minkowski_zero(self) -> None:
        F, g = minkowski()
        Ric = ricci(levi_civita(g))
        assert sp.simplify(ricci_squared(Ric, g)) == 0

    def test_schwarzschild_zero(self) -> None:
        """Vacuum spacetime: Ric = 0 → Ric² = 0."""
        F, g = schwarzschild()
        Ric = ricci(levi_civita(g))
        assert sp.simplify(ricci_squared(Ric, g)) == 0

    def test_de_sitter_nonzero(self) -> None:
        """dS has Ric ∝ g, so Ric² ≠ 0."""
        F, g = de_sitter()
        Ric = ricci(levi_civita(g))
        RS = ricci_squared(Ric, g)
        # dS in 4D: Ric = -Λg → Ric² = 4 Λ² (independent of r/θ/φ)
        Lam = sp.Symbol("Lambda", positive=True)
        expected = 4 * Lam ** 2
        assert sp.simplify(RS - expected) == 0

    def test_signature_validation(self) -> None:
        F, g = minkowski()
        R = curvature(levi_civita(g))  # (1, 3)
        with pytest.raises(ValueError, match="signature"):
            ricci_squared(R, g)


# --------------------------------------------------------------------- #
# Cotton (3D)                                                            #
# --------------------------------------------------------------------- #


class TestCotton:
    def test_flat_3d_zero(self) -> None:
        """Flat R³: Cotton = 0 (trivially conformally flat)."""
        x, y, z = sp.symbols("x y z", real=True)
        F = CoordinateFrame([x, y, z])
        g = ComponentMetric(F, sp.eye(3))
        C = cotton(levi_civita(g), g)
        assert C.is_zero()

    def test_S2_times_R_zero(self) -> None:
        """S² × R is conformally flat: Cotton = 0."""
        theta, phi, z = sp.symbols("theta phi z")
        R0 = sp.Symbol("R0", positive=True)
        F = CoordinateFrame([theta, phi, z])
        g = ComponentMetric(
            F, sp.diag(R0 ** 2, R0 ** 2 * sp.sin(theta) ** 2, 1)
        )
        C = cotton(levi_civita(g), g)
        assert C.is_zero()

    def test_3d_anisotropic_nonzero(self) -> None:
        """3D anisotropic Bianchi-I-like: Cotton ≠ 0."""
        t, x, y = sp.symbols("t x y", real=True)
        a_t = sp.Function("a")(t)
        b_t = sp.Function("b")(t)
        F = CoordinateFrame([t, x, y])
        g = ComponentMetric(F, sp.diag(-1, a_t ** 2, b_t ** 2))
        C = cotton(levi_civita(g), g)
        assert not C.is_zero()

    def test_dim_validation(self) -> None:
        """Cotton requires dim 3."""
        F, g = minkowski()  # 4D
        with pytest.raises(ValueError, match="dim 3"):
            cotton(levi_civita(g), g)

    def test_signature_is_03(self) -> None:
        """Cotton tensor has signature (0, 3), antisymmetric in (a, b)."""
        x, y, z = sp.symbols("x y z", real=True)
        F = CoordinateFrame([x, y, z])
        g = ComponentMetric(F, sp.eye(3))
        C = cotton(levi_civita(g), g)
        assert C.signature == (0, 3)
