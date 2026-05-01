"""Tests for `jacopy.frame_calc.einstein` (Faz 18 Stage F)."""

import pytest
import sympy as sp

from jacopy.frame_calc import (
    ComponentMetric,
    CoordinateFrame,
    EinsteinTensor,
    einstein_from_ricci,
    einstein_tensor,
    levi_civita,
    ricci,
)


# --------------------------------------------------------------------- #
# Fixtures                                                              #
# --------------------------------------------------------------------- #


@pytest.fixture
def schwarzschild() -> tuple[CoordinateFrame, ComponentMetric]:
    t, r, theta, phi = sp.symbols("t r theta phi")
    M = sp.Symbol("M", positive=True)
    F = CoordinateFrame([t, r, theta, phi])
    g = ComponentMetric(F, sp.Matrix([
        [-(1 - 2*M/r), 0, 0, 0],
        [0, 1/(1 - 2*M/r), 0, 0],
        [0, 0, r**2, 0],
        [0, 0, 0, r**2 * sp.sin(theta)**2],
    ]))
    return F, g


@pytest.fixture
def minkowski_cartesian() -> tuple[CoordinateFrame, ComponentMetric]:
    coords = list(sp.symbols("t x y z"))
    F = CoordinateFrame(coords)
    g = ComponentMetric(F, sp.diag(-1, 1, 1, 1))
    return F, g


@pytest.fixture
def two_sphere() -> tuple[CoordinateFrame, ComponentMetric]:
    theta, phi = sp.symbols("theta phi", positive=True)
    R0 = sp.Symbol("R0", positive=True)
    F = CoordinateFrame([theta, phi])
    g = ComponentMetric(F, sp.Matrix([
        [R0**2, 0],
        [0, R0**2 * sp.sin(theta)**2],
    ]))
    return F, g


# --------------------------------------------------------------------- #
# Type checks                                                           #
# --------------------------------------------------------------------- #


class TestEinsteinTypeChecks:
    def test_non_connection_rejected(self) -> None:
        with pytest.raises(TypeError, match="ComponentConnection"):
            einstein_tensor("not", "args")  # type: ignore[arg-type]

    def test_non_metric_rejected(self, schwarzschild) -> None:
        F, g = schwarzschild
        LC = levi_civita(g)
        with pytest.raises(TypeError, match="ComponentMetric"):
            einstein_tensor(LC, "not a metric")  # type: ignore[arg-type]


# --------------------------------------------------------------------- #
# THE punchline test: Schwarzschild vacuum                              #
# --------------------------------------------------------------------- #


class TestSchwarzschildVacuum:
    """Stage F integration test: Einstein tensor identically zero on
    Schwarzschild, symbolic verification of the vacuum solution."""

    def test_einstein_tensor_zero(self, schwarzschild) -> None:
        F, g = schwarzschild
        G = einstein_tensor(levi_civita(g), g)
        assert isinstance(G, EinsteinTensor)
        assert G.signature == (0, 2)
        assert G.shape == (4, 4)
        assert G.is_zero()

    def test_is_vacuum_alias(self, schwarzschild) -> None:
        F, g = schwarzschild
        G = einstein_tensor(levi_civita(g), g)
        assert G.is_vacuum()

    def test_each_entry_individually_zero(self, schwarzschild) -> None:
        """Defensive check: every component is sympy-zero after simplify."""
        F, g = schwarzschild
        G = einstein_tensor(levi_civita(g), g)
        for a in range(4):
            for b in range(4):
                assert (
                    sp.simplify(sp.trigsimp(G[a, b])) == 0
                ), f"G[{a},{b}] = {G[a, b]} ≠ 0"

    def test_einstein_from_ricci_matches_einstein_tensor(
        self, schwarzschild
    ) -> None:
        F, g = schwarzschild
        LC = levi_civita(g)
        G_a = einstein_tensor(LC, g)
        Ric = ricci(LC)
        G_b = einstein_from_ricci(Ric, g)
        assert G_a.equals(G_b)


# --------------------------------------------------------------------- #
# Minkowski → vacuum                                                    #
# --------------------------------------------------------------------- #


class TestMinkowskiVacuum:
    def test_cartesian_vacuum(self, minkowski_cartesian) -> None:
        F, g = minkowski_cartesian
        G = einstein_tensor(levi_civita(g), g)
        assert G.is_zero()

    def test_spherical_vacuum(self) -> None:
        t, r, theta, phi = sp.symbols("t r theta phi")
        F = CoordinateFrame([t, r, theta, phi])
        g = ComponentMetric(F, sp.Matrix([
            [-1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, r**2, 0],
            [0, 0, 0, r**2 * sp.sin(theta)**2],
        ]))
        G = einstein_tensor(levi_civita(g), g)
        assert G.is_zero()


# --------------------------------------------------------------------- #
# 2-sphere → NOT vacuum (constant positive curvature)                   #
# --------------------------------------------------------------------- #


class TestTwoSphereEinstein:
    """The 2-sphere has constant curvature; the Einstein tensor in 2D
    has the property G_{ab} = Ric_{ab} - ½ R g_{ab} = 0 actually,
    because in 2D the Einstein tensor is identically zero (Lovelock
    theorem). This is a nice cross-check."""

    def test_2d_einstein_tensor_is_identically_zero(
        self, two_sphere
    ) -> None:
        """In 2D, G_{ab} ≡ 0 for every metric (Lovelock / Gauss-Bonnet)."""
        F, g = two_sphere
        G = einstein_tensor(levi_civita(g), g)
        assert G.is_zero()


# --------------------------------------------------------------------- #
# Symmetry                                                              #
# --------------------------------------------------------------------- #


class TestEinsteinSymmetry:
    def test_schwarzschild_symmetric(self, schwarzschild) -> None:
        F, g = schwarzschild
        G = einstein_tensor(levi_civita(g), g)
        for a in range(4):
            for b in range(4):
                assert sp.simplify(G[a, b] - G[b, a]) == 0


# --------------------------------------------------------------------- #
# Matrix view                                                           #
# --------------------------------------------------------------------- #


class TestEinsteinMatrixView:
    def test_matrix_returns_sympy_matrix(self, schwarzschild) -> None:
        F, g = schwarzschild
        G = einstein_tensor(levi_civita(g), g)
        M = G.matrix()
        assert isinstance(M, sp.Matrix)
        assert M.shape == (4, 4)
