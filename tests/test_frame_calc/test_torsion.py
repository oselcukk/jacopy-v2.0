"""Tests for `jacopy.frame_calc.torsion` (Faz 18 Stage E)."""

import pytest
import sympy as sp

from jacopy.frame_calc import (
    ComponentConnection,
    ComponentMetric,
    CoordinateFrame,
    TorsionTensor,
    levi_civita,
    torsion,
)


@pytest.fixture
def polar_lc() -> ComponentConnection:
    """Levi-Civita on polar 2D Euclidean."""
    r, theta = sp.symbols("r theta", positive=True)
    F = CoordinateFrame([r, theta])
    g = ComponentMetric(F, sp.Matrix([[1, 0], [0, r**2]]))
    return levi_civita(g)


@pytest.fixture
def schwarzschild_lc() -> ComponentConnection:
    """Levi-Civita on 4D Schwarzschild."""
    t, r, theta, phi = sp.symbols("t r theta phi")
    M = sp.Symbol("M", positive=True)
    F = CoordinateFrame([t, r, theta, phi])
    g = ComponentMetric(F, sp.Matrix([
        [-(1 - 2*M/r), 0, 0, 0],
        [0, 1/(1 - 2*M/r), 0, 0],
        [0, 0, r**2, 0],
        [0, 0, 0, r**2 * sp.sin(theta)**2],
    ]))
    return levi_civita(g)


class TestTorsionTypeChecks:
    def test_non_connection_rejected(self) -> None:
        with pytest.raises(TypeError, match="ComponentConnection"):
            torsion("not a connection")  # type: ignore[arg-type]


class TestLeviCivitaTorsion:
    """The Levi-Civita connection is torsion-free by construction."""

    def test_polar_torsion_zero(self, polar_lc) -> None:
        T = torsion(polar_lc)
        assert isinstance(T, TorsionTensor)
        assert T.signature == (1, 2)
        assert T.shape == (2, 2, 2)
        assert T.is_zero()

    def test_schwarzschild_torsion_zero(self, schwarzschild_lc) -> None:
        T = torsion(schwarzschild_lc)
        assert T.is_zero()

    def test_minkowski_spherical_torsion_zero(self) -> None:
        t, r, theta, phi = sp.symbols("t r theta phi")
        F = CoordinateFrame([t, r, theta, phi])
        g = ComponentMetric(F, sp.Matrix([
            [-1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, r**2, 0],
            [0, 0, 0, r**2 * sp.sin(theta)**2],
        ]))
        T = torsion(levi_civita(g))
        assert T.is_zero()

    def test_minkowski_cartesian_torsion_zero(self) -> None:
        coords = list(sp.symbols("t x y z"))
        F = CoordinateFrame(coords)
        g = ComponentMetric(F, sp.diag(-1, 1, 1, 1))
        T = torsion(levi_civita(g))
        assert T.is_zero()


class TestNonTrivialTorsion:
    """Hand-built non-symmetric connection should produce non-zero torsion."""

    def test_antisymmetric_part_extracted(self) -> None:
        """Build a connection with `Γ^0_{01} = a, Γ^0_{10} = b` (a ≠ b)
        on a coordinate frame (γ ≡ 0). Then `T^0_{01} = a - b`.
        """
        t, x = sp.symbols("t x")
        F = CoordinateFrame([t, x])
        a, b = sp.symbols("a b")
        # 2x2x2 array
        christoffel = sp.MutableDenseNDimArray.zeros(2, 2, 2)
        christoffel[0, 0, 1] = a
        christoffel[0, 1, 0] = b
        conn = ComponentConnection(F, christoffel)
        T = torsion(conn)
        # T^0_{01} = Γ^0_{01} - Γ^0_{10} - γ^0_{01}
        #         = a - b - 0 = a - b
        assert sp.simplify(T[0, 0, 1] - (a - b)) == 0
        # T^0_{10} = Γ^0_{10} - Γ^0_{01} - γ^0_{10}
        #         = b - a - 0 = -(a - b)
        assert sp.simplify(T[0, 1, 0] - (b - a)) == 0
        # Antisymmetric in (b, c): T^0_{00} should be 0 - 0 - 0 = 0
        assert T[0, 0, 0] == 0
        assert T[0, 1, 1] == 0

    def test_lower_pair_antisymmetry(self) -> None:
        """For any connection, `T^a_{bc} = -T^a_{cb}`."""
        t, x, y = sp.symbols("t x y")
        F = CoordinateFrame([t, x, y])
        # Random-ish entries
        christoffel = sp.MutableDenseNDimArray.zeros(3, 3, 3)
        for a in range(3):
            for b in range(3):
                for c in range(3):
                    christoffel[a, b, c] = (
                        sp.Symbol(f"Γ_{a}_{b}_{c}")
                    )
        conn = ComponentConnection(F, christoffel)
        T = torsion(conn)
        for a in range(3):
            for b in range(3):
                for c in range(3):
                    assert sp.simplify(T[a, b, c] + T[a, c, b]) == 0
