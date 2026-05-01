"""Tests for `jacopy.frame_calc.curvature` (Faz 18 Stage E)."""

import pytest
import sympy as sp

from jacopy.frame_calc import (
    ComponentConnection,
    ComponentMetric,
    CoordinateFrame,
    CurvatureStep,
    CurvatureTensor,
    curvature,
    levi_civita,
)


# --------------------------------------------------------------------- #
# Fixtures                                                              #
# --------------------------------------------------------------------- #


@pytest.fixture
def minkowski_cartesian() -> ComponentConnection:
    coords = list(sp.symbols("t x y z"))
    F = CoordinateFrame(coords)
    g = ComponentMetric(F, sp.diag(-1, 1, 1, 1))
    return levi_civita(g)


@pytest.fixture
def minkowski_spherical() -> ComponentConnection:
    t, r, theta, phi = sp.symbols("t r theta phi")
    F = CoordinateFrame([t, r, theta, phi])
    g = ComponentMetric(F, sp.Matrix([
        [-1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, r**2, 0],
        [0, 0, 0, r**2 * sp.sin(theta)**2],
    ]))
    return levi_civita(g)


@pytest.fixture
def schwarzschild_lc() -> ComponentConnection:
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


@pytest.fixture
def polar_lc() -> ComponentConnection:
    r, theta = sp.symbols("r theta", positive=True)
    F = CoordinateFrame([r, theta])
    g = ComponentMetric(F, sp.Matrix([[1, 0], [0, r**2]]))
    return levi_civita(g)


# --------------------------------------------------------------------- #
# Type checks                                                           #
# --------------------------------------------------------------------- #


class TestCurvatureTypeChecks:
    def test_non_connection_rejected(self) -> None:
        with pytest.raises(TypeError, match="ComponentConnection"):
            curvature("not a connection")  # type: ignore[arg-type]


# --------------------------------------------------------------------- #
# Flat space → curvature = 0                                            #
# --------------------------------------------------------------------- #


class TestFlatSpaceCurvature:
    """Riemann tensor must be identically zero on flat space, even
    when the coordinates are curvilinear (Christoffels non-zero)."""

    def test_minkowski_cartesian_flat(self, minkowski_cartesian) -> None:
        R = curvature(minkowski_cartesian)
        assert isinstance(R, CurvatureTensor)
        assert R.signature == (1, 3)
        assert R.shape == (4, 4, 4, 4)
        assert R.is_zero()

    def test_minkowski_spherical_flat(self, minkowski_spherical) -> None:
        """Spherical Minkowski has 9 non-zero Christoffels but is
        still flat, Riemann curvature must vanish identically."""
        R = curvature(minkowski_spherical)
        assert R.is_zero()

    def test_polar_2d_flat(self, polar_lc) -> None:
        """2D Euclidean in polar: 3 non-zero Christoffels, flat space."""
        R = curvature(polar_lc)
        assert R.is_zero()

    def test_minkowski_cylindrical_flat(self) -> None:
        t, z = sp.symbols("t z")
        rho = sp.Symbol("rho", positive=True)
        phi = sp.Symbol("phi", positive=True)
        F = CoordinateFrame([t, rho, phi, z])
        g = ComponentMetric(F, sp.diag(-1, 1, rho**2, 1))
        R = curvature(levi_civita(g))
        assert R.is_zero()

    def test_rindler_2d_flat(self) -> None:
        """Rindler frame is flat space (only the coords are accelerated)."""
        tau = sp.Symbol("tau")
        rho = sp.Symbol("rho", positive=True)
        F = CoordinateFrame([tau, rho])
        g = ComponentMetric(F, sp.Matrix([[-(rho**2), 0], [0, 1]]))
        R = curvature(levi_civita(g))
        assert R.is_zero()


# --------------------------------------------------------------------- #
# Schwarzschild → curvature non-zero                                    #
# --------------------------------------------------------------------- #


class TestSchwarzschildCurvature:
    def test_curvature_not_zero(self, schwarzschild_lc) -> None:
        """Schwarzschild has non-zero curvature (genuine spacetime)."""
        R = curvature(schwarzschild_lc)
        assert not R.is_zero()

    def test_riemann_depends_only_on_M_and_r(self, schwarzschild_lc) -> None:
        """Spherical symmetry: every Riemann component is independent of
        the angular coordinates θ and φ when expressed in terms of `M` and
        `r`. Convention-independent shape check that the curvature was
        computed correctly without committing to a sign convention.
        """
        t, r, theta, phi = sp.symbols("t r theta phi")
        R = curvature(schwarzschild_lc)
        # R^t_{rtr} is the most-cited Riemann component; verify it depends
        # on M and r but not on θ/φ.
        val = sp.simplify(R[0, 1, 0, 1])
        assert val != 0
        # Differentiating w.r.t. θ or φ should vanish (radial-only).
        assert sp.simplify(sp.diff(val, theta)) == 0
        assert sp.simplify(sp.diff(val, phi)) == 0
        # And w.r.t. M it should NOT vanish (M is the source).
        M = sp.Symbol("M", positive=True)
        assert sp.simplify(sp.diff(val, M)) != 0


# --------------------------------------------------------------------- #
# Antisymmetry in lower pair (b, c)                                     #
# --------------------------------------------------------------------- #


class TestRiemannAntisymmetry:
    def test_minkowski_spherical_antisymmetric(self, minkowski_spherical) -> None:
        """`R^a_{bcd} = -R^a_{cbd}` for any connection."""
        R = curvature(minkowski_spherical)
        n = 4
        for a in range(n):
            for b in range(n):
                for c in range(n):
                    for d in range(n):
                        diff = sp.simplify(R[a, b, c, d] + R[a, c, b, d])
                        assert diff == 0

    def test_schwarzschild_antisymmetric(self, schwarzschild_lc) -> None:
        R = curvature(schwarzschild_lc)
        n = 4
        for a in range(n):
            for b in range(n):
                for c in range(n):
                    for d in range(n):
                        diff = sp.simplify(R[a, b, c, d] + R[a, c, b, d])
                        assert diff == 0

    def test_polar_antisymmetric(self, polar_lc) -> None:
        R = curvature(polar_lc)
        n = 2
        for a in range(n):
            for b in range(n):
                for c in range(n):
                    for d in range(n):
                        diff = sp.simplify(R[a, b, c, d] + R[a, c, b, d])
                        assert diff == 0

    def test_b_equals_c_zero(self, schwarzschild_lc) -> None:
        """`R^a_{bbd} = 0` always (antisymmetry in `bc` with `b = c`)."""
        R = curvature(schwarzschild_lc)
        for a in range(4):
            for b in range(4):
                for d in range(4):
                    assert sp.simplify(R[a, b, b, d]) == 0


# --------------------------------------------------------------------- #
# Derivation traces                                                     #
# --------------------------------------------------------------------- #


class TestCurvatureDerivationTraces:
    def test_steps_recorded(self, schwarzschild_lc) -> None:
        R = curvature(schwarzschild_lc)
        steps = R.derivation_steps(0, 1, 0, 1)
        assert len(steps) >= 4
        assert isinstance(steps[0], CurvatureStep)
        assert "Riemann" in steps[0].rule

    def test_format_derivation_text(self, polar_lc) -> None:
        R = curvature(polar_lc)
        # 2D so canonical (b, c) with b < c is just (0, 1)
        text = R.format_derivation(0, 0, 1, 0)
        assert isinstance(text, str)
        assert "Riemann" in text

    def test_antisymmetric_steps_share_canonical(self, polar_lc) -> None:
        """Steps for `(a, b, c, d)` with `b > c` map to the canonical
        `(a, min(b,c), max(b,c), d)` entry."""
        R = curvature(polar_lc)
        steps_canonical = R.derivation_steps(0, 0, 1, 0)
        steps_swapped = R.derivation_steps(0, 1, 0, 0)
        assert steps_canonical == steps_swapped

    def test_index_out_of_range(self, polar_lc) -> None:
        R = curvature(polar_lc)
        with pytest.raises(IndexError):
            R.derivation_steps(5, 0, 0, 0)
