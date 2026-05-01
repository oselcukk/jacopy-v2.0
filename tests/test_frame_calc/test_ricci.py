"""Tests for `jacopy.frame_calc.ricci` (Faz 18 Stage F)."""

import pytest
import sympy as sp

from jacopy.frame_calc import (
    ComponentConnection,
    ComponentMetric,
    CoordinateFrame,
    CurvatureTensor,
    RicciStep,
    RicciTensor,
    curvature,
    levi_civita,
    ricci,
    ricci_from_curvature,
    ricci_scalar,
    ricci_scalar_from_ricci,
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
def minkowski_spherical() -> tuple[CoordinateFrame, ComponentMetric]:
    t, r, theta, phi = sp.symbols("t r theta phi")
    F = CoordinateFrame([t, r, theta, phi])
    g = ComponentMetric(F, sp.Matrix([
        [-1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, r**2, 0],
        [0, 0, 0, r**2 * sp.sin(theta)**2],
    ]))
    return F, g


@pytest.fixture
def two_sphere() -> tuple[CoordinateFrame, ComponentMetric]:
    """Round 2-sphere of radius `R0`: ds² = R₀² dθ² + R₀² sin²θ dφ²."""
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


class TestRicciTypeChecks:
    def test_non_connection_rejected(self) -> None:
        with pytest.raises(TypeError, match="ComponentConnection"):
            ricci("not a connection")  # type: ignore[arg-type]

    def test_ricci_from_curvature_rejects_non_curvature(self) -> None:
        with pytest.raises(TypeError, match="CurvatureTensor"):
            ricci_from_curvature("not a tensor")  # type: ignore[arg-type]

    def test_ricci_scalar_rejects_non_metric(self, schwarzschild) -> None:
        F, g = schwarzschild
        LC = levi_civita(g)
        with pytest.raises(TypeError, match="ComponentMetric"):
            ricci_scalar(LC, "not a metric")  # type: ignore[arg-type]

    def test_ricci_scalar_from_ricci_rejects_frame_mismatch(
        self, schwarzschild
    ) -> None:
        F, g = schwarzschild
        Ric = ricci(levi_civita(g))
        # Build a metric on a different frame
        x = sp.Symbol("x")
        F2 = CoordinateFrame([x])
        g2 = ComponentMetric(F2, sp.Matrix([[1]]))
        with pytest.raises(ValueError, match="share a frame"):
            ricci_scalar_from_ricci(Ric, g2)


# --------------------------------------------------------------------- #
# Schwarzschild, Ric = 0                                               #
# --------------------------------------------------------------------- #


class TestSchwarzschildRicci:
    def test_ricci_is_zero(self, schwarzschild) -> None:
        F, g = schwarzschild
        Ric = ricci(levi_civita(g))
        assert isinstance(Ric, RicciTensor)
        assert Ric.signature == (0, 2)
        assert Ric.shape == (4, 4)
        assert Ric.is_zero()

    def test_ricci_scalar_is_zero(self, schwarzschild) -> None:
        F, g = schwarzschild
        R = ricci_scalar(levi_civita(g), g)
        assert sp.simplify(R) == 0

    def test_ricci_from_curvature_matches_ricci(self, schwarzschild) -> None:
        F, g = schwarzschild
        LC = levi_civita(g)
        Ric_a = ricci(LC)
        Ric_b = ricci_from_curvature(curvature(LC))
        assert Ric_a.equals(Ric_b)


# --------------------------------------------------------------------- #
# Minkowski curvilinear, Ric = 0                                       #
# --------------------------------------------------------------------- #


class TestMinkowskiRicci:
    def test_minkowski_spherical_ricci_zero(
        self, minkowski_spherical
    ) -> None:
        F, g = minkowski_spherical
        Ric = ricci(levi_civita(g))
        assert Ric.is_zero()

    def test_minkowski_cartesian_ricci_zero(self) -> None:
        coords = list(sp.symbols("t x y z"))
        F = CoordinateFrame(coords)
        g = ComponentMetric(F, sp.diag(-1, 1, 1, 1))
        Ric = ricci(levi_civita(g))
        assert Ric.is_zero()


# --------------------------------------------------------------------- #
# 2-sphere, Ricci proportional to metric (Einstein space)              #
# --------------------------------------------------------------------- #


class TestTwoSphere:
    """Round 2-sphere is an Einstein space: ``Ric_{ab} ∝ g_{ab}``.

    Sign convention: prof's definition
    ``R(U,V)W = ∇_U∇_VW − ∇_V∇_UW − ∇_{[U,V]}W`` with contraction
    ``Ric_{ab} = R^c_{acb}`` gives the **opposite** sign of the
    Wald/Carroll physics convention. So
        Ric = −(1/R₀²) g
        R = −2/R₀²
    Schwarzschild vacuum (`G = 0`) is convention-independent, hence
    unaffected.
    """

    def test_ricci_components(self, two_sphere) -> None:
        F, g = two_sphere
        theta = sp.Symbol("theta", positive=True)
        R0 = sp.Symbol("R0", positive=True)
        Ric = ricci(levi_civita(g))
        # Ric_θθ = -1
        assert sp.simplify(Ric[0, 0] - (-1)) == 0
        # Ric_φφ = -sin²θ, needs trigsimp to recognise the form
        assert sp.simplify(
            sp.trigsimp(Ric[1, 1]) - (-sp.sin(theta) ** 2)
        ) == 0
        # Off-diagonal vanishes
        assert sp.simplify(Ric[0, 1]) == 0

    def test_einstein_space_relation(self, two_sphere) -> None:
        """Ric = −(1/R₀²) g, Einstein-space condition (with sign)."""
        F, g = two_sphere
        R0 = sp.Symbol("R0", positive=True)
        Ric = ricci(levi_civita(g))
        n = F.dim
        for a in range(n):
            for b in range(n):
                diff = sp.simplify(
                    sp.trigsimp(Ric[a, b] - (-1 / R0**2) * g[a, b])
                )
                assert diff == 0

    def test_ricci_scalar_on_sphere(self, two_sphere) -> None:
        """R = −2/R₀² on the 2-sphere (prof's sign convention)."""
        F, g = two_sphere
        R0 = sp.Symbol("R0", positive=True)
        R = ricci_scalar(levi_civita(g), g)
        assert sp.simplify(R - (-2 / R0**2)) == 0


# --------------------------------------------------------------------- #
# Symmetry                                                              #
# --------------------------------------------------------------------- #


class TestRicciSymmetry:
    """Ric_{ab} = Ric_{ba} for any Levi-Civita connection."""

    def test_schwarzschild_symmetric(self, schwarzschild) -> None:
        F, g = schwarzschild
        Ric = ricci(levi_civita(g))
        for a in range(4):
            for b in range(4):
                assert (
                    sp.simplify(Ric[a, b] - Ric[b, a]) == 0
                )

    def test_two_sphere_symmetric(self, two_sphere) -> None:
        F, g = two_sphere
        Ric = ricci(levi_civita(g))
        for a in range(2):
            for b in range(2):
                assert (
                    sp.simplify(Ric[a, b] - Ric[b, a]) == 0
                )


# --------------------------------------------------------------------- #
# Derivation traces                                                     #
# --------------------------------------------------------------------- #


class TestRicciDerivationTraces:
    def test_steps_recorded(self, schwarzschild) -> None:
        F, g = schwarzschild
        Ric = ricci(levi_civita(g))
        steps = Ric.derivation_steps(0, 0)
        assert len(steps) >= 2
        assert isinstance(steps[0], RicciStep)
        assert "Ricci" in steps[0].rule

    def test_format_derivation(self, schwarzschild) -> None:
        F, g = schwarzschild
        Ric = ricci(levi_civita(g))
        text = Ric.format_derivation(0, 0)
        assert isinstance(text, str)
        assert "Ricci" in text

    def test_index_out_of_range(self, schwarzschild) -> None:
        F, g = schwarzschild
        Ric = ricci(levi_civita(g))
        with pytest.raises(IndexError):
            Ric.derivation_steps(5, 0)

    def test_canonical_pair_lookup(self, schwarzschild) -> None:
        """Ric is symmetric so (a, b) and (b, a) share the canonical
        derivation."""
        F, g = schwarzschild
        Ric = ricci(levi_civita(g))
        s1 = Ric.derivation_steps(1, 2)
        s2 = Ric.derivation_steps(2, 1)
        assert s1 == s2
