"""Tests for `jacopy.frame_calc.levi_civita` (Faz 18 Stage D)."""

import pytest
import sympy as sp

from jacopy.frame_calc import (
    AbstractFrame,
    ComponentMetric,
    CoordinateFrame,
    KoszulStep,
    LeviCivitaConnection,
    Tetrad,
    levi_civita,
)


# --------------------------------------------------------------------- #
# Type / argument validation                                            #
# --------------------------------------------------------------------- #


class TestLeviCivitaTypeChecks:
    def test_non_metric_rejected(self) -> None:
        with pytest.raises(TypeError, match="ComponentMetric"):
            levi_civita("not a metric")  # type: ignore[arg-type]

    def test_abstract_frame_raises(self) -> None:
        F = AbstractFrame(dim=2)
        # ComponentMetric on AbstractFrame: storage works
        m = ComponentMetric(F, sp.eye(2))
        with pytest.raises(NotImplementedError, match="AbstractFrame"):
            levi_civita(m)

    def test_tetrad_supported(self) -> None:
        """Stage B: Tetrad is now supported via frame protocol."""
        t, r = sp.symbols("t r")
        coord = CoordinateFrame([t, r])
        T = Tetrad(coord, vielbein=sp.eye(2))
        m = ComponentMetric(T, sp.eye(2))
        # Should run without error (even if trivial result for identity tetrad)
        LC = levi_civita(m)
        assert LC.is_zero()  # identity tetrad on Minkowski → flat


# --------------------------------------------------------------------- #
# Polar frame, flat 2D in polar coords                                 #
# --------------------------------------------------------------------- #


@pytest.fixture
def polar() -> tuple[CoordinateFrame, ComponentMetric]:
    """Polar 2D Euclidean: ``ds² = dr² + r² dθ²``."""
    r, theta = sp.symbols("r theta", positive=True)
    F = CoordinateFrame([r, theta])
    g = ComponentMetric(F, sp.Matrix([[1, 0], [0, r**2]]))
    return F, g


class TestPolarMetric:
    def test_christoffel_signature_and_shape(self, polar) -> None:
        F, g = polar
        LC = levi_civita(g)
        assert isinstance(LC, LeviCivitaConnection)
        assert LC.signature == (1, 2)
        assert LC.shape == (2, 2, 2)

    def test_polar_christoffel_values(self, polar) -> None:
        """User-specified expected values: Γ^r_{θθ}=-r, Γ^θ_{rθ}=Γ^θ_{θr}=1/r."""
        F, g = polar
        r, theta = sp.symbols("r theta", positive=True)
        LC = levi_civita(g)
        # Index convention: 0 = r, 1 = θ
        # Γ^r_{θθ} = -r
        assert sp.simplify(LC[0, 1, 1] - (-r)) == 0
        # Γ^θ_{rθ} = 1/r
        assert sp.simplify(LC[1, 0, 1] - 1 / r) == 0
        # Γ^θ_{θr} = 1/r (symmetry)
        assert sp.simplify(LC[1, 1, 0] - 1 / r) == 0

    def test_polar_other_christoffels_are_zero(self, polar) -> None:
        F, g = polar
        LC = levi_civita(g)
        nonzero = LC.nonzero_components()
        # Only three non-zero entries expected
        assert set(nonzero.keys()) == {(0, 1, 1), (1, 0, 1), (1, 1, 0)}

    def test_polar_symmetric_in_lower_indices(self, polar) -> None:
        F, g = polar
        LC = levi_civita(g)
        n = F.dim
        for e in range(n):
            for a in range(n):
                for b in range(n):
                    diff = sp.simplify(LC[e, a, b] - LC[e, b, a])
                    assert diff == 0


# --------------------------------------------------------------------- #
# Schwarzschild metric                                                  #
# --------------------------------------------------------------------- #


@pytest.fixture
def schwarzschild() -> tuple[CoordinateFrame, ComponentMetric]:
    """4D Schwarzschild in Boyer-Lindquist coordinates (t, r, θ, φ)."""
    t, r, theta, phi = sp.symbols("t r theta phi")
    M = sp.Symbol("M", positive=True)
    F = CoordinateFrame([t, r, theta, phi])
    g = ComponentMetric(
        F,
        sp.Matrix([
            [-(1 - 2 * M / r), 0, 0, 0],
            [0, 1 / (1 - 2 * M / r), 0, 0],
            [0, 0, r**2, 0],
            [0, 0, 0, r**2 * sp.sin(theta) ** 2],
        ]),
    )
    return F, g


class TestSchwarzschild:
    def test_known_christoffel_components(self, schwarzschild) -> None:
        """Spot-check known Schwarzschild Christoffel symbols."""
        F, g = schwarzschild
        t, r, theta, phi = sp.symbols("t r theta phi")
        M = sp.Symbol("M", positive=True)
        LC = levi_civita(g)

        # Index convention: 0=t, 1=r, 2=θ, 3=φ
        # Γ^t_{tr} = M / (r² (1 - 2M/r))
        expected_t_tr = M / (r**2 * (1 - 2 * M / r))
        assert sp.simplify(LC[0, 0, 1] - expected_t_tr) == 0

        # Γ^r_{tt} = M (1 - 2M/r) / r²
        expected_r_tt = M * (1 - 2 * M / r) / r**2
        assert sp.simplify(LC[1, 0, 0] - expected_r_tt) == 0

        # Γ^r_{rr} = -M / (r² (1 - 2M/r))
        expected_r_rr = -M / (r**2 * (1 - 2 * M / r))
        assert sp.simplify(LC[1, 1, 1] - expected_r_rr) == 0

        # Γ^θ_{rθ} = 1/r
        assert sp.simplify(LC[2, 1, 2] - 1 / r) == 0

        # Γ^φ_{rφ} = 1/r
        assert sp.simplify(LC[3, 1, 3] - 1 / r) == 0

        # Γ^φ_{θφ} = cot θ
        assert (
            sp.simplify(LC[3, 2, 3] - sp.cos(theta) / sp.sin(theta)) == 0
        )

        # Γ^r_{θθ} = -(r - 2M)
        assert sp.simplify(LC[1, 2, 2] - (-(r - 2 * M))) == 0

        # Γ^r_{φφ} = -(r - 2M) sin² θ
        assert (
            sp.simplify(
                LC[1, 3, 3] - (-(r - 2 * M) * sp.sin(theta) ** 2)
            )
            == 0
        )

        # Γ^θ_{φφ} = -sin θ cos θ
        assert (
            sp.simplify(LC[2, 3, 3] + sp.sin(theta) * sp.cos(theta)) == 0
        )

    def test_lower_indices_symmetric(self, schwarzschild) -> None:
        F, g = schwarzschild
        LC = levi_civita(g)
        for e in range(4):
            for a in range(4):
                for b in range(4):
                    if a != b:
                        diff = sp.simplify(LC[e, a, b] - LC[e, b, a])
                        assert diff == 0


# --------------------------------------------------------------------- #
# Generic 2D off-diagonal metric                                        #
# --------------------------------------------------------------------- #


class TestGeneric2DMetric:
    def test_christoffel_1_11_matches_textbook_formula(self) -> None:
        """User's second example:
        Γ¹_{11} = 1/(2D) [g₂₂ ∂₁g₁₁ − g₁₂ (2 ∂₁g₁₂ − ∂₂g₁₁)]
        where D = g₁₁ g₂₂ − g₁₂².
        """
        x, y = sp.symbols("x y")
        g11 = sp.Function("g_11")(x, y)
        g12 = sp.Function("g_12")(x, y)
        g22 = sp.Function("g_22")(x, y)
        F = CoordinateFrame([x, y])
        g = ComponentMetric(
            F, sp.Matrix([[g11, g12], [g12, g22]])
        )
        LC = levi_civita(g)
        D = g11 * g22 - g12**2
        expected = (
            g22 * sp.diff(g11, x)
            - g12 * (2 * sp.diff(g12, x) - sp.diff(g11, y))
        ) / (2 * D)
        diff = sp.simplify(LC[0, 0, 0] - expected)
        assert diff == 0


# --------------------------------------------------------------------- #
# 1D + Minkowski sanity                                                 #
# --------------------------------------------------------------------- #


class Test1DAndMinkowski:
    def test_1d_flat_christoffel_zero(self) -> None:
        t = sp.Symbol("t")
        F = CoordinateFrame([t])
        g = ComponentMetric(F, sp.Matrix([[1]]))
        LC = levi_civita(g)
        assert LC.is_zero()

    def test_minkowski_cartesian_christoffel_zero(self) -> None:
        coords = list(sp.symbols("t x y z"))
        F = CoordinateFrame(coords)
        g = ComponentMetric(F, sp.diag(-1, 1, 1, 1))
        LC = levi_civita(g)
        assert LC.is_zero()

    def test_minkowski_signature_flip_christoffel_zero(self) -> None:
        """Signature `+ - - -` is also flat in Cartesian coords."""
        coords = list(sp.symbols("t x y z"))
        F = CoordinateFrame(coords)
        g = ComponentMetric(F, sp.diag(1, -1, -1, -1))
        LC = levi_civita(g)
        assert LC.is_zero()


class TestMinkowskiCurvilinear:
    """Minkowski in non-Cartesian coordinates: Christoffel symbols are
    non-zero (the coordinates are curvilinear) even though the underlying
    space-time is flat. Curvature tensor zero-ness is verified in Stage E.
    """

    def test_minkowski_spherical_known_components(self) -> None:
        """`ds² = -dt² + dr² + r² dθ² + r² sin²θ dφ²`.

        Christoffels match the spatial 3-sphere portion of Schwarzschild
        with M = 0, i.e. the angular block of Schwarzschild reduces
        to spherical Minkowski.
        """
        t = sp.Symbol("t")
        r, theta, phi = sp.symbols("r theta phi", positive=True)
        F = CoordinateFrame([t, r, theta, phi])
        g = ComponentMetric(F, sp.Matrix([
            [-1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, r**2, 0],
            [0, 0, 0, r**2 * sp.sin(theta)**2],
        ]))
        LC = levi_civita(g)
        # Index convention: 0=t, 1=r, 2=θ, 3=φ
        # Γ^r_{θθ} = -r
        assert sp.simplify(LC[1, 2, 2] - (-r)) == 0
        # Γ^r_{φφ} = -r sin²θ
        assert sp.simplify(LC[1, 3, 3] - (-r * sp.sin(theta) ** 2)) == 0
        # Γ^θ_{rθ} = 1/r
        assert sp.simplify(LC[2, 1, 2] - 1 / r) == 0
        # Γ^θ_{φφ} = -sin θ cos θ
        assert (
            sp.simplify(LC[2, 3, 3] + sp.sin(theta) * sp.cos(theta)) == 0
        )
        # Γ^φ_{rφ} = 1/r
        assert sp.simplify(LC[3, 1, 3] - 1 / r) == 0
        # Γ^φ_{θφ} = cot θ
        assert (
            sp.simplify(LC[3, 2, 3] - sp.cos(theta) / sp.sin(theta)) == 0
        )

    def test_minkowski_spherical_t_block_zero(self) -> None:
        """`t` is unaffected by the curvilinear angular coords,
        no Christoffel symbol touches the t-slot."""
        t = sp.Symbol("t")
        r, theta, phi = sp.symbols("r theta phi", positive=True)
        F = CoordinateFrame([t, r, theta, phi])
        g = ComponentMetric(F, sp.Matrix([
            [-1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, r**2, 0],
            [0, 0, 0, r**2 * sp.sin(theta)**2],
        ]))
        LC = levi_civita(g)
        # Every Γ^t_{??} = 0 and Γ^?_{t?} = 0
        for a in range(4):
            for b in range(4):
                # Upper t
                assert LC[0, a, b] == 0, f"Γ^t_{{{a}{b}}} should be 0"
                # Lower t in either slot
                assert LC[a, 0, b] == 0, f"Γ^{a}_{{t{b}}} should be 0"
                assert LC[a, b, 0] == 0, f"Γ^{a}_{{{b}t}} should be 0"

    def test_minkowski_cylindrical_known_components(self) -> None:
        """`ds² = -dt² + dρ² + ρ² dφ² + dz²`, flat coords on z and t."""
        t, z = sp.symbols("t z")
        rho = sp.Symbol("rho", positive=True)
        phi = sp.Symbol("phi", positive=True)
        F = CoordinateFrame([t, rho, phi, z])
        g = ComponentMetric(F, sp.diag(-1, 1, rho**2, 1))
        LC = levi_civita(g)

        nz = LC.nonzero_components()
        # Exactly 3 non-zero entries
        assert len(nz) == 3
        # Γ^ρ_{φφ} = -ρ
        assert sp.simplify(LC[1, 2, 2] - (-rho)) == 0
        # Γ^φ_{ρφ} = 1/ρ
        assert sp.simplify(LC[2, 1, 2] - 1 / rho) == 0
        # Γ^φ_{φρ} = 1/ρ
        assert sp.simplify(LC[2, 2, 1] - 1 / rho) == 0

    def test_rindler_2d_known_components(self) -> None:
        """Rindler chart `ds² = -ρ² dτ² + dρ²`, uniformly accelerated
        observer's frame. Non-zero Christoffels reflect the
        pseudo-force; spacetime is still flat."""
        tau = sp.Symbol("tau")
        rho = sp.Symbol("rho", positive=True)
        F = CoordinateFrame([tau, rho])
        g = ComponentMetric(F, sp.Matrix([[-(rho**2), 0], [0, 1]]))
        LC = levi_civita(g)

        nz = LC.nonzero_components()
        assert len(nz) == 3
        # Γ^τ_{τρ} = Γ^τ_{ρτ} = 1/ρ
        assert sp.simplify(LC[0, 0, 1] - 1 / rho) == 0
        assert sp.simplify(LC[0, 1, 0] - 1 / rho) == 0
        # Γ^ρ_{ττ} = ρ
        assert sp.simplify(LC[1, 0, 0] - rho) == 0


# --------------------------------------------------------------------- #
# Derivation traces                                                     #
# --------------------------------------------------------------------- #


class TestDerivationTraces:
    def test_polar_christoffel_has_steps(self, polar) -> None:
        F, g = polar
        LC = levi_civita(g)
        steps = LC.derivation_steps(0, 1, 1)  # Γ^r_{θθ}
        # Should have: open formula, frame-deriv, γ (=0 for coord),
        # contraction, simplify → 5 steps
        assert len(steps) == 5
        assert isinstance(steps[0], KoszulStep)
        assert "Koszul" in steps[0].rule
        assert "Simplify" in steps[-1].rule

    def test_derivation_steps_symmetric_pair(self, polar) -> None:
        """`Γ^θ_{rθ}` and `Γ^θ_{θr}` share the same derivation."""
        F, g = polar
        LC = levi_civita(g)
        s1 = LC.derivation_steps(1, 0, 1)
        s2 = LC.derivation_steps(1, 1, 0)
        assert s1 == s2

    def test_derivation_steps_index_check(self, polar) -> None:
        F, g = polar
        LC = levi_civita(g)
        with pytest.raises(IndexError):
            LC.derivation_steps(5, 0, 0)

    def test_format_derivation_returns_string(self, polar) -> None:
        F, g = polar
        LC = levi_civita(g)
        text = LC.format_derivation(0, 1, 1)
        assert isinstance(text, str)
        assert "Koszul" in text
        assert "Simplify" in text

    def test_final_step_value_matches_component(self, polar) -> None:
        F, g = polar
        r = sp.symbols("r", positive=True)
        LC = levi_civita(g)
        steps = LC.derivation_steps(0, 1, 1)
        # Last step's expression should equal the component value
        assert sp.simplify(steps[-1].expression - LC[0, 1, 1]) == 0

    def test_coordinate_frame_gamma_step_narrated(self, polar) -> None:
        """For a coordinate frame, the γ step should narrate `γ ≡ 0`."""
        F, g = polar
        LC = levi_civita(g)
        steps = LC.derivation_steps(0, 1, 1)
        gamma_steps = [s for s in steps if "γ" in s.rule]
        assert len(gamma_steps) == 1
        assert "coordinate" in gamma_steps[0].rule.lower()
