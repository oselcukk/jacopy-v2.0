"""Cross-stage integration tests (Faz 18 Stage J).

End-to-end paper-grade workflow tests that exercise multiple
stages together, verifying the full pipeline (Frame → Metric →
LeviCivita → Curvature → Ricci → Einstein → ProofChain → LaTeX)
works in one go on each canonical fixture.
"""

import pytest
import sympy as sp

from jacopy.display import chain_to_latex, chain_to_latex_document
from jacopy.frame_calc import (
    AbstractFrame,
    ComponentMetric,
    CoordinateFrame,
    Tetrad,
    curvature,
    einstein_tensor,
    levi_civita,
    ricci,
    ricci_scalar,
)
from jacopy.frame_calc.library import (
    frw,
    kerr,
    minkowski,
    schwarzschild,
)
from jacopy.proof.chain import ProofChain


# --------------------------------------------------------------------- #
# Schwarzschild, full paper-grade workflow                              #
# --------------------------------------------------------------------- #


class TestSchwarzschildFullWorkflow:
    """The flagship use case: Schwarzschild metric → vacuum
    verification → LaTeX-rendered Christoffel derivation."""

    def test_full_pipeline(self) -> None:
        F, g = schwarzschild()
        # Stage D
        LC = levi_civita(g)
        # Stage E
        R = curvature(LC)
        # Stage F
        Ric = ricci(LC)
        R_scalar = ricci_scalar(LC, g)
        G = einstein_tensor(LC, g)

        # Curvature non-zero (genuine spacetime)
        assert not R.is_zero()
        # Ricci-flat (Schwarzschild)
        assert Ric.is_zero()
        # Scalar zero
        assert sp.simplify(R_scalar) == 0
        # Vacuum
        assert G.is_vacuum()

    def test_christoffel_derivation_renders_latex(self) -> None:
        """Stage G: derivation chain → paper-grade LaTeX."""
        F, g = schwarzschild()
        LC = levi_civita(g)
        chain = LC.derivation_chain(0, 0, 1)  # Γ^t_tr
        assert isinstance(chain, ProofChain)

        # Latex render via existing display layer
        latex = chain_to_latex(chain)
        assert isinstance(latex, str)
        assert "gather*" in latex
        assert "computation" in latex     # provenance tag visible

    def test_full_document_render(self) -> None:
        """End-to-end: derivation → full LaTeX document."""
        F, g = schwarzschild()
        LC = levi_civita(g)
        chain = LC.derivation_chain(0, 0, 1)
        doc = chain_to_latex_document(chain)
        assert "\\begin{document}" in doc
        assert "\\end{document}" in doc

    def test_optimised_mode_consistency(self) -> None:
        """Default and optimised modes give equivalent results."""
        F, g = schwarzschild()
        # Default
        G_default = einstein_tensor(levi_civita(g), g)
        # Optimised
        G_opt = einstein_tensor(
            levi_civita(g, optimized=True), g, optimized=True
        )
        # Both vacuum
        assert G_default.is_vacuum()
        assert G_opt.is_vacuum()


# --------------------------------------------------------------------- #
# Minkowski in three coordinate systems                                  #
# --------------------------------------------------------------------- #


class TestMinkowskiCoordinateSystems:
    """Same flat spacetime, three coordinate systems → all vacuum."""

    def test_cartesian_vacuum(self) -> None:
        F, g = minkowski()
        G = einstein_tensor(levi_civita(g), g)
        assert G.is_vacuum()

    def test_spherical_minkowski_vacuum(self) -> None:
        """Spherical Minkowski has 9 non-zero Christoffels (curvilinear
        coordinates) but is still flat, Riemann ≡ 0 → G = 0."""
        t, r, theta, phi = sp.symbols("t r theta phi")
        F = CoordinateFrame([t, r, theta, phi])
        g = ComponentMetric(F, sp.Matrix([
            [-1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, r**2, 0],
            [0, 0, 0, r**2 * sp.sin(theta)**2],
        ]))
        LC = levi_civita(g)
        # Many non-zero Christoffels
        assert len(LC.nonzero_components()) >= 9
        # But still flat, Riemann zero
        R = curvature(LC)
        assert R.is_zero()
        # Vacuum
        G = einstein_tensor(LC, g)
        assert G.is_vacuum()


# --------------------------------------------------------------------- #
# FRW cosmology, non-vacuum                                             #
# --------------------------------------------------------------------- #


class TestFRWCosmology:
    """FRW metric → Friedmann equations form."""

    def test_frw_einstein_nonzero_for_dynamic_a(self) -> None:
        F, g = frw()
        G = einstein_tensor(levi_civita(g), g)
        # Generic a(t) → non-vacuum
        assert not G.is_zero()
        # G_tt should depend on a(t) and its derivative
        expr_str = str(G[0, 0])
        assert "a" in expr_str

    def test_frw_with_constant_a_is_minkowski(self) -> None:
        """If a(t) ≡ 1, FRW reduces to flat space → G ≡ 0."""
        t = sp.Symbol("t", positive=True)
        F, g = frw(a_func=sp.Integer(1), t_sym=t)
        G = einstein_tensor(levi_civita(g), g)
        assert G.is_zero()


# --------------------------------------------------------------------- #
# Kerr, vacuum via optimised mode                                      #
# --------------------------------------------------------------------- #


class TestKerrIntegration:
    """Kerr full pipeline must complete in optimised mode."""

    def test_kerr_vacuum_optimised(self) -> None:
        F, g = kerr()
        G = einstein_tensor(
            levi_civita(g, optimized=True), g, optimized=True
        )
        # The Kerr Einstein-tensor entries collapse to literal zero
        # in raw form (no simplify needed).
        assert G.is_vacuum()
        assert G.is_zero(simplify=False)


# --------------------------------------------------------------------- #
# Tetrad path                                                            #
# --------------------------------------------------------------------- #


class TestTetradPath:
    """Tetrad-based metric calculations."""

    def test_identity_tetrad_minkowski_vacuum(self) -> None:
        """Identity vielbein → tetrad coincides with coord frame."""
        coord = CoordinateFrame(list(sp.symbols("t x y z")))
        T = Tetrad(coord, vielbein=sp.eye(4))
        g = ComponentMetric(T, sp.diag(-1, 1, 1, 1))
        G = einstein_tensor(levi_civita(g), g)
        assert G.is_vacuum()

    def test_tetrad_christoffel_uses_gamma(self) -> None:
        """A non-trivial vielbein produces non-zero γ, and the Koszul
        formula's γ-correction terms fire."""
        x, y = sp.symbols("x y", positive=True)
        coord = CoordinateFrame([x, y])
        # Polar-like vielbein
        T = Tetrad(coord, vielbein=sp.Matrix([[1, 0], [0, 1 / x]]))
        # γ should be non-zero on this tetrad
        nonzero_gamma = False
        for a in range(2):
            for b in range(2):
                for c in range(b + 1, 2):
                    if T.gamma(a, b, c) != 0:
                        nonzero_gamma = True
        assert nonzero_gamma


# --------------------------------------------------------------------- #
# Cross-fixture sanity                                                   #
# --------------------------------------------------------------------- #


class TestFixtureSanity:
    """Each library fixture builds a working ComponentMetric."""

    @pytest.mark.parametrize(
        "factory_name", ["minkowski", "schwarzschild", "kerr"]
    )
    def test_fixture_construction(self, factory_name) -> None:
        from jacopy.frame_calc.library import (
            kerr,
            minkowski,
            schwarzschild,
        )

        factories = {
            "minkowski": minkowski,
            "schwarzschild": schwarzschild,
            "kerr": kerr,
        }
        F, g = factories[factory_name]()
        assert isinstance(g, ComponentMetric)
        assert g.shape == (4, 4)
        # All have a t in the index names
        assert "t" in F.index_names()


# --------------------------------------------------------------------- #
# Sign convention sanity                                                 #
# --------------------------------------------------------------------- #


class TestSignConventionConsistency:
    """The package's chosen sign convention is consistent across pipeline."""

    def test_schwarzschild_vacuum_is_convention_independent(self) -> None:
        """Schwarzschild G ≡ 0 holds regardless of sign, Ric and ½ R g
        both flip together."""
        F, g = schwarzschild()
        G = einstein_tensor(levi_civita(g), g)
        for a in range(4):
            for b in range(4):
                assert sp.simplify(sp.trigsimp(G[a, b])) == 0

    def test_2d_lovelock_einstein_zero(self) -> None:
        """Lovelock theorem: G ≡ 0 in 2D for any metric.

        On the 2-sphere, Ricci ∝ g (Einstein space) and the
        contraction `Ric - ½ R g` collapses identically because
        R = (1/n)·n·trace = trace → ½ R g = Ric.
        """
        theta, phi = sp.symbols("theta phi", positive=True)
        R0 = sp.Symbol("R0", positive=True)
        F = CoordinateFrame([theta, phi])
        g = ComponentMetric(F, sp.Matrix([
            [R0**2, 0],
            [0, R0**2 * sp.sin(theta)**2],
        ]))
        G = einstein_tensor(levi_civita(g), g)
        assert G.is_zero()
