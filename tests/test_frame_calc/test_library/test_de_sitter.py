"""Tests for `jacopy.frame_calc.library.de_sitter` (Faz 19 Chunk B)."""

import pytest
import sympy as sp

from jacopy.frame_calc import (
    ComponentMetric,
    CoordinateFrame,
    levi_civita,
    ricci,
)
from jacopy.frame_calc.library import de_sitter


class TestDeSitterConstruction:
    def test_default_construction(self) -> None:
        F, g = de_sitter()
        assert isinstance(F, CoordinateFrame)
        assert isinstance(g, ComponentMetric)
        assert F.dim == 4
        assert F.index_names() == ("t", "r", "theta", "phi")

    def test_custom_Lambda(self) -> None:
        my_L = sp.Symbol("Lambda0", positive=True)
        F, g = de_sitter(Lambda_sym=my_L)
        assert my_L in g[0, 0].free_symbols

    def test_invalid_Lambda_rejected(self) -> None:
        with pytest.raises(TypeError, match="Symbol"):
            de_sitter(Lambda_sym="not a symbol")  # type: ignore[arg-type]


class TestDeSitterPhysics:
    """de Sitter is maximally symmetric: Ric ∝ g.

    The package's sign convention gives ``Ric_{ab} = -Λ g_{ab}`` (opposite of
    Wald/Carroll); see faz18_progress.md.
    """

    def test_ricci_proportional_to_metric(self) -> None:
        """Maximally symmetric: Ric = c * g for some scalar c."""
        F, g = de_sitter()
        Ric = ricci(levi_civita(g))
        Lambda = sp.Symbol("Lambda", positive=True)
        # In our convention: Ric = -Λ g
        for a in range(4):
            for b in range(4):
                diff = sp.simplify(Ric[a, b] + Lambda * g[a, b])
                assert diff == 0, f"Ric[{a},{b}] + Λg[{a},{b}] = {diff}"

    def test_constant_ricci_scalar(self) -> None:
        """R is constant for max-symmetric space."""
        F, g = de_sitter()
        Ric = ricci(levi_civita(g))
        g_inv = g.inverse()
        R = sp.S.Zero
        for a in range(4):
            for b in range(4):
                R += g_inv[a, b] * Ric[a, b]
        R_simplified = sp.simplify(R)
        # Should not depend on r
        r = sp.Symbol("r", positive=True)
        assert sp.diff(R_simplified, r) == 0
