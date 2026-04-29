"""Tests for `jacopy.frame_calc.library.schwarzschild` (Faz 18 Stage H)."""

import pytest
import sympy as sp

from jacopy.frame_calc import (
    ComponentMetric,
    CoordinateFrame,
    curvature,
    einstein_tensor,
    levi_civita,
    ricci,
)
from jacopy.frame_calc.library import schwarzschild


class TestSchwarzschildConstruction:
    def test_default_construction(self) -> None:
        F, g = schwarzschild()
        assert isinstance(F, CoordinateFrame)
        assert isinstance(g, ComponentMetric)
        assert F.dim == 4
        assert F.index_names() == ("t", "r", "theta", "phi")

    def test_custom_M_symbol(self) -> None:
        my_M = sp.Symbol("M0", positive=True)
        F, g = schwarzschild(M_sym=my_M)
        # g_tt should have M0 in it, not M
        assert my_M in g[0, 0].free_symbols

    def test_invalid_M_rejected(self) -> None:
        with pytest.raises(TypeError, match="Symbol"):
            schwarzschild(M_sym="not a symbol")  # type: ignore[arg-type]


class TestSchwarzschildVacuum:
    """The Stage F integration test, packaged via the library factory."""

    def test_einstein_tensor_vanishes(self) -> None:
        F, g = schwarzschild()
        G = einstein_tensor(levi_civita(g), g)
        assert G.is_vacuum()

    def test_ricci_vanishes(self) -> None:
        F, g = schwarzschild()
        Ric = ricci(levi_civita(g))
        assert Ric.is_zero()

    def test_curvature_does_not_vanish(self) -> None:
        """Schwarzschild has genuine curvature (it's not flat)."""
        F, g = schwarzschild()
        R = curvature(levi_civita(g))
        assert not R.is_zero()
