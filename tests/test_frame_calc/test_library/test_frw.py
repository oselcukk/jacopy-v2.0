"""Tests for `jacopy.frame_calc.library.frw` (Faz 18 Stage H)."""

import pytest
import sympy as sp

from jacopy.frame_calc import (
    ComponentMetric,
    CoordinateFrame,
    einstein_tensor,
    levi_civita,
    ricci,
    ricci_scalar,
)
from jacopy.frame_calc.library import frw


class TestFRWConstruction:
    def test_default_flat_construction(self) -> None:
        F, g = frw()
        assert isinstance(F, CoordinateFrame)
        assert F.dim == 4
        assert F.index_names() == ("t", "r", "theta", "phi")
        # k=0 default
        assert "k=0" in F.name

    def test_invalid_k_rejected(self) -> None:
        with pytest.raises(ValueError, match="k must be"):
            frw(k=2)
        with pytest.raises(ValueError, match="k must be"):
            frw(k=-2)

    def test_closed_universe(self) -> None:
        F, g = frw(k=1)
        assert "k=1" in F.name

    def test_open_universe(self) -> None:
        F, g = frw(k=-1)
        assert "k=-1" in F.name

    def test_custom_scale_factor(self) -> None:
        """Pass an explicit `a(t) = t^(2/3)` (matter-dominated)."""
        t = sp.Symbol("t", positive=True)
        a_explicit = t ** sp.Rational(2, 3)
        F, g = frw(a_func=a_explicit, t_sym=t)
        # g_rr should be a²
        assert sp.simplify(g[1, 1] - t ** sp.Rational(4, 3)) == 0


class TestFRWPipeline:
    """FRW is a non-vacuum solution, Einstein tensor non-zero in general."""

    def test_einstein_tensor_nonzero(self) -> None:
        F, g = frw()
        G = einstein_tensor(levi_civita(g), g)
        # Generically non-vacuum (depends on a(t)); G_tt ∝ ((da/dt)/a)²
        assert not G.is_zero()

    def test_einstein_components_have_a_dependence(self) -> None:
        """G's components should reference the scale factor a(t)."""
        F, g = frw()
        G = einstein_tensor(levi_civita(g), g)
        # G[0, 0] should contain a derivative of a(t)
        expr_str = str(G[0, 0])
        # Either Derivative(a(t), t) or a(t) appears
        assert "a" in expr_str or "Derivative" in expr_str

    def test_constant_a_yields_minkowski(self) -> None:
        """Setting a(t) = 1 should reduce FRW to flat Minkowski → G = 0."""
        t = sp.Symbol("t", positive=True)
        F, g = frw(a_func=sp.Integer(1), t_sym=t)
        G = einstein_tensor(levi_civita(g), g)
        # When a is constant, FRW degenerates to Minkowski-spatial-flat
        # → G should vanish identically.
        assert G.is_zero()
