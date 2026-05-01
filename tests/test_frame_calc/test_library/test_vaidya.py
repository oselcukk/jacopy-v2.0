"""Tests for `jacopy.frame_calc.library.vaidya` (Faz 19 Chunk B)."""

import pytest
import sympy as sp

from jacopy.frame_calc import (
    ComponentMetric,
    CoordinateFrame,
    einstein_tensor,
    levi_civita,
    ricci,
)
from jacopy.frame_calc.library import vaidya


class TestVaidyaConstruction:
    def test_default_construction(self) -> None:
        F, g = vaidya()
        assert isinstance(F, CoordinateFrame)
        assert isinstance(g, ComponentMetric)
        assert F.dim == 4
        # Note: first coord is v (advanced null), not t
        assert F.index_names() == ("v", "r", "theta", "phi")

    def test_off_diagonal_g_vr_present(self) -> None:
        """Hallmark of null coordinates: g_{vr} = 1."""
        F, g = vaidya()
        assert sp.simplify(g[0, 1] - 1) == 0
        assert sp.simplify(g[1, 0] - 1) == 0
        assert g[1, 1] == 0  # g_{rr} = 0 in null coords

    def test_custom_M_func(self) -> None:
        v = sp.Symbol("v", real=True)
        my_M = sp.Function("M_custom")(v)
        F, g = vaidya(M_func=my_M, v_sym=v)
        assert my_M in g[0, 0].atoms(sp.Function)


class TestVaidyaPhysics:
    def test_constant_M_is_vacuum(self) -> None:
        """When M(v) = const, Vaidya reduces to (ingoing-EF) Schwarzschild,
        which is vacuum."""
        v = sp.Symbol("v", real=True)
        M_const = sp.Symbol("M", positive=True)
        F, g = vaidya(M_func=M_const, v_sym=v)
        G = einstein_tensor(levi_civita(g), g)
        assert G.is_vacuum()

    def test_dynamic_M_not_vacuum(self) -> None:
        """Time-dependent M(v) → non-vacuum (incoming radiation)."""
        F, g = vaidya()
        G = einstein_tensor(levi_civita(g), g)
        assert not G.is_vacuum()

    def test_einstein_G_vv_proportional_to_Mdot(self) -> None:
        """G_{vv} encodes the inflowing null energy density."""
        v = sp.Symbol("v", real=True)
        M = sp.Function("M")(v)
        F, g = vaidya(M_func=M, v_sym=v)
        G = einstein_tensor(levi_civita(g), g)
        # G_{vv} should contain dM/dv
        Mdot = sp.diff(M, v)
        assert Mdot in G[0, 0].atoms(sp.Derivative)
