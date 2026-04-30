"""Tests for `jacopy.frame_calc.library.godel` (Faz 19 Chunk B)."""

import pytest
import sympy as sp

from jacopy.frame_calc import (
    ComponentMetric,
    CoordinateFrame,
    einstein_tensor,
    levi_civita,
)
from jacopy.frame_calc.library import godel


class TestGodelConstruction:
    def test_default_construction(self) -> None:
        F, g = godel()
        assert isinstance(F, CoordinateFrame)
        assert isinstance(g, ComponentMetric)
        assert F.dim == 4
        assert F.index_names() == ("t", "x", "y", "z")

    def test_off_diagonal_g_ty_present(self) -> None:
        """Hallmark of rotation: g_{ty} ≠ 0."""
        F, g = godel()
        assert g[0, 2] != 0
        assert g[2, 0] != 0
        # Symmetric
        assert sp.simplify(g[0, 2] - g[2, 0]) == 0

    def test_z_decouples(self) -> None:
        """The z direction is flat: g_{zz} = 1, no off-diagonal in z."""
        F, g = godel()
        assert g[3, 3] == 1
        for a in range(3):
            assert g[a, 3] == 0
            assert g[3, a] == 0

    def test_custom_omega(self) -> None:
        my_w = sp.Symbol("w0", positive=True)
        F, g = godel(omega_sym=my_w)
        assert my_w in g[0, 2].free_symbols


class TestGodelPhysics:
    """Gödel solves Einstein with rotating dust + Λ. Pipeline must close."""

    def test_einstein_tensor_computes(self) -> None:
        """Just verify the full pipeline runs without error."""
        F, g = godel()
        G = einstein_tensor(levi_civita(g), g)
        # Gödel is non-vacuum
        assert not G.is_zero()
