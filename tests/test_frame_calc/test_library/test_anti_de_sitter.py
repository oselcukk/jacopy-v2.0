"""Tests for `jacopy.frame_calc.library.anti_de_sitter` (Faz 19 Chunk B)."""

import pytest
import sympy as sp

from jacopy.frame_calc import (
    ComponentMetric,
    CoordinateFrame,
    levi_civita,
    ricci,
)
from jacopy.frame_calc.library import anti_de_sitter


class TestAntiDeSitterConstruction:
    def test_default_construction(self) -> None:
        F, g = anti_de_sitter()
        assert isinstance(F, CoordinateFrame)
        assert isinstance(g, ComponentMetric)
        assert F.dim == 4

    def test_custom_Lambda(self) -> None:
        my_L = sp.Symbol("Lambda0", positive=True)
        F, g = anti_de_sitter(Lambda_sym=my_L)
        assert my_L in g[0, 0].free_symbols


class TestAntiDeSitterPhysics:
    """AdS is maximally symmetric with Ric proportional to g.

    Sign opposite to dS in our convention: ``Ric_{ab} = +Λ g_{ab}`` for AdS.
    """

    def test_ricci_proportional_to_metric(self) -> None:
        F, g = anti_de_sitter()
        Ric = ricci(levi_civita(g))
        Lambda = sp.Symbol("Lambda", positive=True)
        # In our convention: Ric = +Λ g (opposite sign vs dS)
        for a in range(4):
            for b in range(4):
                diff = sp.simplify(Ric[a, b] - Lambda * g[a, b])
                assert diff == 0, f"Ric[{a},{b}] - Λg[{a},{b}] = {diff}"

    def test_constant_ricci_scalar(self) -> None:
        F, g = anti_de_sitter()
        Ric = ricci(levi_civita(g))
        g_inv = g.inverse()
        R = sp.S.Zero
        for a in range(4):
            for b in range(4):
                R += g_inv[a, b] * Ric[a, b]
        R_simplified = sp.simplify(R)
        r = sp.Symbol("r", positive=True)
        assert sp.diff(R_simplified, r) == 0
