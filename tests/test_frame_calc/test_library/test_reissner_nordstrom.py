"""Tests for `jacopy.frame_calc.library.reissner_nordstrom` (Faz 19 Chunk B)."""

import pytest
import sympy as sp

from jacopy.frame_calc import (
    ComponentMetric,
    CoordinateFrame,
    einstein_tensor,
    levi_civita,
    ricci,
)
from jacopy.frame_calc.library import reissner_nordstrom, schwarzschild


class TestReissnerNordstromConstruction:
    def test_default_construction(self) -> None:
        F, g = reissner_nordstrom()
        assert isinstance(F, CoordinateFrame)
        assert isinstance(g, ComponentMetric)
        assert F.dim == 4
        assert F.index_names() == ("t", "r", "theta", "phi")

    def test_custom_symbols(self) -> None:
        my_M = sp.Symbol("M0", positive=True)
        my_Q = sp.Symbol("Q0", real=True)
        F, g = reissner_nordstrom(M_sym=my_M, Q_sym=my_Q)
        free = g[0, 0].free_symbols
        assert my_M in free
        assert my_Q in free

    def test_invalid_symbols_rejected(self) -> None:
        with pytest.raises(TypeError, match="Symbol"):
            reissner_nordstrom(M_sym="not a symbol")  # type: ignore[arg-type]
        with pytest.raises(TypeError, match="Symbol"):
            reissner_nordstrom(Q_sym="not a symbol")  # type: ignore[arg-type]


class TestReissnerNordstromPhysics:
    def test_not_vacuum(self) -> None:
        """RN is NOT vacuum, has EM stress-energy."""
        F, g = reissner_nordstrom()
        G = einstein_tensor(levi_civita(g), g)
        assert not G.is_vacuum()

    def test_ricci_scalar_zero(self) -> None:
        """RN's electromagnetic stress-energy is traceless: R = 0."""
        F, g = reissner_nordstrom()
        Ric = ricci(levi_civita(g))
        # Trace: g^{ab} Ric_{ab}
        g_inv = g.inverse()
        R = sp.S.Zero
        for a in range(4):
            for b in range(4):
                R += g_inv[a, b] * Ric[a, b]
        assert sp.simplify(R) == 0

    def test_reduces_to_schwarzschild_when_Q_zero(self) -> None:
        """RN with Q=0 reduces to Schwarzschild."""
        F_rn, g_rn = reissner_nordstrom()
        F_sch, g_sch = schwarzschild(M_sym=list(g_rn[0, 0].free_symbols & {sp.Symbol("M", positive=True)})[0])
        Q = sp.Symbol("Q", real=True)
        # Substitute Q -> 0 in RN g_tt and check it matches Schwarzschild
        rn_tt = g_rn[0, 0].subs(Q, 0)
        assert sp.simplify(rn_tt - g_sch[0, 0]) == 0
