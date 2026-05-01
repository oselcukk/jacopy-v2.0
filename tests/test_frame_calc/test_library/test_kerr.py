"""Tests for `jacopy.frame_calc.library.kerr` (Faz 18 Stage H).

Kerr's full pipeline (`einstein_tensor.is_vacuum()`) is computationally
heavy (~60s); the vacuum check is **not** included as a default test.
The construction-only tests below verify the metric shape, types, and
the Schwarzschild limit (`a → 0`).
"""

import pytest
import sympy as sp

from jacopy.frame_calc import (
    ComponentMetric,
    CoordinateFrame,
    levi_civita,
)
from jacopy.frame_calc.library import kerr, schwarzschild


class TestKerrConstruction:
    def test_default_construction(self) -> None:
        F, g = kerr()
        assert isinstance(F, CoordinateFrame)
        assert isinstance(g, ComponentMetric)
        assert F.dim == 4
        assert F.index_names() == ("t", "r", "theta", "phi")
        assert F.name == "kerr"

    def test_off_diagonal_dt_dphi_present(self) -> None:
        """Frame dragging: g_{tφ} ≠ 0."""
        F, g = kerr()
        assert g[0, 3] != 0
        # And symmetric
        assert sp.simplify(g[0, 3] - g[3, 0]) == 0

    def test_a_zero_limit_recovers_schwarzschild(self) -> None:
        """Setting `a = 0` in Kerr should match Schwarzschild's diagonal."""
        F_k, g_k = kerr()
        F_s, g_s = schwarzschild()
        a = sp.Symbol("a", real=True)

        # g_tt in Kerr at a=0
        kerr_tt_at_a_zero = sp.simplify(g_k[0, 0].subs(a, 0))
        # Schwarzschild g_tt
        schwarz_tt = g_s[0, 0]
        assert sp.simplify(kerr_tt_at_a_zero - schwarz_tt) == 0

        # g_rr similarly
        kerr_rr_at_a_zero = sp.simplify(g_k[1, 1].subs(a, 0))
        schwarz_rr = g_s[1, 1]
        assert sp.simplify(kerr_rr_at_a_zero - schwarz_rr) == 0

        # g_{tφ} at a=0 should be 0
        assert sp.simplify(g_k[0, 3].subs(a, 0)) == 0

    def test_custom_M_and_a_symbols(self) -> None:
        my_M = sp.Symbol("M0", positive=True)
        my_a = sp.Symbol("a0", real=True)
        F, g = kerr(M_sym=my_M, a_sym=my_a)
        free = g[0, 0].free_symbols
        assert my_M in free
        assert my_a in free

    def test_invalid_M_rejected(self) -> None:
        with pytest.raises(TypeError, match="Symbol"):
            kerr(M_sym="not a symbol")  # type: ignore[arg-type]

    def test_invalid_a_rejected(self) -> None:
        with pytest.raises(TypeError, match="Symbol"):
            kerr(a_sym="not a symbol")  # type: ignore[arg-type]


class TestKerrVacuumOptimized:
    """Kerr's vacuum solution `G ≡ 0` verified in optimized mode.

    Default mode timed out at 180 s (Ricci alone never finished,
    sympy.simplify on Kerr-complexity expressions blows up).
    Optimized mode (skip mid-formula simplify) completes the entire
    pipeline in ~23 seconds, and the resulting Einstein-tensor
    components are **literal zero** in raw form, SymPy's basic
    arithmetic alone collapses the cancellations cleanly. No
    sympy.simplify is needed for the vacuum check.

    This test gates an important architectural result: the package
    handles Kerr-class research-grade metrics in feasible time.
    """

    def test_kerr_vacuum_in_optimized_mode(self) -> None:
        from jacopy.frame_calc import einstein_tensor

        F, g = kerr()
        G = einstein_tensor(
            levi_civita(g, optimized=True), g, optimized=True
        )
        assert G.is_vacuum()
        # Stronger claim: every entry is literal zero (raw form,
        # no simplify needed). Documents the architectural win.
        assert G.is_zero(simplify=False)
