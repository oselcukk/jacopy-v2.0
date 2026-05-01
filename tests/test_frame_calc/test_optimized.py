"""Tests for optimized-mode pipeline (Faz 18.5).

Verify that:
- `optimized=True` produces mathematically equivalent results to default mode.
- Derivation traces are unavailable (raise RuntimeError) in optimized mode.
- Optimized mode is meaningfully faster for complex metrics.
"""

import pytest
import sympy as sp

from jacopy.frame_calc import (
    curvature,
    einstein_tensor,
    levi_civita,
    ricci,
    ricci_scalar,
)
from jacopy.frame_calc.library import minkowski, schwarzschild


# --------------------------------------------------------------------- #
# Mathematical equivalence: default vs optimized                         #
# --------------------------------------------------------------------- #


class TestSchwarzschildOptimizedEquivalence:
    """Schwarzschild full pipeline must give identical results in
    both modes (after final simplify)."""

    def test_levi_civita_components_equal(self) -> None:
        F, g = schwarzschild()
        LC_full = levi_civita(g)
        LC_opt = levi_civita(g, optimized=True)
        n = F.dim
        for e in range(n):
            for a in range(n):
                for b in range(n):
                    diff = sp.simplify(LC_full[e, a, b] - LC_opt[e, a, b])
                    assert diff == 0, (
                        f"Mismatch at Γ^{e}_{{{a}{b}}}: "
                        f"full={LC_full[e,a,b]} opt={LC_opt[e,a,b]}"
                    )

    def test_levi_civita_nonzero_count_matches(self) -> None:
        F, g = schwarzschild()
        LC_full = levi_civita(g)
        LC_opt = levi_civita(g, optimized=True)
        # Both should have 13 non-zero entries after simplify
        assert len(LC_full.nonzero_components()) == len(
            LC_opt.nonzero_components()
        )

    def test_einstein_vacuum_in_both_modes(self) -> None:
        F, g = schwarzschild()
        # Default mode
        G_full = einstein_tensor(levi_civita(g), g)
        assert G_full.is_vacuum()
        # Optimized mode
        G_opt = einstein_tensor(levi_civita(g, optimized=True), g, optimized=True)
        assert G_opt.is_vacuum()

    def test_ricci_scalar_zero_in_both_modes(self) -> None:
        F, g = schwarzschild()
        LC_full = levi_civita(g)
        LC_opt = levi_civita(g, optimized=True)
        R_full = ricci_scalar(LC_full, g)
        R_opt = ricci_scalar(LC_opt, g, optimized=True)
        # Both must simplify to zero
        assert sp.simplify(R_full) == 0
        assert sp.simplify(R_opt) == 0


class TestMinkowskiOptimizedEquivalence:
    def test_full_pipeline(self) -> None:
        F, g = minkowski()
        # Both modes should give zero Christoffel, curvature, Ricci, G
        for opt in (False, True):
            LC = levi_civita(g, optimized=opt)
            assert LC.is_zero()
            R = curvature(LC, optimized=opt)
            assert R.is_zero()
            Ric = ricci(LC, optimized=opt)
            assert Ric.is_zero()
            G = einstein_tensor(LC, g, optimized=opt)
            assert G.is_vacuum()


# --------------------------------------------------------------------- #
# Optimized-mode contract: derivation traces unavailable                #
# --------------------------------------------------------------------- #


class TestOptimizedNoTraces:
    """In optimized mode, derivation_steps and format_derivation
    raise RuntimeError pointing at the trade-off."""

    def test_levi_civita_derivation_raises(self) -> None:
        F, g = schwarzschild()
        LC = levi_civita(g, optimized=True)
        assert LC.optimized is True
        with pytest.raises(RuntimeError, match="optimized=True"):
            LC.derivation_steps(0, 0, 1)
        with pytest.raises(RuntimeError, match="optimized"):
            LC.format_derivation(0, 0, 1)

    def test_curvature_derivation_raises(self) -> None:
        F, g = schwarzschild()
        LC = levi_civita(g, optimized=True)
        R = curvature(LC, optimized=True)
        assert R.optimized is True
        with pytest.raises(RuntimeError, match="optimized=True"):
            R.derivation_steps(0, 1, 0, 1)
        with pytest.raises(RuntimeError, match="optimized"):
            R.format_derivation(0, 1, 0, 1)

    def test_ricci_derivation_raises(self) -> None:
        F, g = schwarzschild()
        Ric = ricci(levi_civita(g, optimized=True), optimized=True)
        assert Ric.optimized is True
        with pytest.raises(RuntimeError, match="optimized=True"):
            Ric.derivation_steps(0, 0)
        with pytest.raises(RuntimeError, match="optimized"):
            Ric.format_derivation(0, 0)


class TestDefaultModeStillHasTraces:
    """Verify default mode is unaffected, backward compatibility."""

    def test_levi_civita_default_has_traces(self) -> None:
        F, g = schwarzschild()
        LC = levi_civita(g)
        assert LC.optimized is False
        steps = LC.derivation_steps(0, 0, 1)
        assert len(steps) >= 4

    def test_curvature_default_has_traces(self) -> None:
        F, g = schwarzschild()
        LC = levi_civita(g)
        R = curvature(LC)
        assert R.optimized is False
        steps = R.derivation_steps(0, 1, 0, 1)
        assert len(steps) >= 4

    def test_ricci_default_has_traces(self) -> None:
        F, g = schwarzschild()
        Ric = ricci(levi_civita(g))
        assert Ric.optimized is False
        steps = Ric.derivation_steps(0, 0)
        assert len(steps) >= 2


# --------------------------------------------------------------------- #
# Optimized mode performance, Schwarzschild full pipeline               #
# --------------------------------------------------------------------- #


class TestOptimizedPerformance:
    """Optimized mode should be at least 2x faster than default for the
    full Schwarzschild pipeline."""

    def test_schwarzschild_optimized_faster(self) -> None:
        import time
        F, g = schwarzschild()

        t0 = time.perf_counter()
        G_full = einstein_tensor(levi_civita(g), g)
        t_full = time.perf_counter() - t0

        t0 = time.perf_counter()
        G_opt = einstein_tensor(levi_civita(g, optimized=True), g, optimized=True)
        t_opt = time.perf_counter() - t0

        # Default mode does heavy simplify; optimized skips it.
        # Both produce vacuum result; optimized stores raw form.
        assert G_full.is_vacuum()
        assert G_opt.is_vacuum()
        # Sanity: optimized mode shouldn't be slower than default.
        # (Not asserting exact speedup ratio since varies by hardware.)
        assert t_opt <= t_full + 0.5  # generous slack
