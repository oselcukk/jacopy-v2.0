"""Tests for `jacopy.frame_calc.library.bianchi` (Faz 19 Chunk B)."""

import pytest
import sympy as sp

from jacopy.frame_calc import (
    ComponentMetric,
    CoordinateFrame,
    einstein_tensor,
    levi_civita,
)
from jacopy.frame_calc.library import (
    bianchi_I,
    bianchi_IX,
    bianchi_V,
)


class TestBianchiI:
    def test_default_construction(self) -> None:
        F, g = bianchi_I()
        assert isinstance(F, CoordinateFrame)
        assert isinstance(g, ComponentMetric)
        assert F.dim == 4
        assert F.index_names() == ("t", "x", "y", "z")

    def test_diagonal_metric(self) -> None:
        F, g = bianchi_I()
        for a in range(4):
            for b in range(4):
                if a != b:
                    assert g[a, b] == 0

    def test_isotropic_limit_yields_flat_einstein(self) -> None:
        """When a = b = c = 1, Bianchi I reduces to Minkowski → vacuum."""
        F, g = bianchi_I(a_func=sp.Integer(1), b_func=sp.Integer(1), c_func=sp.Integer(1))
        G = einstein_tensor(levi_civita(g), g)
        assert G.is_vacuum()

    def test_dynamic_scale_factors_nonvacuum(self) -> None:
        """Generic a(t), b(t), c(t) → non-vacuum (anisotropic stress)."""
        F, g = bianchi_I()
        G = einstein_tensor(levi_civita(g), g)
        assert not G.is_zero()


class TestBianchiV:
    def test_default_construction(self) -> None:
        F, g = bianchi_V()
        assert isinstance(F, CoordinateFrame)
        assert F.dim == 4

    def test_has_x_dependence_in_metric(self) -> None:
        """Bianchi V's defining feature: e^{2x} factors in y, z parts."""
        F, g = bianchi_V()
        x = sp.Symbol("x", real=True)
        # g_{yy} should have exp(2x) factor
        assert x in g[2, 2].free_symbols


class TestBianchiIX:
    """Bianchi IX is the most complex; just smoke-test construction."""

    def test_default_construction(self) -> None:
        F, g = bianchi_IX()
        assert isinstance(F, CoordinateFrame)
        assert F.dim == 4
        assert F.index_names() == ("t", "psi", "theta", "phi")

    def test_christoffel_computes_optimised(self) -> None:
        """Just verify the pipeline doesn't blow up, Mixmaster is famously
        complex, so we use optimized=True."""
        F, g = bianchi_IX()
        LC = levi_civita(g, optimized=True)
        # Must have non-trivial Christoffels
        assert len(LC.nonzero_components(simplify=False)) > 0
