"""Tests for `jacopy.frame_calc.library.minkowski` (Faz 18 Stage H)."""

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
from jacopy.frame_calc.library import minkowski


class TestMinkowskiConstruction:
    def test_default_signature_minus_plus(self) -> None:
        F, g = minkowski()
        assert isinstance(F, CoordinateFrame)
        assert isinstance(g, ComponentMetric)
        assert F.dim == 4
        assert F.index_names() == ("t", "x", "y", "z")
        # Check signature -+++
        assert g[0, 0] == -1
        assert g[1, 1] == 1
        assert g[2, 2] == 1
        assert g[3, 3] == 1

    def test_alt_signature_plus_minus(self) -> None:
        F, g = minkowski(signature="+---")
        assert g[0, 0] == 1
        assert g[1, 1] == -1
        assert g[2, 2] == -1
        assert g[3, 3] == -1

    def test_invalid_signature_rejected(self) -> None:
        with pytest.raises(ValueError, match="signature"):
            minkowski(signature="++--")


class TestMinkowskiPipeline:
    """Minkowski has zero everything, Christoffel, curvature, Ricci, G."""

    def test_christoffel_zero(self) -> None:
        F, g = minkowski()
        LC = levi_civita(g)
        assert LC.is_zero()

    def test_curvature_zero(self) -> None:
        F, g = minkowski()
        R = curvature(levi_civita(g))
        assert R.is_zero()

    def test_ricci_zero(self) -> None:
        F, g = minkowski()
        Ric = ricci(levi_civita(g))
        assert Ric.is_zero()

    def test_einstein_zero(self) -> None:
        F, g = minkowski()
        G = einstein_tensor(levi_civita(g), g)
        assert G.is_vacuum()
