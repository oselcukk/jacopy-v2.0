"""Tests for `ComponentTensor.subs` numeric evaluation (Faz 19 Chunk C.6)."""

import pytest
import sympy as sp

from jacopy.frame_calc import (
    EinsteinTensor,
    LeviCivitaConnection,
    einstein_tensor,
    levi_civita,
)
from jacopy.frame_calc.library import schwarzschild


class TestSubs:
    def test_single_substitution(self) -> None:
        """T.subs(M, 1) replaces M with 1 in every entry."""
        F, g = schwarzschild()
        LC = levi_civita(g)
        M = sp.Symbol("M", positive=True)
        LC_at_M1 = LC.subs(M, 1)
        # Γ^t_{tr} = M / (r²(1-2M/r)), at M=1: 1/(r²(1-2/r)) = 1/(r²-2r)
        r = sp.Symbol("r", positive=True)
        expected = 1 / (r ** 2 - 2 * r)
        assert sp.simplify(LC_at_M1[0, 0, 1] - expected) == 0

    def test_dict_substitution(self) -> None:
        """T.subs({sym: val}) handles dict form."""
        F, g = schwarzschild()
        LC = levi_civita(g)
        M = sp.Symbol("M", positive=True)
        r = sp.Symbol("r", positive=True)
        LC_at_point = LC.subs({M: 1, r: sp.Rational(5, 2)})
        # Γ^t_{tr} at M=1, r=5/2: 1/(25/4 - 5) = 1/(5/4) = 4/5
        assert sp.simplify(LC_at_point[0, 0, 1] - sp.Rational(4, 5)) == 0

    def test_list_of_pairs_substitution(self) -> None:
        """T.subs([(sym, val), ...]) handles list-of-pairs form."""
        F, g = schwarzschild()
        LC = levi_civita(g)
        M = sp.Symbol("M", positive=True)
        r = sp.Symbol("r", positive=True)
        LC_at = LC.subs([(M, 1), (r, 3)])
        # At M=1, r=3: Γ^t_{tr} = 1/(9 - 6) = 1/3
        assert sp.simplify(LC_at[0, 0, 1] - sp.Rational(1, 3)) == 0

    def test_preserves_type_LC(self) -> None:
        """LC.subs(...) returns a LeviCivitaConnection."""
        F, g = schwarzschild()
        LC = levi_civita(g)
        M = sp.Symbol("M", positive=True)
        out = LC.subs(M, 1)
        assert isinstance(out, LeviCivitaConnection)

    def test_preserves_type_einstein(self) -> None:
        """EinsteinTensor stays EinsteinTensor through subs."""
        F, g = schwarzschild()
        G = einstein_tensor(levi_civita(g), g)
        out = G.subs(sp.Symbol("M", positive=True), 1)
        assert isinstance(out, EinsteinTensor)

    def test_einstein_vacuum_after_subs(self) -> None:
        """G is vacuum at any specific M, r values."""
        F, g = schwarzschild()
        G = einstein_tensor(levi_civita(g), g)
        M = sp.Symbol("M", positive=True)
        r = sp.Symbol("r", positive=True)
        G_at = G.subs({M: 2, r: sp.Rational(7, 2)})
        # Still vacuum
        assert G_at.is_vacuum()
