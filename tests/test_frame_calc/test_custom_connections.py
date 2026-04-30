"""Tests for `jacopy.frame_calc.custom_connections` (Faz 19 Chunk D)."""

import pytest
import sympy as sp

from jacopy.frame_calc import (
    ComponentMetric,
    ComponentTensor,
    CoordinateFrame,
    connection_with_torsion,
    levi_civita,
    projective_connection,
    torsion,
    weyl_connection,
)


# --------------------------------------------------------------------- #
# connection_with_torsion                                                #
# --------------------------------------------------------------------- #


class TestConnectionWithTorsion:
    """The defining property: torsion(connection_with_torsion(g, T)) == T."""

    def test_2d_round_trip(self) -> None:
        """In 2D: prescribe T, build connection, recover T."""
        t, x = sp.symbols("t x")
        F = CoordinateFrame([t, x])
        g = ComponentMetric(F, sp.diag(-1, 1))
        f = sp.Function("f")(t)
        T_arr = sp.MutableDenseNDimArray.zeros(2, 2, 2)
        T_arr[1, 0, 1] = f
        T_arr[1, 1, 0] = -f
        T_in = ComponentTensor(F, T_arr, signature=(1, 2))

        conn = connection_with_torsion(g, T_in)
        T_out = torsion(conn)

        for a in range(2):
            for b in range(2):
                for c in range(2):
                    diff = sp.simplify(T_out[a, b, c] - T_in[a, b, c])
                    assert diff == 0, f"T[{a},{b},{c}] mismatch: {diff}"

    def test_3d_round_trip(self) -> None:
        """3D check with two independent torsion components."""
        t, x, y = sp.symbols("t x y")
        F = CoordinateFrame([t, x, y])
        g = ComponentMetric(F, sp.diag(-1, 1, 1))
        T = sp.MutableDenseNDimArray.zeros(3, 3, 3)
        T[2, 0, 1] = sp.Function("a")(t)
        T[2, 1, 0] = -sp.Function("a")(t)
        T[1, 0, 2] = sp.Function("b")(x)
        T[1, 2, 0] = -sp.Function("b")(x)
        T_in = ComponentTensor(F, T, signature=(1, 2))

        conn = connection_with_torsion(g, T_in)
        T_out = torsion(conn)

        for a in range(3):
            for b in range(3):
                for c in range(3):
                    diff = sp.simplify(T_out[a, b, c] - T_in[a, b, c])
                    assert diff == 0, f"T[{a},{b},{c}] mismatch: {diff}"

    def test_zero_torsion_yields_levi_civita(self) -> None:
        """Pass zero torsion → connection should equal Levi-Civita."""
        t, x = sp.symbols("t x")
        F = CoordinateFrame([t, x])
        g = ComponentMetric(F, sp.diag(-1, 1))
        T_zero = ComponentTensor(
            F,
            sp.MutableDenseNDimArray.zeros(2, 2, 2),
            signature=(1, 2),
        )
        conn = connection_with_torsion(g, T_zero)
        LC = levi_civita(g)
        for a in range(2):
            for b in range(2):
                for c in range(2):
                    assert sp.simplify(conn[a, b, c] - LC[a, b, c]) == 0

    def test_signature_validation(self) -> None:
        t, x = sp.symbols("t x")
        F = CoordinateFrame([t, x])
        g = ComponentMetric(F, sp.diag(-1, 1))
        # Wrong-rank tensor
        bad = ComponentTensor(
            F, sp.eye(2), signature=(0, 2)
        )
        with pytest.raises(ValueError, match="signature"):
            connection_with_torsion(g, bad)


# --------------------------------------------------------------------- #
# weyl_connection                                                        #
# --------------------------------------------------------------------- #


class TestWeylConnection:
    def test_torsion_free(self) -> None:
        """Weyl connection is torsion-free by construction."""
        t, x = sp.symbols("t x")
        F = CoordinateFrame([t, x])
        g = ComponentMetric(F, sp.diag(-1, 1))
        W = [sp.Function("W0")(t), sp.Function("W1")(x)]
        conn = weyl_connection(g, W)
        T = torsion(conn)
        assert T.is_zero()

    def test_zero_W_gives_levi_civita(self) -> None:
        """W = 0 → connection equals Levi-Civita."""
        t, x = sp.symbols("t x")
        F = CoordinateFrame([t, x])
        g = ComponentMetric(F, sp.diag(-1, 1))
        conn = weyl_connection(g, [0, 0])
        LC = levi_civita(g)
        for a in range(2):
            for b in range(2):
                for c in range(2):
                    assert sp.simplify(conn[a, b, c] - LC[a, b, c]) == 0

    def test_W_length_validation(self) -> None:
        t, x = sp.symbols("t x")
        F = CoordinateFrame([t, x])
        g = ComponentMetric(F, sp.diag(-1, 1))
        with pytest.raises(ValueError, match="length"):
            weyl_connection(g, [0])  # too short


# --------------------------------------------------------------------- #
# projective_connection                                                  #
# --------------------------------------------------------------------- #


class TestProjectiveConnection:
    def test_torsion_free(self) -> None:
        """Projective deformation preserves symmetry → no torsion."""
        t, x = sp.symbols("t x")
        F = CoordinateFrame([t, x])
        g = ComponentMetric(F, sp.diag(-1, 1))
        X = [sp.Function("X0")(t), sp.Function("X1")(x)]
        conn = projective_connection(g, X)
        T = torsion(conn)
        assert T.is_zero()

    def test_zero_X_gives_levi_civita(self) -> None:
        t, x = sp.symbols("t x")
        F = CoordinateFrame([t, x])
        g = ComponentMetric(F, sp.diag(-1, 1))
        conn = projective_connection(g, [0, 0])
        LC = levi_civita(g)
        for a in range(2):
            for b in range(2):
                for c in range(2):
                    assert sp.simplify(conn[a, b, c] - LC[a, b, c]) == 0

    def test_difference_from_LC_is_projective_form(self) -> None:
        r"""The deformation is exactly δ^a_b X_c + δ^a_c X_b."""
        t, x = sp.symbols("t x")
        F = CoordinateFrame([t, x])
        g = ComponentMetric(F, sp.diag(-1, 1))
        X = [sp.Symbol("X0"), sp.Symbol("X1")]
        conn = projective_connection(g, X)
        LC = levi_civita(g)

        for a in range(2):
            for b in range(2):
                for c in range(2):
                    expected_def = (
                        (X[c] if a == b else 0)
                        + (X[b] if a == c else 0)
                    )
                    diff = sp.simplify(
                        conn[a, b, c] - LC[a, b, c] - expected_def
                    )
                    assert diff == 0
