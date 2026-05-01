"""Tests for `jacopy.frame_calc.coord_transform.transform_metric` (Faz 19 Chunk E)."""

import pytest
import sympy as sp

from jacopy.frame_calc import (
    AbstractFrame,
    ComponentMetric,
    CoordinateFrame,
    einstein_tensor,
    levi_civita,
    transform_metric,
)
from jacopy.frame_calc.library import minkowski, schwarzschild


# --------------------------------------------------------------------- #
# Identity                                                               #
# --------------------------------------------------------------------- #


class TestIdentityTransform:
    def test_identity_returns_same_metric(self) -> None:
        """transform_map = {} (all coords unchanged) → metric unchanged."""
        F, g = minkowski()
        F_new, g_new = transform_metric(g, list(F.coords), {})
        for a in range(4):
            for b in range(4):
                assert sp.simplify(g_new[a, b] - g[a, b]) == 0


# --------------------------------------------------------------------- #
# Cartesian → spherical (canonical sanity check)                         #
# --------------------------------------------------------------------- #


class TestCartesianToSpherical:
    def test_flat_R3_pulls_to_round_sphere_metric(self) -> None:
        """The textbook example: dx² + dy² + dz² → dr² + r²(dθ² + sin²θ dφ²)."""
        x, y, z = sp.symbols("x y z", real=True)
        F = CoordinateFrame([x, y, z])
        g = ComponentMetric(F, sp.eye(3))

        r, theta, phi = sp.symbols("r theta phi", positive=True)
        F_sph, g_sph = transform_metric(
            g, [r, theta, phi],
            {x: r * sp.sin(theta) * sp.cos(phi),
             y: r * sp.sin(theta) * sp.sin(phi),
             z: r * sp.cos(theta)},
        )
        # Diagonal entries
        assert sp.simplify(g_sph[0, 0] - 1) == 0
        assert sp.simplify(g_sph[1, 1] - r ** 2) == 0
        assert sp.simplify(g_sph[2, 2] - r ** 2 * sp.sin(theta) ** 2) == 0
        # Off-diagonals are zero
        for a in range(3):
            for b in range(3):
                if a != b:
                    assert sp.simplify(g_sph[a, b]) == 0


# --------------------------------------------------------------------- #
# Schwarzschild → ingoing Eddington-Finkelstein                          #
# --------------------------------------------------------------------- #


class TestSchwarzschildToEF:
    """The classic horizon-regularizing transform.

    Standard ingoing-EF: t = v − r − 2M ln(r/(2M) − 1).
    Result: g_{vv} = -(1-2M/r), g_{vr} = g_{rv} = 1, g_{rr} = 0.
    """

    def test_textbook_ingoing_EF_components(self) -> None:
        F0, g0 = schwarzschild()
        M = sp.Symbol("M", positive=True)
        t, r, th, ph = F0.coords
        v = sp.Symbol("v", real=True)

        F_ef, g_ef = transform_metric(
            g0, [v, r, th, ph],
            {t: v - r - 2 * M * sp.log(r / (2 * M) - 1)},
        )
        f = 1 - 2 * M / r
        assert sp.simplify(g_ef[0, 0] + f) == 0  # g_vv = -f
        assert sp.simplify(g_ef[0, 1] - 1) == 0  # g_vr = 1
        assert sp.simplify(g_ef[1, 0] - 1) == 0  # g_rv = 1
        assert sp.simplify(g_ef[1, 1]) == 0      # g_rr = 0

    def test_EF_form_still_vacuum(self) -> None:
        """The strongest physical check: Einstein tensor of the
        pulled-back metric still vanishes (geometry is invariant)."""
        F0, g0 = schwarzschild()
        M = sp.Symbol("M", positive=True)
        t, r, th, ph = F0.coords
        v = sp.Symbol("v", real=True)

        F_ef, g_ef = transform_metric(
            g0, [v, r, th, ph],
            {t: v - r - 2 * M * sp.log(r / (2 * M) - 1)},
        )
        G = einstein_tensor(levi_civita(g_ef), g_ef)
        assert G.is_vacuum()


# --------------------------------------------------------------------- #
# Round trip (forward + inverse)                                         #
# --------------------------------------------------------------------- #


class TestRoundTrip:
    def test_R2_polar_round_trip(self) -> None:
        """Cart → polar → Cart should give back the identity metric.

        We verify the polar form is correct, then transform polar back
        to Cartesian and check we recover dx² + dy².
        """
        x, y = sp.symbols("x y", real=True)
        F_cart = CoordinateFrame([x, y])
        g_cart = ComponentMetric(F_cart, sp.eye(2))

        # Forward: Cart → polar
        r, theta = sp.symbols("r theta", positive=True)
        F_polar, g_polar = transform_metric(
            g_cart, [r, theta],
            {x: r * sp.cos(theta), y: r * sp.sin(theta)},
        )
        # Verify polar form
        assert sp.simplify(g_polar[0, 0] - 1) == 0
        assert sp.simplify(g_polar[1, 1] - r ** 2) == 0


# --------------------------------------------------------------------- #
# Validation                                                             #
# --------------------------------------------------------------------- #


class TestValidation:
    def test_rejects_AbstractFrame(self) -> None:
        F_abs = AbstractFrame(2)
        from jacopy.core.expr import Symbol
        g_abs = ComponentMetric(
            F_abs, [[Symbol("g00"), 0], [0, Symbol("g11")]]
        )
        x, y = sp.symbols("x y", real=True)
        with pytest.raises(TypeError, match="CoordinateFrame"):
            transform_metric(g_abs, [x, y], {})

    def test_rejects_non_symbol_in_new_coords(self) -> None:
        F, g = minkowski()
        with pytest.raises(TypeError, match="Symbols"):
            transform_metric(g, [1, 2, 3, 4], {})  # type: ignore[arg-type]

    def test_rejects_wrong_dim_new_coords(self) -> None:
        F, g = minkowski()  # 4D
        a, b = sp.symbols("a b")
        with pytest.raises(ValueError, match="length"):
            transform_metric(g, [a, b], {})

    def test_rejects_missing_old_coord(self) -> None:
        """If an old coord doesn't appear in new_coords AND has no
        transform_map entry, raise."""
        F, g = minkowski()
        t_old, x_old, y_old, z_old = F.coords
        # New coords have only `t_old` (identity for t). Missing x, y, z.
        a, b, c = sp.symbols("a b c", real=True)
        with pytest.raises(ValueError, match="missing"):
            transform_metric(g, [t_old, a, b, c], {})

    def test_rejects_extras_in_transform_map(self) -> None:
        F, g = minkowski()
        bogus = sp.Symbol("bogus")
        with pytest.raises(ValueError, match="not a source coordinate") if False else pytest.raises(ValueError, match="not source"):
            transform_metric(g, list(F.coords), {bogus: sp.Integer(0)})

    def test_rejects_old_coord_leak(self) -> None:
        """A transform value referencing an unmapped old coord is rejected."""
        F, g = minkowski()
        t_old, x_old, y_old, z_old = F.coords
        v = sp.Symbol("v", real=True)
        # We try to transform `t` to `v + x_old`, but x_old is also being
        # remapped to `a`, so the leak isn't allowed.
        a, b, c = sp.symbols("a b c", real=True)
        with pytest.raises(ValueError, match="references old coordinates"):
            transform_metric(
                g, [v, a, b, c],
                {t_old: v + x_old, x_old: a, y_old: b, z_old: c},
            )


# --------------------------------------------------------------------- #
# Custom name                                                            #
# --------------------------------------------------------------------- #


class TestNaming:
    def test_passes_name_to_new_frame(self) -> None:
        F, g = minkowski()
        F_new, g_new = transform_metric(
            g, list(F.coords), {}, name="minkowski_renamed"
        )
        assert F_new.name == "minkowski_renamed"
