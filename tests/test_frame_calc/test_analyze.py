"""Tests for `jacopy.frame_calc.analyze.analyze_metric` (Faz 19 Chunk C.4)."""

import sympy as sp

from jacopy.frame_calc import (
    ComponentMetric,
    ComponentMetricInverse,
    CoordinateFrame,
    EinsteinTensor,
    LeviCivitaConnection,
    analyze_metric,
)


class TestAnalyzeMetricMinkowski:
    def test_returns_complete_dict(self) -> None:
        t, x, y, z = sp.symbols("t x y z")
        out = analyze_metric(sp.diag(-1, 1, 1, 1), [t, x, y, z])
        for key in (
            "frame",
            "metric",
            "inverse",
            "christoffel",
            "riemann",
            "ricci",
            "ricci_scalar",
            "einstein",
            "kretschmann",
            "ricci_squared",
            "is_vacuum",
        ):
            assert key in out, f"missing key {key!r}"

    def test_minkowski_results(self) -> None:
        t, x, y, z = sp.symbols("t x y z")
        out = analyze_metric(sp.diag(-1, 1, 1, 1), [t, x, y, z])
        assert isinstance(out["frame"], CoordinateFrame)
        assert isinstance(out["metric"], ComponentMetric)
        assert isinstance(out["inverse"], ComponentMetricInverse)
        assert isinstance(out["christoffel"], LeviCivitaConnection)
        assert isinstance(out["einstein"], EinsteinTensor)
        # Flat space invariants
        assert sp.simplify(out["kretschmann"]) == 0
        assert sp.simplify(out["ricci_squared"]) == 0
        assert out["is_vacuum"] is True


class TestAnalyzeMetricSchwarzschild:
    """End-to-end paper-grade workflow on Schwarzschild via the helper."""

    def test_textbook_kretschmann(self) -> None:
        t, r, th, ph = sp.symbols("t r theta phi")
        M = sp.Symbol("M", positive=True)
        f = 1 - 2 * M / r
        out = analyze_metric(
            sp.diag(-f, 1 / f, r ** 2, r ** 2 * sp.sin(th) ** 2),
            [t, r, th, ph],
        )
        expected = 48 * M ** 2 / r ** 6
        assert sp.simplify(out["kretschmann"] - expected) == 0
        assert out["is_vacuum"] is True
        assert sp.simplify(out["ricci_squared"]) == 0


class TestAnalyzeMetricCustomName:
    def test_passes_name_to_frame(self) -> None:
        t, x = sp.symbols("t x")
        out = analyze_metric(
            sp.diag(-1, 1), [t, x], name="my_2d_minkowski"
        )
        assert out["frame"].name == "my_2d_minkowski"
