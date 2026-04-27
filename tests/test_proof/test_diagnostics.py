"""Tests for the diagnostics scaffolding (A.1).

Rule-specific scan behaviour lives in A.2 and gets its own test files;
this suite just pins the data types, the registry, and the empty-path
behaviour of :func:`diagnose`.
"""

from __future__ import annotations

import pytest

from jacopy.core.expr import Integer, Symbol
from jacopy.proof.diagnostics import (
    DiagnosticHint,
    DiagnosticReport,
    _RULES,
    diagnose,
    register_rule,
)


@pytest.fixture
def clean_rules():
    """Snapshot/restore the module-level rule list around each test."""
    saved = list(_RULES)
    _RULES.clear()
    try:
        yield
    finally:
        _RULES.clear()
        _RULES.extend(saved)


class TestDiagnosticHint:
    def test_minimum_fields(self):
        h = DiagnosticHint(category="stalled", message="X didn't fire")
        assert h.category == "stalled"
        assert h.location is None
        assert h.suggestion is None

    def test_optional_fields_set(self):
        loc = Symbol("x")
        h = DiagnosticHint(
            category="cat",
            message="msg",
            location=loc,
            suggestion="try Y",
        )
        assert h.location is loc
        assert h.suggestion == "try Y"


class TestDiagnosticReport:
    def test_empty_report_is_falsy(self):
        r = DiagnosticReport(residual=Integer(1))
        assert not r
        assert len(r) == 0
        assert list(r) == []

    def test_populated_report_is_truthy(self):
        r = DiagnosticReport(residual=Integer(1))
        r.hints.append(DiagnosticHint(category="a", message="b"))
        assert r
        assert len(r) == 1

    def test_by_category_filters(self):
        r = DiagnosticReport(residual=Integer(1))
        r.hints.append(DiagnosticHint(category="a", message="one"))
        r.hints.append(DiagnosticHint(category="b", message="two"))
        r.hints.append(DiagnosticHint(category="a", message="three"))
        assert [h.message for h in r.by_category("a")] == ["one", "three"]
        assert [h.message for h in r.by_category("b")] == ["two"]

    def test_format_no_hints(self):
        r = DiagnosticReport(residual=Symbol("x"))
        assert "No structural hints" in r.format()

    def test_format_renders_hints(self):
        r = DiagnosticReport(residual=Symbol("x"))
        r.hints.append(
            DiagnosticHint(
                category="stalled-d-squared",
                message="d(d(f)) survived",
                location=Symbol("f"),
                suggestion="enable d_squared_mode=\"axiom\"",
            )
        )
        text = r.format()
        assert "Residual: x" in text
        assert "stalled-d-squared" in text
        assert "d(d(f)) survived" in text
        assert "enable" in text


class TestRuleRegistry:
    def test_diagnose_empty_returns_empty(self, clean_rules):
        report = diagnose(Integer(0))
        assert not report
        assert report.residual == Integer(0)

    def test_diagnose_runs_registered_rules(self, clean_rules):
        @register_rule
        def always_fire(residual, registry, engine):
            yield DiagnosticHint(
                category="always", message=f"saw {residual._repr_inner()}"
            )

        report = diagnose(Symbol("x"))
        assert len(report) == 1
        assert report.hints[0].category == "always"
        assert "saw x" in report.hints[0].message

    def test_register_rule_returns_function(self, clean_rules):
        def f(residual, registry, engine):
            return []

        returned = register_rule(f)
        assert returned is f
        assert f in _RULES

    def test_duplicate_hints_deduped(self, clean_rules):
        loc = Symbol("x")

        @register_rule
        def r1(residual, registry, engine):
            yield DiagnosticHint(category="c", message="m", location=loc)

        @register_rule
        def r2(residual, registry, engine):
            yield DiagnosticHint(category="c", message="m", location=loc)

        report = diagnose(Integer(0))
        assert len(report) == 1


class TestPublicAPI:
    def test_exports_on_package(self):
        from jacopy.proof import (
            DiagnosticHint as H,
            DiagnosticReport as R,
            diagnose as d,
            register_rule as reg,
        )

        assert H is DiagnosticHint
        assert R is DiagnosticReport
        assert d is diagnose
        assert reg is register_rule
