"""Integration tests, ``ProofFailure.report`` carries diagnostic hints
when :class:`ExpandAndSimplify` stalls.
"""

from __future__ import annotations

import pytest

from jacopy.algebra.derivation import Act
from jacopy.calculus.exterior_d import d
from jacopy.core.expr import Integer, Symbol
from jacopy.core.properties import Graded
from jacopy.core.registry import PropertyRegistry
from jacopy.proof import (
    ActOverSumOpDefinition,
    DiagnosticReport,
    ExpandAndSimplify,
    ExpansionEngine,
    IotaOnZeroFormDefinition,
    LieDerivativeCartanDefinition,
    ProofFailure,
)


def _engine_without_d_squared(registry):
    """Engine with the Cartan axioms minus ``d² = 0``, lets the
    diagnostic surface residual ``Act(d, Act(d, f))`` shapes that the
    default engine would have eliminated."""
    return ExpansionEngine(
        [
            LieDerivativeCartanDefinition(),
            ActOverSumOpDefinition(),
            IotaOnZeroFormDefinition(registry=registry),
        ]
    )


class TestProofFailureCarriesReport:
    def test_report_attached_on_stalled_d_squared(self):
        reg = PropertyRegistry()
        f = Symbol("f")
        reg.declare(f, Graded(degree=0))

        strat = ExpandAndSimplify()
        eng = _engine_without_d_squared(reg)

        with pytest.raises(ProofFailure) as info:
            strat.prove(
                Act(d, Act(d, f)), Integer(0),
                registry=reg, engine=eng,
            )

        exc = info.value
        assert isinstance(exc.report, DiagnosticReport)
        assert exc.report, "expected at least one hint"
        cats = {h.category for h in exc.report.hints}
        assert "stalled-d-squared" in cats

    def test_str_includes_report_text(self):
        reg = PropertyRegistry()
        f = Symbol("f")
        reg.declare(f, Graded(degree=0))

        strat = ExpandAndSimplify()
        eng = _engine_without_d_squared(reg)

        with pytest.raises(ProofFailure) as info:
            strat.prove(
                Act(d, Act(d, f)), Integer(0),
                registry=reg, engine=eng,
            )

        text = str(info.value)
        assert "Residual:" in text
        assert "stalled-d-squared" in text

    def test_empty_report_does_not_pollute_str(self):
        """Legacy path, a ProofFailure constructed without a report
        should still render the bare message (no trailing empty block)."""
        exc = ProofFailure("plain message")
        assert str(exc) == "plain message"
        assert exc.report is None
