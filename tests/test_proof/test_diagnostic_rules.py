"""Rule-level tests for the built-in diagnostic scanners.

Each test constructs the exact stalled shape the rule is meant to
catch, runs :func:`diagnose`, and asserts the right category fires.
A "negative" test per rule pins the non-matching case so the rule
doesn't over-flag.
"""

from __future__ import annotations

import pytest

from jacopy.algebra.derivation import Act, Derivation
from jacopy.calculus.exterior_d import d
from jacopy.calculus.interior import interior
from jacopy.calculus.lie_derivative import lie_derivative
from jacopy.core.expr import Integer, Neg, Product, Sum, Symbol
from jacopy.core.properties import Graded
from jacopy.core.registry import PropertyRegistry
from jacopy.proof import diagnose


@pytest.fixture
def reg():
    r = PropertyRegistry()
    f = Symbol("f")
    r.declare(f, Graded(degree=0))
    return r, f


# --------------------------------------------------------------------- #
# stalled-d-squared                                                      #
# --------------------------------------------------------------------- #


class TestStalledDSquared:
    def test_fires_on_ddx(self, reg):
        r, f = reg
        residual = Act(d, Act(d, f))
        report = diagnose(residual, registry=r)
        assert report.by_category("stalled-d-squared")

    def test_skips_plain_dx(self, reg):
        r, f = reg
        report = diagnose(Act(d, f), registry=r)
        assert not report.by_category("stalled-d-squared")

    def test_skips_even_derivation(self, reg):
        """``L_X`` is degree 0, two nested L's are not a d² violation."""
        r, f = reg
        L = Derivation("L", degree=0)
        report = diagnose(Act(L, Act(L, f)), registry=r)
        assert not report.by_category("stalled-d-squared")


# --------------------------------------------------------------------- #
# stalled-act-over-zero                                                  #
# --------------------------------------------------------------------- #


class TestStalledActOverZero:
    def test_fires_on_act_op_zero(self, reg):
        r, _ = reg
        residual = Act(d, Integer(0))
        report = diagnose(residual, registry=r)
        assert report.by_category("stalled-act-over-zero")

    def test_skips_act_op_nonzero(self, reg):
        r, f = reg
        report = diagnose(Act(d, f), registry=r)
        assert not report.by_category("stalled-act-over-zero")


# --------------------------------------------------------------------- #
# stalled-act-over-neg-op                                                #
# --------------------------------------------------------------------- #


class TestStalledActOverNegOp:
    def test_fires_on_neg_op(self, reg):
        r, f = reg
        residual = Act(Neg(d), f)
        report = diagnose(residual, registry=r)
        assert report.by_category("stalled-act-over-neg-op")

    def test_skips_plain_op(self, reg):
        r, f = reg
        report = diagnose(Act(d, f), registry=r)
        assert not report.by_category("stalled-act-over-neg-op")


# --------------------------------------------------------------------- #
# unreduced-iota-on-df                                                   #
# --------------------------------------------------------------------- #


class TestUnreducedIotaOnDf:
    def test_fires_on_bracket_vector_field(self, reg):
        r, f = reg
        X = Derivation("X", degree=0)
        Y = Derivation("Y", degree=0)
        # Build ι_{X*Y - Y*X}(d(f)) exactly as the B.2 residual would.
        vf = Sum(Product(X, Y), Neg(Product(Y, X)))
        iota_vf = interior(vf)
        residual = Act(iota_vf, Act(d, f))
        report = diagnose(residual, registry=r)
        assert report.by_category("unreduced-iota-on-df")

    def test_skips_plain_vector_field(self, reg):
        """``ι_X(df)``, plain Derivation is handled by the default
        definition; diagnostic should stay quiet."""
        r, f = reg
        X = Derivation("X", degree=0)
        residual = Act(interior(X), Act(d, f))
        report = diagnose(residual, registry=r)
        assert not report.by_category("unreduced-iota-on-df")


# --------------------------------------------------------------------- #
# symbol-vector-field                                                    #
# --------------------------------------------------------------------- #


class TestSymbolVectorField:
    def test_fires_on_iota_of_symbol(self, reg):
        """``ι_X(df)`` with ``X = Symbol("X")``, pairing never fires."""
        r, f = reg
        X = Symbol("X")
        residual = Act(interior(X), Act(d, f))
        report = diagnose(residual, registry=r)
        hints = report.by_category("symbol-vector-field")
        assert hints
        # Suggestion should name the offending symbol.
        assert any("X" in (h.suggestion or "") for h in hints)

    def test_fires_on_lie_derivative_of_symbol(self, reg):
        """``L_X(f)`` with ``X = Symbol("X")``, same gate."""
        r, f = reg
        X = Symbol("X")
        residual = Act(lie_derivative(X), f)
        report = diagnose(residual, registry=r)
        assert report.by_category("symbol-vector-field")

    def test_fires_on_bracket_of_symbols(self, reg):
        """Bracket ``X*Y − Y*X`` of Symbols, still flags, names deduped."""
        r, f = reg
        X = Symbol("X")
        Y = Symbol("Y")
        vf = Sum(Product(X, Y), Neg(Product(Y, X)))
        residual = Act(interior(vf), Act(d, f))
        report = diagnose(residual, registry=r)
        hints = report.by_category("symbol-vector-field")
        assert hints
        suggestion = hints[0].suggestion or ""
        # Both Symbols named; each listed once (no "X, Y, Y, X").
        assert "X" in suggestion and "Y" in suggestion
        assert suggestion.count("X") == 1
        assert suggestion.count("Y") == 1

    def test_skips_plain_derivation(self, reg):
        """``ι_X(df)`` with ``X = Derivation(...)``, pairing fires, no hint."""
        r, f = reg
        X = Derivation("X", degree=0)
        residual = Act(interior(X), Act(d, f))
        report = diagnose(residual, registry=r)
        assert not report.by_category("symbol-vector-field")

    def test_skips_bracket_of_derivations(self, reg):
        """Derivation bracket lands on :func:`unreduced_iota_on_df`, not here."""
        r, f = reg
        X = Derivation("X", degree=0)
        Y = Derivation("Y", degree=0)
        vf = Sum(Product(X, Y), Neg(Product(Y, X)))
        residual = Act(interior(vf), Act(d, f))
        report = diagnose(residual, registry=r)
        assert not report.by_category("symbol-vector-field")

    def test_skips_operators_without_vector_field(self, reg):
        """``Act(d, f)``, outer op has no vector_field slot, no hint."""
        r, f = reg
        report = diagnose(Act(d, f), registry=r)
        assert not report.by_category("symbol-vector-field")


# --------------------------------------------------------------------- #
# unclassified-factor                                                    #
# --------------------------------------------------------------------- #


class TestUnclassifiedFactor:
    def test_fires_on_undeclared_symbol_factor(self):
        r = PropertyRegistry()
        f = Symbol("f")  # deliberately NOT declared
        g = Symbol("g")
        r.declare(g, Graded(degree=0))
        residual = Product(f, g)
        report = diagnose(residual, registry=r)
        hints = report.by_category("unclassified-factor")
        assert hints
        # The offender is f, not g.
        assert any(h.location == f for h in hints)
        assert not any(h.location == g for h in hints)

    def test_skips_fully_classified_product(self, reg):
        r, f = reg
        g = Symbol("g")
        r.declare(g, Graded(degree=0))
        report = diagnose(Product(f, g), registry=r)
        assert not report.by_category("unclassified-factor")

    def test_derivation_factor_classifies(self, reg):
        r, f = reg
        X = Derivation("X", degree=0)
        report = diagnose(Product(X, f), registry=r)
        assert not report.by_category("unclassified-factor")

    def test_numeric_factor_classifies(self, reg):
        r, f = reg
        report = diagnose(Product(Integer(2), f), registry=r)
        assert not report.by_category("unclassified-factor")


# --------------------------------------------------------------------- #
# Integration smoke                                                      #
# --------------------------------------------------------------------- #


class TestIntegration:
    def test_empty_residual_no_hints(self, reg):
        r, _ = reg
        assert not diagnose(Integer(0), registry=r)

    def test_multiple_rules_compose(self, reg):
        r, f = reg
        # Residual contains both a stalled d² and an Act(op, 0),
        # both rules should fire on the same residual.
        residual = Sum(Act(d, Act(d, f)), Act(d, Integer(0)))
        report = diagnose(residual, registry=r)
        cats = {h.category for h in report.hints}
        assert "stalled-d-squared" in cats
        assert "stalled-act-over-zero" in cats
