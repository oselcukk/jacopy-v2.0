"""
Residual diagnostics for failed proof attempts.

When :class:`ExpandAndSimplify` (or any derived strategy) fails to drive
the obstruction ``Sum(lhs, Neg(rhs))`` to :class:`Integer` ``0``, the
surviving *residual* carries structural information about *why* the
pipeline stalled: a definition never fired, a factor's grading was
missing, an engine pass wasn't interleaved properly. Historically the
user (and the agent driving the proof) had to eyeball that residual and
trace the missing rewrite by hand, a five-fix hunt to close the Cartan
``verify()`` suite lived through exactly this loop.

This module mechanises the first pass of that analysis. :func:`diagnose`
walks the residual and emits :class:`DiagnosticHint` records for each
*stalled shape* it recognises, an ``Act(d_like, Act(d_like, x))`` that
would have cancelled under ``d² = 0``, an ``ι_V(df)`` where ``V`` is a
bracket-like sum the current :class:`IotaOnExactOneFormDefinition`
wouldn't match, an unclassified factor reaching the Koszul sort layer,
and so on. The report is intended as a hypothesis list, not a proof: a
hint says "this shape *could* be why the proof stalled", and it's the
caller's (or the agent's) job to decide whether the suggested rewrite
is the real gap.

The design is open-ended on purpose: new rules plug in via
:func:`register_rule` and each rule is a pure function on the residual
tree, so adding coverage is a one-file change and never mutates the
proof engine.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Iterable, List, Optional, Tuple

from jacopy.core.expr import Expr
from jacopy.core.registry import PropertyRegistry
from jacopy.proof.expansion import ExpansionEngine


# --------------------------------------------------------------------- #
# Data types                                                             #
# --------------------------------------------------------------------- #


@dataclass(frozen=True)
class DiagnosticHint:
    """A single structural observation about a stalled residual.

    ``category`` is a short machine key (``"stalled-d-squared"``,
    ``"unreduced-iota"``, ``"unclassified-factor"``, …) so programmatic
    callers can filter without parsing free text. ``location`` is the
    offending sub-expression, rendered with ``_repr_inner()`` in
    :meth:`DiagnosticReport.format` but kept as an :class:`Expr` here
    so callers can re-inspect it. ``message`` is the prose
    description; ``suggestion`` is a hint at the fix (optional, not
    every hint has one).
    """

    category: str
    message: str
    location: Optional[Expr] = None
    suggestion: Optional[str] = None


@dataclass
class DiagnosticReport:
    """Collection of :class:`DiagnosticHint` observations on a residual.

    Iterating a report yields its hints; truthiness reflects whether
    any hint fired. :meth:`format` produces a human-readable string
    suitable for appending to a :class:`ProofFailure` message.
    """

    residual: Expr
    hints: List[DiagnosticHint] = field(default_factory=list)

    def __bool__(self) -> bool:
        return bool(self.hints)

    def __iter__(self):
        return iter(self.hints)

    def __len__(self) -> int:
        return len(self.hints)

    def by_category(self, category: str) -> List[DiagnosticHint]:
        return [h for h in self.hints if h.category == category]

    def format(self) -> str:
        if not self.hints:
            return (
                f"No structural hints for residual "
                f"{self.residual._repr_inner()}."
            )
        lines = [
            f"Residual: {self.residual._repr_inner()}",
            f"Hints ({len(self.hints)}):",
        ]
        for i, hint in enumerate(self.hints, start=1):
            loc = (
                f" at {hint.location._repr_inner()}"
                if hint.location is not None
                else ""
            )
            lines.append(f"  {i}. [{hint.category}]{loc}: {hint.message}")
            if hint.suggestion is not None:
                lines.append(f"     suggestion: {hint.suggestion}")
        return "\n".join(lines)


# --------------------------------------------------------------------- #
# Rule registry                                                          #
# --------------------------------------------------------------------- #


DiagnosticRule = Callable[
    [Expr, Optional[PropertyRegistry], Optional[ExpansionEngine]],
    Iterable[DiagnosticHint],
]


_RULES: List[DiagnosticRule] = []


def register_rule(rule: DiagnosticRule) -> DiagnosticRule:
    """Register a diagnostic rule. Returns the rule for decorator use."""
    _RULES.append(rule)
    return rule


def _registered_rules() -> Tuple[DiagnosticRule, ...]:
    return tuple(_RULES)


# --------------------------------------------------------------------- #
# Public entry                                                           #
# --------------------------------------------------------------------- #


def diagnose(
    residual: Expr,
    *,
    registry: Optional[PropertyRegistry] = None,
    engine: Optional[ExpansionEngine] = None,
) -> DiagnosticReport:
    """Scan ``residual`` with every registered rule.

    Rules run in registration order; duplicate hints (same category +
    location) are de-duplicated so callers aren't spammed when two
    rules overlap on the same stalled shape. ``registry`` and
    ``engine`` are passed through, rules that need grading or
    definition-coverage context read them, the rest ignore them.
    """
    report = DiagnosticReport(residual=residual)
    seen: set = set()
    for rule in _RULES:
        for hint in rule(residual, registry, engine):
            key = (hint.category, repr(hint.location))
            if key in seen:
                continue
            seen.add(key)
            report.hints.append(hint)
    return report
