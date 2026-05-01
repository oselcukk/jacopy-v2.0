"""
Shared algorithm interface.

An *algorithm* in jacopy is a single rewriting step: a flatten pass, a
distribute pass, a sort-product pass, a collect-terms pass. Each lives
in its own module and exposes a free function, the free function is
what callers actually use. This module gives those algorithms a
shared :class:`Algorithm` facade and a :class:`StepResult` value type
so proof-level machinery (Faz 7+) can record what each pass did.

The plan calls for ``Algorithm.can_apply``, ``Algorithm.apply``,
``Algorithm.run``. ``run`` is the recommended entry point: it returns
a :class:`StepResult` regardless of whether the step actually changed
anything, which is what a proof unroll wants.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from jacopy.core.expr import Expr


@dataclass(frozen=True)
class StepResult:
    """Outcome of a single algorithm invocation.

    ``changed`` is computed against structural equality. A pass that
    rebuilds the tree but lands on an equal expression reports
    ``changed=False``, the fix-point driver uses this to know when to
    stop.
    """

    before: Expr
    after: Expr
    changed: bool

    @classmethod
    def unchanged(cls, expr: Expr) -> "StepResult":
        return cls(before=expr, after=expr, changed=False)

    @classmethod
    def rewrite(cls, before: Expr, after: Expr) -> "StepResult":
        return cls(before=before, after=after, changed=(before != after))


class Algorithm(ABC):
    """Base class for single-step rewriting algorithms.

    Subclasses implement :meth:`can_apply` (cheap gate) and
    :meth:`apply` (the actual rewrite). :meth:`run` combines them into
    a :class:`StepResult`. Keeping the gate separate from the rewrite
    lets callers ask "would this do anything?" without paying the
    rewrite cost.
    """

    @abstractmethod
    def can_apply(self, expr: Expr) -> bool:
        """Cheap check: would :meth:`apply` potentially rewrite ``expr``?"""

    @abstractmethod
    def apply(self, expr: Expr) -> Expr:
        """Rewrite ``expr``. May return the input unchanged."""

    def run(self, expr: Expr) -> StepResult:
        if not self.can_apply(expr):
            return StepResult.unchanged(expr)
        return StepResult.rewrite(expr, self.apply(expr))
