"""
Proof transcript, an ordered sequence of :class:`ProofStep`.

A :class:`ProofChain` is a flat list; nesting lives inside each step's
``children``. That split keeps the chain a simple linear narrative at
the top level while still allowing a strategy to record its internal
work hierarchically. Strategies append steps as they run; the caller
inspects ``initial``/``final`` to see the starting and ending
expressions.

Verbosity choices for :meth:`format` are intentionally coarse,
``"compact"`` prints one line per top-level step and hides children,
``"full"`` renders every step with its children tree. Richer display
modes (LaTeX, coloured terminal) belong in the Faz 8 display layer and
should read the chain as data rather than extending this formatter.
"""

from __future__ import annotations

from typing import Iterable, Iterator, List, Tuple

from jacopy.core.expr import Expr
from jacopy.proof.step import ProofStep


class ProofChain:
    """Ordered list of :class:`ProofStep` forming a single proof transcript."""

    __slots__ = ("_steps",)

    def __init__(self, steps: Iterable[ProofStep] = ()) -> None:
        self._steps: List[ProofStep] = []
        for s in steps:
            self.append(s)

    def append(self, step: ProofStep) -> None:
        if not isinstance(step, ProofStep):
            raise TypeError("ProofChain.append expects a ProofStep")
        self._steps.append(step)

    def extend(self, steps: Iterable[ProofStep]) -> None:
        for s in steps:
            self.append(s)

    @property
    def steps(self) -> Tuple[ProofStep, ...]:
        return tuple(self._steps)

    @property
    def initial(self) -> Expr:
        """First step's ``before``. Raises if the chain is empty."""
        if not self._steps:
            raise ValueError("empty chain has no initial expression")
        return self._steps[0].before

    @property
    def final(self) -> Expr:
        """Last step's ``after``. Raises if the chain is empty."""
        if not self._steps:
            raise ValueError("empty chain has no final expression")
        return self._steps[-1].after

    def __iter__(self) -> Iterator[ProofStep]:
        return iter(self._steps)

    def __len__(self) -> int:
        return len(self._steps)

    def __bool__(self) -> bool:
        return bool(self._steps)

    def format(self, verbosity: str = "full") -> str:
        """Render the chain as text.

        * ``"compact"``, one line per top-level step, children hidden.
        * ``"full"``, every step with its full children tree.
        """
        if verbosity == "compact":
            return "\n".join(
                f"[{s.rule}] {s.before._repr_inner()} → {s.after._repr_inner()}"
                for s in self._steps
            )
        if verbosity == "full":
            return "\n".join(s.format() for s in self._steps)
        raise ValueError(
            f"unknown verbosity {verbosity!r}; expected 'compact' or 'full'"
        )

    def __repr__(self) -> str:
        return f"ProofChain({len(self._steps)} steps)"
