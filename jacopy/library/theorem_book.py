"""
Theorem Book, central registry of proven theorems.

A :class:`Theorem` pairs a human-readable statement with the underlying
:class:`~jacopy.proof.chain.ProofChain`, the axiom list it depends on,
and optional exposition. :class:`TheoremBook` is an ordered registry of
those records plus a module-level singleton :data:`theorem_book`.

The Book is intentionally minimal at this stage: a keyed store that
returns the full proof object on lookup. Seeding with concrete theorems
happens in the later library stages (``symplectic``, ``poisson``,
``lie_algebroid``, ``courant_algebroid``), as those modules are what
produce the proofs in the first place.

:class:`~jacopy.proof.strategies.UnrollToFoundations` already drives
foundational unrolling through the expansion engine and does *not* need
the Book to function; a future extension may let it look up named
theorems here, but Stage A deliberately keeps that coupling absent so
the registry is a pure data structure.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterator, Tuple

from jacopy.proof.chain import ProofChain


@dataclass(frozen=True)
class Theorem:
    """A named, proven mathematical result.

    Parameters
    ----------
    name
        Unique short handle (``"d_squared_zero"``, ``"cartan_magic"``).
        Used as the lookup key inside :class:`TheoremBook`.
    statement
        Human-readable mathematical statement, e.g. ``"d ∘ d = 0"``.
    from_axioms
        Tuple of axiom labels the proof depends on. Purely descriptive
       , not cross-validated against the proof itself, because axioms
        surface through the expansion engine's ``theorem/axiom``
        classification rather than a name list.
    proof
        The :class:`ProofChain` establishing the result. Typically
        produced by a strategy (``ExpandAndSimplify``,
        ``UnrollToFoundations``) or by a dedicated helper such as
        :meth:`jacopy.calculus.cartan.CartanCalculus.verify`.
    notes
        Optional exposition, context, equivalent formulations, caveats.
    """

    name: str
    statement: str
    from_axioms: Tuple[str, ...]
    proof: ProofChain
    notes: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("Theorem.name must be a non-empty string")
        if not isinstance(self.statement, str):
            raise TypeError("Theorem.statement must be a string")
        if not isinstance(self.from_axioms, tuple) or not all(
            isinstance(a, str) for a in self.from_axioms
        ):
            raise TypeError("Theorem.from_axioms must be a tuple of strings")
        if not isinstance(self.proof, ProofChain):
            raise TypeError("Theorem.proof must be a ProofChain")
        if not isinstance(self.notes, str):
            raise TypeError("Theorem.notes must be a string")


class TheoremBook:
    """Ordered registry of :class:`Theorem` records keyed by name.

    Insertion order is preserved so :meth:`names` / :meth:`__iter__`
    return results in the order theorems were added, useful when
    rendering a "table of results" page. Duplicate names raise instead
    of silently overwriting, because registering the same theorem twice
    usually indicates two library modules disagreeing on the canonical
    proof and that disagreement should surface loudly.
    """

    __slots__ = ("_theorems",)

    def __init__(self) -> None:
        # dict preserves insertion order on Python 3.7+.
        self._theorems: dict[str, Theorem] = {}

    def add(self, theorem: Theorem) -> None:
        """Register ``theorem``; raise :class:`KeyError` on name clash."""
        if not isinstance(theorem, Theorem):
            raise TypeError("TheoremBook.add expects a Theorem")
        if theorem.name in self._theorems:
            raise KeyError(
                f"Theorem {theorem.name!r} is already registered; "
                f"use replace() to override"
            )
        self._theorems[theorem.name] = theorem

    def replace(self, theorem: Theorem) -> None:
        """Register ``theorem``, overwriting any existing entry with the same name.

        Use sparingly, the default :meth:`add` path refuses overwrites
        so registration conflicts surface immediately. ``replace`` is
        for deliberate cases where a later library stage supplies a
        tighter proof than an earlier stub.
        """
        if not isinstance(theorem, Theorem):
            raise TypeError("TheoremBook.replace expects a Theorem")
        self._theorems[theorem.name] = theorem

    def get(self, name: str) -> Theorem:
        """Return the :class:`Theorem` registered under ``name``."""
        if name not in self._theorems:
            raise KeyError(f"no theorem named {name!r}; known: {self.names()}")
        return self._theorems[name]

    def names(self) -> Tuple[str, ...]:
        """Registered theorem names, in insertion order."""
        return tuple(self._theorems)

    def remove(self, name: str) -> None:
        """Remove the theorem registered under ``name``.

        Provided mostly for test isolation, production library modules
        should not unregister their own theorems at runtime.
        """
        if name not in self._theorems:
            raise KeyError(f"no theorem named {name!r}")
        del self._theorems[name]

    def clear(self) -> None:
        """Drop every registered theorem. Intended for test fixtures."""
        self._theorems.clear()

    def __contains__(self, name: object) -> bool:
        return isinstance(name, str) and name in self._theorems

    def __len__(self) -> int:
        return len(self._theorems)

    def __iter__(self) -> Iterator[Theorem]:
        return iter(self._theorems.values())

    def __repr__(self) -> str:
        return f"TheoremBook({len(self)} theorems: {', '.join(self.names())})"


#: Process-wide default :class:`TheoremBook`. Library modules register
#: their theorems here at import time; downstream callers look up
#: results via :meth:`TheoremBook.get`. Kept empty at Stage A, seeding
#: is the responsibility of the library stages that produce the proofs.
theorem_book = TheoremBook()
