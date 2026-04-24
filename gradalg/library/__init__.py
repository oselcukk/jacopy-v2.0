"""Hazır yapılar ve teorem kütüphanesi (Faz 9).

Stage A landed: :class:`Theorem` / :class:`TheoremBook` registry plus
the process-wide singleton :data:`theorem_book`. Stages B-D (Symplectic,
Poisson, Lie algebroid, Courant, Dirac) will populate the registry as
they land.
"""

from gradalg.library.theorem_book import Theorem, TheoremBook, theorem_book

__all__ = [
    "Theorem",
    "TheoremBook",
    "theorem_book",
]
