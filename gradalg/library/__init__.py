"""Hazır yapılar ve teorem kütüphanesi (Faz 9).

Stage A landed: :class:`Theorem` / :class:`TheoremBook` registry plus
the process-wide singleton :data:`theorem_book`. Stage B.1 adds
:class:`SymplecticManifold` and :class:`PoissonBracket` together with
the seeded ``poisson_jacobi`` theorem. Later stages (Lie algebroid,
Courant, Dirac) will populate further entries.
"""

from gradalg.library.theorem_book import Theorem, TheoremBook, theorem_book

# Stage B.1 — seeding happens on import of these modules.
from gradalg.library.symplectic import SymplecticManifold
from gradalg.library.poisson import (
    THEOREM_POISSON_JACOBI,
    PoissonBracket,
    poisson_bracket,
)

__all__ = [
    "Theorem",
    "TheoremBook",
    "theorem_book",
    "SymplecticManifold",
    "PoissonBracket",
    "poisson_bracket",
    "THEOREM_POISSON_JACOBI",
]
