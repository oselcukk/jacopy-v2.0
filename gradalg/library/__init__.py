"""Hazır yapılar ve teorem kütüphanesi (Faz 9).

Stage A landed: :class:`Theorem` / :class:`TheoremBook` registry plus
the process-wide singleton :data:`theorem_book`. Stage B adds
:class:`SymplecticManifold` and :class:`PoissonBracket` together with
the seeded Poisson/Koszul theorems. Stage C adds :class:`LieAlgebroid`
with the algebroid Cartan bundle and the seeded
``lie_algebroid_anchor_compat`` axiom theorem. Stage D (Courant,
Dirac) will populate further entries.
"""

from gradalg.library.theorem_book import Theorem, TheoremBook, theorem_book

# Stage B — seeding happens on import of these modules.
from gradalg.library.symplectic import SymplecticManifold
from gradalg.library.poisson import (
    THEOREM_POISSON_JACOBI,
    THEOREM_POISSON_KOSZUL_EQUIVALENCE,
    THEOREM_POISSON_KOSZUL_JACOBI,
    PoissonBracket,
    poisson_bracket,
)

# Stage C — Lie algebroid + algebroid Cartan; seeds
# ``lie_algebroid_anchor_compat`` on import.
from gradalg.library.lie_algebroid import (
    THEOREM_LIE_ALGEBROID_ANCHOR_COMPAT,
    LieAlgebroid,
    lie_algebroid,
)

__all__ = [
    "Theorem",
    "TheoremBook",
    "theorem_book",
    "SymplecticManifold",
    "PoissonBracket",
    "poisson_bracket",
    "LieAlgebroid",
    "lie_algebroid",
    "THEOREM_POISSON_JACOBI",
    "THEOREM_POISSON_KOSZUL_EQUIVALENCE",
    "THEOREM_POISSON_KOSZUL_JACOBI",
    "THEOREM_LIE_ALGEBROID_ANCHOR_COMPAT",
]
