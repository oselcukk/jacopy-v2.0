"""Hazır yapılar ve teorem kütüphanesi (Faz 9).

Stage A landed: :class:`Theorem` / :class:`TheoremBook` registry plus
the process-wide singleton :data:`theorem_book`. Stage B adds
:class:`SymplecticManifold` and :class:`PoissonBracket` together with
the seeded Poisson/Koszul theorems. Stage C adds :class:`LieAlgebroid`
with the algebroid Cartan bundle and the seeded
``lie_algebroid_anchor_compat`` axiom theorem. Stage D adds
:class:`CourantAlgebroid` (with H-twist + Courant–Dorfman bridge) and
:class:`DiracStructure` plus four further seeded theorems.
"""

from gradalg.library.theorem_book import Theorem, TheoremBook, theorem_book
from gradalg.library.declarations import (
    Bivector,
    Forms,
    Functions,
    VectorFields,
)

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

# Stage D — Courant algebroid + Dirac structures; seeds
# ``courant_jacobi_twist``, ``courant_dorfman_bridge``,
# ``dirac_isotropy``, ``dirac_involutivity`` on import.
from gradalg.library.courant_algebroid import (
    THEOREM_COURANT_DORFMAN_BRIDGE,
    THEOREM_COURANT_JACOBI_TWIST,
    CourantAlgebroid,
    courant_algebroid,
)
from gradalg.library.dirac import (
    THEOREM_DIRAC_INVOLUTIVITY,
    THEOREM_DIRAC_ISOTROPY,
    DiracStructure,
    poisson_dirac,
    presymplectic_dirac,
)

# Twisted Cartan bundle — d_H = d + H∧ variant (Faz 10 tutorial gap closure).
from gradalg.library.twisted_cartan import (
    TwistedCartanBundle,
    twisted_cartan_bundle,
)

__all__ = [
    "Theorem",
    "TheoremBook",
    "theorem_book",
    "Functions",
    "VectorFields",
    "Forms",
    "Bivector",
    "SymplecticManifold",
    "PoissonBracket",
    "poisson_bracket",
    "LieAlgebroid",
    "lie_algebroid",
    "CourantAlgebroid",
    "courant_algebroid",
    "DiracStructure",
    "poisson_dirac",
    "presymplectic_dirac",
    "TwistedCartanBundle",
    "twisted_cartan_bundle",
    "THEOREM_POISSON_JACOBI",
    "THEOREM_POISSON_KOSZUL_EQUIVALENCE",
    "THEOREM_POISSON_KOSZUL_JACOBI",
    "THEOREM_LIE_ALGEBROID_ANCHOR_COMPAT",
    "THEOREM_COURANT_JACOBI_TWIST",
    "THEOREM_COURANT_DORFMAN_BRIDGE",
    "THEOREM_DIRAC_ISOTROPY",
    "THEOREM_DIRAC_INVOLUTIVITY",
]
