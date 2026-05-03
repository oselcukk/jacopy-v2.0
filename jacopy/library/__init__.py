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

from jacopy.library.theorem_book import Theorem, TheoremBook, theorem_book
from jacopy.library.declarations import (
    Bivector,
    Forms,
    Functions,
    VectorFields,
)

# Stage B, seeding happens on import of these modules.
from jacopy.library.symplectic import SymplecticManifold
from jacopy.library.symplectic_problem import SymplecticProblem
from jacopy.library.poisson import (
    THEOREM_POISSON_JACOBI,
    THEOREM_POISSON_KOSZUL_EQUIVALENCE,
    THEOREM_POISSON_KOSZUL_JACOBI,
    PoissonBracket,
    poisson_bracket,
)
from jacopy.library.koszul_problem import (
    KoszulBracketExpansionDefinition,
    KoszulProblem,
)

# Stage C, Lie algebroid + algebroid Cartan; seeds
# ``lie_algebroid_anchor_compat`` on import.
from jacopy.library.lie_algebroid import (
    THEOREM_LIE_ALGEBROID_ANCHOR_COMPAT,
    LieAlgebroid,
    lie_algebroid,
)

# Stage D, Courant algebroid + Dirac structures; seeds
# ``courant_jacobi_twist``, ``courant_dorfman_bridge``,
# ``dirac_isotropy``, ``dirac_involutivity`` on import.
from jacopy.library.courant_algebroid import (
    THEOREM_COURANT_DORFMAN_BRIDGE,
    THEOREM_COURANT_JACOBI_TWIST,
    CourantAlgebroid,
    courant_algebroid,
)
from jacopy.library.triangular_lie_bialgebroid import (
    TriangularLieBialgebroid,
    triangular_lie_bialgebroid,
)
from jacopy.library.dirac import (
    THEOREM_DIRAC_INVOLUTIVITY,
    THEOREM_DIRAC_ISOTROPY,
    DiracStructure,
    poisson_dirac,
    presymplectic_dirac,
)

# Twisted Cartan bundle, d_H = d + H∧ variant (Faz 10 tutorial gap closure).
from jacopy.library.twisted_cartan import (
    TwistedCartanBundle,
    twisted_cartan_bundle,
)

# Bianchi-identity wrapper for affine connections (Faz 16.D).
from jacopy.library.bianchi_problem import (
    BianchiProblem,
    BianchiProofResult,
    cyclic_sum_3,
    cyclic_sum_3_fixed_last,
)

# Form-degree (form-property) proofs for ω, Q, T, R on a local frame.
from jacopy.library.cartan_form_property import (
    CartanFormPropertyProblem,
    CartanFormPropertyProofResult,
)

# Cartan structure equations on a local frame.
from jacopy.library.cartan_structure import (
    CartanStructureProblem,
    CartanStructureProofResult,
)

# Q9 Stage 9.F, Koszul-connection capstone wrapper bundling
# Bianchi + form-property + Cartan-structure facets for ∇̃ on T*M.
from jacopy.library.koszul_connection_problem import KoszulConnectionProblem

__all__ = [
    "Theorem",
    "TheoremBook",
    "theorem_book",
    "Functions",
    "VectorFields",
    "Forms",
    "Bivector",
    "SymplecticManifold",
    "SymplecticProblem",
    "PoissonBracket",
    "poisson_bracket",
    "KoszulProblem",
    "KoszulBracketExpansionDefinition",
    "LieAlgebroid",
    "lie_algebroid",
    "CourantAlgebroid",
    "courant_algebroid",
    "TriangularLieBialgebroid",
    "triangular_lie_bialgebroid",
    "DiracStructure",
    "poisson_dirac",
    "presymplectic_dirac",
    "TwistedCartanBundle",
    "twisted_cartan_bundle",
    "BianchiProblem",
    "BianchiProofResult",
    "cyclic_sum_3",
    "cyclic_sum_3_fixed_last",
    "CartanFormPropertyProblem",
    "CartanFormPropertyProofResult",
    "CartanStructureProblem",
    "CartanStructureProofResult",
    "KoszulConnectionProblem",
    "THEOREM_POISSON_JACOBI",
    "THEOREM_POISSON_KOSZUL_EQUIVALENCE",
    "THEOREM_POISSON_KOSZUL_JACOBI",
    "THEOREM_LIE_ALGEBROID_ANCHOR_COMPAT",
    "THEOREM_COURANT_JACOBI_TWIST",
    "THEOREM_COURANT_DORFMAN_BRIDGE",
    "THEOREM_DIRAC_ISOTROPY",
    "THEOREM_DIRAC_INVOLUTIVITY",
]
