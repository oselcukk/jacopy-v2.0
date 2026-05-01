"""
Tilde calculus, Cartan operators on the Koszul (multivector) side.

Three operators parameterise the dual of the standard exterior calculus
on a Poisson manifold ``(M, π)``:

* :class:`TildeInteriorProduct` ``ι̃_ω``, form-indexed contraction on
  multivectors, ``ι̃_ω V := ι_V ω``.
* :class:`TildeExteriorDerivative` ``d̃``, Lichnerowicz differential
  on multivectors, ``d̃ V := [π, V]_SN``.
* :class:`TildeLieDerivative` ``L̃_ω``, Cartan magic on the tilde side,
  ``L̃_ω := d̃ ∘ ι̃_ω + ι̃_ω ∘ d̃``.

This sub-package exposes only the Expr-level operator atoms (Faz 14.A).
The defining-axiom rewrite rules and the auxiliary axioms required to
close the six tilde Cartan relations land in sibling modules
(``axioms.py``, ``aux_axioms.py``) in later stages of Faz 14.
"""

from jacopy.calculus.tilde.cartan_remainder import (
    K_tilde,
    TildeCartanRemainder,
)
from jacopy.calculus.tilde.aux_axioms import (
    TildeDOfFunctionDefinition,
    TildeDSquaredPoissonDefinition,
    TildeIotaActAsScalarDefinition,
    TildeIotaOnZeroVectorDefinition,
    TildeIotaSquaredZeroDefinition,
    TildeLieOnZeroVectorDefinition,
)
from jacopy.calculus.tilde.axioms import (
    TildeExteriorDLichnerowiczDefinition,
    TildeIotaSwapDefinition,
    TildeLieMagicDefinition,
)
from jacopy.calculus.tilde.intrinsic_axioms import (
    TildeDIntrinsicDefinition,
    TildeIotaIntrinsicDefinition,
    TildeLieIntrinsicDefinition,
)
from jacopy.calculus.tilde.intrinsic_engine import (
    TildeIntrinsicFormulaMatch,
    TildeIntrinsicFormulaRecognizer,
    prove_tilde_cartan_relation,
    tilde_intrinsic_engine,
)
from jacopy.calculus.tilde.operators import (
    TildeExteriorDerivative,
    TildeInteriorProduct,
    TildeLieDerivative,
    tilde_d,
    tilde_interior,
    tilde_lie,
)

__all__ = [
    "TildeInteriorProduct",
    "TildeExteriorDerivative",
    "TildeLieDerivative",
    "tilde_interior",
    "tilde_d",
    "tilde_lie",
    "TildeIotaSwapDefinition",
    "TildeExteriorDLichnerowiczDefinition",
    "TildeLieMagicDefinition",
    "TildeIotaOnZeroVectorDefinition",
    "TildeIotaSquaredZeroDefinition",
    "TildeLieOnZeroVectorDefinition",
    "TildeDOfFunctionDefinition",
    "TildeDSquaredPoissonDefinition",
    "TildeIotaActAsScalarDefinition",
    "TildeIotaIntrinsicDefinition",
    "TildeLieIntrinsicDefinition",
    "TildeDIntrinsicDefinition",
    "tilde_intrinsic_engine",
    "TildeIntrinsicFormulaRecognizer",
    "TildeIntrinsicFormulaMatch",
    "prove_tilde_cartan_relation",
    "TildeCartanRemainder",
    "K_tilde",
]
