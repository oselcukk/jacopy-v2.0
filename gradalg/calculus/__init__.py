"""Calculus layer: Cartan operators on the exterior algebra."""

from gradalg.calculus.anchor import (
    Anchor,
    bracket_compatibility_obstruction,
)
from gradalg.calculus.exterior_algebra import ExteriorAlgebra
from gradalg.calculus.exterior_d import (
    ExteriorDerivative,
    apply_d_squared_zero,
    d,
)
from gradalg.calculus.interior import (
    InteriorProduct,
    apply_iota_axioms,
    apply_iota_squared_zero,
    interior,
)
from gradalg.calculus.invariant_d import (
    INVARIANT_D_CLASSIFICATIONS,
    InvariantDOneFormDefinition,
    invariant_d_one_form,
)
from gradalg.calculus.lie_derivative import (
    DEFINITIONS,
    LieDerivative,
    cartan_expansion,
    cartan_obstruction,
    lie_derivative,
)
from gradalg.calculus.cartan import CartanCalculus, RELATIONS
from gradalg.calculus.hamiltonian_vf import (
    HamiltonianVectorField,
    HamiltonianVfDerivedDefinition,
    equivalence_condition,
    hamiltonian_vf,
)
from gradalg.calculus.musical import (
    ArgNegLinearityDefinition,
    Flat,
    IotaFlatDefinition,
    MusicalCompatibility,
    MusicalCompatibilityDefinition,
    Sharp,
    flat,
    sharp,
)
from gradalg.calculus.operator_equation import OperatorEquation
from gradalg.calculus.pairing import Pairing, pairing

__all__ = [
    # exterior derivative
    "ExteriorDerivative",
    "d",
    "apply_d_squared_zero",
    # interior product
    "InteriorProduct",
    "interior",
    "apply_iota_axioms",
    "apply_iota_squared_zero",
    # Lie derivative
    "LieDerivative",
    "lie_derivative",
    "cartan_expansion",
    "cartan_obstruction",
    "DEFINITIONS",
    # anchor
    "Anchor",
    "bracket_compatibility_obstruction",
    # exterior algebra skeleton
    "ExteriorAlgebra",
    # operator equation
    "OperatorEquation",
    # Cartan calculus framework
    "CartanCalculus",
    "RELATIONS",
    # Invariant formula for d on 1-forms
    "invariant_d_one_form",
    "InvariantDOneFormDefinition",
    "INVARIANT_D_CLASSIFICATIONS",
    # Hamiltonian vector field
    "HamiltonianVectorField",
    "HamiltonianVfDerivedDefinition",
    "hamiltonian_vf",
    "equivalence_condition",
    # Pairing
    "Pairing",
    "pairing",
    # Musical isomorphisms
    "Flat",
    "flat",
    "Sharp",
    "sharp",
    "MusicalCompatibility",
    "MusicalCompatibilityDefinition",
    "IotaFlatDefinition",
    "ArgNegLinearityDefinition",
]
