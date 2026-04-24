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


# Invariant-d re-exports are resolved lazily to break a circular import:
# ``invariant_d`` subclasses ``proof.expansion.Definition``, and
# ``proof.expansion`` itself depends on ``calculus.exterior_d``. Eagerly
# pulling ``invariant_d`` here would force ``proof.expansion`` to finish
# before it has had a chance to define ``Definition``. PEP 562's module
# ``__getattr__`` defers the pull to first access, by which time the
# import graph has settled.
_LAZY_INVARIANT_D_NAMES = frozenset(
    {
        "INVARIANT_D_CLASSIFICATIONS",
        "InvariantDOneFormDefinition",
        "invariant_d_one_form",
    }
)


def __getattr__(name: str):
    if name in _LAZY_INVARIANT_D_NAMES:
        from gradalg.calculus import invariant_d as _mod

        value = getattr(_mod, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
