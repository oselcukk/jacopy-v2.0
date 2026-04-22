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
]
