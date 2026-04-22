"""Core layer: expression tree, properties, registry."""

from gradalg.core.expr import (
    Expr,
    Atom,
    Symbol,
    Integer,
    Rational,
    Sum,
    Product,
    Power,
    Neg,
    Zero,
    One,
    NegOne,
)
from gradalg.core.symbolic_degree import (
    Degree,
    DegreeLike,
    as_degree,
)
from gradalg.core.properties import (
    Provenance,
    ProofRef,
    Property,
    Scalar,
    Graded,
    Symmetric,
    Antisymmetric,
    GradedAntisymmetric,
)
from gradalg.core.registry import (
    PropertyRegistry,
    default_registry,
    reset_default_registry,
)

__all__ = [
    # expr
    "Expr",
    "Atom",
    "Symbol",
    "Integer",
    "Rational",
    "Sum",
    "Product",
    "Power",
    "Neg",
    "Zero",
    "One",
    "NegOne",
    # symbolic_degree
    "Degree",
    "DegreeLike",
    "as_degree",
    # properties
    "Provenance",
    "ProofRef",
    "Property",
    "Scalar",
    "Graded",
    "Symmetric",
    "Antisymmetric",
    "GradedAntisymmetric",
    # registry
    "PropertyRegistry",
    "default_registry",
    "reset_default_registry",
]
