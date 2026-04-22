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
from gradalg.core.wildcards import (
    Wildcard,
    SeqWildcard,
    match,
    substitute,
)
from gradalg.core.equality import (
    structural_equal,
    alpha_equal,
    sum_bag_equal,
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
    # wildcards
    "Wildcard",
    "SeqWildcard",
    "match",
    "substitute",
    # equality
    "structural_equal",
    "alpha_equal",
    "sum_bag_equal",
]
