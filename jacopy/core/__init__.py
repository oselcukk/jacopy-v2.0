"""Core layer: expression tree, properties, registry."""

from jacopy.core.expr import (
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
from jacopy.core.symbolic_degree import (
    Degree,
    DegreeLike,
    as_degree,
)
from jacopy.core.properties import (
    Provenance,
    ProofRef,
    Property,
    Scalar,
    Graded,
    Symmetric,
    Antisymmetric,
    GradedAntisymmetric,
    Closed,
    NonDegenerate,
    Poisson,
    NonCommuting,
    AntiCommuting,
    GradedCommutative,
)
from jacopy.core.registry import (
    PropertyRegistry,
    default_registry,
    reset_default_registry,
)
from jacopy.core.wildcards import (
    Wildcard,
    SeqWildcard,
    match,
    substitute,
)
from jacopy.core.equality import (
    structural_equal,
    alpha_equal,
    sum_bag_equal,
)
from jacopy.core.multi_eval import (
    MultiEval,
    has_repeated_arg,
    multi_eval,
    validate_arity,
)
from jacopy.core.indexed_sum import (
    IndexedSum,
    dummy_in,
    indexed_sum,
)
from jacopy.core.wedge import Wedge

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
    "Closed",
    "NonDegenerate",
    "Poisson",
    "NonCommuting",
    "AntiCommuting",
    "GradedCommutative",
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
    # multilinear evaluation (Faz 12.A.0)
    "MultiEval",
    "multi_eval",
    "has_repeated_arg",
    "validate_arity",
    # indexed sum (Faz 17.E)
    "IndexedSum",
    "indexed_sum",
    "dummy_in",
    # wedge product (Faz 17.F.1.5)
    "Wedge",
]
