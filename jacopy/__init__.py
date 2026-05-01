"""jacopy, graded algebra and bracket calculus with step-by-step proofs."""

from jacopy.core.expr import (
    Expr,
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
from jacopy.library.declarations import (
    Bivector,
    Forms,
    Functions,
    VectorFields,
)

__version__ = "0.0.1"

__all__ = [
    "Expr",
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
    "Functions",
    "VectorFields",
    "Forms",
    "Bivector",
]
