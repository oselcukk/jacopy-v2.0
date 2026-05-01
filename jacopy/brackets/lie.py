"""
Lie bracket, the standard (ungraded) commutator of vector fields.

``[X, Y]_Lie`` is degree 0, antisymmetric, satisfies Jacobi, and is a
derivation (Leibniz) in each slot. At the syntactic level we encode it
by translating a :class:`BracketApply` into the commutator
``X*Y − Y*X`` on the expression tree. This matches the standard formula
on commutative-algebra pairs and delegates all downstream simplification
(like-term collection, sign handling) to the regular algorithms.

For operators/derivations the *graded* commutator should be used
instead, see :class:`jacopy.algebra.commutator.Commutator`. The Lie
bracket here represents the particular case of a degree-0
antisymmetric bracket; it is the right object for vector-field
calculus but not for derivation algebra where Koszul signs matter.
"""

from __future__ import annotations

from typing import Optional

from jacopy.brackets.base import GradedBracket
from jacopy.core.expr import Expr, Neg, Product, Sum
from jacopy.core.registry import PropertyRegistry


class LieBracket(GradedBracket):
    """``[X, Y] := X*Y − Y*X``, degree 0, antisymmetric, Jacobi, Leibniz.

    The expansion produces the bare commutator on the product algebra;
    treat the result as syntactic. Downstream simplification (e.g.
    ``collect_terms`` on ``[X, X]`` collapsing to 0) is left to the
    algorithms layer.
    """

    def __init__(self, name: str = "[·,·]") -> None:
        super().__init__(
            name,
            degree=0,
            is_graded_antisymmetric=True,
            satisfies_leibniz=True,
            satisfies_graded_jacobi=True,
        )

    def expand(
        self,
        a: Expr,
        b: Expr,
        registry: Optional[PropertyRegistry] = None,
    ) -> Expr:
        return Sum(Product(a, b), Neg(Product(b, a)))


# Module-level singleton for convenience. Users who want a distinctly
# named bracket construct their own instance.
lie = LieBracket()
