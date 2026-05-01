"""
User-defined custom brackets.

``CustomBracket`` lets a user plug in a concrete expansion function
without subclassing :class:`GradedBracket` explicitly. It's intended
for one-off brackets in tutorials, exploratory proofs, or as the
workbench for a bracket whose axioms haven't been nailed down yet.

Production brackets with well-characterized properties should still
get their own subclass, the CustomBracket route stores only the
expansion callable and a flat set of axiom flags, so richer invariants
(``expand_definition``, obstruction hooks) aren't available.
"""

from __future__ import annotations

from typing import Any, Callable, Optional

from jacopy.brackets.base import GradedBracket
from jacopy.core.expr import Expr
from jacopy.core.registry import PropertyRegistry
from jacopy.core.symbolic_degree import DegreeLike

ExpandFn = Callable[[Expr, Expr, Optional[PropertyRegistry]], Expr]


class CustomBracket(GradedBracket):
    """A bracket whose expansion is supplied as a callable.

    Parameters
    ----------
    name
        Display name.
    expand_fn
        ``(a, b, registry) → Expr``, the definitional expansion. The
        registry argument is always passed, even when the rule ignores
        it, to keep the calling convention uniform with other brackets.
    degree, is_graded_antisymmetric, satisfies_leibniz,
    satisfies_graded_jacobi
        Axiom profile. Same semantics as :class:`GradedBracket` base.
    """

    def __init__(
        self,
        name: str,
        expand_fn: ExpandFn,
        *,
        degree: DegreeLike = 0,
        is_graded_antisymmetric: bool = True,
        satisfies_leibniz: bool = True,
        satisfies_graded_jacobi: Optional[bool] = True,
    ) -> None:
        if not callable(expand_fn):
            raise TypeError("CustomBracket expand_fn must be callable")
        super().__init__(
            name,
            degree=degree,
            is_graded_antisymmetric=is_graded_antisymmetric,
            satisfies_leibniz=satisfies_leibniz,
            satisfies_graded_jacobi=satisfies_graded_jacobi,
        )
        self._expand_fn = expand_fn

    def expand(
        self,
        a: Expr,
        b: Expr,
        registry: Optional[PropertyRegistry] = None,
    ) -> Expr:
        return self._expand_fn(a, b, registry)

    def _identity_key(self) -> Any:
        # Identify custom brackets by their callable too, two
        # CustomBrackets with the same name but different expansion
        # rules must compare unequal. Python functions compare by
        # identity, which is what we want here (users pass distinct
        # lambdas or named functions).
        return super()._identity_key() + (self._expand_fn,)
