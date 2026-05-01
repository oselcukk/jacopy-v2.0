"""
Exterior algebra ``Ω*(M)``, the generator skeleton.

For the proof layer's ``AgreementOnGenerators`` strategy (coming in
Faz 7) we need to know *explicitly* which elements generate the
exterior algebra under wedge and graded Leibniz. For ``Ω*(M)`` those
generators are

* the degree-0 functions ``C^∞(M)``, represented here by the
  ``Scalar`` / ``Graded(degree=0)`` declarations the caller has
  already made,
* the exact 1-forms ``df``, represented syntactically as
  ``d(f)`` on a declared function ``f``.

An :class:`ExteriorAlgebra` instance is a thin record: it carries a
reference to the exterior derivative and a list of *function
generators* (symbolic functions ``f, g, …``). Its two operations are

* :attr:`generators`, return the list of exprs that generate the
  algebra, in a form consumable by a future agreement-on-generators
  strategy,
* :meth:`is_generated_by`, answer whether a candidate set covers the
  algebra in the sense of reaching every generator by closing under
  wedge and ``d``. At this layer the check is deliberately structural
  (containment of the declared generators); the deeper *closure*
  check is a Faz 7 concern.

Treat this module as a skeleton: it provides the shape the proof
layer will hook into. The concrete verification of "these elements
generate" is deferred until the proof-layer ``AgreementOnGenerators``
strategy exists to consume it.
"""

from __future__ import annotations

from typing import Iterable, Optional, Sequence, Tuple

from jacopy.algebra.derivation import Act
from jacopy.calculus.exterior_d import ExteriorDerivative, d as default_d
from jacopy.core.expr import Expr


class ExteriorAlgebra:
    """The exterior algebra ``Ω*(M)`` indexed by its function ring.

    ``functions`` is the tuple of 0-form generators, the symbols
    the caller treats as ``C^∞(M)``. Their exterior derivatives
    ``d(f)`` form the 1-form generators; higher-degree forms are
    reached by wedge products and further applications of ``d``.

    The class is intentionally data-only, no expansion logic, no
    proof strategy. It is the object a future ``AgreementOnGenerators``
    will consume.
    """

    __slots__ = ("_functions", "_d")

    def __init__(
        self,
        functions: Iterable[Expr],
        *,
        d: Optional[ExteriorDerivative] = None,
    ) -> None:
        funcs = tuple(functions)
        for f in funcs:
            if not isinstance(f, Expr):
                raise TypeError(
                    "ExteriorAlgebra functions must be Exprs"
                )
        self._functions = funcs
        self._d = default_d if d is None else d

    @property
    def functions(self) -> Tuple[Expr, ...]:
        return self._functions

    @property
    def d(self) -> ExteriorDerivative:
        return self._d

    @property
    def generators(self) -> Tuple[Expr, ...]:
        """The full generator set: 0-form functions and their differentials.

        Order is ``(f_1, …, f_n, d(f_1), …, d(f_n))`` so a consumer
        can split into degree-0 and degree-1 blocks by halving the
        tuple. The ``d(f_i)`` entries are :class:`Act` nodes built
        with the algebra's own ``d``, in particular, comparing them
        against a user-supplied ``d(f)`` requires the same
        :class:`ExteriorDerivative` instance on both sides, since
        :class:`Derivation` equality is structural on ``(name,
        degree)``.
        """
        one_forms = tuple(Act(self._d, f) for f in self._functions)
        return self._functions + one_forms

    def is_generated_by(self, elements: Sequence[Expr]) -> bool:
        """Structural check: do ``elements`` cover the declared generators?

        Returns ``True`` when every generator returned by
        :attr:`generators` appears (structurally-equal) somewhere in
        ``elements``. This is a conservative, fast test, it does not
        attempt to detect when some generator is *reachable* from
        ``elements`` under wedge/``d``; that closure-style reasoning
        belongs in the Faz 7 proof layer.
        """
        needed = set(self.generators)
        provided = set(elements)
        return needed.issubset(provided)
