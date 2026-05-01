"""
Graded commutator as a first-class Expr node.

``[A, B] := A*B − (−1)^{|A||B|} B*A``.

Kept as an opaque node (:class:`Commutator`) at build time; callers
that want the explicit signed Sum call :func:`expand`, which consults
degrees via :func:`jacopy.algebra.derivation.degree_of` and folds the
Koszul sign when its parity is decidable.

The commutator of two graded derivations is itself a graded derivation
of degree ``|A| + |B|``, that statement is a *theorem* that belongs in
the proof layer, not here. This module only provides the syntactic
object and its definitional expansion.
"""

from __future__ import annotations

from typing import Any, Optional, Tuple

from jacopy.algebra.derivation import degree_of
from jacopy.core.expr import Expr, Neg, Product, Sum
from jacopy.core.registry import PropertyRegistry
from jacopy.core.symbolic_degree import Degree


# --------------------------------------------------------------------- #
# Node                                                                   #
# --------------------------------------------------------------------- #


class Commutator(Expr):
    """``[A, B]``, graded commutator node.

    Inert by construction. :meth:`expand` produces the signed Sum when
    the degrees of ``A`` and ``B`` combine to a decidable parity.
    """

    __slots__ = ("_a", "_b")

    def __init__(self, a: Expr, b: Expr) -> None:
        if not isinstance(a, Expr):
            raise TypeError("Commutator left operand must be an Expr")
        if not isinstance(b, Expr):
            raise TypeError("Commutator right operand must be an Expr")
        self._a = a
        self._b = b

    @property
    def a(self) -> Expr:
        return self._a

    @property
    def b(self) -> Expr:
        return self._b

    @property
    def children(self) -> Tuple[Expr, ...]:
        return (self._a, self._b)

    def _key(self) -> Any:
        return (self._a, self._b)

    def _repr_inner(self) -> str:
        return f"[{self._a._repr_inner()}, {self._b._repr_inner()}]"

    # ---- expansion ------------------------------------------------- #

    def expand(
        self, registry: Optional[PropertyRegistry] = None
    ) -> Expr:
        """Return ``A*B − (−1)^{|A||B|} B*A`` with the sign folded in.

        Falls back to :func:`expand_commutator`, which handles the
        decidable / symbolic-parity distinction. Raises
        :class:`ValueError` when the parity is undecidable, the caller
        is expected to keep the Commutator node around in that case.
        """
        return expand_commutator(self, registry)


# --------------------------------------------------------------------- #
# Helpers                                                                #
# --------------------------------------------------------------------- #


def commutator(a: Expr, b: Expr) -> Commutator:
    """Shorthand factory for :class:`Commutator`."""
    return Commutator(a, b)


def expand_commutator(
    node: Commutator, registry: Optional[PropertyRegistry] = None
) -> Expr:
    """Expand a single commutator node.

    Strategy:

    * Compute ``|A|`` and ``|B|`` via :func:`degree_of`.
    * The sign exponent is ``|A| * |B|``. If parity is even, the result
      is ``A*B − B*A``; if odd, it's ``A*B + B*A``.
    * Undecidable parity raises :class:`ValueError`, representing a
      half-concrete sign would require a ``(−1)^{…}`` expression type
      that the core layer doesn't yet offer.
    """
    deg_a = degree_of(node.a, registry)
    deg_b = degree_of(node.b, registry)
    parity = (deg_a * deg_b).parity()
    if parity is None:
        raise ValueError(
            f"Cannot expand {node!r}: sign parity of "
            f"|{node.a!r}|*|{node.b!r}| is symbolic. Keep the "
            "Commutator node or supply more specific degrees."
        )
    ab = Product(node.a, node.b)
    ba = Product(node.b, node.a)
    if parity == 0:
        return Sum(ab, Neg(ba))
    return Sum(ab, ba)
