r"""
Tilde-calculus defining axioms, Faz 14.B.

Three engine rewrite rules realise the defining identities of the
tilde operators introduced in :mod:`jacopy.calculus.tilde.operators`:

* :class:`TildeIotaSwapDefinition`, bridge axiom
  ``ι̃_ω V → ι_V ω``. The swap rewrites a tilde-interior application
  into a standard interior-product application against the indexing
  form. Unconditional: the registry-aware corner case
  ``ι̃_ω f = 0`` for a 0-vector ``f`` is handled by
  :class:`~jacopy.calculus.tilde.aux_axioms.TildeIotaOnZeroVectorDefinition`
  (Faz 14.D); engine ordering must register the auxiliary rule first
  so it fires on the pre-swap shape.

* :class:`TildeExteriorDLichnerowiczDefinition`, Lichnerowicz
  identity ``d̃ V → [π, V]_SN``. Instance-bound to a specific Poisson
  bivector ``π`` so that two unrelated tilde-d operators (e.g. on
  separate Poisson manifolds in the same proof) don't accidentally
  cross-pollute.

* :class:`TildeLieMagicDefinition`, tilde Cartan magic
  ``L̃_ω V → d̃(ι̃_ω V) + ι̃_ω(d̃ V)``. The right-hand side reuses the
  same ``π`` and ``ω`` instances stored on the matched
  :class:`~jacopy.calculus.tilde.operators.TildeLieDerivative`, so the
  fresh ``TildeExteriorDerivative(π)`` and ``TildeInteriorProduct(ω)``
  built here compare equal to any pre-existing ones the caller may
  have constructed.

All three rules are :class:`Definition` subclasses ready to register on
an :class:`~jacopy.proof.expansion.ExpansionEngine`. They do not
themselves close any of the six tilde Cartan relations, they only
unfold the operators by their definitions. Closure relies on the
auxiliary axioms in :mod:`jacopy.calculus.tilde.aux_axioms` plus the
ambient SN / Sharp / Koszul-bracket rules already in the proof engine.
"""

from __future__ import annotations

from jacopy.algebra.derivation import Act
from jacopy.brackets.base import BracketApply
from jacopy.brackets.schouten import sn as default_sn
from jacopy.calculus.tilde.operators import (
    TildeExteriorDerivative,
    TildeInteriorProduct,
    TildeLieDerivative,
)
from jacopy.calculus.interior import InteriorProduct
from jacopy.core.expr import Expr, Sum
from jacopy.proof.expansion import Definition


# --------------------------------------------------------------------- #
# ι̃_ω V → ι_V ω                                                         #
# --------------------------------------------------------------------- #


class TildeIotaSwapDefinition(Definition):
    r"""``ι̃_ω V → ι_V ω``, defining identity of the tilde interior product.

    Fires on ``Act(TildeInteriorProduct(ω), V)`` for any ``V``,
    rewriting it to ``Act(InteriorProduct(V), ω)``. The form ``ω``
    becomes the operand of the standard interior product and ``V``
    becomes its indexing vector field, this is the role-swap that
    distinguishes the tilde calculus from the ordinary one.

    The rule is registry-free; the corner case where ``V`` is a
    0-vector (a scalar function, for which the standard
    ``ι_V ω`` is identically zero) is owned by
    :class:`~jacopy.calculus.tilde.aux_axioms.TildeIotaOnZeroVectorDefinition`
    (Faz 14.D). Callers register that auxiliary rule before this one
    so the 0-vector reduction fires on the pre-swap shape, leaving
    this swap rule to handle the generic case.
    """

    name = "ι̃_ω V = ι_V ω"

    def matches(self, expr: Expr) -> bool:
        return (
            isinstance(expr, Act)
            and isinstance(expr.op, TildeInteriorProduct)
        )

    def rewrite(self, expr: Expr) -> Expr:
        ti = expr.op
        assert isinstance(ti, TildeInteriorProduct)
        omega = ti.form
        V = expr.arg
        return Act(InteriorProduct(V), omega)


# --------------------------------------------------------------------- #
# d̃ V → [π, V]_SN                                                       #
# --------------------------------------------------------------------- #


class TildeExteriorDLichnerowiczDefinition(Definition):
    r"""``d̃ V → [π, V]_SN``, Lichnerowicz definition of the tilde-d.

    Scoped to a specific Poisson bivector ``π``: matches only when the
    outer head is a :class:`TildeExteriorDerivative` whose
    :attr:`~jacopy.calculus.tilde.operators.TildeExteriorDerivative.bivector`
    equals ``π``. Two ``d̃`` rules with different ``π``'s coexist on the
    same engine without aliasing.

    The rewrite emits an inert :class:`BracketApply` against the
    Schouten-Nijenhuis bracket; the engine's existing SN expansion
    rules then take over (base cases on functions / 1-vectors and
    wedge-Leibniz recursion).
    """

    def __init__(self, pi: Expr) -> None:
        if not isinstance(pi, Expr):
            raise TypeError(
                "TildeExteriorDLichnerowiczDefinition pi must be an Expr"
            )
        self._pi = pi
        self._head = TildeExteriorDerivative(pi)
        self.name = f"d̃ V = [π, V]_SN [{pi._repr_inner()}]"

    @property
    def pi(self) -> Expr:
        return self._pi

    def matches(self, expr: Expr) -> bool:
        return (
            isinstance(expr, Act)
            and isinstance(expr.op, TildeExteriorDerivative)
            and expr.op == self._head
        )

    def rewrite(self, expr: Expr) -> Expr:
        V = expr.arg
        return BracketApply(default_sn, self._pi, V)


# --------------------------------------------------------------------- #
# L̃_ω V → d̃(ι̃_ω V) + ι̃_ω(d̃ V)                                         #
# --------------------------------------------------------------------- #


class TildeLieMagicDefinition(Definition):
    r"""``L̃_ω V → d̃(ι̃_ω V) + ι̃_ω(d̃ V)``, tilde Cartan magic formula.

    Scoped to a Poisson bivector ``π``: matches only when the outer
    head is a :class:`TildeLieDerivative` whose
    :attr:`~jacopy.calculus.tilde.operators.TildeLieDerivative.bivector`
    equals ``π``. The form parameter ``ω`` is read off the matched
    operator at rewrite time, so a single instance of this rule
    handles every ``L̃_ω`` on the bound ``π``.

    The rewrite constructs fresh ``TildeExteriorDerivative(π)`` and
    ``TildeInteriorProduct(ω)`` heads. Structural equality on those
    classes makes the freshly built operators compare equal to any
    pre-existing instances the caller may have constructed, so
    downstream rules that match on ``d̃`` or ``ι̃_ω`` fire on the
    rewritten subtree without needing to know it came from this rule.
    """

    def __init__(self, pi: Expr) -> None:
        if not isinstance(pi, Expr):
            raise TypeError(
                "TildeLieMagicDefinition pi must be an Expr"
            )
        self._pi = pi
        self.name = f"L̃_ω V = d̃(ι̃_ω V) + ι̃_ω(d̃ V) [{pi._repr_inner()}]"

    @property
    def pi(self) -> Expr:
        return self._pi

    def matches(self, expr: Expr) -> bool:
        return (
            isinstance(expr, Act)
            and isinstance(expr.op, TildeLieDerivative)
            and expr.op.bivector == self._pi
        )

    def rewrite(self, expr: Expr) -> Expr:
        head = expr.op
        assert isinstance(head, TildeLieDerivative)
        omega = head.form
        V = expr.arg
        d_t = TildeExteriorDerivative(self._pi)
        iota_t = TildeInteriorProduct(omega)
        return Sum(
            Act(d_t, Act(iota_t, V)),
            Act(iota_t, Act(d_t, V)),
        )
