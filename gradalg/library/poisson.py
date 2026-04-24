"""
Poisson bracket library.

A :class:`PoissonBracket` wraps a bivector ``π`` and exposes three
complementary views of ``{f, g}_π``:

* the **derived** form via
  :class:`~gradalg.brackets.derived.DerivedBracket` ``(sn, π, degree_Q=1)``,
  which reduces to ``[[f, π]_SN, g]_SN``;
* the **Hamiltonian** form via
  :func:`~gradalg.calculus.hamiltonian_vf.hamiltonian_vf`, where
  ``{f, g}_π = X_f(g)``;
* the universal **Jacobi obstruction** ``[π, π]_SN`` as a
  :class:`~gradalg.brackets.derived.VanishingCondition`.

The Poisson Jacobi identity reduces to a single equation on ``π`` —
the Derived Bracket Theorem says Jacobi on ``{·, ·}_π`` holds iff
``[π, π]_SN = 0``. This module records that reduction as the seeded
theorem :data:`THEOREM_POISSON_JACOBI` in the package-wide
:data:`~gradalg.library.theorem_book.theorem_book`, so downstream code
can cite the result without rebuilding its :class:`ProofChain`.
"""

from __future__ import annotations

from typing import Optional

from gradalg.algebra.derivation import Act
from gradalg.brackets.derived import DerivedBracket, VanishingCondition
from gradalg.brackets.schouten import sn
from gradalg.calculus.hamiltonian_vf import (
    HamiltonianVectorField,
    hamiltonian_vf as _hamiltonian_vf,
)
from gradalg.core.expr import Expr, Symbol
from gradalg.core.properties import Graded
from gradalg.core.registry import PropertyRegistry
from gradalg.core.symbolic_degree import DegreeLike
from gradalg.library.theorem_book import Theorem, theorem_book
from gradalg.proof.chain import ProofChain
from gradalg.proof.step import ProofStep


# --------------------------------------------------------------------- #
# Poisson bracket wrapper                                                #
# --------------------------------------------------------------------- #


class PoissonBracket:
    """``{·, ·}_π`` — the Poisson bracket of a bivector ``π``.

    Parameters
    ----------
    pi
        The Poisson bivector. Treated as an SN multivector of degree
        ``degree_bivector`` in the shifted grading.
    degree_bivector
        Symbolic degree of ``π`` in the SN grading. Default ``1``
        (i.e. ``π`` is a 2-vector, SN-degree ``2 − 1 = 1``). Lift this
        if you need the machinery to work on a non-standard grading.
    name
        Optional display name; defaults to ``f"{{·,·}}_{π}"``.

    Notes
    -----
    The class is deliberately thin — every computation delegates into
    :class:`~gradalg.brackets.derived.DerivedBracket` or
    :func:`~gradalg.calculus.hamiltonian_vf.hamiltonian_vf`. The value
    it adds is naming the bracket (``PoissonBracket.from_bivector(π)``
    reads better than constructing the derived bracket directly at call
    sites) and bundling the three equivalent views so they stay in
    sync. The Jacobi reduction is surfaced as
    :meth:`prove_jacobi_reduction` — a single-step chain asserting the
    Derived Bracket Theorem — rather than as a ``prove_jacobi`` that
    would require ``[π, π]_SN`` to collapse under :func:`simplify`
    (which it doesn't, for atomic ``π``).
    """

    __slots__ = ("_pi", "_derived", "_name")

    def __init__(
        self,
        pi: Expr,
        *,
        degree_bivector: DegreeLike = 1,
        name: Optional[str] = None,
    ) -> None:
        if not isinstance(pi, Expr):
            raise TypeError("PoissonBracket requires an Expr bivector")
        display = name if name is not None else f"{{·,·}}_{pi._repr_inner()}"
        self._pi = pi
        self._derived = DerivedBracket(
            sn, pi, degree_Q=degree_bivector, name=display,
        )
        self._name = display

    @classmethod
    def from_bivector(
        cls,
        pi: Expr,
        *,
        degree_bivector: DegreeLike = 1,
        name: Optional[str] = None,
    ) -> "PoissonBracket":
        """Factory mirror of the constructor with a functional name."""
        return cls(pi, degree_bivector=degree_bivector, name=name)

    # ---- accessors -------------------------------------------------- #

    @property
    def bivector(self) -> Expr:
        return self._pi

    @property
    def derived(self) -> DerivedBracket:
        """The underlying :class:`DerivedBracket` ``(sn, π)``."""
        return self._derived

    @property
    def name(self) -> str:
        return self._name

    # ---- three equivalent views ------------------------------------ #

    def expand(
        self,
        f: Expr,
        g: Expr,
        registry: Optional[PropertyRegistry] = None,
    ) -> Expr:
        """``{f, g}_π = [[f, π]_SN, g]_SN`` — the derived form."""
        return self._derived.expand(f, g, registry)

    def via_hamiltonian(
        self,
        f: Expr,
        g: Expr,
    ) -> Expr:
        """``{f, g}_π = X_f(g)`` — the Hamiltonian form.

        Returns an :class:`Act` of
        :class:`~gradalg.calculus.hamiltonian_vf.HamiltonianVectorField`
        on ``g``. The shape leaves ``X_f`` symbolic; the caller pipes it
        through the expansion layer (or through
        :meth:`HamiltonianVectorField.derived_expansion`) when they want
        the SN expansion.
        """
        if not isinstance(f, Expr) or not isinstance(g, Expr):
            raise TypeError("via_hamiltonian requires Expr operands")
        return Act(self.hamiltonian_vf(f), g)

    def hamiltonian_vf(self, f: Expr) -> HamiltonianVectorField:
        """Build the Hamiltonian vector field ``X_f`` over ``π``."""
        return _hamiltonian_vf(f, bivector=self._pi)

    # ---- Jacobi -------------------------------------------------- #

    def jacobi_obstruction(
        self,
        registry: Optional[PropertyRegistry] = None,
    ) -> Expr:
        """``[π, π]_SN`` — the Poisson condition as an :class:`Expr`."""
        return self._derived.jacobi_obstruction(registry)

    def jacobi_condition(
        self,
        registry: Optional[PropertyRegistry] = None,
    ) -> VanishingCondition:
        """The Poisson condition as a :class:`VanishingCondition`."""
        return VanishingCondition(
            obstruction=self.jacobi_obstruction(registry),
            name=f"Poisson Jacobi condition on {self._name}",
        )

    def prove_jacobi_reduction(
        self,
        f: Expr,
        g: Expr,
        h: Expr,
        *,
        registry: Optional[PropertyRegistry] = None,
    ) -> ProofChain:
        """Reduce triple Jacobi ``(f, g, h)`` to the obstruction ``[π, π]_SN``.

        Returns a :class:`ProofChain` whose single top-level step cites
        the Derived Bracket Theorem: the cyclic Jacobi sum rewrites to
        ``[π, π]_SN``. The chain does *not* discharge the obstruction —
        for atomic ``π`` it stays opaque, and the caller is expected to
        supply ``[π, π]_SN = 0`` as a hypothesis (that is the defining
        property of a Poisson bivector). Callers that have a concrete
        ``π`` whose self-bracket simplifies to zero should use
        :class:`~gradalg.proof.strategies.DerivedBracketStrategy` via
        :func:`gradalg.proof.verifier.prove_jacobi` instead — that path
        closes the ProofChain all the way to :class:`Integer` ``0``.
        """
        jacobi_sum = self._derived.graded_jacobi_obstruction(f, g, h, registry)
        obstruction_raw = self._derived.jacobi_obstruction_raw()
        obstruction = self._derived.jacobi_obstruction(registry)
        chain = ProofChain()
        chain.append(
            ProofStep(
                jacobi_sum,
                obstruction_raw,
                rule="DerivedBracketTheorem",
                justification=(
                    f"Jacobi on {self._name} ⟺ [π, π]_SN = 0 "
                    f"(Derived Bracket Theorem)"
                ),
                provenance_tag="theorem",
            )
        )
        if obstruction != obstruction_raw:
            chain.append(
                ProofStep(
                    obstruction_raw,
                    obstruction,
                    rule="sn-expand",
                    justification="apply SN definition to [π, π]",
                )
            )
        return chain

    # ---- dunder ---------------------------------------------------- #

    def __repr__(self) -> str:
        return f"PoissonBracket(π={self._pi._repr_inner()})"


# --------------------------------------------------------------------- #
# Factory                                                                #
# --------------------------------------------------------------------- #


def poisson_bracket(
    pi: Expr,
    *,
    degree_bivector: DegreeLike = 1,
    name: Optional[str] = None,
) -> PoissonBracket:
    """Build ``{·, ·}_π`` — mirror of :meth:`PoissonBracket.from_bivector`."""
    return PoissonBracket(pi, degree_bivector=degree_bivector, name=name)


# --------------------------------------------------------------------- #
# Seed the canonical Poisson Jacobi theorem in the Theorem Book          #
# --------------------------------------------------------------------- #


def _build_poisson_jacobi_theorem() -> Theorem:
    """Construct the canonical ``poisson_jacobi`` theorem record.

    Uses generic symbols ``(π, f, g, h)`` with Graded SN degrees so the
    :meth:`graded_jacobi_obstruction` parity computation succeeds. The
    resulting :class:`ProofChain` is the one-step reduction ``triple
    cyclic Jacobi → [π, π]_SN`` that
    :meth:`PoissonBracket.prove_jacobi_reduction` emits on arbitrary
    ``(f, g, h)``; here we fix the operand triple to the canonical
    ``(f, g, h)`` so the theorem record is concrete.

    The theorem's ``from_axioms`` lists the Derived Bracket Theorem and
    the hypothesis ``[π, π]_SN = 0`` — together they close the
    conditional chain into an unconditional Jacobi identity on
    ``{·, ·}_π``.
    """
    pi = Symbol("π")
    f, g, h = Symbol("f"), Symbol("g"), Symbol("h")
    reg = PropertyRegistry()
    reg.declare(pi, Graded(degree=1))
    reg.declare(f, Graded(degree=-1))
    reg.declare(g, Graded(degree=-1))
    reg.declare(h, Graded(degree=-1))
    poisson = PoissonBracket.from_bivector(pi)
    chain = poisson.prove_jacobi_reduction(f, g, h, registry=reg)
    return Theorem(
        name="poisson_jacobi",
        statement="{f, g, h}_π cyclic sum = 0 when [π, π]_SN = 0",
        from_axioms=(
            "Derived Bracket Theorem",
            "[π, π]_SN = 0 (Poisson hypothesis)",
        ),
        proof=chain,
        notes=(
            "Derived via DerivedBracketStrategy on {·,·}_π = "
            "DerivedBracket(sn, π). The chain reduces the triple cyclic "
            "Jacobi sum on (f, g, h) to the universal obstruction "
            "[π, π]_SN; supplying the Poisson hypothesis discharges it."
        ),
    )


#: The Poisson-Jacobi reduction theorem. Seeded into
#: :data:`~gradalg.library.theorem_book.theorem_book` at import time so
#: downstream code can cite it via ``theorem_book.get("poisson_jacobi")``.
THEOREM_POISSON_JACOBI = _build_poisson_jacobi_theorem()


if "poisson_jacobi" not in theorem_book:
    theorem_book.add(THEOREM_POISSON_JACOBI)
