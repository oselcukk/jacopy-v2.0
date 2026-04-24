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
from gradalg.brackets.koszul import KoszulBracket
from gradalg.brackets.schouten import sn
from gradalg.calculus.hamiltonian_vf import (
    HamiltonianVectorField,
    hamiltonian_vf as _hamiltonian_vf,
)
from gradalg.calculus.musical import Sharp
from gradalg.core.expr import Expr, Symbol
from gradalg.core.properties import Graded
from gradalg.core.registry import PropertyRegistry
from gradalg.core.symbolic_degree import DegreeLike
from gradalg.library.theorem_book import Theorem, theorem_book
from gradalg.proof.chain import ProofChain
from gradalg.proof.step import ProofStep
from gradalg.proof.verifier import prove_equivalence


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

    __slots__ = (
        "_pi",
        "_derived",
        "_sharp",
        "_koszul_derived",
        "_koszul_classical",
        "_name",
    )

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
        # Sharp (``π^♯``) plus the two form-level views it unlocks. The
        # sharp is the canonical T*M → TM lift on a Poisson manifold and
        # lets both the derived bracket and the classical Koszul
        # formula run directly on 1-form operands without the caller
        # supplying a separate anchor.
        self._sharp = Sharp(pi)
        self._koszul_derived = DerivedBracket(
            sn,
            pi,
            degree_Q=degree_bivector,
            acting_on=self._sharp,
            name=f"{{·,·}}_{pi._repr_inner()},♯",
        )
        self._koszul_classical = KoszulBracket(
            self._sharp,
            name=f"[·,·]_K,{pi._repr_inner()}",
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
    def sharp(self) -> Sharp:
        """``π^♯`` — the musical map ``T*M → TM`` induced by the bivector."""
        return self._sharp

    @property
    def koszul_derived(self) -> DerivedBracket:
        """Form-level derived bracket ``DerivedBracket(sn, π, acting_on=π^♯)``.

        This is the shape the library uses when evaluating ``{·, ·}_π``
        on 1-forms — the anchor lifts the forms to vector fields via
        ``π^♯`` and the expansion emits the classical Koszul three-term
        formula.
        """
        return self._koszul_derived

    @property
    def koszul_classical(self) -> KoszulBracket:
        """Classical :class:`KoszulBracket` with anchor ``π^♯``.

        Structurally the ``KoszulBracket(Sharp(π))`` built at
        construction, kept as a handle so
        :meth:`prove_koszul_equivalence` can reference a fixed anchor
        instance and the proof close in a single reflexive step.
        """
        return self._koszul_classical

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

    # ---- form-level (Koszul) view ---------------------------------- #

    def koszul_expand(
        self,
        alpha: Expr,
        beta: Expr,
        registry: Optional[PropertyRegistry] = None,
    ) -> Expr:
        """``{α, β}_π = L_{π^♯(α)} β − L_{π^♯(β)} α − d⟨π^♯(α), β⟩``.

        The form-level view of the Poisson bracket — the classical
        Koszul three-term formula on 1-forms, produced through the
        :attr:`koszul_derived` bracket so the anchor ``π^♯`` is fixed to
        this bivector.

        Structurally equal to
        ``poisson.koszul_classical.expand(α, β, registry)``: both paths
        go through the same ``Sharp(π)`` instance and emit the same
        :class:`Sum`. Use this method when expanding by value; use
        :meth:`prove_koszul_equivalence` to obtain the equality as a
        transcripted :class:`ProofChain`.
        """
        return self._koszul_derived.expand(alpha, beta, registry)

    def prove_koszul_equivalence(
        self,
        alpha: Expr,
        beta: Expr,
        *,
        registry: Optional[PropertyRegistry] = None,
    ) -> ProofChain:
        """Close ``koszul_classical.expand(α, β) = koszul_derived.expand(α, β)``.

        Both sides are computed with the same ``π^♯`` anchor, so their
        :class:`Expr` outputs are structurally equal and
        :func:`~gradalg.proof.verifier.prove_equivalence` closes the
        chain in a single reflexive step. The value isn't the depth of
        the proof — it is having the classical/derived agreement
        recorded as a citable :class:`ProofChain` on this specific
        ``(α, β)`` triple.
        """
        return prove_equivalence(
            self._koszul_classical,
            self._koszul_derived,
            alpha,
            beta,
            registry=registry,
        )

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
        return self._prove_jacobi_reduction_chain(
            self._derived, f, g, h, registry
        )

    def koszul_jacobi_condition(
        self,
        registry: Optional[PropertyRegistry] = None,
    ) -> VanishingCondition:
        """Form-level Jacobi condition — same ``[π, π]_SN`` obstruction.

        The universal obstruction only depends on ``(base, Q)``, so this
        condition wraps the same :class:`Expr` as :meth:`jacobi_condition`
        — the ``π^♯`` anchor in :attr:`koszul_derived` doesn't shift it.
        What differs from the function-level condition is the *name*,
        which is keyed to the Koszul view for display / theorem-book
        citations.
        """
        return VanishingCondition(
            obstruction=self._koszul_derived.jacobi_obstruction(registry),
            name=f"Koszul Jacobi condition on {self._name}",
        )

    def prove_koszul_jacobi_reduction(
        self,
        alpha: Expr,
        beta: Expr,
        gamma: Expr,
        *,
        registry: Optional[PropertyRegistry] = None,
    ) -> ProofChain:
        """Reduce triple Koszul Jacobi ``(α, β, γ)`` to ``[π, π]_SN``.

        Form-level counterpart of :meth:`prove_jacobi_reduction` — same
        Derived Bracket Theorem citation, same ``[π, π]_SN`` obstruction,
        just driven by the :attr:`koszul_derived` bracket so the Jacobi
        sum is written on 1-form operands lifted through ``π^♯``. The
        structural identity witnessed in B.2 (Koszul Jacobi obstruction
        equals Poisson Jacobi obstruction) is what lets a single helper
        serve both views.
        """
        return self._prove_jacobi_reduction_chain(
            self._koszul_derived, alpha, beta, gamma, registry
        )

    def _prove_jacobi_reduction_chain(
        self,
        bracket: DerivedBracket,
        a: Expr,
        b: Expr,
        c: Expr,
        registry: Optional[PropertyRegistry],
    ) -> ProofChain:
        """Shared Jacobi-reduction chain for a DerivedBracket(sn, π, …).

        Produces the Derived Bracket Theorem step that rewrites the
        cyclic Jacobi sum on ``(a, b, c)`` to the raw
        :class:`BracketApply` ``[π, π]_SN``, then appends an ``sn-expand``
        step when the base bracket's own expansion narrows the
        obstruction further. Callers (function-level vs form-level) only
        differ in which DerivedBracket they hand in.
        """
        jacobi_sum = bracket.graded_jacobi_obstruction(a, b, c, registry)
        obstruction_raw = bracket.jacobi_obstruction_raw()
        obstruction = bracket.jacobi_obstruction(registry)
        chain = ProofChain()
        chain.append(
            ProofStep(
                jacobi_sum,
                obstruction_raw,
                rule="DerivedBracketTheorem",
                justification=(
                    f"Jacobi on {bracket.name} ⟺ [π, π]_SN = 0 "
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


def _build_poisson_koszul_equivalence_theorem() -> Theorem:
    """Construct the canonical ``poisson_koszul_equivalence`` theorem.

    On 1-forms the Poisson bracket agrees with the classical Koszul
    bracket whose anchor is ``π^♯``:

        {α, β}_π = L_{π^♯(α)} β − L_{π^♯(β)} α − d⟨π^♯(α), β⟩.

    Both :attr:`PoissonBracket.koszul_derived` and
    :attr:`PoissonBracket.koszul_classical` emit the same
    :class:`Expr` — they share the ``Sharp(π)`` anchor by construction
    — so the proof closes in a single reflexive step. The seeded
    record fixes generic symbols ``(π, α, β)`` as a concrete witness;
    downstream callers produce their own chain on the concrete operands
    they care about via :meth:`PoissonBracket.prove_koszul_equivalence`.
    """
    pi = Symbol("π")
    alpha = Symbol("α")
    beta = Symbol("β")
    reg = PropertyRegistry()
    reg.declare(pi, Graded(degree=1))
    reg.declare(alpha, Graded(degree=1))
    reg.declare(beta, Graded(degree=1))
    poisson = PoissonBracket.from_bivector(pi)
    chain = poisson.prove_koszul_equivalence(alpha, beta, registry=reg)
    return Theorem(
        name="poisson_koszul_equivalence",
        statement=(
            "{α, β}_π = L_{π^♯(α)} β − L_{π^♯(β)} α − d⟨π^♯(α), β⟩ "
            "(derived = classical Koszul via π^♯)"
        ),
        from_axioms=(
            "derived bracket definition",
            "classical Koszul bracket definition",
            "π^♯ = Sharp(π) as common anchor",
        ),
        proof=chain,
        notes=(
            "Both views are built with the same Sharp(π) instance, so "
            "their Exprs are structurally equal and prove_equivalence "
            "closes in one reflexive step. The theorem records that "
            "structural identity as a transcribable ProofChain."
        ),
    )


#: The Poisson–Koszul equivalence theorem — the form-level counterpart
#: of :data:`THEOREM_POISSON_JACOBI`. Seeded into
#: :data:`~gradalg.library.theorem_book.theorem_book` at import time.
THEOREM_POISSON_KOSZUL_EQUIVALENCE = _build_poisson_koszul_equivalence_theorem()


if "poisson_koszul_equivalence" not in theorem_book:
    theorem_book.add(THEOREM_POISSON_KOSZUL_EQUIVALENCE)


def _build_poisson_koszul_jacobi_theorem() -> Theorem:
    """Construct the canonical ``poisson_koszul_jacobi`` theorem.

    Form-level counterpart of :data:`THEOREM_POISSON_JACOBI`: the cyclic
    Koszul Jacobi sum on ``(α, β, γ)`` (1-forms, lifted through ``π^♯``)
    reduces to the *same* universal obstruction ``[π, π]_SN``. The
    structural identity ``koszul_derived.jacobi_obstruction ==
    derived.jacobi_obstruction`` — the ``acting_on`` anchor doesn't
    shift the ``[Q, Q]_base`` on a DerivedBracket — is what lets this
    record share its Poisson hypothesis with the function-level
    theorem.
    """
    pi = Symbol("π")
    alpha = Symbol("α")
    beta = Symbol("β")
    gamma = Symbol("γ")
    reg = PropertyRegistry()
    reg.declare(pi, Graded(degree=1))
    reg.declare(alpha, Graded(degree=1))
    reg.declare(beta, Graded(degree=1))
    reg.declare(gamma, Graded(degree=1))
    poisson = PoissonBracket.from_bivector(pi)
    chain = poisson.prove_koszul_jacobi_reduction(
        alpha, beta, gamma, registry=reg,
    )
    return Theorem(
        name="poisson_koszul_jacobi",
        statement=(
            "Koszul Jacobi on {·,·}_π cyclic sum = 0 when [π, π]_SN = 0"
        ),
        from_axioms=(
            "Derived Bracket Theorem",
            "π^♯ = Sharp(π) as form-lift anchor",
            "[π, π]_SN = 0 (Poisson hypothesis)",
        ),
        proof=chain,
        notes=(
            "Form-level analogue of poisson_jacobi. The Koszul view's "
            "Jacobi obstruction coincides with the SN self-bracket "
            "[π, π]_SN — anchor ``acting_on=Sharp(π)`` reshapes the "
            "expansion but leaves [Q, Q]_base untouched — so one "
            "Poisson hypothesis discharges both views at once."
        ),
    )


#: The form-level Poisson–Koszul Jacobi reduction theorem. Seeded into
#: :data:`~gradalg.library.theorem_book.theorem_book` at import time.
THEOREM_POISSON_KOSZUL_JACOBI = _build_poisson_koszul_jacobi_theorem()


if "poisson_koszul_jacobi" not in theorem_book:
    theorem_book.add(THEOREM_POISSON_KOSZUL_JACOBI)
