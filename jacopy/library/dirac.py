"""
Dirac structures on the standard Courant algebroid.

A :class:`DiracStructure` pins down the data of a Dirac subbundle
``L ⊂ TM ⊕ T*M`` that is

* **maximally isotropic** with respect to the canonical symmetric
  pairing ``⟨(X, α), (Y, β)⟩ = ½(ι_X β + ι_Y α)``, and
* **involutive**, ``[Γ(L), Γ(L)]_C ⊂ Γ(L)``, with respect to the
  Courant bracket of the ambient :class:`CourantAlgebroid`.

The wrapper does not model subbundle membership symbolically (no
machinery for ``a ∈ Γ(L)`` as a proposition); what it does do is:

* surface the **isotropy obstruction** ``ι_X α`` for a single section
  (polarisation of the symmetric pairing) and the full pairing
  ``⟨a, b⟩`` for two sections, and
* record both isotropy and involutivity as **axiom-tagged** proof
  steps so downstream code can cite them as Dirac-defining properties.

Seeded theorems:

* ``dirac_isotropy``, isotropy is definitional on ``L``; the chain is a
  single axiom-tagged step on the generic section ``(X, α)``.
* ``dirac_involutivity``, involutivity is definitional on ``L``; the
  chain records the axiom on the generic section pairs.

Convenience factories:

* :func:`poisson_dirac`, graph of ``π♯ : T*M → TM`` for a Poisson
  bivector, ``L_π = {(π♯ α, α)}``.
* :func:`presymplectic_dirac`, graph of ``ω♭ : TM → T*M`` for a closed
  2-form, ``L_ω = {(X, ω♭ X)}``.

Both factories only record the subbundle name / generator; the isotropy
and involutivity axioms remain axioms on the Dirac wrapper. Proving
``dω = 0 ⇒ L_ω`` is Dirac (or ``[π, π] = 0 ⇒ L_π`` is Dirac) is a
separate theorem that slots in naturally here but is out of scope for
Stage D, both special cases inherit the axiom-step proofs on this
wrapper.
"""

from __future__ import annotations

from typing import Optional

from jacopy.algebra.derivation import Act
from jacopy.brackets.derived import VanishingCondition
from jacopy.brackets.dorfman import SectionPair
from jacopy.core.expr import Expr, Integer, Product, Rational, Sum, Symbol
from jacopy.core.properties import Graded
from jacopy.core.registry import PropertyRegistry
from jacopy.library.courant_algebroid import CourantAlgebroid
from jacopy.library.theorem_book import Theorem, theorem_book
from jacopy.proof.chain import ProofChain
from jacopy.proof.step import ProofStep


# --------------------------------------------------------------------- #
# DiracStructure                                                         #
# --------------------------------------------------------------------- #


class DiracStructure:
    """``L ⊂ TM ⊕ T*M``, a Dirac subbundle of a Courant algebroid.

    Parameters
    ----------
    courant
        The ambient :class:`CourantAlgebroid`. Cartan operators for the
        pairing (``ι``) are read from it.
    subbundle
        Symbolic name for the Dirac subbundle ``L``. Carried for
        display only, the wrapper does not model section membership.
    name
        Optional display name; defaults to ``f"Dirac({subbundle})"``.

    Notes
    -----
    The pairing returned by :meth:`pairing` is literally
    ``½(ι_X β + ι_Y α)``; its diagonal ``⟨a, a⟩ = ι_X α`` is the
    single-section isotropy obstruction exposed by
    :meth:`isotropy_obstruction`. Full bilinear isotropy follows from
    the diagonal via polarisation, downstream callers who need the
    bilinear form can cite :meth:`pairing` directly.

    Involutivity is definitional on ``L`` and therefore represented as
    a single axiom step: we cannot symbolically assert
    ``[a, b]_C ∈ Γ(L)`` without a subbundle-membership predicate, so
    :meth:`prove_involutivity` discharges a placeholder *involutivity
    obstruction symbol* through the axiom, keeping the shape of the
    proof consistent with every other axiom-step chain in the library.
    """

    __slots__ = ("_courant", "_subbundle", "_name")

    def __init__(
        self,
        courant: CourantAlgebroid,
        subbundle: Expr,
        *,
        name: Optional[str] = None,
    ) -> None:
        if not isinstance(courant, CourantAlgebroid):
            raise TypeError(
                "DiracStructure courant must be a CourantAlgebroid"
            )
        if not isinstance(subbundle, Expr):
            raise TypeError("DiracStructure subbundle must be an Expr")
        self._courant = courant
        self._subbundle = subbundle
        self._name = (
            name if name is not None
            else f"Dirac({subbundle._repr_inner()})"
        )

    # ---- accessors -------------------------------------------------- #

    @property
    def courant(self) -> CourantAlgebroid:
        return self._courant

    @property
    def subbundle(self) -> Expr:
        return self._subbundle

    @property
    def name(self) -> str:
        return self._name

    # ---- pairing ---------------------------------------------------- #

    def pairing(self, a: SectionPair, b: SectionPair) -> Expr:
        """``⟨a, b⟩ = ½(ι_X β + ι_Y α)`` on the ambient Courant algebroid.

        Uses the Courant algebroid's own interior factory so the pairing
        stays consistent with whatever Cartan operators the algebroid
        was constructed with.
        """
        if not isinstance(a, SectionPair) or not isinstance(b, SectionPair):
            raise TypeError("DiracStructure.pairing requires SectionPair operands")
        X, alpha = a.vector, a.form
        Y, beta = b.vector, b.form
        iota = self._courant.interior
        return Product(
            Rational(1, 2),
            Sum(Act(iota(X), beta), Act(iota(Y), alpha)),
        )

    def isotropy_obstruction(self, a: SectionPair) -> Expr:
        """``⟨a, a⟩ = ι_X α``, the self-pairing of a single section.

        Isotropy on ``L`` asserts this vanishes for every ``a ∈ Γ(L)``.
        Polarisation lifts the vanishing of the diagonal to vanishing
        of the full bilinear form, so this diagonal obstruction is
        sufficient.
        """
        if not isinstance(a, SectionPair):
            raise TypeError(
                "DiracStructure.isotropy_obstruction requires a SectionPair"
            )
        X, alpha = a.vector, a.form
        return Act(self._courant.interior(X), alpha)

    def isotropy_condition(self, a: SectionPair) -> VanishingCondition:
        """Wrap :meth:`isotropy_obstruction` as a :class:`VanishingCondition`."""
        return VanishingCondition(
            obstruction=self.isotropy_obstruction(a),
            name=f"isotropy of {self._name}",
        )

    def prove_isotropy(self, a: SectionPair) -> ProofChain:
        """Single axiom-step chain discharging the isotropy obstruction.

        On a Dirac structure, ``⟨a, a⟩ = 0`` for every ``a ∈ Γ(L)`` by
        definition. The chain has a single ``axiom``-tagged step that
        maps ``ι_X α`` to ``0`` citing the Dirac isotropy axiom. Callers
        that want the symbolic expansion should skip this method and
        work directly with :meth:`isotropy_obstruction`.
        """
        obs = self.isotropy_obstruction(a)
        chain = ProofChain()
        chain.append(
            ProofStep(
                obs,
                Integer(0),
                rule="DiracIsotropyAxiom",
                justification=(
                    f"⟨a, a⟩ = ι_X α = 0 on {self._name}, Dirac "
                    f"isotropy axiom on the subbundle {self._subbundle._repr_inner()}."
                ),
                provenance_tag="axiom",
            )
        )
        return chain

    # ---- involutivity ---------------------------------------------- #

    def _involutivity_obstruction_symbol(
        self, a: SectionPair, b: SectionPair
    ) -> Symbol:
        """Placeholder Expr for the involutivity obstruction.

        ``[a, b]_C ∈ Γ(L)`` cannot be asserted symbolically without a
        subbundle-membership predicate; we surface a named symbol that
        stands in for "involutivity obstruction of ``(a, b)`` against
        ``L``" so the axiom step has a concrete before-term to point at.
        """
        return Symbol(
            f"involutivity_obstruction({a._repr_inner()}, "
            f"{b._repr_inner()}; {self._subbundle._repr_inner()})"
        )

    def involutivity_condition(
        self, a: SectionPair, b: SectionPair
    ) -> VanishingCondition:
        """Involutivity of ``L`` as a :class:`VanishingCondition`.

        The obstruction is a placeholder symbol because subbundle
        membership is outside the Expr algebra. Its vanishing stands for
        ``[a, b]_C ∈ Γ(L)``.
        """
        return VanishingCondition(
            obstruction=self._involutivity_obstruction_symbol(a, b),
            name=f"involutivity of {self._name}",
        )

    def prove_involutivity(
        self, a: SectionPair, b: SectionPair
    ) -> ProofChain:
        """Single axiom-step chain discharging the involutivity axiom.

        Same shape as :meth:`prove_isotropy`: a single ``axiom``-tagged
        step whose before-term is the placeholder involutivity symbol
        and whose after-term is ``Integer(0)``, the citation form of
        "``[a, b]_C ∈ Γ(L)`` on a Dirac subbundle".
        """
        if not isinstance(a, SectionPair) or not isinstance(b, SectionPair):
            raise TypeError(
                "DiracStructure.prove_involutivity requires SectionPair operands"
            )
        obs = self._involutivity_obstruction_symbol(a, b)
        chain = ProofChain()
        chain.append(
            ProofStep(
                obs,
                Integer(0),
                rule="DiracInvolutivityAxiom",
                justification=(
                    f"[a, b]_C ∈ Γ({self._subbundle._repr_inner()}), "
                    f"Dirac involutivity axiom on {self._name}."
                ),
                provenance_tag="axiom",
            )
        )
        return chain

    # ---- dunder ---------------------------------------------------- #

    def __repr__(self) -> str:
        return (
            f"DiracStructure(subbundle={self._subbundle._repr_inner()}, "
            f"courant={self._courant.name})"
        )


# --------------------------------------------------------------------- #
# Special-case factories                                                 #
# --------------------------------------------------------------------- #


def poisson_dirac(
    pi: Expr,
    *,
    courant: Optional[CourantAlgebroid] = None,
    name: Optional[str] = None,
) -> DiracStructure:
    """Graph of ``π♯ : T*M → TM`` as a Dirac subbundle.

    For a Poisson bivector ``π``, ``L_π = {(π♯ α, α) : α ∈ T*M}`` is a
    Dirac structure iff ``[π, π]_{SN} = 0`` (i.e. iff ``π`` satisfies
    Jacobi). The factory records the subbundle name only, isotropy and
    involutivity ride through the Dirac wrapper's axiom steps the same
    way the generic case does. Downstream code that needs the
    ``[π,π] = 0 ⇒ L_π`` Dirac implication should compose this with
    :data:`~jacopy.library.poisson.THEOREM_POISSON_JACOBI`.
    """
    if not isinstance(pi, Expr):
        raise TypeError("poisson_dirac requires an Expr bivector")
    if courant is None:
        courant = CourantAlgebroid()
    elif not isinstance(courant, CourantAlgebroid):
        raise TypeError("poisson_dirac courant must be a CourantAlgebroid")
    subbundle = Symbol(f"graph(π♯_{pi._repr_inner()})")
    return DiracStructure(
        courant,
        subbundle,
        name=name if name is not None else f"Dirac_π({pi._repr_inner()})",
    )


def presymplectic_dirac(
    omega: Expr,
    *,
    courant: Optional[CourantAlgebroid] = None,
    name: Optional[str] = None,
) -> DiracStructure:
    """Graph of ``ω♭ : TM → T*M`` as a Dirac subbundle.

    For a closed 2-form ``ω``, ``L_ω = {(X, ω♭ X) : X ∈ TM}`` is a
    Dirac structure (the "presymplectic Dirac"). The factory records
    only the subbundle name; the Dirac wrapper discharges isotropy and
    involutivity via its axiom steps. As with :func:`poisson_dirac`,
    the closure implication ``dω = 0 ⇒ L_ω`` is Dirac is left to
    downstream composition.
    """
    if not isinstance(omega, Expr):
        raise TypeError("presymplectic_dirac requires an Expr 2-form")
    if courant is None:
        courant = CourantAlgebroid()
    elif not isinstance(courant, CourantAlgebroid):
        raise TypeError(
            "presymplectic_dirac courant must be a CourantAlgebroid"
        )
    subbundle = Symbol(f"graph(ω♭_{omega._repr_inner()})")
    return DiracStructure(
        courant,
        subbundle,
        name=name if name is not None else f"Dirac_ω({omega._repr_inner()})",
    )


# --------------------------------------------------------------------- #
# Seeded theorems                                                        #
# --------------------------------------------------------------------- #


def _build_dirac_isotropy_theorem() -> Theorem:
    """Generic witness: ``⟨(X, α), (X, α)⟩ = 0`` on ``L``."""
    X = Symbol("X")
    alpha = Symbol("α")
    reg = PropertyRegistry()
    reg.declare(X, Graded(degree=0))
    reg.declare(alpha, Graded(degree=1))
    L = Symbol("L")
    D = DiracStructure(CourantAlgebroid(), L)
    chain = D.prove_isotropy(SectionPair(X, alpha))
    return Theorem(
        name="dirac_isotropy",
        statement="⟨a, a⟩ = ι_X α = 0 for a = (X, α) ∈ Γ(L)",
        from_axioms=(
            "Dirac isotropy axiom (L ⊆ L^⊥ under canonical pairing)",
        ),
        proof=chain,
        notes=(
            "Isotropy is part of the definition of a Dirac subbundle; "
            "the chain records it as a one-step axiom citation. "
            "Polarisation lifts this diagonal vanishing to the full "
            "bilinear ``⟨a, b⟩ = 0`` on sections of ``L``."
        ),
    )


def _build_dirac_involutivity_theorem() -> Theorem:
    """Generic witness: ``[a, b]_C ∈ Γ(L)`` on ``L``."""
    X = Symbol("X")
    Y = Symbol("Y")
    alpha = Symbol("α")
    beta = Symbol("β")
    reg = PropertyRegistry()
    reg.declare(X, Graded(degree=0))
    reg.declare(Y, Graded(degree=0))
    reg.declare(alpha, Graded(degree=1))
    reg.declare(beta, Graded(degree=1))
    L = Symbol("L")
    D = DiracStructure(CourantAlgebroid(), L)
    chain = D.prove_involutivity(
        SectionPair(X, alpha), SectionPair(Y, beta)
    )
    return Theorem(
        name="dirac_involutivity",
        statement="[a, b]_C ∈ Γ(L) for a, b ∈ Γ(L)",
        from_axioms=(
            "Dirac involutivity axiom ([Γ(L), Γ(L)]_C ⊆ Γ(L))",
        ),
        proof=chain,
        notes=(
            "Involutivity is the second defining axiom of a Dirac "
            "structure; the chain is a one-step axiom citation. "
            "Subbundle membership is not modelled symbolically, so the "
            "obstruction is surfaced as a placeholder symbol, the "
            "axiom step maps it to 0 as the definitional citation."
        ),
    )


#: The Dirac isotropy axiom, catalogued for downstream citation.
THEOREM_DIRAC_ISOTROPY = _build_dirac_isotropy_theorem()

#: The Dirac involutivity axiom, catalogued for downstream citation.
THEOREM_DIRAC_INVOLUTIVITY = _build_dirac_involutivity_theorem()


if "dirac_isotropy" not in theorem_book:
    theorem_book.add(THEOREM_DIRAC_ISOTROPY)

if "dirac_involutivity" not in theorem_book:
    theorem_book.add(THEOREM_DIRAC_INVOLUTIVITY)
