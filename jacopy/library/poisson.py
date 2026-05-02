"""
Poisson bracket library.

A :class:`PoissonBracket` wraps a bivector ``π`` and exposes three
complementary views of ``{f, g}_π``:

* the **derived** form via
  :class:`~jacopy.brackets.derived.DerivedBracket` ``(sn, π, degree_Q=1)``,
  which reduces to ``[[f, π]_SN, g]_SN``;
* the **Hamiltonian** form via
  :func:`~jacopy.calculus.hamiltonian_vf.hamiltonian_vf`, where
  ``{f, g}_π = X_f(g)``;
* the universal **Jacobi obstruction** ``[π, π]_SN`` as a
  :class:`~jacopy.brackets.derived.VanishingCondition`.

The Poisson Jacobi identity reduces to a single equation on ``π``,
the Derived Bracket Theorem says Jacobi on ``{·, ·}_π`` holds iff
``[π, π]_SN = 0``. This module records that reduction as the seeded
theorem :data:`THEOREM_POISSON_JACOBI` in the package-wide
:data:`~jacopy.library.theorem_book.theorem_book`, so downstream code
can cite the result without rebuilding its :class:`ProofChain`.
"""

from __future__ import annotations

from typing import Optional

from jacopy.algebra.derivation import Act
from jacopy.brackets.derived import DerivedBracket, VanishingCondition
from jacopy.brackets.koszul import KoszulBracket
from jacopy.brackets.schouten import sn
from jacopy.calculus.hamiltonian_vf import (
    HamiltonianVectorField,
    hamiltonian_vf as _hamiltonian_vf,
)
from jacopy.calculus.musical import Sharp
from jacopy.core.expr import Expr, Symbol
from jacopy.core.properties import Graded
from jacopy.core.registry import PropertyRegistry
from jacopy.core.symbolic_degree import DegreeLike
from jacopy.library.theorem_book import Theorem, theorem_book
from jacopy.proof.chain import ProofChain
from jacopy.proof.step import ProofStep
from jacopy.proof.verifier import prove_equivalence


# --------------------------------------------------------------------- #
# Poisson bracket wrapper                                                #
# --------------------------------------------------------------------- #


class PoissonBracket:
    """``{·, ·}_π``, the Poisson bracket of a bivector ``π``.

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
    The class is deliberately thin, every computation delegates into
    :class:`~jacopy.brackets.derived.DerivedBracket` or
    :func:`~jacopy.calculus.hamiltonian_vf.hamiltonian_vf`. The value
    it adds is naming the bracket (``PoissonBracket.from_bivector(π)``
    reads better than constructing the derived bracket directly at call
    sites) and bundling the three equivalent views so they stay in
    sync. The Jacobi reduction is surfaced as
    :meth:`prove_jacobi_reduction`, a single-step chain asserting the
    Derived Bracket Theorem, rather than as a ``prove_jacobi`` that
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
        """``π^♯``, the musical map ``T*M → TM`` induced by the bivector."""
        return self._sharp

    @property
    def koszul_derived(self) -> DerivedBracket:
        """Form-level derived bracket ``DerivedBracket(sn, π, acting_on=π^♯)``.

        This is the shape the library uses when evaluating ``{·, ·}_π``
        on 1-forms, the anchor lifts the forms to vector fields via
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
        """``{f, g}_π = [[f, π]_SN, g]_SN``, the derived form."""
        return self._derived.expand(f, g, registry)

    def via_hamiltonian(
        self,
        f: Expr,
        g: Expr,
    ) -> Expr:
        """``{f, g}_π = X_f(g)``, the Hamiltonian form.

        Returns an :class:`Act` of
        :class:`~jacopy.calculus.hamiltonian_vf.HamiltonianVectorField`
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

    def bivector_eval(
        self,
        f: Expr,
        g: Expr,
        *,
        d: Optional["ExteriorDerivative"] = None,  # type: ignore[name-defined]
    ) -> Expr:
        r"""``{f, g}_π = π(df, dg)``, the bivector-evaluation view.

        Returns ``MultiEval(π, d(f), d(g), slot_kind="covector",
        alternating=True)``. The ``alternating=True`` flag arms the
        :class:`~jacopy.calculus.multi_eval_axioms.MultiEvalAlternatingNormalDefinition`
        canonicalisation and the
        :class:`~jacopy.calculus.multi_eval_axioms.MultiEvalRepeatArgZeroDefinition`
        zero-rule on this node, so swapping ``f`` and ``g`` introduces
        a sign and ``π(df, df) = 0`` collapses on its own, no inline
        antisymmetry axiom needed.

        Composes with :class:`~jacopy.calculus.multi_eval_scalar_axioms.MultiEvalScalarPullDefinition`
        (Faz 12.B #6) so ``{f·g, h}_π = f·{g, h}_π + g·{f, h}_π``-style
        Leibniz expansions surface through the engine, and with
        :class:`~jacopy.calculus.musical.MusicalCompatibilityBilinearDefinition`
        (Faz 12.B #8) when the same ``π`` is the musical inverse of a
        symplectic ``ω``.

        Parameters
        ----------
        f, g
            Function operands. The expression ``d(f), d(g)`` is wrapped
            via :func:`~jacopy.calculus.exterior_d.d` (or the supplied
            override).
        d
            Optional :class:`~jacopy.calculus.exterior_d.ExteriorDerivative`
            override, use a bundle-specific ``d_E`` when this Poisson
            bracket lives on a Lie algebroid.
        """
        # Late import: ``calculus`` depends on this module's siblings,
        # and pulling :mod:`exterior_d` at module import would tighten
        # the already-fragile library⇆calculus boundary. Resolving on
        # first call lets the import graph stay relaxed.
        from jacopy.calculus.exterior_d import d as _default_d
        from jacopy.core.multi_eval import multi_eval

        if not isinstance(f, Expr) or not isinstance(g, Expr):
            raise TypeError("bivector_eval requires Expr operands")
        d_op = _default_d if d is None else d
        return multi_eval(
            self._pi,
            Act(d_op, f),
            Act(d_op, g),
            slot_kind="covector",
            alternating=True,
        )

    # ---- form-level (Koszul) view ---------------------------------- #

    def koszul_expand(
        self,
        alpha: Expr,
        beta: Expr,
        registry: Optional[PropertyRegistry] = None,
    ) -> Expr:
        """``{α, β}_π = L_{π^♯(α)} β − L_{π^♯(β)} α − d⟨π^♯(α), β⟩``.

        The form-level view of the Poisson bracket, the classical
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
        :func:`~jacopy.proof.verifier.prove_equivalence` closes the
        chain in a single reflexive step. The value isn't the depth of
        the proof, it is having the classical/derived agreement
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
        """``[π, π]_SN``, the Poisson condition as an :class:`Expr`."""
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
        ``[π, π]_SN``. The chain does *not* discharge the obstruction,
        for atomic ``π`` it stays opaque, and the caller is expected to
        supply ``[π, π]_SN = 0`` as a hypothesis (that is the defining
        property of a Poisson bivector). Callers that have a concrete
        ``π`` whose self-bracket simplifies to zero should use
        :class:`~jacopy.proof.strategies.DerivedBracketStrategy` via
        :func:`jacopy.proof.verifier.prove_jacobi` instead, that path
        closes the ProofChain all the way to :class:`Integer` ``0``.
        """
        return self._prove_jacobi_reduction_chain(
            self._derived, f, g, h, registry
        )

    def prove_jacobi_by_definitions(
        self,
        f: Expr,
        g: Expr,
        h: Expr,
        *,
        registry: Optional[PropertyRegistry] = None,
    ) -> ProofChain:
        r"""Vaisman-style step-by-step expansion of cyclic Poisson Jacobi.

        Builds a three-step :class:`ProofChain` that takes the cyclic
        Poisson Jacobi sum

        .. math::

            \sum_{\mathrm{cyc}}\{f,\{g,h\}_\pi\}_\pi

        through to

        .. math::

            \tfrac12\,[\pi,\pi]_{SN}(df,\,dg,\,dh)

        by applying three definitional rewrites — *no* citation of the
        Derived Bracket Theorem. Each step's ``provenance_tag`` is
        ``"axiom"`` because each rewrite is a definition of one of the
        underlying objects:

        1. **Hamiltonian-VF view**: :math:`\{f, X\}_\pi = X_f(X)`
           (definition of :math:`X_f` as the Hamiltonian vector field).
        2. **Bivector view**: :math:`\{\varphi, \psi\}_\pi
           = \pi(d\varphi, d\psi)` (definition of the Poisson bracket
           via :math:`\pi`).
        3. **SN-on-bivector formula**: :math:`\tfrac12 [\pi,\pi]_{SN}
           (\alpha,\beta,\gamma) = \sum_{\mathrm{cyc}}
           X_f(\pi(\beta,\gamma))` (definition of
           :math:`[\pi,\pi]_{SN}` evaluated on three covectors).

        Both sides reach the common form

        .. math::

            X_f(\{g,h\}_\pi) + X_g(\{h,f\}_\pi) + X_h(\{f,g\}_\pi),

        and the chain records the rewrites that take the cyclic Jacobi
        sum on the left through this common form to
        :math:`\tfrac12 [\pi,\pi]_{SN}(df, dg, dh)` on the right.

        The classical Vaisman / Marsden-Ratiu derivation. Compare with
        :meth:`prove_jacobi_reduction`, which cites the Derived Bracket
        Theorem in a single ``"theorem"``-tagged step. Both produce the
        same identity; they differ in *granularity* (3 axiom steps vs.
        1 theorem citation) and *provenance* (definitional unfolding
        vs. seeded theorem reference).

        The final operator-level statement
        :math:`[\pi,\pi]_{SN} = 0`, deduced from
        :math:`[\pi,\pi]_{SN}(df, dg, dh) = 0` for all
        :math:`f, g, h`, is the multilinearity / locality argument and
        sits *outside* the chain — it is the user's responsibility to
        supply that interpretation.

        Parameters
        ----------
        f, g, h
            Function-like arguments (``Graded(degree=0)`` symbols, or
            ``Functions("f g h", registry=reg)`` output).
        registry
            Optional :class:`PropertyRegistry`; forwarded to the
            internal calls that need degree information.

        Returns
        -------
        ProofChain
            Three axiom-tagged steps; the chain's ``initial`` is the
            cyclic Poisson Jacobi compact form, ``final`` is
            :math:`\tfrac12 [\pi,\pi]_{SN}` evaluated on the three
            exterior derivatives.
        """
        # Lazy imports to keep the module-level import graph quiet:
        from jacopy.algebra.derivation import Act
        from jacopy.brackets.base import BracketApply
        from jacopy.brackets.schouten import sn as default_sn
        from jacopy.calculus.exterior_d import d as default_d
        from jacopy.core.multi_eval import MultiEval

        # LHS compact: cyclic Poisson Jacobi sum on (f, g, h)
        def _P(a: Expr, b: Expr) -> Expr:
            return BracketApply(self._derived, a, b)

        lhs_compact = (
            _P(f, _P(g, h)) + _P(g, _P(h, f)) + _P(h, _P(f, g))
        )

        # LHS Hamiltonian view (after axiom 1): X_f({g,h}_π) + cyclic
        lhs_ham = (
            self.via_hamiltonian(f, _P(g, h))
            + self.via_hamiltonian(g, _P(h, f))
            + self.via_hamiltonian(h, _P(f, g))
        )

        # RHS form0: ½[π, π]_SN(df, dg, dh)  (the SN evaluation)
        sn_self = BracketApply(default_sn, self._pi, self._pi)
        df = Act(default_d, f)
        dg = Act(default_d, g)
        dh = Act(default_d, h)
        rhs_form0 = MultiEval(
            sn_self, df, dg, dh,
            alternating=True, slot_kind="covector",
        )

        # RHS form1 (after axiom 3): X_f(π(dg, dh)) + cyclic
        pi_gh = self.bivector_eval(g, h)
        pi_hf = self.bivector_eval(h, f)
        pi_fg = self.bivector_eval(f, g)
        rhs_form1 = (
            self.via_hamiltonian(f, pi_gh)
            + self.via_hamiltonian(g, pi_hf)
            + self.via_hamiltonian(h, pi_fg)
        )

        # RHS form2 (after axiom 2): X_f({g, h}_π) + cyclic = lhs_ham
        # (no separate variable needed — same Expr as lhs_ham)

        # Build the chain: lhs_compact → lhs_ham → rhs_form1 → rhs_form0
        chain = ProofChain()
        chain.append(ProofStep(
            lhs_compact, lhs_ham,
            rule="hamiltonian-view",
            justification=(
                "{f, X}_π = X_f(X)  (Hamiltonian-VF definition)"
            ),
            provenance_tag="axiom",
        ))
        chain.append(ProofStep(
            lhs_ham, rhs_form1,
            rule="bivector-eq-bracket",
            justification=(
                "{φ, ψ}_π = π(dφ, dψ)  (bivector definition of {·,·}_π)"
            ),
            provenance_tag="axiom",
        ))
        chain.append(ProofStep(
            rhs_form1, rhs_form0,
            rule="sn-bivector-formula",
            justification=(
                "½ [π, π]_SN(α, β, γ) = X_f(π(β, γ)) + cyclic  "
                "(SN-on-bivector definition)"
            ),
            provenance_tag="axiom",
        ))
        return chain

    def koszul_jacobi_condition(
        self,
        registry: Optional[PropertyRegistry] = None,
    ) -> VanishingCondition:
        """Form-level Jacobi condition, same ``[π, π]_SN`` obstruction.

        The universal obstruction only depends on ``(base, Q)``, so this
        condition wraps the same :class:`Expr` as :meth:`jacobi_condition`
       , the ``π^♯`` anchor in :attr:`koszul_derived` doesn't shift it.
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

        Form-level counterpart of :meth:`prove_jacobi_reduction`, same
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
    """Build ``{·, ·}_π``, mirror of :meth:`PoissonBracket.from_bivector`."""
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
    the hypothesis ``[π, π]_SN = 0``, together they close the
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
#: :data:`~jacopy.library.theorem_book.theorem_book` at import time so
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
    :class:`Expr`, they share the ``Sharp(π)`` anchor by construction
   , so the proof closes in a single reflexive step. The seeded
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


#: The Poisson–Koszul equivalence theorem, the form-level counterpart
#: of :data:`THEOREM_POISSON_JACOBI`. Seeded into
#: :data:`~jacopy.library.theorem_book.theorem_book` at import time.
THEOREM_POISSON_KOSZUL_EQUIVALENCE = _build_poisson_koszul_equivalence_theorem()


if "poisson_koszul_equivalence" not in theorem_book:
    theorem_book.add(THEOREM_POISSON_KOSZUL_EQUIVALENCE)


def _build_poisson_koszul_jacobi_theorem() -> Theorem:
    """Construct the canonical ``poisson_koszul_jacobi`` theorem.

    Form-level counterpart of :data:`THEOREM_POISSON_JACOBI`: the cyclic
    Koszul Jacobi sum on ``(α, β, γ)`` (1-forms, lifted through ``π^♯``)
    reduces to the *same* universal obstruction ``[π, π]_SN``. The
    structural identity ``koszul_derived.jacobi_obstruction ==
    derived.jacobi_obstruction``, the ``acting_on`` anchor doesn't
    shift the ``[Q, Q]_base`` on a DerivedBracket, is what lets this
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
            "[π, π]_SN, anchor ``acting_on=Sharp(π)`` reshapes the "
            "expansion but leaves [Q, Q]_base untouched, so one "
            "Poisson hypothesis discharges both views at once."
        ),
    )


#: The form-level Poisson–Koszul Jacobi reduction theorem. Seeded into
#: :data:`~jacopy.library.theorem_book.theorem_book` at import time.
THEOREM_POISSON_KOSZUL_JACOBI = _build_poisson_koszul_jacobi_theorem()


if "poisson_koszul_jacobi" not in theorem_book:
    theorem_book.add(THEOREM_POISSON_KOSZUL_JACOBI)
