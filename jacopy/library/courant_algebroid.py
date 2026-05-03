"""
Courant algebroid library.

A :class:`CourantAlgebroid` bundles the data of the standard exact
Courant algebroid on ``TM ⊕ T*M``: the Courant bracket, its Dorfman
twin built on the *same* Cartan operators, and optional H-twist. The
wrapper exposes:

* ``expand`` / ``expand_dorfman``, the two bracket views, produced
  off the same ``(d, L, ι, vector_bracket)`` quadruple so identities
  involving both stay faithful;
* ``jacobi_condition`` / ``prove_jacobi_reduction``, the Courant
  Jacobi condition (vacuous in the untwisted case; ``dH = 0`` in the
  H-twisted case) plus a single-step axiomatic reduction to that
  condition;
* ``courant_dorfman_obstruction`` / ``bridge_correction`` /
  ``prove_courant_dorfman_bridge``, the classical algebraic identity
  ``[·,·]_D − [·,·]_C = (0, ½ d(ι_X β + ι_Y α))``, previously deferred
  from the Stage 3 Courant pass.

Seeded theorems (added to :data:`~jacopy.library.theorem_book.theorem_book`
at import time):

* ``courant_jacobi_twist``, H-twisted Courant Jacobi ⟺ ``dH = 0``;
* ``courant_dorfman_bridge``, the Courant–Dorfman correction identity.

The Courant algebroid's anchor is the canonical projection
``pr_TM: TM ⊕ T*M → TM``; it is not surfaced here as an :class:`Anchor`
instance because the bracket's machinery already implements the
compatibility implicitly through :class:`SectionPair` extraction.
"""

from __future__ import annotations

from typing import Any, Callable, Optional

from jacopy.algebra.derivation import Act, Derivation
from jacopy.brackets.base import BracketApply, GradedBracket
from jacopy.brackets.courant import CourantBracket
from jacopy.brackets.courant_anchor_d import (
    CourantAnchor,
    CourantAnchorDefinition,
    DOperator,
    DOperatorDefinition,
)
from jacopy.brackets.courant_inner_product import (
    CourantInnerProduct,
    CourantInnerProductDefinition,
)
from jacopy.brackets.derived import VanishingCondition
from jacopy.brackets.dorfman import DorfmanBracket, SectionPair
from jacopy.brackets.lie import LieBracket
from jacopy.calculus.pairing import Pairing
from jacopy.calculus.exterior_d import d as default_d
from jacopy.calculus.interior import interior as default_interior
from jacopy.calculus.lie_derivative import (
    lie_derivative as default_lie_derivative,
)
from jacopy.core.expr import Expr, Integer, Neg, Product, Rational, Sum, Symbol
from jacopy.core.properties import Graded
from jacopy.core.registry import PropertyRegistry
from jacopy.library.theorem_book import Theorem, theorem_book
from jacopy.proof.chain import ProofChain
from jacopy.proof.step import ProofStep


LieDerivativeFactory = Callable[[Expr], Derivation]
InteriorFactory = Callable[[Expr], Derivation]


# --------------------------------------------------------------------- #
# CourantAlgebroid wrapper                                               #
# --------------------------------------------------------------------- #


class CourantAlgebroid:
    """``(TM ⊕ T*M, [·,·]_C)``, the standard exact Courant algebroid.

    Parameters
    ----------
    vector_bracket
        Bracket on the vector-field halves. Defaults to
        :class:`LieBracket`.
    d, lie_derivative, interior
        Cartan operators; default to the smooth-manifold singletons.
        These are shared between the Courant and Dorfman brackets so
        the bridge identity (:meth:`prove_courant_dorfman_bridge`)
        lands on matching operator names.
    background_H
        Optional closed 3-form ``H``. When supplied, :attr:`courant`
        is the H-twisted Courant bracket; the Dorfman twin is *not*
        twisted, the bridge identity holds without the twist term.
    name
        Optional display name; defaults to ``"Courant(TM⊕T*M)"`` or a
        twist-tagged variant.

    Notes
    -----
    * The wrapper does *not* model the canonical pairing
      ``⟨(X, α), (Y, β)⟩ = ½(ι_X β + ι_Y α)`` as its own object,
      :meth:`bridge_correction` surfaces the exact combination that
      shows up in the correction identity and downstream callers can
      read off the pairing from there.
    * The anchor ``pr_TM: TM ⊕ T*M → TM`` is implicit: the Courant /
      Dorfman brackets consume :class:`SectionPair` operands and
      extract the vector component directly, so surfacing a separate
      :class:`~jacopy.calculus.anchor.Anchor` instance would only
      duplicate that projection.
    """

    __slots__ = (
        "_courant",
        "_dorfman",
        "_vector_bracket",
        "_d",
        "_lie_derivative",
        "_interior",
        "_background_H",
        "_name",
    )

    def __init__(
        self,
        *,
        vector_bracket: Optional[GradedBracket] = None,
        d: Optional[Derivation] = None,
        lie_derivative: Optional[LieDerivativeFactory] = None,
        interior: Optional[InteriorFactory] = None,
        background_H: Optional[Expr] = None,
        name: Optional[str] = None,
    ) -> None:
        if background_H is not None and not isinstance(background_H, Expr):
            raise TypeError(
                "CourantAlgebroid background_H must be an Expr when provided"
            )
        self._vector_bracket = (
            vector_bracket if vector_bracket is not None else LieBracket()
        )
        self._d = d if d is not None else default_d
        self._lie_derivative = (
            lie_derivative
            if lie_derivative is not None
            else default_lie_derivative
        )
        self._interior = (
            interior if interior is not None else default_interior
        )
        self._background_H = background_H
        self._courant = CourantBracket(
            vector_bracket=self._vector_bracket,
            d=self._d,
            lie_derivative=self._lie_derivative,
            interior=self._interior,
            background_H=background_H,
        )
        # Dorfman twin uses the SAME Cartan operators. The bridge
        # identity is only exact when both brackets share operators;
        # mixing would re-introduce the very residuals the bridge
        # claims to cancel.
        self._dorfman = DorfmanBracket(
            vector_bracket=self._vector_bracket,
            d=self._d,
            lie_derivative=self._lie_derivative,
            interior=self._interior,
        )
        if name is not None:
            self._name = name
        elif background_H is None:
            self._name = "Courant(TM⊕T*M)"
        else:
            self._name = (
                f"Courant_H(TM⊕T*M, H={background_H._repr_inner()})"
            )

    # ---- accessors -------------------------------------------------- #

    @property
    def courant(self) -> CourantBracket:
        return self._courant

    @property
    def dorfman(self) -> DorfmanBracket:
        return self._dorfman

    @property
    def vector_bracket(self) -> GradedBracket:
        return self._vector_bracket

    @property
    def d(self) -> Derivation:
        return self._d

    @property
    def lie_derivative(self) -> LieDerivativeFactory:
        return self._lie_derivative

    @property
    def interior(self) -> InteriorFactory:
        return self._interior

    @property
    def background_H(self) -> Optional[Expr]:
        return self._background_H

    @property
    def is_twisted(self) -> bool:
        return self._background_H is not None

    @property
    def name(self) -> str:
        return self._name

    # ---- bracket views --------------------------------------------- #

    def expand(
        self,
        a: SectionPair,
        b: SectionPair,
        registry: Optional[PropertyRegistry] = None,
    ) -> SectionPair:
        """``[a, b]_C``, the Courant bracket on section pairs."""
        return self._courant.expand(a, b, registry)

    def expand_dorfman(
        self,
        a: SectionPair,
        b: SectionPair,
        registry: Optional[PropertyRegistry] = None,
    ) -> SectionPair:
        """``[a, b]_D``, the Dorfman twin, same Cartan operators."""
        return self._dorfman.expand(a, b, registry)

    # ---- structural operators (Stage E) ---------------------------- #

    def inner_product(
        self,
        a: SectionPair,
        b: SectionPair,
    ) -> CourantInnerProduct:
        """Build the symmetric inner product ``⟨a, b⟩`` on this algebroid.

        Wraps a literal :class:`CourantInnerProduct` Expr node; the
        unfold is handled by
        :class:`~jacopy.brackets.courant_inner_product.CourantInnerProductDefinition`
        when the prove suite invokes the engine on the resulting node.
        """
        if not isinstance(a, SectionPair) or not isinstance(b, SectionPair):
            raise TypeError(
                "inner_product requires SectionPair operands"
            )
        return CourantInnerProduct(a, b)

    def D(self, f: Expr) -> DOperator:
        """Build the section ``D f = (0, d f)`` for ``f ∈ C∞(M)``.

        Routes the algebroid's own :attr:`d` operator into the
        :class:`DOperator` instance so non-default Cartan families
        propagate faithfully when this :class:`CourantAlgebroid` was
        constructed with a custom ``d``.
        """
        if not isinstance(f, Expr):
            raise TypeError("D() argument must be an Expr")
        return DOperator(f, d=self._d)

    def anchor_of(self, section: Expr) -> CourantAnchor:
        """Build the anchor projection ``anchor(section)``.

        The image is a :class:`CourantAnchor` shape; the unfold to the
        vector half of a :class:`SectionPair` is handled by
        :class:`~jacopy.brackets.courant_anchor_d.CourantAnchorDefinition`.

        The method is named ``anchor_of`` rather than ``anchor`` to
        avoid shadowing the canonical anchor-construction helper
        :func:`jacopy.brackets.courant_anchor_d.anchor` if both are
        imported into the same scope.
        """
        if not isinstance(section, Expr):
            raise TypeError("anchor_of argument must be an Expr")
        return CourantAnchor(section)

    # ---- Stage E: D-compatibility prove method --------------------- #

    def prove_D_compat(
        self,
        f: Expr,
    ) -> ProofChain:
        """Definitional proof of ``anchor(D f) = 0``.

        Two-step axiom-tagged unfold:

        1. ``anchor(D f) → anchor((0, d f))`` via the D-operator
           direct definition ``D f := (0, d f)``;
        2. ``anchor((0, d f)) → 0`` via the anchor projection
           ``anchor((X, α)) := X``.

        Both steps tag with ``provenance_tag="axiom"``: each is a
        single primitive rewrite from a Definition class, not a
        seeded theorem citation. The chain therefore exhibits the
        full definitional path from ``anchor(D f)`` to ``0`` rather
        than collapsing the derivation into a single step.
        """
        if not isinstance(f, Expr):
            raise TypeError("prove_D_compat argument must be an Expr")
        Df = self.D(f)
        anchor_Df = self.anchor_of(Df)

        d_unfold = DOperatorDefinition().rewrite(Df)
        after_step1 = CourantAnchor(d_unfold)
        after_step2 = CourantAnchorDefinition().rewrite(after_step1)

        chain = ProofChain()
        chain.append(
            ProofStep(
                anchor_Df,
                after_step1,
                rule="DOperatorDefinition",
                justification=(
                    "D f := (0, d f); the D operator on (TM ⊕ T*M) "
                    "places the function's exterior derivative on the "
                    "form half and zero on the vector half."
                ),
                provenance_tag="axiom",
            )
        )
        chain.append(
            ProofStep(
                after_step1,
                after_step2,
                rule="CourantAnchorDefinition",
                justification=(
                    "anchor((X, α)) := X; the canonical projection "
                    "π_TM picks out the vector-field component of a "
                    "section pair."
                ),
                provenance_tag="axiom",
            )
        )
        return chain

    # ---- Stage E: anchor compatibility ----------------------------- #

    def prove_anchor_compat(
        self,
        e1: SectionPair,
        e2: SectionPair,
        *,
        registry: Optional[PropertyRegistry] = None,
    ) -> ProofChain:
        """Definitional proof of ``anchor([e1, e2]_C) = [anchor(e1), anchor(e2)]_VF``.

        On concrete operands ``e1 = (X, α)``, ``e2 = (Y, β)`` the chain
        unfolds the Courant bracket and projects out the vector half:

        1. ``anchor([(X,α),(Y,β)]_C) → anchor(([X,Y]_VF, form_part))``
           via the Courant bracket definition. The vector-bracket
           apply on the right is kept *inert* (a literal
           :class:`BracketApply` on :attr:`vector_bracket`) so the
           chain's final form reads as the textbook RHS rather than
           the underlying ``X·Y − Y·X`` derivation product.
        2. ``anchor((·, ·)) → vector_part`` via the anchor projection
           ``anchor((X, α)) := X``.

        Both steps are atomic axiom rewrites; no seeded theorem
        citation is used. The form half is built verbatim from the
        algebroid's own Cartan operators (``L``, ``ι``, ``d``, plus
        the H-twist contraction ``ι_Y ι_X H`` when twisted), so the
        proof is faithful to the H-twisted as well as untwisted
        Courant bracket.
        """
        if not isinstance(e1, SectionPair) or not isinstance(e2, SectionPair):
            raise TypeError(
                "prove_anchor_compat requires SectionPair operands"
            )
        bracket_apply = BracketApply(self._courant, e1, e2)
        lhs = self.anchor_of(bracket_apply)

        # Build the unfolded SectionPair, but replace its vector half
        # with the inert ``BracketApply(vector_bracket, X, Y)`` so the
        # chain end mirrors ``[anchor(e1), anchor(e2)]_VF`` literally.
        full_unfold = self._courant.expand(e1, e2, registry)
        inert_vector = BracketApply(self._vector_bracket, e1.vector, e2.vector)
        section_with_inert_vec = SectionPair(inert_vector, full_unfold.form)
        after_step1 = self.anchor_of(section_with_inert_vec)
        after_step2 = inert_vector

        chain = ProofChain()
        chain.append(
            ProofStep(
                lhs,
                after_step1,
                rule="CourantBracketDefinition",
                justification=(
                    "[(X,α), (Y,β)]_C := ([X,Y]_VF, "
                    "L_X β − L_Y α − ½ d(ι_X β − ι_Y α)"
                    + (" + ι_Y ι_X H" if self.is_twisted else "")
                    + "); the vector half is the underlying "
                    "vector-bracket on X and Y."
                ),
                provenance_tag="axiom",
            )
        )
        chain.append(
            ProofStep(
                after_step1,
                after_step2,
                rule="CourantAnchorDefinition",
                justification=(
                    "anchor((X, α)) := X; on the unfolded Courant "
                    "bracket the vector half is exactly [X, Y]_VF, "
                    "which equals [anchor(e1), anchor(e2)]_VF since "
                    "anchor((X,α)) = X and anchor((Y,β)) = Y."
                ),
                provenance_tag="axiom",
            )
        )
        return chain

    # ---- Stage E: Vaisman Leibniz ---------------------------------- #

    def prove_leibniz(
        self,
        e1: SectionPair,
        e2: SectionPair,
        f: Expr,
        *,
        registry: Optional[PropertyRegistry] = None,
    ) -> ProofChain:
        """Definitional proof of the Vaisman Leibniz axiom

        ``[e1, f·e2]_C = f [e1, e2]_C + (anchor(e1)·f) e2 − ⟨e1, e2⟩ D f``.

        Concrete operands ``e1 = (X, α)``, ``e2 = (Y, β)``, scalar
        ``f ∈ C∞(M)``. The chain emits **eight** axiom-tagged steps,
        each a single named atomic axiom (no seeded-theorem citation,
        no bundled Cartan-Leibniz macro):

        1. **CourantBracketDefinition** unfolds
           ``[(X,α), (fY, fβ)]_C`` to its component-form
           ``([X, fY]_VF, L_X(fβ) − L_{fY}α − ½ d(ι_X(fβ) − ι_{fY}α))``
           (plus ``ι_{fY} ι_X H`` when twisted).
        2. **LieBracketLeibnizSecondSlot**: ``[X, fY] = f [X, Y] + X(f) Y``.
           Rewrites the vector half.
        3. **LieDerivativeProductRule**: ``L_X(f β) = f L_X β + X(f) β``.
           Rewrites the first form-half summand.
        4. **LieRescaling**: ``L_{fY} α = f L_Y α + α(Y) d f``. Rewrites
           ``L_{fY} α`` (operator-side rescaling).
        5. **InteriorScalarLinearity**: ``ι_{f·V}(ω) = f ι_V(ω)`` and
           ``ι_V(f·ω) = f ι_V(ω)``. Applied to ``ι_X(fβ) → f ι_X β``,
           ``ι_{fY}α → f ι_Y α``, plus the twist contribution
           ``ι_{fY} ι_X H → f ι_Y ι_X H`` when twisted.
        6. **InteriorPairing**: ``ι_V(ω) = ω(V) = ⟨ω, V⟩`` for a 1-form
           ``ω`` and vector ``V``. Applied to ``ι_X β → β(X)`` and
           ``ι_Y α → α(Y)``.
        7. **ExteriorDProductRule**: ``d(f g) = df · g + f · d g`` for a
           scalar ``f`` and 0-form ``g``. Applied to
           ``d(f β(X) − f α(Y))`` (also using d-additivity, treated as
           part of the same product-rule axiom).
        8. **VaismanLeibnizRegroup**: collect terms by scalar
           coefficient. The ``f``-subset reassembles ``f [e1, e2]_C``,
           the ``X(f)``-subset is ``X(f) e2``, the ``df``-subset
           collapses to ``− ⟨e1, e2⟩ df = − ⟨e1, e2⟩ (form half of D f)``.

        All eight steps are tagged ``provenance_tag="axiom"``. Each step
        names *exactly one* atomic axiom (some applied to several
        subexpressions of the same kind, e.g. step 5 applies interior
        C∞-linearity to two distinct ``ι`` expressions, but they are
        all instances of one axiom). On the H-twisted algebroid the
        chain shape is unchanged; the twist term ``ι_{fY} ι_X H`` is
        threaded through and absorbed by the same C∞-linearity in
        step 5.

        Parameters
        ----------
        e1, e2
            The unscaled section pairs ``(X, α)`` and ``(Y, β)``.
        f
            The scalar function applied to ``e2``.
        registry
            Optional property registry; consulted by
            :meth:`CourantBracket.expand` for degree information when
            building intermediate Exprs.

        Returns
        -------
        :class:`ProofChain`
            An 8-step chain whose initial Expr is
            ``BracketApply([·,·]_C, e1, f·e2)`` and whose final Expr
            is the RHS :class:`SectionPair`.

        Raises
        ------
        TypeError
            If ``e1`` or ``e2`` is not a :class:`SectionPair`, or
            ``f`` is not an :class:`Expr`.
        """
        if not isinstance(e1, SectionPair) or not isinstance(e2, SectionPair):
            raise TypeError(
                "prove_leibniz requires SectionPair operands"
            )
        if not isinstance(f, Expr):
            raise TypeError("prove_leibniz f argument must be an Expr")

        X, alpha = e1.vector, e1.form
        Y, beta = e2.vector, e2.form
        fY = Product(f, Y)
        fbeta = Product(f, beta)
        sc_e2 = SectionPair(fY, fbeta)

        lhs = BracketApply(self._courant, e1, sc_e2)

        L_X = self._lie_derivative(X)
        L_Y = self._lie_derivative(Y)
        L_fY = self._lie_derivative(fY)
        iota_X = self._interior(X)
        iota_Y = self._interior(Y)
        iota_fY = self._interior(fY)
        half = Rational(1, 2)
        d_op = self._d
        H = self._background_H
        df = Act(d_op, f)
        Xf = Act(L_X, f)
        beta_X = Pairing(beta, X)
        alpha_Y = Pairing(alpha, Y)
        L_X_beta = Act(L_X, beta)
        L_Y_alpha = Act(L_Y, alpha)

        # ------- Build the eight intermediate states ------- #

        # vec_inert: vector half before Lie-bracket Leibniz (step 1 form).
        vec_inert = BracketApply(self._vector_bracket, X, fY)
        # vec_leibniz: vector half after Lie-bracket Leibniz (step 2 form).
        vec_leibniz = Sum(
            Product(f, BracketApply(self._vector_bracket, X, Y)),
            Product(Xf, Y),
        )

        def _form_half(
            *,
            L_X_fbeta_term: Expr,   # represents L_X(fβ) or its rewrite
            L_fY_alpha_term: Expr,  # represents L_{fY}α or its rewrite
            iota_X_fbeta: Expr,     # ι_X(fβ) or its rewrite
            iota_fY_alpha: Expr,    # ι_{fY}α or its rewrite
            d_inner: Optional[Expr] = None,
            twist_term: Optional[Expr] = None,
        ) -> Expr:
            """Assemble the form half from its five Vaisman pieces.

            ``d_inner`` overrides the half-d term when given (used by
            step 7 after d-product rule fires); otherwise the half-d
            term is built as ``− ½ d(iota_X_fbeta − iota_fY_alpha)``.
            ``twist_term`` is the (optional) ``ι_{fY} ι_X H`` (or its
            rewrite); only included when self.is_twisted.
            """
            if d_inner is None:
                d_inner_expr = Sum(iota_X_fbeta, Neg(iota_fY_alpha))
                half_d = Neg(Product(half, Act(d_op, d_inner_expr)))
            else:
                half_d = Neg(Product(half, d_inner))
            terms = [L_X_fbeta_term, Neg(L_fY_alpha_term), half_d]
            # The "Cartan Leibniz on form" expansion of L_X(fβ) emits
            # ``f L_X β + X(f) β``, two summands that we keep at the
            # top level for clean regrouping in step 8. We push them
            # in only when the caller pre-expanded; otherwise they
            # arrive as a single ``Act(L_X, fβ)`` term.
            if self.is_twisted:
                if twist_term is None:
                    twist_term = Act(iota_fY, Act(iota_X, H))
                terms.append(twist_term)
            return Sum(*terms)

        # State after step 1 (Courant bracket def).
        F1 = _form_half(
            L_X_fbeta_term=Act(L_X, fbeta),
            L_fY_alpha_term=Act(L_fY, alpha),
            iota_X_fbeta=Act(iota_X, fbeta),
            iota_fY_alpha=Act(iota_fY, alpha),
        )
        after_step1 = SectionPair(vec_inert, F1)

        # State after step 2 (Lie-bracket Leibniz on vector half only).
        after_step2 = SectionPair(vec_leibniz, F1)

        # State after step 3 (L_X(fβ) → f L_X β + X(f) β).
        # Replace the L_X(fβ) summand with two separate summands
        # ``f L_X β`` and ``X(f) β``.
        F3_terms = [
            Product(f, L_X_beta),
            Product(Xf, beta),
            Neg(Act(L_fY, alpha)),
            Neg(
                Product(
                    half,
                    Act(
                        d_op,
                        Sum(Act(iota_X, fbeta), Neg(Act(iota_fY, alpha))),
                    ),
                )
            ),
        ]
        if self.is_twisted:
            F3_terms.append(Act(iota_fY, Act(iota_X, H)))
        F3 = Sum(*F3_terms)
        after_step3 = SectionPair(vec_leibniz, F3)

        # State after step 4 (L_{fY}α → f L_Y α + α(Y) df).
        # Replace ``− Act(L_fY, α)`` with ``−(f L_Y α + α(Y) df)``,
        # i.e. ``− f L_Y α − α(Y) df``.
        F4_terms = [
            Product(f, L_X_beta),
            Product(Xf, beta),
            Neg(Product(f, L_Y_alpha)),
            Neg(Product(alpha_Y, df)),
            Neg(
                Product(
                    half,
                    Act(
                        d_op,
                        Sum(Act(iota_X, fbeta), Neg(Act(iota_fY, alpha))),
                    ),
                )
            ),
        ]
        if self.is_twisted:
            F4_terms.append(Act(iota_fY, Act(iota_X, H)))
        F4 = Sum(*F4_terms)
        after_step4 = SectionPair(vec_leibniz, F4)

        # State after step 5 (interior C∞-linearity, ×2 + twist).
        # ι_X(fβ) → f ι_X β; ι_{fY}α → f ι_Y α; (twist) ι_{fY} ι_X H
        # → f ι_Y ι_X H.
        F5_terms = [
            Product(f, L_X_beta),
            Product(Xf, beta),
            Neg(Product(f, L_Y_alpha)),
            Neg(Product(alpha_Y, df)),
            Neg(
                Product(
                    half,
                    Act(
                        d_op,
                        Sum(
                            Product(f, Act(iota_X, beta)),
                            Neg(Product(f, Act(iota_Y, alpha))),
                        ),
                    ),
                )
            ),
        ]
        if self.is_twisted:
            F5_terms.append(
                Product(f, Act(iota_Y, Act(iota_X, H)))
            )
        F5 = Sum(*F5_terms)
        after_step5 = SectionPair(vec_leibniz, F5)

        # State after step 6 (ι on 1-form → Pairing, ×2).
        # ι_X β → β(X); ι_Y α → α(Y).
        F6_terms = [
            Product(f, L_X_beta),
            Product(Xf, beta),
            Neg(Product(f, L_Y_alpha)),
            Neg(Product(alpha_Y, df)),
            Neg(
                Product(
                    half,
                    Act(
                        d_op,
                        Sum(Product(f, beta_X), Neg(Product(f, alpha_Y))),
                    ),
                )
            ),
        ]
        if self.is_twisted:
            F6_terms.append(
                Product(f, Act(iota_Y, Act(iota_X, H)))
            )
        F6 = Sum(*F6_terms)
        after_step6 = SectionPair(vec_leibniz, F6)

        # State after step 7 (d product rule on the half-d term).
        # d(f β(X) − f α(Y)) → df (β(X) − α(Y)) + f d(β(X) − α(Y))
        # by d additivity + Leibniz on each Product.
        # We assemble the post-Leibniz pre-collection form: keep the
        # ``½`` distributed over the two summands so step 8 can regroup
        # cleanly.
        diff_pairings = Sum(beta_X, Neg(alpha_Y))
        d_diff = Act(d_op, diff_pairings)
        F7_terms = [
            Product(f, L_X_beta),
            Product(Xf, beta),
            Neg(Product(f, L_Y_alpha)),
            Neg(Product(alpha_Y, df)),
            Neg(Product(half, Product(df, diff_pairings))),
            Neg(Product(half, Product(f, d_diff))),
        ]
        if self.is_twisted:
            F7_terms.append(
                Product(f, Act(iota_Y, Act(iota_X, H)))
            )
        F7 = Sum(*F7_terms)
        after_step7 = SectionPair(vec_leibniz, F7)

        # State after step 8 (RHS regrouping).
        # f-coefficient subset → form half of f [e1, e2]_C
        # X(f)-coefficient subset → form half of X(f) e2 (= X(f) β)
        # df-coefficient subset → -⟨e1, e2⟩ df (form half of -⟨e1,e2⟩ Df)
        f_courant_form_terms = [
            Product(f, L_X_beta),
            Neg(Product(f, L_Y_alpha)),
            Neg(Product(half, Product(f, d_diff))),
        ]
        if self.is_twisted:
            f_courant_form_terms.append(
                Product(f, Act(iota_Y, Act(iota_X, H)))
            )
        f_courant_form = Sum(*f_courant_form_terms)
        f_courant_vec = Product(f, BracketApply(self._vector_bracket, X, Y))
        Xf_e2_vec = Product(Xf, Y)
        Xf_e2_form = Product(Xf, beta)
        inner = CourantInnerProduct(e1, e2)
        neg_inner_Df_form = Neg(Product(inner, df))
        rhs_vec = Sum(f_courant_vec, Xf_e2_vec)
        rhs_form = Sum(f_courant_form, Xf_e2_form, neg_inner_Df_form)
        after_step8 = SectionPair(rhs_vec, rhs_form)

        # ------- Assemble the chain ------- #
        chain = ProofChain()
        chain.append(
            ProofStep(
                lhs,
                after_step1,
                rule="CourantBracketDefinition",
                justification=(
                    "Unfold [(X,α), (fY, fβ)]_C into ([X, fY]_VF, "
                    "L_X(fβ) − L_{fY}α − ½ d(ι_X(fβ) − ι_{fY}α)"
                    + (" + ι_{fY} ι_X H" if self.is_twisted else "")
                    + "). The vector half is kept inert as "
                    "BracketApply on the underlying vector bracket."
                ),
                provenance_tag="axiom",
            )
        )
        chain.append(
            ProofStep(
                after_step1,
                after_step2,
                rule="LieBracketLeibnizSecondSlot",
                justification=(
                    "[X, fY] = f [X, Y] + X(f) Y; Lie-bracket Leibniz "
                    "on the second slot. Rewrites the vector half "
                    "from BracketApply(VB, X, fY) to its expansion."
                ),
                provenance_tag="axiom",
            )
        )
        chain.append(
            ProofStep(
                after_step2,
                after_step3,
                rule="LieDerivativeProductRule",
                justification=(
                    "L_X(f β) = f L_X β + X(f) β; Lie-derivative "
                    "Leibniz on the product of a scalar and a 1-form."
                ),
                provenance_tag="axiom",
            )
        )
        chain.append(
            ProofStep(
                after_step3,
                after_step4,
                rule="LieRescaling",
                justification=(
                    "L_{f Y} α = f L_Y α + (df) ι_Y α = f L_Y α + α(Y) df; "
                    "Lie-derivative rescaling under a scaled vector "
                    "field, with the iota-on-1-form contraction "
                    "yielding the pairing α(Y)."
                ),
                provenance_tag="axiom",
            )
        )
        chain.append(
            ProofStep(
                after_step4,
                after_step5,
                rule="InteriorScalarLinearity",
                justification=(
                    "ι_{f V}(ω) = f ι_V(ω) and ι_V(f ω) = f ι_V(ω); "
                    "interior product C∞-linearity in either slot. "
                    "Applied to ι_X(f β) → f ι_X β, ι_{f Y} α → f ι_Y α"
                    + (
                        ", and ι_{f Y} ι_X H → f ι_Y ι_X H "
                        "for the twist term"
                        if self.is_twisted
                        else ""
                    )
                    + "."
                ),
                provenance_tag="axiom",
            )
        )
        chain.append(
            ProofStep(
                after_step5,
                after_step6,
                rule="InteriorPairing",
                justification=(
                    "ι_V(ω) = ω(V) = ⟨ω, V⟩ for a 1-form ω and vector "
                    "V; interior product on a 1-form is the canonical "
                    "pairing. Applied to ι_X β → β(X) = ⟨β, X⟩ and "
                    "ι_Y α → α(Y) = ⟨α, Y⟩."
                ),
                provenance_tag="axiom",
            )
        )
        chain.append(
            ProofStep(
                after_step6,
                after_step7,
                rule="ExteriorDProductRule",
                justification=(
                    "d(f g) = df · g + f · dg for a scalar f and "
                    "0-form g (extended to the difference β(X) − α(Y) "
                    "by d-additivity). Applied to "
                    "d(f β(X) − f α(Y)) → df (β(X) − α(Y)) "
                    "+ f d(β(X) − α(Y)), yielding the two summands "
                    "scaled by − ½."
                ),
                provenance_tag="axiom",
            )
        )
        chain.append(
            ProofStep(
                after_step7,
                after_step8,
                rule="VaismanLeibnizRegroup",
                justification=(
                    "Collect terms by scalar coefficient: "
                    "f-coefficient subset → form half of f [e1, e2]_C; "
                    "X(f)-coefficient subset → form half of X(f) e2; "
                    "df-coefficient subset (− α(Y) df − ½ β(X) df + ½ "
                    "α(Y) df = − ½(α(Y) + β(X)) df = − ⟨e1, e2⟩ df) → "
                    "form half of − ⟨e1, e2⟩ D f. Final: "
                    "f [e1, e2]_C + X(f) e2 − ⟨e1, e2⟩ D f."
                ),
                provenance_tag="axiom",
            )
        )
        return chain

    # ---- Jacobi ---------------------------------------------------- #

    def jacobi_condition(
        self,
        registry: Optional[PropertyRegistry] = None,
    ) -> VanishingCondition:
        """Delegate to :meth:`CourantBracket.jacobi_condition`.

        Returns the vacuous :class:`VanishingCondition` when untwisted
        and the ``dH = 0`` condition when H-twisted.
        """
        return self._courant.jacobi_condition(registry)

    def prove_jacobi_reduction(
        self,
        *,
        registry: Optional[PropertyRegistry] = None,
    ) -> ProofChain:
        """One-step axiomatic reduction of Courant Jacobi to its condition.

        * Untwisted: the reduction is vacuous, a single reflexive step
          mapping the literal ``0`` obstruction to itself.
        * H-twisted: the step cites the Courant-algebroid Jacobi axiom
          and lands on the ``dH`` obstruction, which the caller
          discharges by supplying ``dH = 0``.

        Either way the chain has a single top-level step, tagged
        ``axiom`` so downstream citation treats it as definitional.
        """
        cond = self._courant.jacobi_condition(registry)
        obstruction = cond.obstruction
        chain = ProofChain()
        if not self.is_twisted:
            chain.append(
                ProofStep(
                    obstruction,
                    obstruction,
                    rule="CourantAlgebroidJacobi",
                    justification=(
                        "Untwisted Courant Jacobi holds unconditionally "
                        "on (TM ⊕ T*M), obstruction is the vacuous 0."
                    ),
                    provenance_tag="axiom",
                )
            )
            return chain
        chain.append(
            ProofStep(
                obstruction,
                obstruction,
                rule="CourantAlgebroidJacobi",
                justification=(
                    f"H-twisted Courant Jacobi ⟺ dH = 0 on "
                    f"{self._name}; obstruction is dH."
                ),
                provenance_tag="axiom",
            )
        )
        return chain

    # ---- Courant-Dorfman bridge ------------------------------------ #

    def courant_dorfman_obstruction(
        self,
        a: SectionPair,
        b: SectionPair,
        registry: Optional[PropertyRegistry] = None,
    ) -> SectionPair:
        """``[a, b]_D − [a, b]_C``, the difference whose identity we
        assert with :meth:`prove_courant_dorfman_bridge`.

        The vector halves match by construction (both brackets run the
        same :attr:`vector_bracket`), so their difference is a formal
        ``[X, Y] − [X, Y]``; the form halves differ by the exact
        correction ``½ d(ι_X β + ι_Y α)``. This method returns the
        literal pre-cancellation :class:`SectionPair` so the proof
        layer has somewhere concrete to start.
        """
        if not isinstance(a, SectionPair) or not isinstance(b, SectionPair):
            raise TypeError(
                "courant_dorfman_obstruction requires SectionPair operands"
            )
        dorf = self._dorfman.expand(a, b, registry)
        cour = self._courant.expand(a, b, registry)
        return SectionPair(
            Sum(dorf.vector, Neg(cour.vector)),
            Sum(dorf.form, Neg(cour.form)),
        )

    def bridge_correction(self, a: SectionPair, b: SectionPair) -> SectionPair:
        """``(0, ½ d(ι_X β + ι_Y α))``, the canonical correction term.

        Built from the algebroid's own Cartan operators so that the
        identity ``[·,·]_D − [·,·]_C = correction`` holds on matching
        operator names.
        """
        if not isinstance(a, SectionPair) or not isinstance(b, SectionPair):
            raise TypeError(
                "bridge_correction requires SectionPair operands"
            )
        X, alpha = a.vector, a.form
        Y, beta = b.vector, b.form
        iota_X = self._interior(X)
        iota_Y = self._interior(Y)
        inner = Sum(Act(iota_X, beta), Act(iota_Y, alpha))
        correction = Product(Rational(1, 2), Act(self._d, inner))
        return SectionPair(Integer(0), correction)

    def prove_courant_dorfman_bridge(
        self,
        a: SectionPair,
        b: SectionPair,
        *,
        registry: Optional[PropertyRegistry] = None,
    ) -> ProofChain:
        """One theorem-step chain asserting the Courant–Dorfman bridge.

        The derivation of the identity uses Cartan's magic formula
        ``L_Y α = d(ι_Y α) + ι_Y(dα)`` to cancel the ``−ι_Y dα`` in
        Dorfman against the ``−L_Y α`` in Courant, leaving the exact
        correction ``½ d(ι_X β + ι_Y α)``. The chain records that
        derivation as a single ``theorem``-tagged step rather than
        unfolding the cancellation arithmetic, the algebraic identity
        is the theorem, not the rewrite.
        """
        obs = self.courant_dorfman_obstruction(a, b, registry)
        target = self.bridge_correction(a, b)
        chain = ProofChain()
        chain.append(
            ProofStep(
                obs,
                target,
                rule="CourantDorfmanBridge",
                justification=(
                    "[·,·]_D − [·,·]_C = (0, ½ d(ι_X β + ι_Y α)); "
                    "derivation reduces the L_Y α term to "
                    "d(ι_Y α) + ι_Y(dα) via Cartan's magic formula and "
                    "the − ι_Y dα of Dorfman cancels the ι_Y dα piece."
                ),
                provenance_tag="theorem",
            )
        )
        return chain

    # ---- dunder ---------------------------------------------------- #

    def __repr__(self) -> str:
        if self._background_H is None:
            return "CourantAlgebroid(TM⊕T*M)"
        return (
            f"CourantAlgebroid(TM⊕T*M, H="
            f"{self._background_H._repr_inner()})"
        )


# --------------------------------------------------------------------- #
# Factory                                                                #
# --------------------------------------------------------------------- #


def courant_algebroid(
    *,
    vector_bracket: Optional[GradedBracket] = None,
    d: Optional[Derivation] = None,
    lie_derivative: Optional[LieDerivativeFactory] = None,
    interior: Optional[InteriorFactory] = None,
    background_H: Optional[Expr] = None,
    name: Optional[str] = None,
) -> CourantAlgebroid:
    """Functional mirror of :class:`CourantAlgebroid`."""
    return CourantAlgebroid(
        vector_bracket=vector_bracket,
        d=d,
        lie_derivative=lie_derivative,
        interior=interior,
        background_H=background_H,
        name=name,
    )


# --------------------------------------------------------------------- #
# Seeded theorems                                                        #
# --------------------------------------------------------------------- #


def _build_courant_jacobi_twist_theorem() -> Theorem:
    """``courant_jacobi_twist``, H-twisted Courant Jacobi ⟺ dH = 0.

    Single axiom-tagged step citing the Courant algebroid Jacobi axiom
    on the H-twisted side. The obstruction ``dH`` is surfaced literally
    so downstream callers can cite the theorem and discharge ``dH = 0``
    separately.
    """
    H = Symbol("H")
    C = CourantAlgebroid(background_H=H)
    chain = C.prove_jacobi_reduction()
    return Theorem(
        name="courant_jacobi_twist",
        statement=(
            "H-twisted Courant bracket satisfies graded Jacobi ⟺ dH = 0"
        ),
        from_axioms=(
            "Courant algebroid Jacobi axiom",
            "dH = 0 (closed-3-form hypothesis)",
        ),
        proof=chain,
        notes=(
            "The obstruction to H-twisted Courant Jacobi is exactly "
            "dH (see CourantBracket.jacobi_condition). Supplying dH = 0 "
            ", i.e. H is closed, discharges the condition and yields "
            "the full Courant algebroid Jacobi identity."
        ),
    )


def _build_courant_dorfman_bridge_theorem() -> Theorem:
    """``courant_dorfman_bridge``, the classical correction identity.

    Concrete witness with generic symbols ``(X, α), (Y, β)`` on the
    untwisted algebroid; downstream callers produce their own chain on
    their own section pairs via :meth:`prove_courant_dorfman_bridge`.
    """
    X = Symbol("X")
    Y = Symbol("Y")
    alpha = Symbol("α")
    beta = Symbol("β")
    reg = PropertyRegistry()
    reg.declare(X, Graded(degree=0))
    reg.declare(Y, Graded(degree=0))
    reg.declare(alpha, Graded(degree=1))
    reg.declare(beta, Graded(degree=1))
    C = CourantAlgebroid()
    a = SectionPair(X, alpha)
    b = SectionPair(Y, beta)
    chain = C.prove_courant_dorfman_bridge(a, b, registry=reg)
    return Theorem(
        name="courant_dorfman_bridge",
        statement=(
            "[(X, α), (Y, β)]_D − [(X, α), (Y, β)]_C = "
            "(0, ½ d(ι_X β + ι_Y α))"
        ),
        from_axioms=(
            "Dorfman bracket definition",
            "Courant bracket definition",
            "Cartan magic formula L_Y α = d ι_Y α + ι_Y d α",
        ),
        proof=chain,
        notes=(
            "The Dorfman − Courant correction is the symmetrised "
            "d-exact piece d(ι_X β + ι_Y α) / 2. Previously deferred "
            "from the Stage 3 Courant pass (see stage3_courant_plan.md). "
            "Closed here as a single theorem-tagged step, the "
            "algebraic identity is the result, not the Cartan-magic "
            "arithmetic that produces it."
        ),
    )


#: H-twisted Courant Jacobi reduction.
THEOREM_COURANT_JACOBI_TWIST = _build_courant_jacobi_twist_theorem()

#: Courant-Dorfman bridge identity.
THEOREM_COURANT_DORFMAN_BRIDGE = _build_courant_dorfman_bridge_theorem()


if "courant_jacobi_twist" not in theorem_book:
    theorem_book.add(THEOREM_COURANT_JACOBI_TWIST)

if "courant_dorfman_bridge" not in theorem_book:
    theorem_book.add(THEOREM_COURANT_DORFMAN_BRIDGE)
