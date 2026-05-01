r"""
Cartan structure problem wrapper.

The textbook **Cartan structure equations** read off a local frame
``F = (X_a)`` for an affine connection ``∇`` are:

.. math::

    T^a(U, V)        &\;=\; (d e^a)(U, V)
                          + \sum_b (\omega^a{}_b \wedge e^b)(U, V),  \\
    R^a{}_b(U, V)    &\;=\; (d \omega^a{}_b)(U, V)
                          + \sum_c (\omega^a{}_c \wedge \omega^c{}_b)(U, V),

i.e. the components of the torsion 2-form decompose into the exterior
derivative of the dual coframe plus a wedge with the connection
1-forms, and analogously the curvature 2-form decomposes into
``dω^a_b`` plus a connection-form wedge.

This module ships :class:`CartanStructureProblem`, a thin bundle that
pairs

* an :class:`~jacopy.calculus.connection.AffineConnection` ``∇``,
* a :class:`~jacopy.calculus.local_frame.LocalFrame` ``F``,

with a pre-configured :class:`~jacopy.proof.expansion.ExpansionEngine`
carrying every axiom needed to mechanise the Cartan I structure
equation:

* :class:`~jacopy.calculus.cartan_forms.TorsionFormDefinition` /
  :class:`~jacopy.calculus.cartan_forms.CurvatureFormDefinition`,
  open ``T^a(U, V) → ⟨e^a, T(∇)(U, V)⟩`` and
  ``R^a_b(U, V) → ⟨e^a, R(∇)(U, V) X_b⟩`` respectively;
* :class:`~jacopy.calculus.torsion_curvature.TorsionDefinitionDefinition`
  / :class:`~jacopy.calculus.torsion_curvature.CurvatureDefinitionDefinition`
 , open the structural definitions
  ``T(∇)(U, V) → ∇_U V − ∇_V U − [U, V]_VF`` and
  ``R(∇)(U, V) Z → ∇_U ∇_V Z − ∇_V ∇_U Z − ∇_{[U,V]_VF} Z``;
* the four connection X-/Y-slot rules (linearity, scalar pull,
  Y-additivity, Y-Leibniz) that carry the inner ``∇`` shapes;
* the two opt-in frame-decomposition rules
  (:class:`~jacopy.calculus.frame_decomposition.ConnectionEvalYFrameDecompositionDefinition`
  and :class:`~jacopy.calculus.frame_decomposition.ConnectionFormDecompositionDefinition`)
  that frame-expand a non-frame ``Y`` and collapse ``∇_V X_b →
  Σ_c ω^c_b(V) X_c`` respectively;
* every :class:`~jacopy.core.indexed_sum.IndexedSum` engine rule from
  17.E.3-E.6 plus the 17.F.1/F.2 push-in rules over
  :class:`~jacopy.calculus.connection.ConnectionEvalExpr` and
  :class:`~jacopy.core.multi_eval.MultiEval`;
* the per-frame Pairing duality ``⟨e^a, X_b⟩ → δ^a_b``;
* :class:`~jacopy.calculus.pairing.Pairing` C∞-linearity (scalar pull
  + Sum/Neg distribution) so ``⟨e^a, f·V⟩``-shapes split correctly;
* the Wedge alternating expansion + arity-1
  :class:`MultiEval` → :class:`Pairing` bridge (Faz 17.F.1.5/.6), these
  fire on the RHS ``(ω^a_b ∧ e^b)(U, V)`` once the
  :class:`IndexedSum` is pushed out of the
  :class:`MultiEval` head;
* the intrinsic Cartan-Koszul ``d``-formula
  (:class:`~jacopy.calculus.intrinsic_axioms.ExteriorDIntrinsicDefinition`)
  that opens ``(d e^a)(U, V) → U(e^a(V)) − V(e^a(U)) − e^a([U,V]_VF)``,
  and analogously ``(d ω^a_b)(U, V)`` for Cartan II.

The bundle deliberately *omits*
:class:`~jacopy.calculus.cartan_forms.ConnectionFormDefinition`: that
rule would unfold ``ω^a_b(V) → ⟨e^a, ∇_V X_b⟩``, which the
:class:`ConnectionFormDecompositionDefinition` immediately re-folds,
a loop. The Cartan-structure proof keeps ``ω^a_b`` packaged on the
RHS and lets the LHS unfold all the way to ``∇_V X_b`` shapes that
the decomposition rule then re-expresses in terms of ``ω``.

The wrapper exposes a per-problem
:class:`~jacopy.core.registry.PropertyRegistry` that declares
:class:`~jacopy.calculus.local_frame.FrameCovector` and
:class:`~jacopy.calculus.cartan_forms.ConnectionForm` as degree-1 forms
, the wedge alternating expansion and the
:class:`MultiEvalOneFormPairingBridgeDefinition` consult that registry
to confirm the factors are one-forms before firing.

The :meth:`prove_first_cartan` and :meth:`prove_second_cartan`
methods build the Cartan I / II equalities for caller-supplied
``U, V`` and free upper / lower indices, expand the difference
``LHS − RHS`` through the engine + :func:`simplify` fix-point, and
return a :class:`CartanStructureProofResult` whose ``ok`` flag is
``True`` iff the difference reduces to ``0``.

The fix-point loop uses :func:`simplify` (full pipeline including
:func:`sort_product`) rather than the lighter ``canonicalize +
collect_terms`` pair :class:`~jacopy.library.bianchi_problem.BianchiProblem`
uses: the Cartan I residue exposes :class:`~jacopy.core.expr.Product`'s
in different factor orders (e.g. ``P(e,V)·P(ω,U)`` and
``P(ω,U)·P(e,V)``) that only :func:`sort_product` puts into a common
canonical order, without which collect_terms cannot cancel the pair.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

from jacopy.algorithms.simplify import simplify
from jacopy.calculus.cartan_forms import (
    ConnectionForm,
    CurvatureForm,
    CurvatureFormDefinition,
    TorsionForm,
    TorsionFormDefinition,
)
from jacopy.calculus.connection import (
    AffineConnection,
    ConnectionXLinearityDefinition,
    ConnectionXScalarPullDefinition,
    ConnectionYAdditivityDefinition,
    ConnectionYLeibnizDefinition,
)
from jacopy.calculus.exterior_d import d
from jacopy.calculus.frame_decomposition import (
    ConnectionEvalYFrameDecompositionDefinition,
    ConnectionFormDecompositionDefinition,
)
from jacopy.calculus.indexed_sum_axioms import (
    ConnectionEvalIndexedSumPushInDefinition,
    IndexedSumKroneckerContractDefinition,
    IndexedSumNegPullDefinition,
    IndexedSumPairingPushInLeftDefinition,
    IndexedSumPairingPushInRightDefinition,
    IndexedSumScalarPullDefinition,
    IndexedSumSumDistributeDefinition,
    MultiEvalIndexedSumPushInDefinition,
)
from jacopy.calculus.intrinsic_axioms import (
    ExteriorDIntrinsicDefinition,
    KoszulExteriorDIntrinsicDefinition,
)
from jacopy.calculus.local_frame import (
    FrameCovector,
    FrameIndex,
    LocalFrame,
)
from jacopy.calculus.pairing_axioms import (
    MultiEvalOneFormPairingBridgeDefinition,
    PairingLinearityDefinition,
)
from jacopy.calculus.pairing_linearity_axioms import (
    PairingScalarPullDefinition,
)
from jacopy.calculus.torsion_curvature import (
    CurvatureDefinitionDefinition,
    TorsionDefinitionDefinition,
)
from jacopy.calculus.wedge_axioms import WedgeMultiEvalAlternatingDefinition
from jacopy.algebra.derivation import Act
from jacopy.core.expr import Expr, Integer, Neg, Sum, Zero
from jacopy.core.indexed_sum import IndexedSum
from jacopy.core.multi_eval import MultiEval
from jacopy.core.properties import Graded
from jacopy.core.registry import PropertyRegistry
from jacopy.core.symbolic_degree import Degree
from jacopy.core.wedge import Wedge
from jacopy.proof.expansion import ExpansionEngine
from jacopy.proof.step import ProofStep


# --------------------------------------------------------------------- #
# Proof result                                                           #
# --------------------------------------------------------------------- #


@dataclass(frozen=True)
class CartanStructureProofResult:
    """Outcome of a Cartan-structure equation proof attempt.

    Attributes
    ----------
    lhs_initial, rhs_initial
        The two sides of the claimed structure equation before
        expansion.
    final
        The fully expanded canonical form of ``LHS − RHS``. ``Zero``
        on success.
    steps
        Transcript of the engine's rewriting passes on the difference.
    ok
        ``True`` iff ``final`` reduces to ``0`` after expansion.
    """

    lhs_initial: Expr
    rhs_initial: Expr
    final: Expr
    steps: Tuple[ProofStep, ...]
    ok: bool


# --------------------------------------------------------------------- #
# CartanStructureProblem                                                 #
# --------------------------------------------------------------------- #


class CartanStructureProblem:
    """``(∇, F)``, Cartan structure equation problem bundle.

    Parameters
    ----------
    connection
        :class:`AffineConnection` whose Cartan structure equations are
        to be proved.
    frame
        :class:`LocalFrame` providing the dual coframe ``e^a`` and the
        basis vector fields ``X_a`` against which the structure
        equations are stated.
    name
        Display name; defaults to a ``(∇, F)`` summary.
    """

    __slots__ = ("_conn", "_frame", "_registry", "_engine", "_name")

    def __init__(
        self,
        connection: AffineConnection,
        frame: LocalFrame,
        *,
        name: Optional[str] = None,
    ) -> None:
        if not isinstance(connection, AffineConnection):
            raise TypeError(
                "CartanStructureProblem requires an AffineConnection"
            )
        if not isinstance(frame, LocalFrame):
            raise TypeError(
                "CartanStructureProblem requires a LocalFrame"
            )
        self._conn = connection
        self._frame = frame
        self._registry = self._build_registry()
        self._engine = self._build_engine()
        self._name = (
            name
            if name is not None
            else (
                f"CartanStructureProblem({connection._repr_inner()},"
                f"{frame.name})"
            )
        )

    def _build_registry(self) -> PropertyRegistry:
        """Per-problem registry declaring the one-form atoms.

        :class:`FrameCovector` and :class:`ConnectionForm` carry degree
        ``1``. The wedge alternating expansion and the arity-1
        :class:`MultiEval` → :class:`Pairing` bridge consult this
        registry before firing, without these declarations both rules
        decline to rewrite, and the RHS never reduces.
        """
        reg = PropertyRegistry()
        reg.declare_for_class(FrameCovector, Graded(Degree.const(1)))
        reg.declare_for_class(ConnectionForm, Graded(Degree.const(1)))
        return reg

    def _build_engine(self) -> ExpansionEngine:
        rules = [
            # 17.C: open ``T^a(U, V) → ⟨e^a, T(∇)(U, V)⟩`` and
            # ``R^a_b(U, V) → ⟨e^a, R(∇)(U, V) X_b⟩`` so the standard
            # ∇-axioms can carry the LHS.
            TorsionFormDefinition(self._conn, self._frame),
            CurvatureFormDefinition(self._conn, self._frame),
            # 16.B: open ``T(∇)(U, V) → ∇_U V − ∇_V U − [U, V]_VF`` and
            # ``R(∇)(U, V) Z → ∇_U ∇_V Z − ∇_V ∇_U Z − ∇_{[U,V]_VF} Z``.
            TorsionDefinitionDefinition(self._conn),
            CurvatureDefinitionDefinition(self._conn),
            # 16.A + 17.D: connection X-slot linearity / scalar pull +
            # Y-additivity. Y-Leibniz uses the per-problem registry to
            # spot the degree-0 Pairing prefactor that arises after
            # frame-decomposition surfaces ``∇_U(P(e^c, V) · X_c)``.
            ConnectionXLinearityDefinition(self._conn),
            ConnectionXScalarPullDefinition(self._conn),
            ConnectionYAdditivityDefinition(self._conn),
            ConnectionYLeibnizDefinition(
                self._conn, registry=self._registry
            ),
            # 17.F.2: positional Y-frame decomposition + bound-index
            # connection-form decomposition. The pair is loop-safe
            # because the Y-decomposition skips frame VFs of this frame,
            # and the form decomposition only fires on those.
            ConnectionEvalYFrameDecompositionDefinition(
                self._conn, self._frame
            ),
            ConnectionFormDecompositionDefinition(
                self._conn, self._frame
            ),
            # 17.F.1: push ``∇_X`` past an :class:`IndexedSum` binder so
            # Y-Leibniz can fire on the body.
            ConnectionEvalIndexedSumPushInDefinition(self._conn),
            # 17.E.3-E.6: structural :class:`IndexedSum` rewrites
            # (Sum/Neg distribute, scalar pull, Pairing push-in both
            # slots, Kronecker contraction).
            IndexedSumSumDistributeDefinition(),
            IndexedSumNegPullDefinition(),
            IndexedSumScalarPullDefinition(),
            IndexedSumPairingPushInLeftDefinition(),
            IndexedSumPairingPushInRightDefinition(),
            IndexedSumKroneckerContractDefinition(),
            # 17.F.2: lift an IndexedSum out of a MultiEval head so the
            # wedge alternating expansion can fire on the body.
            MultiEvalIndexedSumPushInDefinition(),
            # 17.A: ``⟨e^a, X_b⟩ → δ^a_b`` for this frame.
            self._frame.duality_definition(),
            # 12.B: scalar pull-out of either Pairing slot, required
            # once the Y-decomposition surfaces ``⟨e^a, P(e^c, V)·X_c⟩``.
            PairingScalarPullDefinition(),
            # 13.B: Sum / Neg distribution through either Pairing slot.
            PairingLinearityDefinition(),
            # 17.F.1.5: alternating expansion of a wedge of one-forms.
            WedgeMultiEvalAlternatingDefinition(registry=self._registry),
            # 17.F.1.6: arity-1 :class:`MultiEval(α, V)` → :class:`Pairing(α, V)`
            # bridge for one-form ``α``.
            MultiEvalOneFormPairingBridgeDefinition(
                registry=self._registry
            ),
            # 12.A.3: intrinsic Cartan-Koszul ``d`` formula. Connection
            # without a bracket gets the standard rule
            # ``(de^a)(U, V) → U(e^a(V)) − V(e^a(U)) − e^a([U, V]_VF)``;
            # a bracket-equipped connection (Q9 Koszul mode) gets the
            # anchor-pulled, connection-bracket variant
            # ``(d̃e^a)(α, β) → ρ(α)(e^a(β)) − ρ(β)(e^a(α)) − e^a([α, β]_K)``
            # so the LHS's ρ-shapes from Y-Leibniz cancel against the RHS.
            self._intrinsic_d_rule(),
        ]
        return ExpansionEngine(rules)

    def _intrinsic_d_rule(self) -> object:
        """Pick the intrinsic d̃ rule that matches this connection's mode."""
        if self._conn.bracket is None:
            return ExteriorDIntrinsicDefinition()
        return KoszulExteriorDIntrinsicDefinition(self._conn)

    # ---- accessors ------------------------------------------------- #

    @property
    def connection(self) -> AffineConnection:
        return self._conn

    @property
    def frame(self) -> LocalFrame:
        return self._frame

    @property
    def registry(self) -> PropertyRegistry:
        return self._registry

    @property
    def engine(self) -> ExpansionEngine:
        return self._engine

    @property
    def name(self) -> str:
        return self._name

    # ---- builders -------------------------------------------------- #

    def torsion_form(self, upper: FrameIndex | str) -> TorsionForm:
        r"""``T^upper(∇)`` bound to this problem's connection / frame."""
        return TorsionForm(self._conn, self._frame, upper)

    def connection_form(
        self, upper: FrameIndex | str, lower: FrameIndex | str
    ) -> ConnectionForm:
        r"""``ω^upper{}_lower(∇)`` bound to this problem's connection / frame."""
        return ConnectionForm(self._conn, self._frame, upper, lower)

    def curvature_form(
        self, upper: FrameIndex | str, lower: FrameIndex | str
    ) -> CurvatureForm:
        r"""``R^upper{}_lower(∇)`` bound to this problem's connection / frame."""
        return CurvatureForm(self._conn, self._frame, upper, lower)

    def coframe(self, idx: FrameIndex | str) -> FrameCovector:
        r"""``e^idx`` for this problem's frame."""
        return self._frame.coframe(idx)

    def first_cartan_lhs(
        self, U: Expr, V: Expr, upper_a: FrameIndex | str
    ) -> Expr:
        r"""``T^a(U, V)``, Cartan I LHS as a :class:`MultiEval`."""
        return MultiEval(
            self.torsion_form(upper_a), U, V, alternating=True
        )

    def first_cartan_rhs(
        self, U: Expr, V: Expr, upper_a: FrameIndex | str
    ) -> Expr:
        r"""``(de^a)(U, V) + Σ_b (ω^a_b ∧ e^b)(U, V)``, Cartan I RHS.

        The bound dummy ``b`` is freshly minted on each call as a
        bound :class:`FrameIndex`, caller-supplied ``upper_a`` is the
        only free index in the result.
        """
        e_a = self.coframe(upper_a)
        d_term = MultiEval(Act(d, e_a), U, V, alternating=True)
        b = self._frame.index("b", bound=True)
        omega_ab = self.connection_form(upper_a, b)
        e_b = self._frame.coframe(b)
        wedge_body = Wedge(omega_ab, e_b)
        wedge_term = MultiEval(
            IndexedSum(b, self._frame, wedge_body),
            U,
            V,
            alternating=True,
        )
        return Sum.make(d_term, wedge_term)

    def second_cartan_lhs(
        self,
        U: Expr,
        V: Expr,
        upper_a: FrameIndex | str,
        lower_b: FrameIndex | str,
    ) -> Expr:
        r"""``R^a_b(U, V)``, Cartan II LHS as a :class:`MultiEval`."""
        return MultiEval(
            self.curvature_form(upper_a, lower_b),
            U,
            V,
            alternating=True,
        )

    def second_cartan_rhs(
        self,
        U: Expr,
        V: Expr,
        upper_a: FrameIndex | str,
        lower_b: FrameIndex | str,
    ) -> Expr:
        r"""``(dω^a_b)(U, V) + Σ_c (ω^a_c ∧ ω^c_b)(U, V)``, Cartan II RHS.

        The bound dummy ``c`` is freshly minted on each call as a bound
        :class:`FrameIndex`. The caller-supplied ``upper_a`` /
        ``lower_b`` are the only free indices in the result.
        """
        omega_ab = self.connection_form(upper_a, lower_b)
        d_term = MultiEval(Act(d, omega_ab), U, V, alternating=True)
        c = self._frame.index("c", bound=True)
        omega_ac = self.connection_form(upper_a, c)
        omega_cb = self.connection_form(c, lower_b)
        wedge_body = Wedge(omega_ac, omega_cb)
        wedge_term = MultiEval(
            IndexedSum(c, self._frame, wedge_body),
            U,
            V,
            alternating=True,
        )
        return Sum.make(d_term, wedge_term)

    # ---- proof helpers -------------------------------------------- #

    def _expand_to_canonical(
        self, expr: Expr, *, max_steps: int
    ) -> Tuple[Expr, Tuple[ProofStep, ...]]:
        """Iterate engine + :func:`simplify` to a fix-point.

        Distinct from :class:`~jacopy.library.bianchi_problem.BianchiProblem`
        and :class:`~jacopy.library.cartan_form_property.CartanFormPropertyProblem`: the
        Cartan-structure residue contains :class:`Product`'s in
        different factor orders that only :func:`sort_product` (inside
        :func:`simplify`) puts into a common canonical order. The
        per-problem registry provides degree information so
        :func:`sort_product` can fold decidable sign parities.
        """
        all_steps: List[ProofStep] = []
        current = expr
        for _ in range(8):
            new, steps = self._engine.expand(current, max_steps=max_steps)
            all_steps.extend(steps)
            new = simplify(new, registry=self._registry)
            if new == current:
                break
            current = new
        return current, tuple(all_steps)

    def _prove_equality(
        self,
        lhs: Expr,
        rhs: Expr,
        *,
        max_steps: int,
    ) -> CartanStructureProofResult:
        diff = Sum.make(lhs, Neg(rhs))
        final, steps = self._expand_to_canonical(diff, max_steps=max_steps)
        ok = final == Zero or final == Integer(0)
        return CartanStructureProofResult(
            lhs_initial=lhs,
            rhs_initial=rhs,
            final=final,
            steps=steps,
            ok=ok,
        )

    # ---- Cartan I proof ------------------------------------------- #

    def prove_first_cartan(
        self,
        U: Expr,
        V: Expr,
        upper_a: FrameIndex | str,
        *,
        max_steps: int = 4096,
    ) -> CartanStructureProofResult:
        r"""Cartan I: ``T^a(U, V) = (de^a)(U, V) + Σ_b (ω^a_b ∧ e^b)(U, V)``.

        Builds both sides via :meth:`first_cartan_lhs` /
        :meth:`first_cartan_rhs`, expands the difference through the
        engine + :func:`simplify` fix-point, and reports
        :attr:`CartanStructureProofResult.ok` iff the residue reduces to
        ``0``.
        """
        lhs = self.first_cartan_lhs(U, V, upper_a)
        rhs = self.first_cartan_rhs(U, V, upper_a)
        return self._prove_equality(lhs, rhs, max_steps=max_steps)

    # ---- Cartan II proof ------------------------------------------ #

    def prove_second_cartan(
        self,
        U: Expr,
        V: Expr,
        upper_a: FrameIndex | str,
        lower_b: FrameIndex | str,
        *,
        max_steps: int = 8192,
    ) -> CartanStructureProofResult:
        r"""Cartan II: ``R^a_b(U, V) = (dω^a_b)(U, V) + Σ_c (ω^a_c ∧ ω^c_b)(U, V)``.

        Builds both sides via :meth:`second_cartan_lhs` /
        :meth:`second_cartan_rhs`, expands the difference through the
        engine + :func:`simplify` fix-point, and reports
        :attr:`CartanStructureProofResult.ok` iff the residue reduces to
        ``0``. The deeper LHS (curvature unfolds to nested ``∇_U ∇_V``
        + ``∇_{[U,V]_VF}`` shapes) means a higher
        :attr:`max_steps` default than Cartan I.
        """
        lhs = self.second_cartan_lhs(U, V, upper_a, lower_b)
        rhs = self.second_cartan_rhs(U, V, upper_a, lower_b)
        return self._prove_equality(lhs, rhs, max_steps=max_steps)

    # ---- dunder --------------------------------------------------- #

    def __repr__(self) -> str:
        return (
            f"CartanStructureProblem({self._conn._repr_inner()},"
            f"{self._frame.name})"
        )
