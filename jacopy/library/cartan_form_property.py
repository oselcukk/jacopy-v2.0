r"""
Cartan form-property problem wrapper.

Given an affine connection ``∇`` (and optionally a metric ``g``) on a
local frame ``F``, the four local-component objects

* ``ω^a{}_b(∇)``, connection 1-form,
* ``Q_{ab}(∇, g)``, non-metricity 1-form (needs ``g``),
* ``T^a(∇)``, torsion 2-form,
* ``R^a{}_b(∇)``, curvature 2-form,

obey textbook *form-degree* properties: ``ω`` and ``Q`` are
``C^∞``-linear in ``V``, while ``T`` and ``R`` are ``C^∞``-bilinear
in ``(U, V)`` and antisymmetric (i.e. genuine 2-forms).

The 1-form claim says ``ω(fV) = f·ω(V)`` (and ``+`` over the ``V``
slot); the 2-form claim says the same ``C^∞``-linearity in each of
``(U, V)`` plus antisymmetry ``T(U, V) = −T(V, U)``.

This module ships :class:`CartanFormPropertyProblem`, a thin bundle
that pairs

* an :class:`~jacopy.calculus.connection.AffineConnection` ``∇``,
* a :class:`~jacopy.calculus.local_frame.LocalFrame` ``F``,
* an optional :class:`~jacopy.calculus.metric.MetricTensor` ``g``
  (only required for the ``Q`` 1-form proofs)

with a pre-configured :class:`~jacopy.proof.expansion.ExpansionEngine`
carrying every axiom needed to mechanise the four claims:

* the four Cartan-form definitions opening ``⟨ω^a_b, V⟩``,
  ``⟨Q_{ab}, V⟩``, ``T^a(U, V)``, ``R^a_b(U, V)`` into their bare
  ``∇`` / ``T`` / ``R`` / ``Q`` shapes,
* the connection X-linearity / X-scalar-pull rules,
* the torsion / curvature ``C^∞``-bilinearity + ``(X, Y)``-antisymmetry
  rules,
* the non-metricity V-linearity / V-scalar-pull / X↔Y symmetry rules,
* the Pairing ``C^∞``-linearity rule.

The two pure-V claims (1-form for ``ω`` and ``Q``) are stated as the
single equality ``form(f·V) = f·form(V)`` plus the ``+``-additive
``form(V₁ + V₂) = form(V₁) + form(V₂)``. The two ``(U, V)`` claims are
the same equalities in each slot plus an antisymmetry test
``form(U, V) + form(V, U) = 0``.

Each ``prove_*`` method returns a :class:`CartanFormPropertyProofResult`
whose ``ok`` flag is ``True`` iff the difference ``LHS − RHS`` reduces
to ``0`` after the engine + canonicalize + collect_terms fixpoint, the
same protocol :class:`~jacopy.library.bianchi_problem.BianchiProblem`
uses.

Termination notes. The form definitions only fire on closed-shape
``Pairing(form, V)`` / ``MultiEval(form, U, V)`` patterns; their
right-hand sides contain no form atom, so a single open-and-canonicalise
pass per term suffices. Antisymmetry rules are repr-canonicalize so
they apply at most once per node, same termination story as
:class:`~jacopy.calculus.metric.MetricEvalSymmetryDefinition`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

from jacopy.algorithms.canonicalize import canonicalize
from jacopy.algorithms.collect_terms import collect_terms
from jacopy.calculus.cartan_forms import (
    ConnectionForm,
    ConnectionFormDefinition,
    CurvatureForm,
    CurvatureFormDefinition,
    NonMetricityForm,
    NonMetricityFormDefinition,
    TorsionForm,
    TorsionFormDefinition,
)
from jacopy.calculus.connection import (
    AffineConnection,
    ConnectionXLinearityDefinition,
    ConnectionXScalarPullDefinition,
    ConnectionYAdditivityDefinition,
)
from jacopy.calculus.local_frame import FrameIndex, LocalFrame
from jacopy.calculus.metric import MetricTensor
from jacopy.calculus.non_metricity import (
    NonMetricityVLinearityDefinition,
    NonMetricityVScalarPullDefinition,
    NonMetricityXYSymmetryDefinition,
)
from jacopy.calculus.pairing import Pairing
from jacopy.calculus.pairing_axioms import PairingLinearityDefinition
from jacopy.calculus.pairing_linearity_axioms import PairingScalarPullDefinition
from jacopy.calculus.torsion_curvature import (
    CurvatureXLinearityDefinition,
    CurvatureXScalarPullDefinition,
    CurvatureXYAntiSymmetryDefinition,
    CurvatureYLinearityDefinition,
    CurvatureYScalarPullDefinition,
    TorsionAntiSymmetryDefinition,
    TorsionXLinearityDefinition,
    TorsionXScalarPullDefinition,
    TorsionYLinearityDefinition,
    TorsionYScalarPullDefinition,
)
from jacopy.core.expr import Expr, Integer, Neg, Product, Sum, Zero
from jacopy.core.multi_eval import MultiEval
from jacopy.proof.expansion import ExpansionEngine
from jacopy.proof.step import ProofStep


# --------------------------------------------------------------------- #
# Proof result                                                           #
# --------------------------------------------------------------------- #


@dataclass(frozen=True)
class CartanFormPropertyProofResult:
    """Outcome of a Cartan form-property proof attempt.

    Attributes
    ----------
    lhs_initial, rhs_initial
        The two sides of the claimed equality before expansion.
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
# CartanFormPropertyProblem                                              #
# --------------------------------------------------------------------- #


class CartanFormPropertyProblem:
    """``(∇, F, g?)``, Cartan form-property problem bundle.

    Parameters
    ----------
    connection
        :class:`AffineConnection` whose Cartan-form package is to be
        examined. Required.
    frame
        :class:`LocalFrame` against which the Cartan-form components
        ``ω^a_b``, ``Q_{ab}``, ``T^a``, ``R^a_b`` are read off. Required.
    metric
        Optional :class:`MetricTensor`. Required only for the ``Q``
        1-form proofs; ``ω`` / ``T`` / ``R`` proofs work without it.
    name
        Display name; defaults to a ``(∇, F)`` summary.
    """

    __slots__ = ("_conn", "_frame", "_metric", "_engine", "_name")

    def __init__(
        self,
        connection: AffineConnection,
        frame: LocalFrame,
        *,
        metric: Optional[MetricTensor] = None,
        name: Optional[str] = None,
    ) -> None:
        if not isinstance(connection, AffineConnection):
            raise TypeError(
                "CartanFormPropertyProblem requires an AffineConnection"
            )
        if not isinstance(frame, LocalFrame):
            raise TypeError(
                "CartanFormPropertyProblem requires a LocalFrame"
            )
        if metric is not None and not isinstance(metric, MetricTensor):
            raise TypeError(
                "CartanFormPropertyProblem metric must be a "
                "MetricTensor or None"
            )
        self._conn = connection
        self._frame = frame
        self._metric = metric
        self._engine = self._build_engine()
        self._name = (
            name
            if name is not None
            else (
                f"CartanFormPropertyProblem({connection._repr_inner()},"
                f"{frame.name})"
            )
        )

    def _build_engine(self) -> ExpansionEngine:
        rules = [
            # Open the form heads so the slot-linearity rules below can
            # fire on the underlying ∇ / Q / T / R.
            ConnectionFormDefinition(self._conn, self._frame),
            TorsionFormDefinition(self._conn, self._frame),
            CurvatureFormDefinition(self._conn, self._frame),
            # Connection X-slot linearity / scalar pull, carries
            # ∇_{fV} X_b → f·∇_V X_b for the ω 1-form proof, and the
            # Sum / Neg additivity for ω(V₁ + V₂).
            ConnectionXLinearityDefinition(self._conn),
            ConnectionXScalarPullDefinition(self._conn),
            ConnectionYAdditivityDefinition(self._conn),
            # Torsion / curvature primitive C∞-bilinearity +
            # (X, Y)-antisymmetry. These are the workhorses for the
            # T^a / R^a_b 2-form proofs.
            TorsionXLinearityDefinition(self._conn),
            TorsionYLinearityDefinition(self._conn),
            TorsionXScalarPullDefinition(self._conn),
            TorsionYScalarPullDefinition(self._conn),
            TorsionAntiSymmetryDefinition(self._conn),
            CurvatureXLinearityDefinition(self._conn),
            CurvatureYLinearityDefinition(self._conn),
            CurvatureXScalarPullDefinition(self._conn),
            CurvatureYScalarPullDefinition(self._conn),
            CurvatureXYAntiSymmetryDefinition(self._conn),
            # Pull the scalar coefficient out of either Pairing slot
            # once it surfaces (e.g. ⟨e^a, f·∇_V X_b⟩ → f·⟨…⟩).
            PairingScalarPullDefinition(),
            # Distribute Sum / Neg through either Pairing slot
            # (e.g. ⟨e^a, T(U,V₁) + T(U,V₂)⟩ → ⟨e^a, T(U,V₁)⟩ +
            # ⟨e^a, T(U,V₂)⟩, and ⟨e^a, −T(V,U)⟩ → −⟨e^a, T(V,U)⟩).
            # Required for the additivity / antisymmetry side of the
            # 2-form proofs once Torsion/Curvature linearity has
            # surfaced a Sum or Neg in the Pairing's right slot.
            PairingLinearityDefinition(),
        ]
        if self._metric is not None:
            rules.extend([
                NonMetricityFormDefinition(
                    self._conn, self._metric, self._frame
                ),
                NonMetricityVLinearityDefinition(
                    self._conn, self._metric
                ),
                NonMetricityVScalarPullDefinition(
                    self._conn, self._metric
                ),
                NonMetricityXYSymmetryDefinition(
                    self._conn, self._metric
                ),
            ])
        return ExpansionEngine(rules)

    # ---- accessors ------------------------------------------------- #

    @property
    def connection(self) -> AffineConnection:
        return self._conn

    @property
    def frame(self) -> LocalFrame:
        return self._frame

    @property
    def metric(self) -> Optional[MetricTensor]:
        return self._metric

    @property
    def engine(self) -> ExpansionEngine:
        return self._engine

    @property
    def name(self) -> str:
        return self._name

    # ---- builders -------------------------------------------------- #

    def omega(self, upper: FrameIndex | str, lower: FrameIndex | str) -> ConnectionForm:
        r"""``ω^upper{}_lower(∇)`` bound to this problem's connection / frame."""
        return ConnectionForm(self._conn, self._frame, upper, lower)

    def Q(self, lower_a: FrameIndex | str, lower_b: FrameIndex | str) -> NonMetricityForm:
        r"""``Q_{ab}(∇, g)`` bound to this problem's connection / metric / frame.

        Raises a :class:`ValueError` when no metric was supplied at
        construction time, the non-metricity form needs one.
        """
        if self._metric is None:
            raise ValueError(
                "CartanFormPropertyProblem.Q requires a metric, "
                "construct the problem with metric=..."
            )
        return NonMetricityForm(
            self._conn, self._metric, self._frame, lower_a, lower_b
        )

    def T(self, upper: FrameIndex | str) -> TorsionForm:
        r"""``T^upper(∇)`` bound to this problem's connection / frame."""
        return TorsionForm(self._conn, self._frame, upper)

    def R(self, upper: FrameIndex | str, lower: FrameIndex | str) -> CurvatureForm:
        r"""``R^upper{}_lower(∇)`` bound to this problem's connection / frame."""
        return CurvatureForm(self._conn, self._frame, upper, lower)

    # ---- proof helpers -------------------------------------------- #

    def _expand_to_canonical(
        self, expr: Expr, *, max_steps: int
    ) -> Tuple[Expr, Tuple[ProofStep, ...]]:
        """Iterate engine + canonicalize + collect_terms to a fixpoint.

        Same shape as :meth:`BianchiProblem._expand_to_canonical`. The
        four form-property proofs all amount to "two sides cancel
        after a few rewrites"; the canonicalize / collect_terms passes
        between engine cycles let the engine see Sum-level residues
        the Sum-flatten pass exposes.
        """
        all_steps: List[ProofStep] = []
        current = expr
        for _ in range(8):
            new, steps = self._engine.expand(current, max_steps=max_steps)
            all_steps.extend(steps)
            new = canonicalize(new)
            new = collect_terms(new)
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
    ) -> CartanFormPropertyProofResult:
        diff = Sum.make(lhs, Neg(rhs))
        final, steps = self._expand_to_canonical(diff, max_steps=max_steps)
        ok = final == Zero or final == Integer(0)
        return CartanFormPropertyProofResult(
            lhs_initial=lhs,
            rhs_initial=rhs,
            final=final,
            steps=steps,
            ok=ok,
        )

    # ---- ω 1-form proofs ----------------------------------------- #

    def omega_eval(
        self, upper: FrameIndex | str, lower: FrameIndex | str, V: Expr
    ) -> Pairing:
        r"""``⟨ω^upper{}_lower, V⟩``, the engine entry point."""
        return Pairing(self.omega(upper, lower), V)

    def prove_omega_scalar_linear_in_V(
        self,
        upper: FrameIndex | str,
        lower: FrameIndex | str,
        f: Expr,
        V: Expr,
        *,
        max_steps: int = 256,
    ) -> CartanFormPropertyProofResult:
        r"""``⟨ω^a_b, f·V⟩ = f·⟨ω^a_b, V⟩``."""
        lhs = self.omega_eval(upper, lower, Product.make(f, V))
        rhs = Product.make(f, self.omega_eval(upper, lower, V))
        return self._prove_equality(lhs, rhs, max_steps=max_steps)

    def prove_omega_additive_in_V(
        self,
        upper: FrameIndex | str,
        lower: FrameIndex | str,
        V1: Expr,
        V2: Expr,
        *,
        max_steps: int = 256,
    ) -> CartanFormPropertyProofResult:
        r"""``⟨ω^a_b, V₁ + V₂⟩ = ⟨ω^a_b, V₁⟩ + ⟨ω^a_b, V₂⟩``."""
        lhs = self.omega_eval(upper, lower, Sum.make(V1, V2))
        rhs = Sum.make(
            self.omega_eval(upper, lower, V1),
            self.omega_eval(upper, lower, V2),
        )
        return self._prove_equality(lhs, rhs, max_steps=max_steps)

    # ---- Q 1-form proofs ----------------------------------------- #

    def Q_eval(
        self,
        lower_a: FrameIndex | str,
        lower_b: FrameIndex | str,
        V: Expr,
    ) -> Pairing:
        r"""``⟨Q_{ab}, V⟩``, the engine entry point."""
        return Pairing(self.Q(lower_a, lower_b), V)

    def prove_Q_scalar_linear_in_V(
        self,
        lower_a: FrameIndex | str,
        lower_b: FrameIndex | str,
        f: Expr,
        V: Expr,
        *,
        max_steps: int = 256,
    ) -> CartanFormPropertyProofResult:
        r"""``⟨Q_{ab}, f·V⟩ = f·⟨Q_{ab}, V⟩``."""
        lhs = self.Q_eval(lower_a, lower_b, Product.make(f, V))
        rhs = Product.make(f, self.Q_eval(lower_a, lower_b, V))
        return self._prove_equality(lhs, rhs, max_steps=max_steps)

    def prove_Q_additive_in_V(
        self,
        lower_a: FrameIndex | str,
        lower_b: FrameIndex | str,
        V1: Expr,
        V2: Expr,
        *,
        max_steps: int = 256,
    ) -> CartanFormPropertyProofResult:
        r"""``⟨Q_{ab}, V₁ + V₂⟩ = ⟨Q_{ab}, V₁⟩ + ⟨Q_{ab}, V₂⟩``."""
        lhs = self.Q_eval(lower_a, lower_b, Sum.make(V1, V2))
        rhs = Sum.make(
            self.Q_eval(lower_a, lower_b, V1),
            self.Q_eval(lower_a, lower_b, V2),
        )
        return self._prove_equality(lhs, rhs, max_steps=max_steps)

    # ---- T^a 2-form proofs --------------------------------------- #

    def T_eval(
        self, upper: FrameIndex | str, U: Expr, V: Expr
    ) -> MultiEval:
        r"""``T^a(U, V)``, the engine entry point."""
        return MultiEval(self.T(upper), U, V)

    def prove_T_scalar_linear_in_first(
        self,
        upper: FrameIndex | str,
        f: Expr,
        U: Expr,
        V: Expr,
        *,
        max_steps: int = 256,
    ) -> CartanFormPropertyProofResult:
        r"""``T^a(f·U, V) = f·T^a(U, V)``."""
        lhs = self.T_eval(upper, Product.make(f, U), V)
        rhs = Product.make(f, self.T_eval(upper, U, V))
        return self._prove_equality(lhs, rhs, max_steps=max_steps)

    def prove_T_scalar_linear_in_second(
        self,
        upper: FrameIndex | str,
        f: Expr,
        U: Expr,
        V: Expr,
        *,
        max_steps: int = 256,
    ) -> CartanFormPropertyProofResult:
        r"""``T^a(U, f·V) = f·T^a(U, V)``."""
        lhs = self.T_eval(upper, U, Product.make(f, V))
        rhs = Product.make(f, self.T_eval(upper, U, V))
        return self._prove_equality(lhs, rhs, max_steps=max_steps)

    def prove_T_additive_in_first(
        self,
        upper: FrameIndex | str,
        U1: Expr,
        U2: Expr,
        V: Expr,
        *,
        max_steps: int = 256,
    ) -> CartanFormPropertyProofResult:
        r"""``T^a(U₁ + U₂, V) = T^a(U₁, V) + T^a(U₂, V)``."""
        lhs = self.T_eval(upper, Sum.make(U1, U2), V)
        rhs = Sum.make(
            self.T_eval(upper, U1, V),
            self.T_eval(upper, U2, V),
        )
        return self._prove_equality(lhs, rhs, max_steps=max_steps)

    def prove_T_additive_in_second(
        self,
        upper: FrameIndex | str,
        U: Expr,
        V1: Expr,
        V2: Expr,
        *,
        max_steps: int = 256,
    ) -> CartanFormPropertyProofResult:
        r"""``T^a(U, V₁ + V₂) = T^a(U, V₁) + T^a(U, V₂)``."""
        lhs = self.T_eval(upper, U, Sum.make(V1, V2))
        rhs = Sum.make(
            self.T_eval(upper, U, V1),
            self.T_eval(upper, U, V2),
        )
        return self._prove_equality(lhs, rhs, max_steps=max_steps)

    def prove_T_antisymmetric(
        self,
        upper: FrameIndex | str,
        U: Expr,
        V: Expr,
        *,
        max_steps: int = 256,
    ) -> CartanFormPropertyProofResult:
        r"""``T^a(U, V) + T^a(V, U) = 0``."""
        lhs = Sum.make(
            self.T_eval(upper, U, V),
            self.T_eval(upper, V, U),
        )
        rhs = Zero
        return self._prove_equality(lhs, rhs, max_steps=max_steps)

    # ---- R^a_b 2-form proofs ------------------------------------- #

    def R_eval(
        self,
        upper: FrameIndex | str,
        lower: FrameIndex | str,
        U: Expr,
        V: Expr,
    ) -> MultiEval:
        r"""``R^a_b(U, V)``, the engine entry point."""
        return MultiEval(self.R(upper, lower), U, V)

    def prove_R_scalar_linear_in_first(
        self,
        upper: FrameIndex | str,
        lower: FrameIndex | str,
        f: Expr,
        U: Expr,
        V: Expr,
        *,
        max_steps: int = 256,
    ) -> CartanFormPropertyProofResult:
        r"""``R^a_b(f·U, V) = f·R^a_b(U, V)``."""
        lhs = self.R_eval(upper, lower, Product.make(f, U), V)
        rhs = Product.make(f, self.R_eval(upper, lower, U, V))
        return self._prove_equality(lhs, rhs, max_steps=max_steps)

    def prove_R_scalar_linear_in_second(
        self,
        upper: FrameIndex | str,
        lower: FrameIndex | str,
        f: Expr,
        U: Expr,
        V: Expr,
        *,
        max_steps: int = 256,
    ) -> CartanFormPropertyProofResult:
        r"""``R^a_b(U, f·V) = f·R^a_b(U, V)``."""
        lhs = self.R_eval(upper, lower, U, Product.make(f, V))
        rhs = Product.make(f, self.R_eval(upper, lower, U, V))
        return self._prove_equality(lhs, rhs, max_steps=max_steps)

    def prove_R_additive_in_first(
        self,
        upper: FrameIndex | str,
        lower: FrameIndex | str,
        U1: Expr,
        U2: Expr,
        V: Expr,
        *,
        max_steps: int = 256,
    ) -> CartanFormPropertyProofResult:
        r"""``R^a_b(U₁ + U₂, V) = R^a_b(U₁, V) + R^a_b(U₂, V)``."""
        lhs = self.R_eval(upper, lower, Sum.make(U1, U2), V)
        rhs = Sum.make(
            self.R_eval(upper, lower, U1, V),
            self.R_eval(upper, lower, U2, V),
        )
        return self._prove_equality(lhs, rhs, max_steps=max_steps)

    def prove_R_additive_in_second(
        self,
        upper: FrameIndex | str,
        lower: FrameIndex | str,
        U: Expr,
        V1: Expr,
        V2: Expr,
        *,
        max_steps: int = 256,
    ) -> CartanFormPropertyProofResult:
        r"""``R^a_b(U, V₁ + V₂) = R^a_b(U, V₁) + R^a_b(U, V₂)``."""
        lhs = self.R_eval(upper, lower, U, Sum.make(V1, V2))
        rhs = Sum.make(
            self.R_eval(upper, lower, U, V1),
            self.R_eval(upper, lower, U, V2),
        )
        return self._prove_equality(lhs, rhs, max_steps=max_steps)

    def prove_R_antisymmetric(
        self,
        upper: FrameIndex | str,
        lower: FrameIndex | str,
        U: Expr,
        V: Expr,
        *,
        max_steps: int = 256,
    ) -> CartanFormPropertyProofResult:
        r"""``R^a_b(U, V) + R^a_b(V, U) = 0``."""
        lhs = Sum.make(
            self.R_eval(upper, lower, U, V),
            self.R_eval(upper, lower, V, U),
        )
        rhs = Zero
        return self._prove_equality(lhs, rhs, max_steps=max_steps)

    # ---- dunder --------------------------------------------------- #

    def __repr__(self) -> str:
        if self._metric is None:
            return (
                f"CartanFormPropertyProblem({self._conn._repr_inner()},"
                f"{self._frame.name})"
            )
        return (
            f"CartanFormPropertyProblem({self._conn._repr_inner()},"
            f"{self._frame.name},{self._metric._repr_inner()})"
        )
