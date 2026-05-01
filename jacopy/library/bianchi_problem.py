r"""
Bianchi-identity problem wrapper, Faz 16.D.

A :class:`BianchiProblem` is a thin bundle that pairs an
:class:`~jacopy.calculus.connection.AffineConnection` ``∇`` with a
pre-configured :class:`~jacopy.proof.expansion.ExpansionEngine`
carrying every axiom a Bianchi-identity proof typically cites:

* the three connection-linearity rules
  (:class:`ConnectionXLinearityDefinition`,
  :class:`ConnectionYAdditivityDefinition`,
  :class:`ConnectionYLeibnizDefinition`),
* the two structural-tensor definitions
  (:class:`TorsionDefinitionDefinition`,
  :class:`CurvatureDefinitionDefinition`),
* the two tensor-Leibniz rules
  (:class:`TorsionCovariantDerivativeDefinition`,
  :class:`CurvatureCovariantDerivativeDefinition`),
* the LBVF closure axioms
  (:class:`LieBracketVfAntiSymmetryDefinition`,
  :class:`LieBracketVfJacobiDefinition`).

The wrapper also exposes a small algebra of cyclic-sum constructors
for the textbook Bianchi shapes:

* :meth:`cyclic_sum`, generic ``cycl_{U,V,W} f(U,V,W)`` over a
  3-permutation, ``cycl_{U,V,W,Z} f(U,V,W,Z)`` over a fixed last slot,
* :meth:`first_bianchi_lhs` / :meth:`first_bianchi_rhs`,
  ``cycl R(U,V)W`` and ``cycl[(∇_U T)(V,W) + T(T(U,V),W)]``,
* :meth:`second_bianchi_lhs` / :meth:`second_bianchi_rhs`,
  ``cycl (∇_U R)(V,W)Z`` and ``cycl R(U, T(V,W)) Z``.

:meth:`prove_first_bianchi` and :meth:`prove_second_bianchi` expand
both sides through the bundled engine and return a
:class:`BianchiProofResult` with both transcripts plus an ``ok``
verdict (``LHS_canonical == RHS_canonical``).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

from jacopy.algorithms.canonicalize import canonicalize
from jacopy.algorithms.collect_terms import collect_terms
from jacopy.calculus.bracket_apply_axioms import (
    BracketApplyAntiSymmetryDefinition,
    BracketApplyArgAntisymmetryDefinition,
    BracketApplyJacobiDefinition,
    BracketApplyNegLinearityDefinition,
    BracketApplySumLinearityDefinition,
)
from jacopy.calculus.closure_axioms import (
    LieBracketVfAntiSymmetryDefinition,
    LieBracketVfJacobiDefinition,
)
from jacopy.calculus.sn_function_axiom import (
    LieBracketVfAntisymmetryDefinition as LieBracketVfArgAntisymmetryDefinition,
    LieBracketVfNegLinearityDefinition,
    LieBracketVfSumLinearityDefinition,
)
from jacopy.core.expr import Integer, Neg, Zero
from jacopy.calculus.connection import (
    AffineConnection,
    ConnectionEvalExpr,
    ConnectionXLinearityDefinition,
    ConnectionYAdditivityDefinition,
    ConnectionYLeibnizDefinition,
)
from jacopy.calculus.torsion_curvature import (
    Curvature,
    CurvatureCovariantDerivative,
    CurvatureCovariantDerivativeDefinition,
    CurvatureDefinitionDefinition,
    Torsion,
    TorsionCovariantDerivative,
    TorsionCovariantDerivativeDefinition,
    TorsionDefinitionDefinition,
)
from jacopy.core.expr import Expr, Sum
from jacopy.core.registry import PropertyRegistry
from jacopy.proof.expansion import ExpansionEngine
from jacopy.proof.step import ProofStep


# --------------------------------------------------------------------- #
# Cyclic sums                                                            #
# --------------------------------------------------------------------- #


def cyclic_sum_3(
    factory: Callable[..., Expr], a: Expr, b: Expr, c: Expr
) -> Expr:
    r"""``factory(a,b,c) + factory(b,c,a) + factory(c,a,b)``, 3-cycle.

    The standard cyclic sum used in Bianchi I.
    """
    return Sum.make(
        factory(a, b, c),
        factory(b, c, a),
        factory(c, a, b),
    )


def cyclic_sum_3_fixed_last(
    factory: Callable[..., Expr],
    a: Expr,
    b: Expr,
    c: Expr,
    last: Expr,
) -> Expr:
    r"""``factory(a,b,c,last) + factory(b,c,a,last) + factory(c,a,b,last)``.

    The cyclic sum over the first three slots with a fixed final slot,
    used in Bianchi II where ``Z`` is held fixed across the cycle.
    """
    return Sum.make(
        factory(a, b, c, last),
        factory(b, c, a, last),
        factory(c, a, b, last),
    )


# --------------------------------------------------------------------- #
# Proof result                                                           #
# --------------------------------------------------------------------- #


@dataclass(frozen=True)
class BianchiProofResult:
    """Outcome of a Bianchi-identity proof attempt.

    Attributes
    ----------
    lhs_initial, rhs_initial
        Top-level LHS / RHS expressions before expansion.
    lhs_final, rhs_final
        Fully expanded canonical forms returned by the engine.
    lhs_steps, rhs_steps
        Transcripts of the engine's rewriting passes for each side.
    ok
        ``True`` iff ``lhs_final == rhs_final`` after expansion.
    """

    lhs_initial: Expr
    rhs_initial: Expr
    lhs_final: Expr
    rhs_final: Expr
    lhs_steps: Tuple[ProofStep, ...]
    rhs_steps: Tuple[ProofStep, ...]
    ok: bool


# --------------------------------------------------------------------- #
# BianchiProblem                                                         #
# --------------------------------------------------------------------- #


class BianchiProblem:
    """``(∇, registry)``, Bianchi-identity problem bundle.

    Parameters
    ----------
    connection
        :class:`AffineConnection` whose Bianchi identities are to be
        explored. Required.
    registry
        Optional :class:`PropertyRegistry`. The Y-Leibniz rule consults
        the registry for the ``Graded(degree=0)`` test on the function
        factor of a :class:`~jacopy.core.expr.Product`. If you only
        care about pure-vector-field arguments, ``None`` suffices.
    name
        Display name; defaults to ``f"BianchiProblem({∇})"``.
    """

    __slots__ = ("_conn", "_registry", "_engine", "_name")

    def __init__(
        self,
        connection: AffineConnection,
        *,
        registry: Optional[PropertyRegistry] = None,
        name: Optional[str] = None,
    ) -> None:
        if not isinstance(connection, AffineConnection):
            raise TypeError(
                "BianchiProblem requires an AffineConnection"
            )
        if registry is not None and not isinstance(registry, PropertyRegistry):
            raise TypeError(
                "BianchiProblem registry must be a PropertyRegistry or None"
            )
        self._conn = connection
        self._registry = registry
        self._engine = self._build_engine()
        self._name = (
            name
            if name is not None
            else f"BianchiProblem({connection._repr_inner()})"
        )

    def _build_engine(self) -> ExpansionEngine:
        rules = [
            # Definition unfolds first, turns Torsion/Curvature into
            # ∇-commutator + bracket terms (LBVF or BracketApply
            # depending on whether the connection has a custom bracket).
            TorsionCovariantDerivativeDefinition(self._conn),
            CurvatureCovariantDerivativeDefinition(self._conn),
            TorsionDefinitionDefinition(self._conn),
            CurvatureDefinitionDefinition(self._conn),
            # Connection linearity / Leibniz pushes Sum/Neg/Product
            # through the X / Y slots.
            ConnectionXLinearityDefinition(self._conn),
            ConnectionYAdditivityDefinition(self._conn),
            ConnectionYLeibnizDefinition(self._conn, registry=self._registry),
        ]
        rules.extend(self._bracket_axioms())
        return ExpansionEngine(rules)

    def _bracket_axioms(self) -> List:
        """Bracket-side axioms: LBVF rules or BracketApply rules.

        When the connection has no custom bracket (``connection.bracket
        is None``), the Torsion/Curvature definitions emit
        :class:`~jacopy.algebra.lie_bracket_vf.LieBracketVF` and the
        engine bundles the LBVF closure family. With a custom bracket
        (Q9 ``koszul_connection``) the same definitions emit
        :class:`~jacopy.brackets.base.BracketApply` headed by that
        bracket, and the engine swaps in the
        :mod:`jacopy.calculus.bracket_apply_axioms` parallel, same
        five rule shapes (Sum/Neg-linearity, atom-antisym,
        Sum-antisym, cyclic Jacobi) but matching the opaque bracket
        node instead of the LBVF atom. The two rule sets are mutually
        exclusive: only one bracket flavor surfaces in any given
        connection's residues.
        """
        bracket = self._conn.bracket
        if bracket is None:
            return [
                LieBracketVfSumLinearityDefinition(),
                LieBracketVfNegLinearityDefinition(),
                LieBracketVfArgAntisymmetryDefinition(),
                LieBracketVfAntiSymmetryDefinition(),
                LieBracketVfJacobiDefinition(),
            ]
        return [
            BracketApplySumLinearityDefinition(bracket),
            BracketApplyNegLinearityDefinition(bracket),
            BracketApplyArgAntisymmetryDefinition(bracket),
            BracketApplyAntiSymmetryDefinition(bracket),
            BracketApplyJacobiDefinition(bracket),
        ]

    # ---- accessors ------------------------------------------------- #

    @property
    def connection(self) -> AffineConnection:
        return self._conn

    @property
    def registry(self) -> Optional[PropertyRegistry]:
        return self._registry

    @property
    def engine(self) -> ExpansionEngine:
        return self._engine

    @property
    def name(self) -> str:
        return self._name

    # ---- builders -------------------------------------------------- #

    def torsion(self, X: Expr, Y: Expr) -> Torsion:
        r"""``T(∇)(X, Y)`` bound to this problem's connection."""
        return Torsion(self._conn, X, Y)

    def curvature(self, X: Expr, Y: Expr, Z: Expr) -> Curvature:
        r"""``R(∇)(X, Y) Z`` bound to this problem's connection."""
        return Curvature(self._conn, X, Y, Z)

    def cov_deriv_torsion(
        self, U: Expr, V: Expr, W: Expr
    ) -> TorsionCovariantDerivative:
        r"""``(∇_U T)(V, W)`` bound to this problem's connection."""
        return TorsionCovariantDerivative(self._conn, U, V, W)

    def cov_deriv_curvature(
        self, U: Expr, V: Expr, W: Expr, Z: Expr
    ) -> CurvatureCovariantDerivative:
        r"""``(∇_U R)(V, W) Z`` bound to this problem's connection."""
        return CurvatureCovariantDerivative(self._conn, U, V, W, Z)

    # ---- Bianchi I LHS/RHS ---------------------------------------- #

    def first_bianchi_lhs(self, U: Expr, V: Expr, W: Expr) -> Expr:
        r"""``cycl_{U,V,W} R(U, V) W``."""
        return cyclic_sum_3(self.curvature, U, V, W)

    def first_bianchi_rhs(self, U: Expr, V: Expr, W: Expr) -> Expr:
        r"""``cycl_{U,V,W} [(∇_U T)(V, W) + T(T(U, V), W)]``."""
        terms: List[Expr] = []
        for a, b, c in ((U, V, W), (V, W, U), (W, U, V)):
            terms.append(self.cov_deriv_torsion(a, b, c))
            terms.append(self.torsion(self.torsion(a, b), c))
        return Sum.make(*terms)

    # ---- Bianchi II LHS/RHS --------------------------------------- #

    def second_bianchi_lhs(
        self, U: Expr, V: Expr, W: Expr, Z: Expr
    ) -> Expr:
        r"""``cycl_{U,V,W} (∇_U R)(V, W) Z``."""
        return cyclic_sum_3_fixed_last(
            self.cov_deriv_curvature, U, V, W, Z
        )

    def second_bianchi_rhs(
        self, U: Expr, V: Expr, W: Expr, Z: Expr
    ) -> Expr:
        r"""``cycl_{U,V,W} R(U, T(V, W)) Z``."""
        return Sum.make(
            self.curvature(U, self.torsion(V, W), Z),
            self.curvature(V, self.torsion(W, U), Z),
            self.curvature(W, self.torsion(U, V), Z),
        )

    # ---- proofs --------------------------------------------------- #

    def _expand_to_canonical(
        self, expr: Expr, *, max_steps: int
    ) -> Tuple[Expr, Tuple[ProofStep, ...]]:
        """Iterate engine + canonicalize + collect_terms to a fixpoint.

        The engine handles the algebraic rewrites (definitions / linearity /
        Jacobi); :func:`canonicalize` pushes ``Neg`` through ``Sum``,
        flattens nested sums and folds integer coefficients;
        :func:`collect_terms` cancels structurally-equal sibling terms with
        opposite signs. Cycling these three lets the engine "see" residues
        the canonicalize pass exposes (e.g. a ``Neg(Sum(...))`` that opens
        into per-term LBVFs the antisym rule can canonicalize).
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

    def prove_first_bianchi(
        self, U: Expr, V: Expr, W: Expr, *, max_steps: int = 512
    ) -> BianchiProofResult:
        r"""Expand both sides of the first Bianchi identity and compare.

        Identity: ``cycl R(U,V)W = cycl[(∇_U T)(V,W) + T(T(U,V),W)]``.

        The proof builds the difference ``LHS − RHS``, expands and
        canonicalizes it through the engine, and reports ``ok=True`` iff
        the result reduces to ``0``. Both ``lhs_final`` and ``rhs_final``
        on the result point at the same canonical reduction of the
        difference (the value is ``Zero`` on success); the per-side
        transcripts in ``lhs_steps`` / ``rhs_steps`` come from the
        difference's expansion. The split exists so the
        :class:`BianchiProofResult` shape stays uniform with the
        per-side approach.
        """
        lhs = self.first_bianchi_lhs(U, V, W)
        rhs = self.first_bianchi_rhs(U, V, W)
        diff = Sum.make(lhs, Neg(rhs))
        final, steps = self._expand_to_canonical(diff, max_steps=max_steps)
        ok = final == Zero or final == Integer(0)
        return BianchiProofResult(
            lhs_initial=lhs,
            rhs_initial=rhs,
            lhs_final=final,
            rhs_final=Zero,
            lhs_steps=steps,
            rhs_steps=(),
            ok=ok,
        )

    def prove_second_bianchi(
        self,
        U: Expr,
        V: Expr,
        W: Expr,
        Z: Expr,
        *,
        max_steps: int = 512,
    ) -> BianchiProofResult:
        r"""Expand both sides of the second Bianchi identity and compare.

        Identity: ``cycl (∇_U R)(V,W) Z = cycl R(U, T(V,W)) Z``.

        Same difference-and-canonicalize protocol as
        :meth:`prove_first_bianchi`.
        """
        lhs = self.second_bianchi_lhs(U, V, W, Z)
        rhs = self.second_bianchi_rhs(U, V, W, Z)
        diff = Sum.make(lhs, Neg(rhs))
        final, steps = self._expand_to_canonical(diff, max_steps=max_steps)
        ok = final == Zero or final == Integer(0)
        return BianchiProofResult(
            lhs_initial=lhs,
            rhs_initial=rhs,
            lhs_final=final,
            rhs_final=Zero,
            lhs_steps=steps,
            rhs_steps=(),
            ok=ok,
        )

    # ---- dunder --------------------------------------------------- #

    def __repr__(self) -> str:
        return f"BianchiProblem({self._conn._repr_inner()})"
