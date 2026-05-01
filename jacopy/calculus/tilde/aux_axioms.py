r"""
Tilde-calculus auxiliary axioms, Faz 14.D.

Five engine rewrite rules that close the gap between the three
defining-axiom rules in :mod:`jacopy.calculus.tilde.axioms` and the
six tilde Cartan relations on a 0-vector ``f`` and a 1-vector ``X``.
Each rule mirrors a standard Cartan-side identity:

* :class:`TildeIotaOnZeroVectorDefinition`, ``ι̃_ω f → 0`` for ``f``
  registered as ``Graded(degree=0)``. Splits the 0-vector corner case
  out of the unconditional :class:`TildeIotaSwapDefinition` so the
  engine transcript shows it as a labelled step instead of a generic
  swap-then-cancel chain.

* :class:`TildeIotaSquaredZeroDefinition`, ``ι̃_ω(ι̃_ω V) → 0``. The
  ``η = ω`` special case of the anti-commute relation
  ``ι̃_ω ι̃_η + ι̃_η ι̃_ω = 0`` (relation 1 in §3.1.3); follows from the
  fact that ``ι̃_ω`` is a degree-(-1) derivation. Rule fires only when
  the two interior heads carry the *same* form.

* :class:`TildeLieOnZeroVectorDefinition`, ``L̃_ω f → π^♯(ω)(f)``.
  Mathematically ``L̃_ω f = X_ω(f)`` where ``X_ω = π^♯(ω)`` is the
  anchor of ``ω``; expressed as a nested ``Act(Act(Sharp(π), ω), f)``
  so existing Sharp / pairing rules pick the result up downstream.

* :class:`TildeDOfFunctionDefinition`, ``d̃ f → −π^♯(df)`` for
  ``f`` degree 0. The Lichnerowicz definition ``[π, f]_SN`` collapses
  via SN base cases to ``−X_f``; this rule shortcuts the multi-step
  derivation to a single named step.

* :class:`TildeDSquaredPoissonDefinition`, ``d̃² V → 0`` when ``π``
  carries the :class:`~jacopy.core.properties.Poisson` flag. Consumes
  the declarative bit set by
  :meth:`~jacopy.library.koszul_problem.KoszulProblem.assume_poisson`.

All five are :class:`Definition` subclasses ready to register on an
:class:`~jacopy.proof.expansion.ExpansionEngine`. Aux-1 must be
registered *before* :class:`TildeIotaSwapDefinition` so the 0-vector
short-circuit fires on the pre-swap shape; the other four are
position-independent.
"""

from __future__ import annotations

from typing import Optional

from jacopy.algebra.derivation import Act, degree_of
from jacopy.calculus.exterior_d import ExteriorDerivative, d as default_d
from jacopy.calculus.musical import Sharp
from jacopy.calculus.tilde.operators import (
    TildeExteriorDerivative,
    TildeInteriorProduct,
    TildeLieDerivative,
)
from jacopy.core.expr import Expr, Neg, Zero
from jacopy.core.multi_eval import MultiEval
from jacopy.core.properties import Graded, Poisson
from jacopy.core.registry import PropertyRegistry
from jacopy.core.symbolic_degree import Degree
from jacopy.proof.expansion import Definition


def _is_degree_zero(
    expr: Expr, registry: Optional[PropertyRegistry]
) -> bool:
    try:
        return degree_of(expr, registry) == Degree.const(0)
    except ValueError:
        return False


# --------------------------------------------------------------------- #
# Aux-1, ι̃_ω f → 0   (f degree 0)                                      #
# --------------------------------------------------------------------- #


class TildeIotaOnZeroVectorDefinition(Definition):
    r"""``ι̃_ω f → 0``, tilde interior product annihilates 0-vectors.

    Fires on ``Act(TildeInteriorProduct(ω), f)`` whenever ``f`` resolves
    to degree 0 in the registry. The standard interior product ``ι_f ω``
    of a function ``f`` against a form ``ω`` is identically zero, and
    ``ι̃_ω f := ι_f ω``, so this is the registry-aware tilde shortcut.

    Registering this rule *before* :class:`TildeIotaSwapDefinition` on the
    engine ensures the 0-vector short-circuit fires on the pre-swap
    shape ``Act(ι̃_ω, f)`` and produces a labelled "ι̃_ω f = 0" step in
    the transcript instead of the unconditional swap to ``ι_f ω``.
    """

    name = "ι̃_ω f = 0  (f deg 0)"

    def __init__(self, *, registry: Optional[PropertyRegistry] = None) -> None:
        self._registry = registry

    def matches(self, expr: Expr) -> bool:
        return (
            isinstance(expr, Act)
            and isinstance(expr.op, TildeInteriorProduct)
            and _is_degree_zero(expr.arg, self._registry)
        )

    def rewrite(self, expr: Expr) -> Expr:
        return Zero


# --------------------------------------------------------------------- #
# Aux-2, ι̃_ω(ι̃_ω V) → 0                                                #
# --------------------------------------------------------------------- #


class TildeIotaSquaredZeroDefinition(Definition):
    r"""``ι̃_ω(ι̃_ω V) → 0``, squared tilde interior product vanishes.

    The ``η = ω`` special case of the anti-commute relation
    ``ι̃_ω ι̃_η + ι̃_η ι̃_ω = 0``; follows from ``ι̃_ω`` being a graded
    derivation of odd degree. Fires only when the outer and inner
    interior heads carry the *same* form (structural equality on the
    stored ``form`` attribute).

    Distinct-form pairs ``ι̃_ω(ι̃_η V)`` are handled by the broader
    anti-commute axiom (closes via the swap rule + standard
    ``ι_X ι_Y ω`` antisymmetry); this rule only owns the strict-square
    shortcut.
    """

    name = "ι̃_ω² V = 0"

    def matches(self, expr: Expr) -> bool:
        if not (isinstance(expr, Act) and isinstance(expr.op, TildeInteriorProduct)):
            return False
        inner = expr.arg
        if not (isinstance(inner, Act) and isinstance(inner.op, TildeInteriorProduct)):
            return False
        return expr.op == inner.op

    def rewrite(self, expr: Expr) -> Expr:
        return Zero


# --------------------------------------------------------------------- #
# Aux-3, L̃_ω f → π^♯(ω)(f)   (f degree 0)                              #
# --------------------------------------------------------------------- #


class TildeLieOnZeroVectorDefinition(Definition):
    r"""``L̃_ω f → (π^♯(ω))(f)``, tilde Lie derivative of a function.

    Scoped to a Poisson bivector ``π``: matches only when the outer
    head is a :class:`TildeLieDerivative` whose
    :attr:`~jacopy.calculus.tilde.operators.TildeLieDerivative.bivector`
    equals ``π``, and the operand is degree 0.

    The rewrite emits ``Act(Act(Sharp(π), ω), f)``, i.e. the anchor
    ``π^♯(ω)`` (a vector field) acting on ``f`` (a function). Existing
    Sharp axioms (Faz 13.A) and ordinary derivation rules then take
    over; downstream proofs do not need to know this short-circuit
    bypassed the Cartan-magic + Aux-1 + Aux-4 chain.

    Mathematically: ``L̃_ω f = d̃(ι̃_ω f) + ι̃_ω(d̃ f) = 0 + ι̃_ω(−X_f)
    = −ι_{X_f} ω = ω(X_f) = (π^♯(ω))(f)``. The intermediate identity
    ``ω(π^♯(df)) = (π^♯(ω))(df) = (π^♯(ω))(f)`` uses ``π``'s
    antisymmetry; that step is owned by
    :class:`~jacopy.calculus.antisym_axioms.RegistryAntiSymCanonicalDefinition`,
    not by this rule.
    """

    def __init__(
        self,
        pi: Expr,
        *,
        registry: Optional[PropertyRegistry] = None,
    ) -> None:
        if not isinstance(pi, Expr):
            raise TypeError(
                "TildeLieOnZeroVectorDefinition pi must be an Expr"
            )
        self._pi = pi
        self._sharp = Sharp(pi)
        self._registry = registry
        self.name = f"L̃_ω f = π^♯(ω)(f) [{pi._repr_inner()}]"

    @property
    def pi(self) -> Expr:
        return self._pi

    def matches(self, expr: Expr) -> bool:
        return (
            isinstance(expr, Act)
            and isinstance(expr.op, TildeLieDerivative)
            and expr.op.bivector == self._pi
            and _is_degree_zero(expr.arg, self._registry)
        )

    def rewrite(self, expr: Expr) -> Expr:
        head = expr.op
        assert isinstance(head, TildeLieDerivative)
        omega = head.form
        f = expr.arg
        return Act(Act(self._sharp, omega), f)


# --------------------------------------------------------------------- #
# Aux-4, d̃ f → −π^♯(df)   (f degree 0)                                 #
# --------------------------------------------------------------------- #


class TildeDOfFunctionDefinition(Definition):
    r"""``d̃ f → −π^♯(df)``, Lichnerowicz of a function shortcut.

    Scoped to a Poisson bivector ``π`` (and an exterior derivative
    ``d``): matches only when the outer head is a
    :class:`TildeExteriorDerivative` whose
    :attr:`~jacopy.calculus.tilde.operators.TildeExteriorDerivative.bivector`
    equals ``π``, and the operand is degree 0.

    The Lichnerowicz definition ``d̃ f = [π, f]_SN`` collapses via the
    SN base case ``[π, f]_SN = −π^♯(df) = −X_f`` after a wedge-Leibniz
    expansion and a Sharp/exterior-d rewrite; this rule installs the
    closed-form result as a single named step, cutting roughly five
    SN engine steps.

    ``d`` defaults to the :mod:`exterior_d` module singleton; pass an
    explicit instance when a Lie-algebroid ``d_E`` coexists with the
    standard ``d``.
    """

    def __init__(
        self,
        pi: Expr,
        *,
        d: Optional[ExteriorDerivative] = None,
        registry: Optional[PropertyRegistry] = None,
    ) -> None:
        if not isinstance(pi, Expr):
            raise TypeError(
                "TildeDOfFunctionDefinition pi must be an Expr"
            )
        self._pi = pi
        self._head = TildeExteriorDerivative(pi)
        self._sharp = Sharp(pi)
        self._d = default_d if d is None else d
        self._registry = registry
        self.name = f"d̃ f = −π^♯(df) [{pi._repr_inner()}]"

    @property
    def pi(self) -> Expr:
        return self._pi

    @property
    def d(self) -> ExteriorDerivative:
        return self._d

    def matches(self, expr: Expr) -> bool:
        return (
            isinstance(expr, Act)
            and isinstance(expr.op, TildeExteriorDerivative)
            and expr.op == self._head
            and _is_degree_zero(expr.arg, self._registry)
        )

    def rewrite(self, expr: Expr) -> Expr:
        f = expr.arg
        return Neg(Act(self._sharp, Act(self._d, f)))


# --------------------------------------------------------------------- #
# Aux-5, d̃² V → 0   (π Poisson)                                         #
# --------------------------------------------------------------------- #


class TildeDSquaredPoissonDefinition(Definition):
    r"""``d̃² V → 0``, tilde-d squared vanishes when ``π`` is Poisson.

    Scoped to a Poisson bivector ``π``: matches ``Act(d̃_π, Act(d̃_π, V))``
    only when the registry marks ``π`` with
    :class:`~jacopy.core.properties.Poisson`. The flag is set by
    :meth:`~jacopy.library.koszul_problem.KoszulProblem.assume_poisson`
    or by a direct ``registry.declare(π, Poisson())`` call.

    Equivalent to the Jacobi identity ``[π, π]_SN = 0`` collapsed to a
    single rewrite: without the flag the rule is a strict no-op, so a
    proof script that omits ``assume_poisson()`` will see the engine
    *not* close ``d̃²``-shaped obstructions, exactly the behaviour
    desired when checking whether a candidate ``π`` is Poisson at all.
    """

    def __init__(
        self,
        pi: Expr,
        *,
        registry: PropertyRegistry,
    ) -> None:
        if not isinstance(pi, Expr):
            raise TypeError(
                "TildeDSquaredPoissonDefinition pi must be an Expr"
            )
        if not isinstance(registry, PropertyRegistry):
            raise TypeError(
                "TildeDSquaredPoissonDefinition requires a PropertyRegistry"
            )
        self._pi = pi
        self._head = TildeExteriorDerivative(pi)
        self._registry = registry
        self.name = f"d̃² V = 0  (π Poisson) [{pi._repr_inner()}]"

    @property
    def pi(self) -> Expr:
        return self._pi

    def matches(self, expr: Expr) -> bool:
        if not self._registry.has(self._pi, Poisson):
            return False
        if not (isinstance(expr, Act) and isinstance(expr.op, TildeExteriorDerivative)):
            return False
        if expr.op != self._head:
            return False
        inner = expr.arg
        if not (isinstance(inner, Act) and isinstance(inner.op, TildeExteriorDerivative)):
            return False
        return inner.op == self._head

    def rewrite(self, expr: Expr) -> Expr:
        return Zero


# --------------------------------------------------------------------- #
# Aux-6, Act(D, Act(ι̃_ω, V)) → Act(D, MultiEval(V, ω))   (V deg 1)      #
# --------------------------------------------------------------------- #


class TildeIotaActAsScalarDefinition(Definition):
    r"""Bridge bare ``Act(ι̃_ω, V)`` inside ``Act(D, _)`` into a MultiEval.

    Fires on ``Act(D, Act(ι̃_ω, V))`` when ``V`` is registered
    :class:`~jacopy.core.properties.Graded` with ``degree=1``. The outer
    ``Act(D, _)`` syntactically asserts its argument is a 0-form scalar,
    which is consistent with ``ι̃_ω(V)`` for a 1-vector ``V``: the result
    is the function ``V(ω)``. The rewrite

    .. math::

        D\bigl(\tilde{\iota}_\omega V\bigr) \;\longrightarrow\;
            D\bigl(V(\omega)\bigr)

    encodes that scalar reduction so subsequent MultiEval-linearity /
    Sharp-axiom rules can collapse the residue further.

    Why this rule is needed: :class:`TildeIotaIntrinsicDefinition` only
    fires when ``Act(ι̃_ω, V)`` sits as the head of an enclosing
    :class:`MultiEval`. The arity-1 branch of
    :class:`~jacopy.calculus.tilde.intrinsic_axioms.TildeDIntrinsicDefinition`
    (``(d̃ f)(η) = π^♯(η)·f``) produces residues like
    ``Act(Act(Sharp(π), η), Act(ι̃_ω, V))`` where the inner
    ``Act(ι̃_ω, V)`` is bare, no enclosing MultiEval, so the iota
    intrinsic can't reach it. Without this bridge the tilde Cartan-magic
    relation ``L̃_ω V = (d̃ ι̃_ω + ι̃_ω d̃) V``, evaluated at a single
    1-form, leaves a residue ``π^♯(η)(ι̃_ω V) − π^♯(η)(V(ω))``.

    The ``Graded(degree=1)`` guard is essential: for a higher-degree
    ``V`` the contraction ``ι̃_ω V`` is not a scalar, and replacing
    ``Act(ι̃_ω, V)`` with the arity-1 ``MultiEval(V, ω)`` would produce
    a malformed evaluation (slot count would not match V's degree).
    The rule unconditionally accepts any outer ``D``: the registry
    guard on V already makes the rewrite type-safe.
    """

    name = "bare ι̃_ω(V) inside Act(D, _) → Act(D, MultiEval(V, ω))  (V deg 1)"

    def __init__(self, *, registry: PropertyRegistry) -> None:
        if not isinstance(registry, PropertyRegistry):
            raise TypeError(
                "TildeIotaActAsScalarDefinition requires a PropertyRegistry"
            )
        self._registry = registry

    def matches(self, expr: Expr) -> bool:
        if not isinstance(expr, Act):
            return False
        inner = expr.arg
        if not (
            isinstance(inner, Act)
            and isinstance(inner.op, TildeInteriorProduct)
        ):
            return False
        V = inner.arg
        graded = self._registry.get(V, Graded)
        if graded is None:
            return False
        return graded.degree == Degree.const(1)

    def rewrite(self, expr: Expr) -> Expr:
        outer_op = expr.op
        inner_act = expr.arg
        iota = inner_act.op
        V = inner_act.arg
        omega = iota.form
        return Act(
            outer_op,
            MultiEval(V, omega, alternating=True, slot_kind="covector"),
        )
