"""
Definition-based expansion.

A :class:`Definition` is a local rewrite rule keyed to a predicate on
an expression subtree: *if this shape appears, replace it with this*.
The :class:`ExpansionEngine` walks an expression bottom-up, fires
registered definitions at the first matching site, and records each
rewrite as a :class:`ProofStep` so the caller can reconstruct the
sequence of unfolds.

The engine is deliberately narrow: it only expands definitions. Koszul
signs, Leibniz distribution, and collecting like terms are *not* its
job, those live in :mod:`jacopy.algorithms` and the strategies layer
runs them afterward. Keeping the two concerns separate matches the
plan's "expand definitions first, then simplify" proof shape, and it
means a strategy can choose when to interleave or defer each pass.

Built-in definitions live in this module. More can be plugged in via
:meth:`ExpansionEngine.register`. Future fazlar will add
:class:`DerivedBracket` and operator-equation definitions here.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Optional, Tuple

from jacopy.algebra.derivation import Act, Derivation, compose, degree_of
from jacopy.core.expr import Expr, Integer, Neg, Product, Sum
from jacopy.core.registry import PropertyRegistry
from jacopy.core.symbolic_degree import Degree
from jacopy.proof.step import ProofStep

# Imported at module bottom to break a circular import: the three
# ``jacopy.calculus.*`` submodules below eagerly trigger
# ``jacopy.calculus.__init__``, which in turn pulls five calculus
# modules that import names from *this* module (``Definition``,
# ``ExpansionEngine``). Deferring these imports until after every
# class/function in this file is defined lets that reverse edge find
# the names it needs. ``from __future__ import annotations`` (above)
# keeps type annotations lazy, so the forward references in the
# function signatures below resolve at runtime without needing these
# imports in scope at definition time.


# --------------------------------------------------------------------- #
# Definition base class                                                  #
# --------------------------------------------------------------------- #


class Definition(ABC):
    """A single rewrite rule keyed to a local pattern.

    Subclasses provide :meth:`matches` (does ``expr`` fit the left-hand
    side?) and :meth:`rewrite` (produce the right-hand side). The
    :attr:`name` is used verbatim in the :class:`ProofStep` rule field.

    A definition is classified as an **axiom** by default, the engine
    tags its :class:`ProofStep` with ``provenance_tag="axiom"``. A
    subclass that represents a *theorem* overrides
    :meth:`theorem_proof_builder` to return a callable producing the
    sub-proof; in foundational mode the engine then attaches that
    sub-proof under the step as children. Efficient mode still tags the
    step ``"theorem"`` but skips sub-proof construction, so the same
    definition serves both ispat modes.
    """

    name: str = "definition"

    @abstractmethod
    def matches(self, expr: Expr) -> bool:
        """True when ``expr`` is an instance of this definition's LHS."""

    @abstractmethod
    def rewrite(self, expr: Expr) -> Expr:
        """Return the RHS. Caller guarantees ``matches(expr)`` is True."""

    def theorem_proof_builder(self):
        """Return a callable producing this definition's sub-proof, or ``None``.

        The returned callable has signature
        ``(matched_expr: Expr) -> ProofChain`` and is only invoked by the
        engine when ``mode="foundational"``. ``None`` (the default)
        marks the definition as a pure axiom.
        """
        return None

    @property
    def is_theorem(self) -> bool:
        """True when this definition carries a sub-proof builder."""
        return self.theorem_proof_builder() is not None


# --------------------------------------------------------------------- #
# Built-in definitions                                                   #
# --------------------------------------------------------------------- #


def _is_degree_zero(expr: Expr, registry: Optional[PropertyRegistry]) -> bool:
    """Safe degree-zero check: returns False on any undecidable case."""
    try:
        return degree_of(expr, registry) == Degree.const(0)
    except ValueError:
        return False


class ActOverSumOpDefinition(Definition):
    """``Act(A + B, x) → Act(A, x) + Act(B, x)``, linearity in the operator.

    ``product_rule`` distributes an operator across a sum in its
    operand (``D(a+b) → D(a) + D(b)``) but not across a sum in its
    operator position. The agreement-on-generators strategy routinely
    builds :class:`Act` nodes whose operator is a :class:`Sum` of
    compositions (the Cartan magic formula RHS is the canonical
    example), so this rewrite is what lets such proofs close via a
    structural cancellation.
    """

    name = "Act linearity: (A + B)(x) = A(x) + B(x)"

    def matches(self, expr: Expr) -> bool:
        return isinstance(expr, Act) and isinstance(expr.op, Sum)

    def rewrite(self, expr: Expr) -> Expr:
        ops = expr.op.children
        return Sum(*(Act(op, expr.arg) for op in ops))


#: The two classifications a :class:`DSquaredZeroDefinition` can carry.
D_SQUARED_CLASSIFICATIONS = ("axiom", "theorem")


class DSquaredZeroDefinition(Definition):
    """``d(d(x)) → 0``, element-level ``d² = 0``.

    Fires on nested ``Act(d, Act(d, x))`` shapes. A composed form
    ``Act(d ∘ d, x)`` is reduced by :mod:`product_rule` into the
    element-level shape first, so this single rule covers both after
    the strategy runs Leibniz expansion.

    ``target`` pins the rewrite to a specific :class:`ExteriorDerivative`
    instance, useful when a standard ``d`` coexists with a
    Lie-algebroid ``d_E`` and only one has the ``d² = 0`` axiom
    asserted. ``target=None`` (the default) accepts any
    :class:`ExteriorDerivative`, provided both layers use the *same*
    instance.

    ``classification`` selects how the rule is recorded in a proof
    transcript:

    * ``"axiom"`` (default), ``d² = 0`` is taken as a primitive axiom
      of the exterior calculus. The :class:`ProofStep` is tagged
      ``provenance_tag="axiom"``; no sub-proof is attached even under
      ``mode="foundational"``. This is the "efficient" presentation the
      plan refers to: fast, no structural justification required.
    * ``"theorem"``, ``d² = 0`` is presented as a derived operator
      identity. Its only primitive input is the generator-level axiom
      ``d(df) = 0`` on 0-forms; everything else extends by agreement on
      the generators of ``Ω*(M)`` and graded Leibniz. In efficient mode
      the step fires opaquely but is tagged ``"theorem"``; in
      foundational mode a sub-proof is attached that cites the
      generator-level axiom as the sole primitive.
    """

    name = "d² = 0"

    def __init__(
        self,
        target: Optional[ExteriorDerivative] = None,
        *,
        classification: str = "axiom",
    ) -> None:
        if classification not in D_SQUARED_CLASSIFICATIONS:
            raise ValueError(
                f"DSquaredZeroDefinition classification must be one of "
                f"{D_SQUARED_CLASSIFICATIONS}, got {classification!r}"
            )
        self._target = target
        self._classification = classification

    @property
    def classification(self) -> str:
        return self._classification

    def matches(self, expr: Expr) -> bool:
        if not (isinstance(expr, Act) and isinstance(expr.op, ExteriorDerivative)):
            return False
        inner = expr.arg
        if not (isinstance(inner, Act) and isinstance(inner.op, ExteriorDerivative)):
            return False
        if expr.op != inner.op:
            return False
        if self._target is not None and expr.op != self._target:
            return False
        return True

    def rewrite(self, expr: Expr) -> Expr:
        return Integer(0)

    def theorem_proof_builder(self):
        if self._classification != "theorem":
            return None

        def _build(matched: Expr) -> "ProofChain":
            # Minimal honest sub-proof: the operator identity d² = 0 is
            # derived from the generator-level axiom d(df) = 0 via
            # agreement on the generators of Ω*(M). A user who wants the
            # full structural unroll composes AgreementOnGenerators on
            # an ExteriorAlgebra, this builder cites the generator
            # axiom as the single primitive so the transcript bottoms
            # out at the foundational fact rather than at d² = 0 itself.
            from jacopy.proof.chain import ProofChain
            from jacopy.proof.step import ProofStep

            chain = ProofChain()
            chain.append(
                ProofStep(
                    matched,
                    Integer(0),
                    rule="d(df) = 0 on 0-forms (generator axiom)",
                    justification=(
                        "operator identity d ∘ d = 0 extends from the "
                        "generator-level axiom d(df) = 0 by agreement "
                        "on the generators of Ω*(M)"
                    ),
                    provenance_tag="axiom",
                )
            )
            return chain

        return _build


class IotaSquaredZeroDefinition(Definition):
    """``ι_X(ι_X(x)) → 0``, interior product applied twice to the same field."""

    name = "ι_X ∘ ι_X = 0"

    def matches(self, expr: Expr) -> bool:
        if not (isinstance(expr, Act) and isinstance(expr.op, InteriorProduct)):
            return False
        inner = expr.arg
        if not (isinstance(inner, Act) and isinstance(inner.op, InteriorProduct)):
            return False
        return expr.op == inner.op

    def rewrite(self, expr: Expr) -> Expr:
        return Integer(0)


class IotaOnZeroFormDefinition(Definition):
    """``ι_X(f) → 0`` when ``f`` is a declared 0-form.

    Uses :func:`degree_of` against the supplied registry; undecidable
    degrees leave the rewrite pending rather than silently assume
    zero. Passing ``registry=None`` effectively disables this rule on
    registry-resolved shapes.
    """

    name = "ι_X(f) = 0 on 0-forms"

    def __init__(self, *, registry: Optional[PropertyRegistry] = None) -> None:
        self._registry = registry

    def matches(self, expr: Expr) -> bool:
        if not (isinstance(expr, Act) and isinstance(expr.op, InteriorProduct)):
            return False
        return _is_degree_zero(expr.arg, self._registry)

    def rewrite(self, expr: Expr) -> Expr:
        return Integer(0)


def _is_derivation_combination(expr: Expr) -> bool:
    """True when ``expr`` is a :class:`Derivation` or a Sum/Product/Neg
    whose leaves are all Derivations.

    The pairing axiom ``ι_V(df) = V(f)`` is meaningful whenever ``V``
    represents a vector field built from Derivations, including
    composites like the Lie bracket ``X*Y − Y*X`` produced by
    ``vector_bracket.expand(X, Y)``. Restricting to strict
    ``Derivation`` instances would miss those composites and leave
    Cartan relations like ``[L_X, L_Y] = L_{[X,Y]}`` half-reduced.
    """
    if isinstance(expr, Derivation):
        return True
    if isinstance(expr, (Sum, Product, Neg)):
        return all(
            _is_derivation_combination(c) for c in expr.children
        )
    return False


class IotaOnExactOneFormDefinition(Definition):
    """``ι_X(df) → X(f)`` when the vector field ``X`` is a :class:`Derivation`
    or a composition of such.

    The rewrite fires on the shape ``Act(ι_X, Act(d, f))`` with ``f``
    resolving to degree 0 in the registry. ``d`` pins the rewrite to
    a specific :class:`ExteriorDerivative`, so a Lie-algebroid ``d_E``
    paired against the standard ``d`` doesn't accidentally pair. The
    vector field is accepted as a :class:`Derivation` or any
    ``Sum``/``Product``/``Neg`` composite whose leaves are all
    Derivations, this covers linear combinations produced by bracket
    expansion on vector fields. If the field contains a non-Derivation
    leaf the rule stays inert (the user hasn't given that field an
    action on functions yet).
    """

    name = "ι_X(df) = X(f)"

    def __init__(
        self,
        *,
        d: Optional[ExteriorDerivative] = None,
        registry: Optional[PropertyRegistry] = None,
    ) -> None:
        self._d = d if d is not None else default_d
        self._registry = registry

    def matches(self, expr: Expr) -> bool:
        if not (isinstance(expr, Act) and isinstance(expr.op, InteriorProduct)):
            return False
        iota: InteriorProduct = expr.op  # type: ignore[assignment]
        if not _is_derivation_combination(iota.vector_field):
            return False
        inner = expr.arg
        if not (isinstance(inner, Act) and isinstance(inner.op, ExteriorDerivative)):
            return False
        if inner.op != self._d:
            return False
        return _is_degree_zero(inner.arg, self._registry)

    def rewrite(self, expr: Expr) -> Expr:
        X = expr.op.vector_field  # type: ignore[union-attr]
        f = expr.arg.arg  # type: ignore[union-attr]
        return Act(X, f)


class LieDerivativeCartanDefinition(Definition):
    """``L_X(ω) := (d ∘ ι_X + ι_X ∘ d)(ω)``, Cartan definition of ``L_X``.

    Fires only on :class:`LieDerivative` instances whose
    :attr:`definition` is ``"cartan"``; an instance built in ``"flow"``
    mode is left alone so that Cartan's magic formula remains a theorem
    in that mode rather than a tautology.

    The rewrite returns the expansion already distributed as a
    :class:`Sum` of two :class:`Act` nodes, the composed operator form
    ``Act(d ∘ ι_X, ω) + Act(ι_X ∘ d, ω)``. Distributing here rather
    than producing ``Act(Sum(…), ω)`` keeps the result in the shape
    that :mod:`product_rule` and :mod:`simplify` can drive to a normal
    form without an additional Act-over-Sum pass.
    """

    name = "L_X := d∘ι_X + ι_X∘d (Cartan definition)"

    def matches(self, expr: Expr) -> bool:
        return (
            isinstance(expr, Act)
            and isinstance(expr.op, LieDerivative)
            and expr.op.definition == "cartan"
        )

    def rewrite(self, expr: Expr) -> Expr:
        L: LieDerivative = expr.op  # type: ignore[assignment]
        arg = expr.arg
        # Honour the bundle-specific ``d``/``iota_factory`` slots on
        # ``LieDerivative``: algebroid-constructed ``L_{E,X}`` carries
        # its own ``d_E`` and ``ι_{E,·}`` factory, and without this
        # routing the expansion would reintroduce the TM default ``d``
        # and ``ι_X``, leaving the magic-formula residual wedged in
        # mismatched operator names. Fallback is still the TM default,
        # so existing callers and the ``TM`` Cartan bundle behave as
        # before.
        dop = L.d if L.d is not None else default_d
        iota_X = L.iota_factory(L.vector_field) if L.iota_factory is not None else interior(L.vector_field)
        return Sum(
            Act(compose(dop, iota_X), arg),
            Act(compose(iota_X, dop), arg),
        )


class LieDerivativeOnZeroFormDefinition(Definition):
    """``L_X(f) → X(f)`` on 0-forms for flow-mode ``L_X``.

    Fires only on :class:`LieDerivative` instances whose
    :attr:`definition` is ``"flow"``, cartan-mode ``L_X`` unfolds via
    :class:`LieDerivativeCartanDefinition` and reaches the same result
    through the magic formula. The vector field must be a Derivation or
    a Sum/Product/Neg composite of Derivations (the pairing gate used
    by :class:`IotaOnExactOneFormDefinition`); otherwise ``X(f)``
    wouldn't be meaningful.
    """

    name = "L_X(f) = X(f) on 0-forms (flow)"

    def __init__(self, *, registry: Optional[PropertyRegistry] = None) -> None:
        self._registry = registry

    def matches(self, expr: Expr) -> bool:
        if not (isinstance(expr, Act) and isinstance(expr.op, LieDerivative)):
            return False
        L: LieDerivative = expr.op  # type: ignore[assignment]
        if L.definition != "flow":
            return False
        if not _is_derivation_combination(L.vector_field):
            return False
        return _is_degree_zero(expr.arg, self._registry)

    def rewrite(self, expr: Expr) -> Expr:
        X = expr.op.vector_field  # type: ignore[union-attr]
        return Act(X, expr.arg)


class LieDerivativeCommutesWithDDefinition(Definition):
    """``L_X(d(ω)) → d(L_X(ω))`` for flow-mode ``L_X``.

    The Cartan-mode ``L_X`` picks up the same identity through the
    magic-formula expansion; the flow-mode presentation keeps ``L_X``
    opaque, so this rule is what lets ``[d, L_X] = 0`` close as a
    rewrite cascade. If ``L_X`` carries a bundle-specific ``d`` slot
    the rewrite only fires on that exterior derivative, avoiding
    accidental pairing with a TM ``d`` that happens to coexist in the
    same expression tree.
    """

    name = "L_X ∘ d = d ∘ L_X (flow)"

    def matches(self, expr: Expr) -> bool:
        if not (isinstance(expr, Act) and isinstance(expr.op, LieDerivative)):
            return False
        L: LieDerivative = expr.op  # type: ignore[assignment]
        if L.definition != "flow":
            return False
        inner = expr.arg
        if not (isinstance(inner, Act) and isinstance(inner.op, ExteriorDerivative)):
            return False
        if L.d is not None and inner.op != L.d:
            return False
        return True

    def rewrite(self, expr: Expr) -> Expr:
        L = expr.op
        d_inner = expr.arg.op  # type: ignore[union-attr]
        omega = expr.arg.arg  # type: ignore[union-attr]
        return Act(d_inner, Act(L, omega))


# --------------------------------------------------------------------- #
# Engine                                                                 #
# --------------------------------------------------------------------- #


#: The two ispat modes recognised by :class:`ExpansionEngine`.
MODES = ("efficient", "foundational")


class ExpansionEngine:
    """Apply registered :class:`Definition` rules bottom-up to fix-point.

    On each iteration :meth:`expand_once` finds the leftmost-innermost
    matching site and rewrites it, recording a :class:`ProofStep`.
    :meth:`expand` loops until no definition fires. A ``max_steps``
    guard protects against a misbehaving rule that produces an
    ever-growing expression.

    The engine's :attr:`mode` controls how theorem-classified
    definitions are recorded:

    * ``"efficient"``, theorems fire like axioms; the resulting step
      is tagged ``"theorem"`` but carries no sub-proof, matching the
      "property taken as given" mode of the plan.
    * ``"foundational"``, theorems fire and their
      :meth:`Definition.theorem_proof_builder` is invoked; the
      resulting sub-proof is attached under the step as children,
      exposing the derivation down to axioms.
    """

    __slots__ = ("_definitions", "_mode")

    def __init__(
        self,
        definitions: Optional[List[Definition]] = None,
        *,
        mode: str = "efficient",
    ) -> None:
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
        self._definitions: List[Definition] = list(definitions) if definitions else []
        self._mode = mode

    def register(self, definition: Definition) -> None:
        """Append ``definition`` to the rule list."""
        if not isinstance(definition, Definition):
            raise TypeError("ExpansionEngine.register expects a Definition")
        self._definitions.append(definition)

    @property
    def definitions(self) -> Tuple[Definition, ...]:
        return tuple(self._definitions)

    @property
    def mode(self) -> str:
        return self._mode

    def with_mode(self, mode: str) -> "ExpansionEngine":
        """Return a new engine with the same definitions but a different mode."""
        return ExpansionEngine(list(self._definitions), mode=mode)

    # -- single-step ----------------------------------------------------- #

    def _match(self, expr: Expr) -> Optional[Definition]:
        for d in self._definitions:
            if d.matches(expr):
                return d
        return None

    def _build_step(self, expr: Expr, after: Expr, d: Definition) -> ProofStep:
        """Create the :class:`ProofStep` for a fired rewrite, with mode-dependent children."""
        tag = "theorem" if d.is_theorem else "axiom"
        step = ProofStep(
            expr,
            after,
            rule=d.name,
            justification=f"apply {tag}: {d.name}",
            provenance_tag=tag,
        )
        if self._mode == "foundational" and d.is_theorem:
            builder = d.theorem_proof_builder()
            if builder is not None:
                sub_chain = builder(expr)
                for sub_step in sub_chain:
                    step.add_child(sub_step)
        return step

    def expand_once(self, expr: Expr) -> Tuple[Expr, Optional[ProofStep]]:
        """Apply a single rewrite at the leftmost-innermost matching site.

        Children are visited first, so the rule that fires is the one
        closest to the leaves. Returns the (possibly unchanged)
        expression and the :class:`ProofStep` produced by the fired
        rule, or ``None`` when no definition matches anywhere.
        """
        if not expr.is_atom:
            children = list(expr.children)
            for i, c in enumerate(children):
                new_c, step = self.expand_once(c)
                if step is not None:
                    children[i] = new_c
                    return expr._rebuild(tuple(children)), step
        d = self._match(expr)
        if d is not None:
            after = d.rewrite(expr)
            return after, self._build_step(expr, after, d)
        return expr, None

    # -- driver ---------------------------------------------------------- #

    def expand(
        self, expr: Expr, *, max_steps: int = 256
    ) -> Tuple[Expr, List[ProofStep]]:
        """Apply definitions repeatedly until no rule fires.

        Returns the fully expanded expression and the list of steps
        taken. Raises :class:`RuntimeError` if the fix-point isn't
        reached within ``max_steps`` iterations, that would indicate a
        cyclic or divergent rule set.
        """
        steps: List[ProofStep] = []
        current = expr
        for _ in range(max_steps):
            nxt, step = self.expand_once(current)
            if step is None:
                return current, steps
            steps.append(step)
            current = nxt
        raise RuntimeError(
            f"ExpansionEngine did not converge within {max_steps} steps; "
            "check for cyclic definitions"
        )


def default_engine(
    *,
    registry: Optional[PropertyRegistry] = None,
    d: Optional[ExteriorDerivative] = None,
    mode: str = "efficient",
    d_squared_mode: str = "axiom",
) -> ExpansionEngine:
    """Engine with the standard built-in definitions registered.

    The default rule set covers the Cartan-calculus axioms most proofs
    need: the Cartan definition of ``L_X``, linearity of :class:`Act`
    in its operator, ``d² = 0``, ``ι_X² = 0``, ``ι_X(f) = 0`` on
    0-forms, and the pairing ``ι_X(df) = X(f)``. The registry-aware
    rules stay inert when the relevant grading isn't declared, so
    passing ``registry=None`` is safe, just less capable.

    Pass a specific :class:`ExteriorDerivative` via ``d`` to pin the
    exact-1-form pairing to a non-default exterior derivative.
    ``mode`` selects efficient vs foundational ispat (see
    :class:`ExpansionEngine`).

    ``d_squared_mode`` picks how ``d² = 0`` is classified in the
    transcript, ``"axiom"`` (default) treats it as primitive;
    ``"theorem"`` presents it as a derived operator identity whose only
    foundational input is the generator-level axiom ``d(df) = 0``. The
    rewrite semantics are identical; only the :class:`ProofStep`
    classification and the foundational-mode sub-proof differ.
    """
    dop = d if d is not None else default_d
    return ExpansionEngine(
        [
            LieDerivativeCartanDefinition(),
            LieDerivativeOnZeroFormDefinition(registry=registry),
            LieDerivativeCommutesWithDDefinition(),
            ActOverSumOpDefinition(),
            DSquaredZeroDefinition(target=dop, classification=d_squared_mode),
            IotaSquaredZeroDefinition(),
            IotaOnZeroFormDefinition(registry=registry),
            IotaOnExactOneFormDefinition(d=dop, registry=registry),
        ],
        mode=mode,
    )


# See the header comment above the top-of-module imports: these three
# pulls happen after every class/function is defined so the circular
# import from ``jacopy.calculus.*`` back into this module can resolve.
from jacopy.calculus.exterior_d import ExteriorDerivative, d as default_d  # noqa: E402
from jacopy.calculus.interior import InteriorProduct, interior  # noqa: E402
from jacopy.calculus.lie_derivative import LieDerivative  # noqa: E402
