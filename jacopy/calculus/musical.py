"""
Musical isomorphisms ``♭`` and ``♯`` between ``TM`` and ``T^*M``.

A non-degenerate 2-form ``ω`` induces a bundle map
``ω^♭ : TM → T^*M`` by ``X ↦ ω(X, ·)``; a non-degenerate bivector
``π`` induces the dual map ``π^♯ : T^*M → TM`` by
``α ↦ π(α, ·)``. When ``ω`` and ``π`` arise from the same geometric
structure (a symplectic manifold, or a Poisson manifold whose
bivector is the inverse of a symplectic form), the two maps are
mutual inverses, this is the **musical compatibility** that bridges
the symplectic and derived definitions of the Hamiltonian vector
field.

The package represents the two maps as degree-0
:class:`~jacopy.algebra.derivation.Derivation` atoms parametric in
their generating form / bivector. They are applied via
:class:`~jacopy.algebra.derivation.Act` just like any other
operator; no graded Leibniz expansion is attached because the maps
are tensorial, not derivative.

Compatibility is surfaced in two shapes:

* :class:`MusicalCompatibility`, a frozen dataclass carrying the
  ``(ω, π)`` pair, the specific :class:`Flat` and :class:`Sharp`
  instances built from them, and a
  :meth:`~MusicalCompatibility.as_definition` factory returning the
  proof-engine rewrite rule.
* :class:`MusicalCompatibilityDefinition`, an
  :class:`~jacopy.proof.expansion.Definition` that rewrites
  ``ω^♭(π^♯(α)) → α`` whenever the nested Act targets the registered
  compatible pair.

Both shapes share their data so the user declares the compatibility
once and lets the proof layer consume it. The reverse composition
``π^♯(ω^♭(X)) → X`` is covered by the same definition, the axiom
``ω^♭ ∘ π^♯ = id`` and its partner ``π^♯ ∘ ω^♭ = id`` are logically
a single fact in the non-degenerate case.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Tuple

from jacopy.algebra.derivation import Act, Derivation
from jacopy.calculus.interior import InteriorProduct
from jacopy.core.expr import Expr, Neg
from jacopy.core.multi_eval import MultiEval
from jacopy.proof.expansion import Definition


# --------------------------------------------------------------------- #
# Flat ω♭: TM → T*M                                                      #
# --------------------------------------------------------------------- #


class Flat(Derivation):
    """``ω^♭``, the flat musical map of a 2-form ``ω``.

    A degree-0 derivation atom. ``Act(Flat(ω), X)`` represents the
    1-form ``ω(X, ·)``. Equality is structural over the generating
    form, matching the pattern used by
    :class:`~jacopy.calculus.interior.InteriorProduct`.
    """

    __slots__ = ("_form",)

    def __init__(self, omega: Expr, *, name: Optional[str] = None) -> None:
        if not isinstance(omega, Expr):
            raise TypeError("Flat requires an Expr 2-form")
        display = name if name is not None else f"{omega._repr_inner()}♭"
        super().__init__(display, degree=0)
        self._form = omega

    @property
    def form(self) -> Expr:
        return self._form

    def _key(self) -> Any:
        return (self._name, self._degree, self._form)


def flat(omega: Expr, *, name: Optional[str] = None) -> Flat:
    """Build the flat operator ``ω^♭``."""
    return Flat(omega, name=name)


# --------------------------------------------------------------------- #
# Sharp π♯: T*M → TM                                                     #
# --------------------------------------------------------------------- #


class Sharp(Derivation):
    """``π^♯``, the sharp musical map of a bivector ``π``.

    Degree-0 derivation atom mirror of :class:`Flat`. ``Act(Sharp(π),
    α)`` represents the vector field ``π(α, ·)``.
    """

    __slots__ = ("_bivector",)

    def __init__(self, pi: Expr, *, name: Optional[str] = None) -> None:
        if not isinstance(pi, Expr):
            raise TypeError("Sharp requires an Expr bivector")
        display = name if name is not None else f"{pi._repr_inner()}♯"
        super().__init__(display, degree=0)
        self._bivector = pi

    @property
    def bivector(self) -> Expr:
        return self._bivector

    def _key(self) -> Any:
        return (self._name, self._degree, self._bivector)


def sharp(pi: Expr, *, name: Optional[str] = None) -> Sharp:
    """Build the sharp operator ``π^♯``."""
    return Sharp(pi, name=name)


# --------------------------------------------------------------------- #
# Musical compatibility, the axiom ω♭ ∘ π♯ = id                        #
# --------------------------------------------------------------------- #


@dataclass(frozen=True)
class MusicalCompatibility:
    """The declaration that ``ω`` and ``π`` are musically inverse.

    Holding one of these is equivalent to holding the axiom
    ``ω^♭ ∘ π^♯ = id_{T^*M}`` (and its dual ``π^♯ ∘ ω^♭ = id_{TM}``).
    The object bundles the defining form, bivector, and the specific
    :class:`Flat` / :class:`Sharp` instances the axiom pairs together
    so callers don't build inconsistent operators by mistake.

    Hand this to :meth:`as_definition` to get the proof-engine rewrite
    rule, or to :meth:`~jacopy.calculus.hamiltonian_vf.HamiltonianVectorField.prove_equivalence`
    which consumes it directly.
    """

    omega: Expr
    pi: Expr
    flat: Flat
    sharp: Sharp
    name: str = "musical compatibility"

    @classmethod
    def between(
        cls,
        omega: Expr,
        pi: Expr,
        *,
        flat_instance: Optional[Flat] = None,
        sharp_instance: Optional[Sharp] = None,
        name: Optional[str] = None,
    ) -> "MusicalCompatibility":
        """Build a compatibility axiom from a form and a bivector.

        Defaults :attr:`flat` to ``Flat(omega)`` and :attr:`sharp` to
        ``Sharp(pi)``; pass custom instances when the caller needs
        bespoke naming. The optional ``name`` supplies the display
        string on the resulting axiom object.
        """
        fl = flat_instance if flat_instance is not None else Flat(omega)
        sh = sharp_instance if sharp_instance is not None else Sharp(pi)
        label = (
            name
            if name is not None
            else f"musical compatibility of {omega._repr_inner()}, {pi._repr_inner()}"
        )
        return cls(omega=omega, pi=pi, flat=fl, sharp=sh, name=label)

    def as_definition(self) -> "MusicalCompatibilityDefinition":
        """Return the proof-engine rule realising this axiom.

        Registering the returned definition on an
        :class:`~jacopy.proof.expansion.ExpansionEngine` makes the
        engine rewrite ``ω^♭(π^♯(α)) → α`` and ``π^♯(ω^♭(X)) → X``
        for exactly this ``(ω, π)`` pair.
        """
        return MusicalCompatibilityDefinition(self)

    def musical_definitions(self) -> Tuple[Definition, ...]:
        """Return the definitions that realise this compatibility as rewrites.

        Used by downstream proof helpers that need to register the
        axiom on an :class:`~jacopy.proof.expansion.ExpansionEngine`.
        The tuple bundles:

        1. :class:`IotaFlatDefinition`, for this ``ω``,
           ``ι_X ω → ω^♭(X)`` (shape identity on 2-forms).
        2. :class:`ArgNegLinearityDefinition`, linearity
           ``D(−x) → −D(x)`` for Derivation ``D``. Needed so the
           compatibility rule can see through a wrapping ``Neg`` left
           over from the Hamiltonian's derived sign convention.
        3. :class:`MusicalCompatibilityDefinition`,
           ``ω^♭ ∘ π^♯ = id`` (and ``π^♯ ∘ ω^♭ = id``).

        The Hamiltonian-specific rewrite ``X_f → −π^♯(df)`` is kept in
        :mod:`~jacopy.calculus.hamiltonian_vf` so this module remains
        independent of the Hamiltonian machinery.
        """
        return (
            IotaFlatDefinition(self),
            ArgNegLinearityDefinition(),
            MusicalCompatibilityDefinition(self),
            MusicalCompatibilityBilinearDefinition(self),
        )


# --------------------------------------------------------------------- #
# Proof-engine rule                                                      #
# --------------------------------------------------------------------- #


class MusicalCompatibilityDefinition(Definition):
    """``ω^♭(π^♯(α)) = α`` and ``π^♯(ω^♭(X)) = X``, the musical axiom.

    Registered on an :class:`~jacopy.proof.expansion.ExpansionEngine`
    this rule peels the nested composition whenever the outer and
    inner operators match the compatibility's declared
    :class:`Flat` / :class:`Sharp` pair. Other flat/sharp operators
    (a different form, a different bivector) are ignored, two
    distinct symplectic structures on the same manifold coexist
    without rewrites bleeding across them.
    """

    def __init__(self, compatibility: MusicalCompatibility) -> None:
        if not isinstance(compatibility, MusicalCompatibility):
            raise TypeError(
                "MusicalCompatibilityDefinition expects a MusicalCompatibility"
            )
        self._compat = compatibility
        self.name = compatibility.name

    @property
    def compatibility(self) -> MusicalCompatibility:
        return self._compat

    def matches(self, expr: Expr) -> bool:
        if not isinstance(expr, Act):
            return False
        outer, inner = expr.op, expr.arg
        if not isinstance(inner, Act):
            return False
        inner_op = inner.op
        if outer == self._compat.flat and inner_op == self._compat.sharp:
            return True
        if outer == self._compat.sharp and inner_op == self._compat.flat:
            return True
        return False

    def rewrite(self, expr: Expr) -> Expr:
        # matches() guarantees the shape Act(outer, Act(inner, payload));
        # the innermost argument is what survives.
        return expr.arg.arg  # type: ignore[union-attr]


# --------------------------------------------------------------------- #
# Iota-flat identity, ι_X ω = ω♭(X) on the compatibility's 2-form       #
# --------------------------------------------------------------------- #


class IotaFlatDefinition(Definition):
    """``ι_X ω = ω^♭(X)`` on the declared compatibility's ``ω``.

    Tied to a specific :class:`MusicalCompatibility` so the rewrite
    fires only when the 2-form in the ``Act`` argument is the one the
    user has declared compatible. ``ι`` on a 2-form contracts in the
    first slot, producing a 1-form, exactly the image of the flat
    map. Treating this as an engine rule (rather than a structural
    identity on :class:`InteriorProduct`) keeps the rewrite scoped:
    only proofs that have opted in via a compatibility axiom see it
    fire.
    """

    def __init__(self, compatibility: MusicalCompatibility) -> None:
        if not isinstance(compatibility, MusicalCompatibility):
            raise TypeError(
                "IotaFlatDefinition expects a MusicalCompatibility"
            )
        self._compat = compatibility
        self.name = f"ι_X ω = ω♭(X) [{compatibility.name}]"

    @property
    def compatibility(self) -> MusicalCompatibility:
        return self._compat

    def matches(self, expr: Expr) -> bool:
        return (
            isinstance(expr, Act)
            and isinstance(expr.op, InteriorProduct)
            and expr.arg == self._compat.omega
        )

    def rewrite(self, expr: Expr) -> Expr:
        X = expr.op.vector_field  # type: ignore[union-attr]
        return Act(self._compat.flat, X)


# --------------------------------------------------------------------- #
# Argument linearity, D(-x) = -D(x) on any Derivation D                 #
# --------------------------------------------------------------------- #


class ArgNegLinearityDefinition(Definition):
    """``D(−x) → −D(x)`` for any :class:`Derivation` ``D``.

    A universal linearity rewrite needed by the Hamiltonian bridge:
    after the derived-form rewrite turns ``X_f`` into ``−π^♯(df)``,
    the outer ``ω^♭`` gets wrapped around a ``Neg`` argument, and the
    musical compatibility rule cannot descend through that ``Neg``.
    Pulling the sign outward with this rule lets the compatibility
    fire on the clean ``ω^♭(π^♯(df))`` shape underneath. The rule is
    safe for any Derivation since Derivations are linear by
    construction in this package.
    """

    name = "D(-x) = -D(x)"

    def matches(self, expr: Expr) -> bool:
        return (
            isinstance(expr, Act)
            and isinstance(expr.op, Derivation)
            and isinstance(expr.arg, Neg)
        )

    def rewrite(self, expr: Expr) -> Expr:
        return Neg(Act(expr.op, expr.arg.arg))


# --------------------------------------------------------------------- #
# Musical bilinear, ω(π♯α, π♯β) = π(α, β)                                #
# --------------------------------------------------------------------- #


class MusicalCompatibilityBilinearDefinition(Definition):
    r"""``ω(π^♯α, π^♯β) → π(α, β)``, the bilinear face of musical compat.

    Tied to a specific :class:`MusicalCompatibility` so the rule fires
    only when the outer 2-form and both ``π^♯`` operators match the
    declared pair. The rewrite turns a form-on-vectors evaluation
    whose two args are both sharped covectors into the equivalent
    bivector-on-covectors evaluation, preserving both the
    ``alternating`` flag and the slot-kind invariant (``"vector"`` →
    ``"covector"``).

    Why arity 2 only: the identity ``ω(π^♯α, π^♯β) = π(α, β)`` is the
    rank-2 musical identity. Higher-rank generalisations exist (a
    metric musical-compatibility on rank-``p`` forms) but require
    extra structure that the current Cartan stack does not model. A
    user wanting the rank-``p`` case can write a problem-specific
    definition; this rule covers the symplectic / Poisson textbook
    use.
    """

    def __init__(self, compatibility: "MusicalCompatibility") -> None:
        if not isinstance(compatibility, MusicalCompatibility):
            raise TypeError(
                "MusicalCompatibilityBilinearDefinition expects a "
                "MusicalCompatibility"
            )
        self._compat = compatibility
        self.name = (
            f"ω(π♯α, π♯β) = π(α, β) [{compatibility.name}]"
        )

    @property
    def compatibility(self) -> "MusicalCompatibility":
        return self._compat

    def matches(self, expr: Expr) -> bool:
        if not isinstance(expr, MultiEval):
            return False
        if len(expr.args) != 2:
            return False
        if expr.head != self._compat.omega:
            return False
        if expr.slot_kind != "vector":
            return False
        return all(
            isinstance(a, Act) and a.op == self._compat.sharp
            for a in expr.args
        )

    def rewrite(self, expr: Expr) -> Expr:
        alpha = expr.args[0].arg  # type: ignore[union-attr]
        beta = expr.args[1].arg  # type: ignore[union-attr]
        return MultiEval(
            self._compat.pi,
            alpha,
            beta,
            alternating=expr.alternating,
            slot_kind="covector",
        )
