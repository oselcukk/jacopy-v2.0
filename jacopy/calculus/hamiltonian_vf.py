"""
Hamiltonian vector field ``X_f``.

Two standard definitions:

* **Symplectic**, ``X_f`` is the unique vector field with
  ``ι_{X_f} ω = −df`` for a non-degenerate closed 2-form ``ω``.
* **Derived**, ``X_f := −[f, π]_SN`` for a Poisson bivector ``π``.

The two agree whenever ``ω`` and ``π`` are *compatible* (musical
inverses): non-degeneracy of ``ω`` gives a bundle isomorphism
``T^*M → TM`` whose inverse is contraction against ``π``. The package
does not yet carry an Expr-level encoding of that musical isomorphism,
so the equivalence of the two definitions surfaces here as a
:class:`VanishingCondition`: the obstruction to compatibility is
``ι_{X_f} ω + df``, zero iff the symplectic definition matches the
derived one on this ``f``.

:class:`HamiltonianVectorField` is a :class:`Derivation` of degree 0.
It can be created from a bivector, from a symplectic form, or from
both, the extra field only adds access to the relevant methods
(:meth:`~HamiltonianVectorField.derived_expansion` needs ``π``;
:meth:`~HamiltonianVectorField.symplectic_obstruction` needs ``ω``).
Downstream code treats the operator like any other
:class:`Derivation`: ``Act(X_f, g)`` is ``X_f(g)``, Leibniz applies
through :mod:`product_rule`, and ``{f, g}_π = X_f(g)`` closes in the
derived-bracket machinery once Stage 1 SN is in place.
"""

from __future__ import annotations

from typing import Any, Optional

from jacopy.algebra.derivation import Act, Derivation
from jacopy.brackets.derived import VanishingCondition
from jacopy.brackets.schouten import sn
from jacopy.calculus.exterior_d import ExteriorDerivative, d as default_d
from jacopy.calculus.interior import InteriorProduct, interior as default_interior
from jacopy.calculus.musical import MusicalCompatibility
from jacopy.core.expr import Expr, Neg, Sum
from jacopy.core.registry import PropertyRegistry
from jacopy.proof.chain import ProofChain
from jacopy.proof.expansion import Definition, ExpansionEngine
from jacopy.proof.strategies import ExpandAndSimplify


_VALID_SIGNS = ("+", "-")


# --------------------------------------------------------------------- #
# HamiltonianVectorField                                                 #
# --------------------------------------------------------------------- #


class HamiltonianVectorField(Derivation):
    """``X_f``, the Hamiltonian vector field of a function ``f``.

    A degree-0 :class:`Derivation` named ``X_{f}`` by default. Carries
    the defining function and, optionally, either (or both) of:

    * ``bivector``, a Poisson bivector ``π`` used by the derived
      definition ``X_f = −[f, π]_SN``.
    * ``symplectic_form``, a symplectic 2-form ``ω`` used by the
      symplectic definition ``ι_{X_f} ω = −df``.

    At least one of the two must be supplied; without it a
    ``HamiltonianVectorField`` has no meaning distinct from an arbitrary
    degree-0 derivation and creation is rejected.
    """

    __slots__ = ("_function", "_bivector", "_symplectic_form", "_sign")

    def __init__(
        self,
        f: Expr,
        *,
        bivector: Optional[Expr] = None,
        symplectic_form: Optional[Expr] = None,
        name: Optional[str] = None,
        sign: str = "-",
    ) -> None:
        if not isinstance(f, Expr):
            raise TypeError("HamiltonianVectorField function must be an Expr")
        if bivector is None and symplectic_form is None:
            raise ValueError(
                "HamiltonianVectorField requires at least one of "
                "'bivector' or 'symplectic_form'"
            )
        if bivector is not None and not isinstance(bivector, Expr):
            raise TypeError("bivector must be an Expr when provided")
        if symplectic_form is not None and not isinstance(symplectic_form, Expr):
            raise TypeError("symplectic_form must be an Expr when provided")
        if sign not in _VALID_SIGNS:
            raise ValueError(
                f"HamiltonianVectorField sign must be '+' or '-', got {sign!r}"
            )
        display = name if name is not None else f"X_{f._repr_inner()}"
        super().__init__(display, degree=0)
        self._function = f
        self._bivector = bivector
        self._symplectic_form = symplectic_form
        self._sign = sign

    # ---- accessors -------------------------------------------------- #

    @property
    def function(self) -> Expr:
        return self._function

    @property
    def bivector(self) -> Optional[Expr]:
        return self._bivector

    @property
    def symplectic_form(self) -> Optional[Expr]:
        return self._symplectic_form

    @property
    def sign(self) -> str:
        """Sign convention for ``ι_{X_f} ω = sign·df``, ``'+'`` or ``'-'``.

        Defaults to ``'-'`` (geometer's convention ``ι_{X_f} ω = -df``).
        Notebook problems following the textbook ``ι_{X_f} ω = +df``
        convention should pass ``sign='+'`` at construction time.
        """
        return self._sign

    # ---- derived definition ---------------------------------------- #

    def derived_expansion(
        self, registry: Optional[PropertyRegistry] = None
    ) -> Expr:
        """Return ``X_f = −[f, π]_SN`` as an :class:`Expr`.

        Dispatches to the :mod:`schouten` SN singleton's expansion.
        For atomic ``π`` (a bare :class:`Symbol` declared
        ``Graded(degree=1)``) the inner bracket stays opaque, which is
        the right shape for the derived-bracket machinery, the
        caller's proof layer consumes the outer ``Neg`` directly.

        Raises :class:`ValueError` when no bivector is attached.
        """
        if self._bivector is None:
            raise ValueError(
                "derived_expansion requires a bivector; "
                "construct HamiltonianVectorField with bivector=..."
            )
        return Neg(sn.expand(self._function, self._bivector, registry))

    # ---- symplectic definition ------------------------------------- #

    def symplectic_obstruction(
        self,
        *,
        d: Optional[ExteriorDerivative] = None,
        interior: Optional[Any] = None,
    ) -> Expr:
        """Return the obstruction ``ι_{X_f} ω − sign·df``.

        Zero iff the symplectic definition ``ι_{X_f} ω = sign·df`` holds
        on this ``X_f`` (sign ``'-'`` ⇒ ``ι_{X_f} ω + df``; sign ``'+'``
        ⇒ ``ι_{X_f} ω − df``). The helper is registry-free, the shape
        itself doesn't depend on grading, and downstream simplification
        is where the grading gets consulted.

        ``d`` defaults to the :mod:`exterior_d` singleton; ``interior``
        defaults to the :func:`jacopy.calculus.interior.interior`
        factory so the ``ι_{X_f}`` operator is a standard
        :class:`InteriorProduct`.

        Raises :class:`ValueError` when no symplectic form is attached.
        """
        if self._symplectic_form is None:
            raise ValueError(
                "symplectic_obstruction requires a symplectic_form; "
                "construct HamiltonianVectorField with symplectic_form=..."
            )
        dop = default_d if d is None else d
        iota_factory = default_interior if interior is None else interior
        iota_Xf = iota_factory(self)
        df = Act(dop, self._function)
        # sign='-': obstruction = ι_X ω - (-df) = ι_X ω + df
        # sign='+': obstruction = ι_X ω - (+df) = ι_X ω - df
        df_term = df if self._sign == "-" else Neg(df)
        return Sum(
            Act(iota_Xf, self._symplectic_form),
            df_term,
        )

    def symplectic_condition(
        self,
        *,
        d: Optional[ExteriorDerivative] = None,
        interior: Optional[Any] = None,
    ) -> VanishingCondition:
        """Return the symplectic definition as a :class:`VanishingCondition`.

        Thin wrapper around :meth:`symplectic_obstruction` that hands
        the result to the proof layer in its typed form. The condition
        vanishes iff ``ι_{X_f} ω = −df``.
        """
        return VanishingCondition(
            obstruction=self.symplectic_obstruction(d=d, interior=interior),
            name=f"symplectic definition of {self.name}",
        )

    # ---- musical bridge proof -------------------------------------- #

    def prove_equivalence(
        self,
        compatibility: MusicalCompatibility,
        *,
        registry: Optional[PropertyRegistry] = None,
        d: Optional[ExteriorDerivative] = None,
        interior: Optional[Any] = None,
    ) -> ProofChain:
        """Close ``ι_{X_f} ω + df = 0`` using a musical compatibility axiom.

        Given a :class:`~jacopy.calculus.musical.MusicalCompatibility`
        axiom declaring ``ω`` and ``π`` as musical inverses, this
        builds a proof engine seeded with

        * the Hamiltonian-to-sharp rewrite ``X_f → −π^♯(df)``
          (:class:`HamiltonianVfDerivedDefinition`, scoped to this
          ``X_f`` instance);
        * the musical triplet returned by
          :meth:`MusicalCompatibility.musical_definitions`,
          :class:`~jacopy.calculus.musical.IotaFlatDefinition`,
          :class:`~jacopy.calculus.musical.ArgNegLinearityDefinition`,
          and :class:`~jacopy.calculus.musical.MusicalCompatibilityDefinition`.

        Runs :class:`~jacopy.proof.strategies.ExpandAndSimplify` on
        the symplectic obstruction against :class:`Integer` ``0``; the
        returned :class:`~jacopy.proof.chain.ProofChain` records each
        axiom application in order and closes on cancellation.

        Raises :class:`ValueError` when this Hamiltonian is missing
        ``bivector`` or ``symplectic_form``, or when the attached
        values don't match the compatibility's declared pair,
        proceeding with a mismatched axiom would produce a
        superficially closed proof that doesn't actually control the
        stated obstruction.
        """
        if self._bivector is None:
            raise ValueError(
                "prove_equivalence requires a bivector; "
                "construct HamiltonianVectorField with bivector=..."
            )
        if self._symplectic_form is None:
            raise ValueError(
                "prove_equivalence requires a symplectic_form; "
                "construct HamiltonianVectorField with symplectic_form=..."
            )
        if self._bivector != compatibility.pi:
            raise ValueError(
                "Hamiltonian bivector does not match compatibility's π"
            )
        if self._symplectic_form != compatibility.omega:
            raise ValueError(
                "Hamiltonian symplectic form does not match compatibility's ω"
            )

        dop = default_d if d is None else d
        obstruction = self.symplectic_obstruction(d=dop, interior=interior)

        definitions = (
            HamiltonianVfDerivedDefinition(self, compatibility, d=dop),
            *compatibility.musical_definitions(),
        )
        engine = ExpansionEngine(list(definitions))
        from jacopy.core.expr import Integer  # local to avoid top-level cycle
        return ExpandAndSimplify().prove(
            obstruction,
            Integer(0),
            registry=registry,
            engine=engine,
        )


# --------------------------------------------------------------------- #
# Factory                                                                #
# --------------------------------------------------------------------- #


def hamiltonian_vf(
    f: Expr,
    *,
    bivector: Optional[Expr] = None,
    symplectic_form: Optional[Expr] = None,
    name: Optional[str] = None,
    sign: str = "-",
) -> HamiltonianVectorField:
    """Build the Hamiltonian vector field ``X_f``.

    Mirror of the :class:`HamiltonianVectorField` constructor with a
    functional name, preferred at call sites for readability.
    """
    return HamiltonianVectorField(
        f,
        bivector=bivector,
        symplectic_form=symplectic_form,
        name=name,
        sign=sign,
    )


# --------------------------------------------------------------------- #
# Equivalence bridge                                                     #
# --------------------------------------------------------------------- #


def equivalence_condition(
    f: Expr,
    *,
    bivector: Expr,
    symplectic_form: Expr,
    d: Optional[ExteriorDerivative] = None,
    interior: Optional[Any] = None,
) -> VanishingCondition:
    """``X_f^{symplectic} = X_f^{derived}`` as a :class:`VanishingCondition`.

    Builds ``X_f`` with both data attached and returns its symplectic
    obstruction wrapped as a named condition. Vanishing here means
    ``ω`` and ``π`` are musically inverse on ``df``, the exact
    condition the two Hamiltonian definitions need to coincide. The
    caller discharges this against whatever compatibility axiom their
    setup declares (typically ``ω ∘ π♯ = id`` or its 1-form image
    ``ω(π(df), ·) = df``).

    The helper deliberately stays at :class:`VanishingCondition` level
    rather than issuing a :class:`ProofChain`, closing the condition
    requires a musical-isomorphism axiom the Expr layer does not yet
    encode. Feed the returned condition to
    :func:`~jacopy.proof.verifier.prove_equivalence` once the caller
    has that axiom on hand.
    """
    Xf = hamiltonian_vf(
        f, bivector=bivector, symplectic_form=symplectic_form
    )
    return VanishingCondition(
        obstruction=Xf.symplectic_obstruction(d=d, interior=interior),
        name=f"equivalence of symplectic and derived definitions of X_{f._repr_inner()}",
    )


# --------------------------------------------------------------------- #
# Proof-engine rule, X_f as −π♯(df) under a declared compatibility     #
# --------------------------------------------------------------------- #


class HamiltonianVfDerivedDefinition(Definition):
    """Rewrite ``X_f → −π^♯(df)`` when the compatibility declares the pair.

    Scoped to a specific :class:`HamiltonianVectorField` instance so
    unrelated Hamiltonians in the same proof don't get swept up. The
    rewrite is the atom-level counterpart of
    :meth:`HamiltonianVectorField.derived_expansion`, reshaped into
    the sharp form that pairs with
    :class:`~jacopy.calculus.musical.MusicalCompatibilityDefinition`
    to collapse the symplectic obstruction.

    The sign lives in the rewrite's output ``Neg``, with
    :class:`~jacopy.calculus.musical.ArgNegLinearityDefinition` in
    the engine, it pulls through the outer ``ω^♭`` so the
    compatibility can match the clean ``ω^♭(π^♯(df))`` shape
    underneath and the residual ``−df + df`` cancels under simplify.
    """

    def __init__(
        self,
        hamiltonian: "HamiltonianVectorField",
        compatibility: MusicalCompatibility,
        *,
        d: Optional[ExteriorDerivative] = None,
    ) -> None:
        if not isinstance(hamiltonian, HamiltonianVectorField):
            raise TypeError(
                "HamiltonianVfDerivedDefinition expects a HamiltonianVectorField"
            )
        if not isinstance(compatibility, MusicalCompatibility):
            raise TypeError(
                "HamiltonianVfDerivedDefinition expects a MusicalCompatibility"
            )
        self._hamiltonian = hamiltonian
        self._compat = compatibility
        self._d = default_d if d is None else d
        sign_str = "−" if hamiltonian.sign == "-" else ""
        self.name = f"X_f = {sign_str}π♯(df) [{hamiltonian.name}]"

    def matches(self, expr: Expr) -> bool:
        return expr is self._hamiltonian or expr == self._hamiltonian

    def rewrite(self, expr: Expr) -> Expr:
        f = self._hamiltonian.function
        body = Act(self._compat.sharp, Act(self._d, f))
        return Neg(body) if self._hamiltonian.sign == "-" else body


# --------------------------------------------------------------------- #
# Hamiltonian defining-relation engine rule (Faz 12.C(c))                #
# --------------------------------------------------------------------- #


class HamiltonianDefiningRelationDefinition(Definition):
    """Rewrite ``ι_X ω → sign·df`` for a designated ``(X, f, ω)`` triple.

    Engine-level encoding of the symplectic Hamiltonian defining
    relation. Scoped to a specific vector field ``X``, function ``f``,
    and 2-form ``ω`` so unrelated interior products in the same proof
    are left untouched. The rewrite shape is

    * ``sign='-'`` (geometer's convention): ``ι_X ω → −df``
    * ``sign='+'`` (textbook convention): ``ι_X ω → +df``

    Together with :class:`ClosedFormDefinition` (``dω = 0``) this is
    enough to discharge the symplectic Lie-derivative identity
    ``L_X ω = 0`` via Cartan magic without spelling out the bridge by
    hand each time.
    """

    def __init__(
        self,
        X: Expr,
        f: Expr,
        omega: Expr,
        *,
        sign: str = "-",
        d: Optional[ExteriorDerivative] = None,
    ) -> None:
        if not isinstance(X, Expr):
            raise TypeError(
                "HamiltonianDefiningRelationDefinition X must be an Expr"
            )
        if not isinstance(f, Expr):
            raise TypeError(
                "HamiltonianDefiningRelationDefinition f must be an Expr"
            )
        if not isinstance(omega, Expr):
            raise TypeError(
                "HamiltonianDefiningRelationDefinition omega must be an Expr"
            )
        if sign not in _VALID_SIGNS:
            raise ValueError(
                f"HamiltonianDefiningRelationDefinition sign must be "
                f"'+' or '-', got {sign!r}"
            )
        self._X = X
        self._f = f
        self._omega = omega
        self._sign = sign
        self._d = default_d if d is None else d
        sign_str = "−" if sign == "-" else ""
        self.name = (
            f"ι_{X._repr_inner()} {omega._repr_inner()} = "
            f"{sign_str}d{f._repr_inner()}"
        )

    def matches(self, expr: Expr) -> bool:
        if not isinstance(expr, Act):
            return False
        if not isinstance(expr.op, InteriorProduct):
            return False
        if expr.op.vector_field != self._X:
            return False
        return expr.arg == self._omega

    def rewrite(self, expr: Expr) -> Expr:
        df = Act(self._d, self._f)
        return Neg(df) if self._sign == "-" else df


def register_hamiltonian_defining_relation(
    X: Expr,
    f: Expr,
    omega: Expr,
    engine: ExpansionEngine,
    *,
    sign: str = "-",
    d: Optional[ExteriorDerivative] = None,
) -> HamiltonianDefiningRelationDefinition:
    """Register ``ι_X ω = sign·df`` on ``engine`` and return the rule.

    One-line replacement for the inline ``IotaXfOmegaIs±Df`` definitions
    that notebooks 2a/2b/2c hand-rolled. The returned
    :class:`HamiltonianDefiningRelationDefinition` is the same object
    handed to the engine, so callers that want to introspect or remove
    the rule keep the reference.

    The default ``sign='-'`` matches the geometer's convention
    ``ι_{X_f} ω = -df`` used by
    :meth:`HamiltonianVectorField.symplectic_obstruction`. Pass
    ``sign='+'`` for problems following the textbook ``ι_{X_f} ω = df``
    convention.
    """
    rule = HamiltonianDefiningRelationDefinition(
        X, f, omega, sign=sign, d=d
    )
    engine.register(rule)
    return rule
