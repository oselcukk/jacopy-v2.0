"""
Symplectic problem wrapper.

A :class:`SymplecticProblem` bundles a :class:`SymplecticManifold`
together with a tuple of designated functions (the Hamiltonians of
interest in a problem statement) and a pre-configured
:class:`ExpansionEngine` carrying every axiom that statement implies:

* ``dω = 0`` (:class:`ClosedFormDefinition`),
* the Hamiltonian defining relation ``ι_{X_f} ω = sign·df`` for each
  designated ``f``,
* if a Poisson bivector is attached: the musical-compatibility quartet
  (:class:`MusicalCompatibilityDefinition`,
  :class:`MusicalCompatibilityBilinearDefinition`,
  :class:`IotaFlatDefinition`,
  :class:`ArgNegLinearityDefinition`),
* the registry-free C∞-linearity rules
  (:class:`MultiEvalScalarPullDefinition`,
  :class:`PairingScalarPullDefinition`,
  :class:`LieRescalingDefinition`),
* :class:`RegistryAntiSymCanonicalDefinition` keyed on ``π`` when a
  bivector is supplied.

The wrapper is the problem-book counterpart to
:class:`SymplecticManifold`: where the manifold names the geometric
data, ``SymplecticProblem`` names the *proof setup*, the engine is
ready to discharge ``L_{X_f} ω = 0``, ``ω(X_f, X_g) = π(df, dg)``, or
the symplectic↔derived equivalence in one call.

Auto-declarations are deliberate but minimal: ``ω`` gets the
:class:`Closed` flag if absent, and ``π`` gets :class:`Antisymmetric`
if absent. Grading declarations (``Graded(degree=2)`` for ``ω``,
``Graded(degree=1)`` for ``π``, ``Graded(degree=0)`` for the
functions) stay the caller's responsibility, they affect
:mod:`sort_product` and other grading-driven layers and a wrapper
should not silently change them.
"""

from __future__ import annotations

from typing import Any, Iterable, Optional, Tuple

from jacopy.calculus.antisym_axioms import RegistryAntiSymCanonicalDefinition
from jacopy.calculus.closed_axioms import ClosedFormDefinition
from jacopy.calculus.exterior_d import ExteriorDerivative, d as default_d
from jacopy.calculus.hamiltonian_vf import (
    HamiltonianDefiningRelationDefinition,
    HamiltonianVectorField,
    register_hamiltonian_defining_relation,
)
from jacopy.calculus.lie_rescaling_axioms import LieRescalingDefinition
from jacopy.calculus.multi_eval_scalar_axioms import (
    MultiEvalScalarPullDefinition,
)
from jacopy.calculus.nondegenerate_axioms import (
    NonDegenerateInteriorEqualityDefinition,
)
from jacopy.calculus.pairing_linearity_axioms import (
    PairingScalarPullDefinition,
)
from jacopy.core.expr import Expr, Integer
from jacopy.core.properties import Antisymmetric, Closed, NonDegenerate
from jacopy.core.registry import PropertyRegistry
from jacopy.library.symplectic import SymplecticManifold
from jacopy.proof.chain import ProofChain
from jacopy.proof.expansion import ExpansionEngine, default_engine


_VALID_SIGNS = ("+", "-")


class SymplecticProblem:
    """Problem-book bundle: ``(M, ω, π?, {f_i}, registry, engine)``.

    Parameters
    ----------
    omega
        The symplectic 2-form ``ω``. Required.
    functions
        Iterable of :class:`Expr` operands designated as Hamiltonians
        in the problem statement. Each gets a registered
        :class:`HamiltonianVectorField` and its defining relation
        ``ι_{X_f} ω = sign·df`` registered on the engine.
    bivector
        Optional Poisson bivector ``π`` assumed musically inverse to
        ``ω``. Adds the musical quartet to the engine.
    registry
        :class:`PropertyRegistry` carrying the grading and any other
        flags. The wrapper auto-declares ``Closed(ω)`` and (if
        ``bivector`` is present) ``Antisymmetric(π)`` if not already
        declared. Required.
    sign
        Sign convention for ``ι_{X_f} ω = sign·df``. ``'-'`` (default)
        matches :meth:`HamiltonianVectorField.symplectic_obstruction`;
        ``'+'`` matches the textbook convention used in problem texts.
    base_engine
        Optional pre-built :class:`ExpansionEngine` to clone the
        defining set from. Defaults to
        :func:`~jacopy.proof.expansion.default_engine` with the
        provided ``registry``.
    name
        Display name; defaults to ``f"SympProblem({ω}; {f_1, …})"``.
    """

    __slots__ = (
        "_manifold",
        "_functions",
        "_hamiltonians",
        "_registry",
        "_engine",
        "_sign",
        "_name",
        "_defining_rules",
        "_d",
    )

    def __init__(
        self,
        omega: Expr,
        functions: Iterable[Expr],
        *,
        bivector: Optional[Expr] = None,
        registry: PropertyRegistry,
        sign: str = "-",
        base_engine: Optional[ExpansionEngine] = None,
        d: Optional[ExteriorDerivative] = None,
        name: Optional[str] = None,
    ) -> None:
        if not isinstance(omega, Expr):
            raise TypeError("SymplecticProblem omega must be an Expr")
        if bivector is not None and not isinstance(bivector, Expr):
            raise TypeError(
                "SymplecticProblem bivector must be an Expr when provided"
            )
        if not isinstance(registry, PropertyRegistry):
            raise TypeError(
                "SymplecticProblem requires a PropertyRegistry"
            )
        if sign not in _VALID_SIGNS:
            raise ValueError(
                f"SymplecticProblem sign must be '+' or '-', got {sign!r}"
            )
        functions_tuple: Tuple[Expr, ...] = tuple(functions)
        if not functions_tuple:
            raise ValueError(
                "SymplecticProblem requires at least one designated function"
            )
        for f in functions_tuple:
            if not isinstance(f, Expr):
                raise TypeError(
                    "SymplecticProblem functions must all be Expr instances"
                )

        # Auto-declare the structural flags. Closed/NonDegenerate/
        # Antisymmetric are part of the *problem statement* (a
        # symplectic form is closed and non-degenerate by definition; a
        # Poisson bivector is antisymmetric by definition), so the
        # wrapper records them as axioms, but only if absent; we
        # never overwrite a caller's prior declaration.
        if not registry.has(omega, Closed):
            registry.declare(omega, Closed())
        if not registry.has(omega, NonDegenerate):
            registry.declare(omega, NonDegenerate())
        if bivector is not None and not registry.has(bivector, Antisymmetric):
            registry.declare(bivector, Antisymmetric())

        manifold = SymplecticManifold(omega, bivector=bivector)
        d_op = default_d if d is None else d

        # Build the hamiltonians dict. Each X_f carries the manifold's
        # ω (always) and π (when present), and the chosen sign.
        hamiltonians = {}
        for f in functions_tuple:
            hamiltonians[f] = manifold.hamiltonian_vf(f, sign=sign)

        # Build the engine: start from base_engine or default_engine,
        # then layer the symplectic-specific rules on top.
        if base_engine is None:
            engine = default_engine(registry=registry, d=d_op)
        else:
            engine = ExpansionEngine(
                list(base_engine.definitions),
                mode=base_engine.mode,
            )

        # Closed form rule needs the registry the wrapper was given.
        engine.register(ClosedFormDefinition(registry=registry))
        # Non-degeneracy injectivity rule (peels ι_Y ω - ι_Z ω → Y - Z).
        engine.register(
            NonDegenerateInteriorEqualityDefinition(registry=registry)
        )
        # Registry-free C∞-linearity rules.
        engine.register(MultiEvalScalarPullDefinition())
        engine.register(PairingScalarPullDefinition())
        engine.register(LieRescalingDefinition())
        # Bivector-side rules.
        if bivector is not None:
            engine.register(RegistryAntiSymCanonicalDefinition(registry=registry))
            for rule in manifold.compatibility.musical_definitions():
                engine.register(rule)

        # Per-function defining relation ι_{X_f} ω = sign·df.
        defining_rules = []
        for f in functions_tuple:
            rule = register_hamiltonian_defining_relation(
                hamiltonians[f],
                f,
                omega,
                engine,
                sign=sign,
                d=d_op,
            )
            defining_rules.append(rule)

        self._manifold = manifold
        self._functions = functions_tuple
        self._hamiltonians = hamiltonians
        self._registry = registry
        self._engine = engine
        self._sign = sign
        self._d = d_op
        self._defining_rules = tuple(defining_rules)
        if name is not None:
            self._name = name
        else:
            funs = ", ".join(f._repr_inner() for f in functions_tuple)
            self._name = f"SympProblem({omega._repr_inner()}; {funs})"

    # ---- accessors ------------------------------------------------- #

    @property
    def manifold(self) -> SymplecticManifold:
        return self._manifold

    @property
    def omega(self) -> Expr:
        return self._manifold.omega

    @property
    def bivector(self) -> Optional[Expr]:
        return self._manifold.bivector

    @property
    def functions(self) -> Tuple[Expr, ...]:
        return self._functions

    @property
    def registry(self) -> PropertyRegistry:
        return self._registry

    @property
    def engine(self) -> ExpansionEngine:
        return self._engine

    @property
    def sign(self) -> str:
        return self._sign

    @property
    def name(self) -> str:
        return self._name

    @property
    def defining_rules(self) -> Tuple[HamiltonianDefiningRelationDefinition, ...]:
        return self._defining_rules

    # ---- Hamiltonian access --------------------------------------- #

    def hamiltonian(self, f: Expr) -> HamiltonianVectorField:
        """Return the registered ``X_f`` for a designated function ``f``.

        Raises :class:`KeyError` when ``f`` was not part of the
        ``functions`` tuple at construction time, the engine has no
        defining relation for it, so silently building a fresh
        ``X_f`` would produce a Hamiltonian the proof layer can't
        actually reduce.
        """
        if f not in self._hamiltonians:
            raise KeyError(
                f"function {f!r} is not registered with this "
                f"SymplecticProblem; pass it in 'functions=' at construction"
            )
        return self._hamiltonians[f]

    # ---- proof helpers -------------------------------------------- #

    def prove_hamiltonian_invariance(self, f: Expr) -> ProofChain:
        """Close ``L_{X_f} ω = 0`` using this problem's engine.

        Cartan magic plus the closed-form axiom and the per-``f``
        defining relation give a clean closure: ``L_{X_f} ω =
        d(ι_{X_f} ω) + ι_{X_f}(dω) = d(±df) + 0 = 0``. Equivalent in
        spirit to notebook 2a's chain, but driven entirely off the
        wrapper's pre-built engine.
        """
        from jacopy.algebra.derivation import Act
        from jacopy.calculus.lie_derivative import lie_derivative
        from jacopy.proof.strategies import ExpandAndSimplify

        Xf = self.hamiltonian(f)
        residual = Act(lie_derivative(Xf), self.omega)
        return ExpandAndSimplify().prove(
            residual,
            Integer(0),
            registry=self._registry,
            engine=self._engine,
        )

    def prove_vector_field_equality(self, Y: Expr, Z: Expr) -> ProofChain:
        r"""Close ``Y = Z`` (as vector fields) via non-degeneracy of ``ω``.

        Builds the obstruction ``ι_Y ω − ι_Z ω`` and lets the engine's
        :class:`NonDegenerateInteriorEqualityDefinition` peel off the
        ``ι_(·) ω`` shell, under non-degeneracy this is exactly the
        injectivity statement ``ι_Y ω = ι_Z ω ⇒ Y = Z``, encoded as a
        term-rewriting step. The remaining ``Y − Z`` then reduces to
        ``0`` whenever the engine has enough machinery to bring ``Y``
        and ``Z`` to a common syntactic form.

        Typical use:

        * Trivial, passing the same vector field twice gives a
          one-step proof (``ι_Y ω − ι_Y ω → Y − Y → 0``).
        * Engine-derived, when ``Y`` and ``Z`` are different
          syntactic shapes that simplify identically (e.g. via
          algebroid Cartan magic, Lie-bracket axioms, sharp
          axioms), the chain closes through the standard
          :class:`ExpandAndSimplify` pipeline.

        Raises :class:`ValueError` when ``ω`` is not declared
        :class:`NonDegenerate` on the registry, the rule then
        produces no rewrite and the chain cannot close. The
        :class:`SymplecticProblem` constructor auto-declares the flag,
        so the typical caller never sees this error.
        """
        from jacopy.algebra.derivation import Act
        from jacopy.calculus.interior import interior
        from jacopy.core.expr import Neg, Sum
        from jacopy.proof.strategies import ExpandAndSimplify

        if not isinstance(Y, Expr):
            raise TypeError(
                "prove_vector_field_equality Y must be an Expr"
            )
        if not isinstance(Z, Expr):
            raise TypeError(
                "prove_vector_field_equality Z must be an Expr"
            )
        if not self._registry.has(self.omega, NonDegenerate):
            raise ValueError(
                "prove_vector_field_equality requires NonDegenerate(ω) "
                "on the registry; SymplecticProblem normally auto-declares "
                "this, only triggered when the caller deliberately "
                "removed the flag"
            )
        obstruction = Sum(
            Act(interior(Y), self.omega),
            Neg(Act(interior(Z), self.omega)),
        )
        return ExpandAndSimplify().prove(
            obstruction,
            Integer(0),
            registry=self._registry,
            engine=self._engine,
        )

    def prove_hamiltonian_equality(self, Y: Expr, h: Expr) -> ProofChain:
        r"""Close ``ι_Y ω = sign·dh`` as a cited-axiom chain.

        Witnesses that ``Y`` plays the Hamiltonian role for ``h`` on
        this problem's ``(M, ω)``. With non-degeneracy of ``ω``
        declared on the registry (auto-declared in the
        :class:`SymplecticProblem` constructor), this equality is
        equivalent to ``Y = X_h``: see
        :meth:`prove_vector_field_equality` for the direct reduction
        of the vector-field equality.

        The chain still proceeds by citing the
        :class:`HamiltonianDefiningRelationDefinition` for the
        ``(Y, h, ω)`` triple as a one-off axiom, that captures the
        caller's hypothesis that ``Y`` does in fact satisfy the
        defining relation. Non-degeneracy then licenses the *implicit*
        upgrade from ``ι_Y ω = sign·dh`` to ``Y = X_h`` (the rule
        sitting on the engine takes care of it for callers who phrase
        their goal as a vector-field equality directly).

        Typical use: closing
        ``[X_f, X_g] = X_{\{f,g\}}`` in problem 2c, the caller passes
        ``Y = lie_bracket(X_f, X_g)`` and ``h = {f, g}``.

        Parameters
        ----------
        Y
            The vector field side of the equality. Any :class:`Expr`
            representing a vector field, typically a
            :class:`Derivation`, a Lie bracket, or a designated
            :class:`HamiltonianVectorField`.
        h
            The function whose Hamiltonian ``Y`` is asserted to be.

        Returns
        -------
        :class:`ProofChain`
            One axiom-cited rewrite step witnessing the equality,
            plus simplification to ``0`` on the obstruction form.

        Notes
        -----
        The temporary defining relation does *not* persist on
        :attr:`engine`, successive calls with different ``(Y, h)``
        pairs do not pollute one another's engines.
        """
        from jacopy.algebra.derivation import Act
        from jacopy.calculus.interior import interior
        from jacopy.core.expr import Neg, Sum
        from jacopy.proof.strategies import ExpandAndSimplify

        if not isinstance(Y, Expr):
            raise TypeError(
                "prove_hamiltonian_equality Y must be an Expr"
            )
        if not isinstance(h, Expr):
            raise TypeError(
                "prove_hamiltonian_equality h must be an Expr"
            )

        # Build the obstruction ι_Y ω − sign·dh, mirroring the shape
        # produced by HamiltonianVectorField.symplectic_obstruction.
        dh = Act(self._d, h)
        df_term = dh if self._sign == "-" else Neg(dh)
        obstruction = Sum(Act(interior(Y), self.omega), df_term)

        # Fork the engine: clone the existing definitions, append a
        # one-off defining relation for the (Y, h, ω) triple, but do
        # *not* register on self._engine so successive calls stay
        # independent.
        forked = ExpansionEngine(
            list(self._engine.definitions),
            mode=self._engine.mode,
        )
        forked.register(
            HamiltonianDefiningRelationDefinition(
                Y, h, self.omega, sign=self._sign, d=self._d,
            )
        )
        return ExpandAndSimplify().prove(
            obstruction,
            Integer(0),
            registry=self._registry,
            engine=forked,
        )

    def prove_hamiltonian_equivalence(self, f: Expr) -> ProofChain:
        """Close ``ι_{X_f} ω + df = 0`` (or its sign-flipped sibling).

        Delegates to :meth:`SymplecticManifold.prove_hamiltonian_equivalence`
        so the manifold's compatibility object drives the bridge,
        identical to constructing the manifold directly and calling
        the method, just guaranteed to use the same registry/sign as
        the rest of the problem.
        """
        if self.bivector is None:
            raise ValueError(
                "prove_hamiltonian_equivalence requires a bivector; "
                "construct SymplecticProblem with bivector=..."
            )
        if f not in self._hamiltonians:
            raise KeyError(
                f"function {f!r} is not registered with this SymplecticProblem"
            )
        return self._manifold.prove_hamiltonian_equivalence(
            f, registry=self._registry
        )

    # ---- dunder --------------------------------------------------- #

    def __repr__(self) -> str:
        funs = ", ".join(f._repr_inner() for f in self._functions)
        if self.bivector is None:
            return (
                f"SymplecticProblem({self.omega._repr_inner()}; "
                f"functions={{{funs}}}, sign='{self._sign}')"
            )
        return (
            f"SymplecticProblem({self.omega._repr_inner()}, "
            f"π={self.bivector._repr_inner()}; "
            f"functions={{{funs}}}, sign='{self._sign}')"
        )
