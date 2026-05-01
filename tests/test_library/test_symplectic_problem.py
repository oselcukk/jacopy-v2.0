"""Tests for ``jacopy.library.symplectic_problem.SymplecticProblem``."""

from __future__ import annotations

import pytest

from jacopy.calculus.hamiltonian_vf import (
    HamiltonianDefiningRelationDefinition,
    HamiltonianVectorField,
)
from jacopy.calculus.musical import (
    MusicalCompatibilityBilinearDefinition,
    MusicalCompatibilityDefinition,
)
from jacopy.calculus.closed_axioms import ClosedFormDefinition
from jacopy.calculus.antisym_axioms import RegistryAntiSymCanonicalDefinition
from jacopy.calculus.nondegenerate_axioms import (
    NonDegenerateInteriorEqualityDefinition,
)
from jacopy.core.expr import Integer, Symbol
from jacopy.core.properties import Antisymmetric, Closed, Graded, NonDegenerate
from jacopy.core.registry import PropertyRegistry
from jacopy.library.symplectic import SymplecticManifold
from jacopy.library.symplectic_problem import SymplecticProblem
from jacopy.proof.chain import ProofChain


# --------------------------------------------------------------------- #
# Fixtures                                                              #
# --------------------------------------------------------------------- #


@pytest.fixture
def omega():
    return Symbol("ω")


@pytest.fixture
def pi():
    return Symbol("π")


@pytest.fixture
def f():
    return Symbol("f")


@pytest.fixture
def g():
    return Symbol("g")


@pytest.fixture
def reg(omega, pi, f, g):
    r = PropertyRegistry()
    r.declare(omega, Graded(degree=2))
    r.declare(pi, Graded(degree=1))
    r.declare(f, Graded(degree=0))
    r.declare(g, Graded(degree=0))
    return r


# --------------------------------------------------------------------- #
# Construction + auto-declaration                                       #
# --------------------------------------------------------------------- #


class TestConstruction:
    def test_omega_only(self, omega, f, g, reg):
        sp = SymplecticProblem(omega, (f, g), registry=reg)
        assert sp.omega is omega
        assert sp.bivector is None
        assert sp.functions == (f, g)
        assert isinstance(sp.manifold, SymplecticManifold)
        assert sp.sign == "-"

    def test_with_bivector(self, omega, pi, f, reg):
        sp = SymplecticProblem(omega, (f,), bivector=pi, registry=reg)
        assert sp.bivector is pi
        assert sp.manifold.compatibility is not None

    def test_auto_declares_closed(self, omega, f, reg):
        assert not reg.has(omega, Closed)
        SymplecticProblem(omega, (f,), registry=reg)
        assert reg.has(omega, Closed)

    def test_does_not_overwrite_existing_closed(self, omega, f, reg):
        prior = Closed()
        reg.declare(omega, prior)
        SymplecticProblem(omega, (f,), registry=reg)
        assert reg.get(omega, Closed) is prior

    def test_auto_declares_antisymmetric_when_bivector(self, omega, pi, f, reg):
        assert not reg.has(pi, Antisymmetric)
        SymplecticProblem(omega, (f,), bivector=pi, registry=reg)
        assert reg.has(pi, Antisymmetric)

    def test_no_antisymmetric_without_bivector(self, omega, pi, f, reg):
        SymplecticProblem(omega, (f,), registry=reg)
        assert not reg.has(pi, Antisymmetric)

    def test_default_sign_is_minus(self, omega, f, reg):
        sp = SymplecticProblem(omega, (f,), registry=reg)
        assert sp.sign == "-"
        assert sp.hamiltonian(f).sign == "-"

    def test_explicit_sign_threads_to_hamiltonians(self, omega, f, reg):
        sp = SymplecticProblem(omega, (f,), registry=reg, sign="+")
        assert sp.sign == "+"
        assert sp.hamiltonian(f).sign == "+"

    def test_sign_threads_to_defining_rules(self, omega, f, reg):
        sp = SymplecticProblem(omega, (f,), registry=reg, sign="+")
        assert len(sp.defining_rules) == 1
        rule = sp.defining_rules[0]
        assert rule._sign == "+"

    def test_default_name(self, omega, f, g, reg):
        sp = SymplecticProblem(omega, (f, g), registry=reg)
        assert "ω" in sp.name and "f" in sp.name and "g" in sp.name

    def test_custom_name(self, omega, f, reg):
        sp = SymplecticProblem(omega, (f,), registry=reg, name="My Problem")
        assert sp.name == "My Problem"

    # ---- validation ---------------------------------------------- #

    def test_rejects_non_expr_omega(self, f, reg):
        with pytest.raises(TypeError, match="omega"):
            SymplecticProblem("ω", (f,), registry=reg)  # type: ignore[arg-type]

    def test_rejects_non_expr_bivector(self, omega, f, reg):
        with pytest.raises(TypeError, match="bivector"):
            SymplecticProblem(
                omega, (f,), bivector="π", registry=reg  # type: ignore[arg-type]
            )

    def test_rejects_non_expr_function(self, omega, reg):
        with pytest.raises(TypeError, match="functions"):
            SymplecticProblem(omega, ("f",), registry=reg)  # type: ignore[arg-type]

    def test_rejects_invalid_sign(self, omega, f, reg):
        with pytest.raises(ValueError, match="sign"):
            SymplecticProblem(omega, (f,), registry=reg, sign="?")

    def test_rejects_missing_registry(self, omega, f):
        with pytest.raises(TypeError, match="PropertyRegistry"):
            SymplecticProblem(omega, (f,), registry=None)  # type: ignore[arg-type]

    def test_rejects_empty_functions(self, omega, reg):
        with pytest.raises(ValueError, match="function"):
            SymplecticProblem(omega, (), registry=reg)


# --------------------------------------------------------------------- #
# Engine pre-loaded with the right axioms                               #
# --------------------------------------------------------------------- #


class TestEngineAxioms:
    def _kinds(self, sp: SymplecticProblem):
        return [type(d) for d in sp.engine.definitions]

    def test_engine_carries_closed_form_rule(self, omega, f, reg):
        sp = SymplecticProblem(omega, (f,), registry=reg)
        assert ClosedFormDefinition in self._kinds(sp)

    def test_engine_carries_per_function_defining_relation(
        self, omega, f, g, reg
    ):
        sp = SymplecticProblem(omega, (f, g), registry=reg)
        rules = [
            d for d in sp.engine.definitions
            if isinstance(d, HamiltonianDefiningRelationDefinition)
        ]
        assert len(rules) == 2

    def test_engine_carries_antisym_when_bivector(self, omega, pi, f, reg):
        sp = SymplecticProblem(omega, (f,), bivector=pi, registry=reg)
        assert RegistryAntiSymCanonicalDefinition in self._kinds(sp)

    def test_engine_carries_musical_quartet_when_bivector(
        self, omega, pi, f, reg
    ):
        sp = SymplecticProblem(omega, (f,), bivector=pi, registry=reg)
        kinds = self._kinds(sp)
        assert MusicalCompatibilityDefinition in kinds
        assert MusicalCompatibilityBilinearDefinition in kinds

    def test_engine_omits_musical_when_no_bivector(self, omega, f, reg):
        sp = SymplecticProblem(omega, (f,), registry=reg)
        kinds = self._kinds(sp)
        assert MusicalCompatibilityDefinition not in kinds
        assert MusicalCompatibilityBilinearDefinition not in kinds


# --------------------------------------------------------------------- #
# hamiltonian(f) accessor                                               #
# --------------------------------------------------------------------- #


class TestHamiltonianAccessor:
    def test_returns_registered_vf(self, omega, f, reg):
        sp = SymplecticProblem(omega, (f,), registry=reg)
        Xf = sp.hamiltonian(f)
        assert isinstance(Xf, HamiltonianVectorField)
        assert Xf.function is f
        assert Xf.symplectic_form is omega

    def test_carries_bivector_when_present(self, omega, pi, f, reg):
        sp = SymplecticProblem(omega, (f,), bivector=pi, registry=reg)
        Xf = sp.hamiltonian(f)
        assert Xf.bivector is pi

    def test_unregistered_function_raises(self, omega, f, reg):
        sp = SymplecticProblem(omega, (f,), registry=reg)
        with pytest.raises(KeyError, match="registered"):
            sp.hamiltonian(Symbol("h"))

    def test_same_call_returns_same_instance(self, omega, f, reg):
        sp = SymplecticProblem(omega, (f,), registry=reg)
        assert sp.hamiltonian(f) is sp.hamiltonian(f)


# --------------------------------------------------------------------- #
# prove_hamiltonian_invariance, L_{X_f} ω = 0                          #
# --------------------------------------------------------------------- #


class TestProveInvariance:
    def test_closes_to_zero(self, omega, f, reg):
        sp = SymplecticProblem(omega, (f,), registry=reg, sign="+")
        chain = sp.prove_hamiltonian_invariance(f)
        assert isinstance(chain, ProofChain)
        assert chain.final == Integer(0)
        assert len(chain) > 0

    def test_closes_with_minus_sign(self, omega, f, reg):
        sp = SymplecticProblem(omega, (f,), registry=reg, sign="-")
        chain = sp.prove_hamiltonian_invariance(f)
        assert chain.final == Integer(0)

    def test_unregistered_function_raises(self, omega, f, reg):
        sp = SymplecticProblem(omega, (f,), registry=reg)
        with pytest.raises(KeyError):
            sp.prove_hamiltonian_invariance(Symbol("h"))


# --------------------------------------------------------------------- #
# prove_hamiltonian_equivalence delegation                              #
# --------------------------------------------------------------------- #


class TestProveEquivalence:
    @pytest.fixture
    def reg_neg(self, omega, pi, f):
        r = PropertyRegistry()
        r.declare(omega, Graded(degree=2))
        r.declare(pi, Graded(degree=1))
        r.declare(f, Graded(degree=-1))
        return r

    def test_requires_bivector(self, omega, f, reg):
        sp = SymplecticProblem(omega, (f,), registry=reg)
        with pytest.raises(ValueError, match="bivector"):
            sp.prove_hamiltonian_equivalence(f)

    def test_unregistered_function_raises(self, omega, pi, f, reg_neg):
        sp = SymplecticProblem(omega, (f,), bivector=pi, registry=reg_neg)
        with pytest.raises(KeyError):
            sp.prove_hamiltonian_equivalence(Symbol("h"))

    def test_closes_with_bivector(self, omega, pi, f, reg_neg):
        sp = SymplecticProblem(omega, (f,), bivector=pi, registry=reg_neg)
        chain = sp.prove_hamiltonian_equivalence(f)
        assert isinstance(chain, ProofChain)
        assert chain.final == Integer(0)


# --------------------------------------------------------------------- #
# prove_hamiltonian_equality, Y = X_h modulo non-degeneracy            #
# --------------------------------------------------------------------- #


class TestProveEquality:
    def test_closes_with_minus_sign(self, omega, f, g, reg):
        from jacopy.algebra.derivation import Derivation

        sp = SymplecticProblem(omega, (f,), registry=reg, sign="-")
        Y = Derivation("[X_f,X_g]", degree=0)
        h = Symbol("{f,g}")
        reg.declare(h, Graded(degree=0))
        chain = sp.prove_hamiltonian_equality(Y, h)
        assert isinstance(chain, ProofChain)
        assert chain.final == Integer(0)
        assert len(chain) > 0

    def test_closes_with_plus_sign(self, omega, f, reg):
        from jacopy.algebra.derivation import Derivation

        sp = SymplecticProblem(omega, (f,), registry=reg, sign="+")
        Y = Derivation("[X_f,X_g]", degree=0)
        h = Symbol("{f,g}")
        reg.declare(h, Graded(degree=0))
        chain = sp.prove_hamiltonian_equality(Y, h)
        assert chain.final == Integer(0)

    def test_does_not_pollute_engine(self, omega, f, reg):
        from jacopy.algebra.derivation import Derivation

        sp = SymplecticProblem(omega, (f,), registry=reg)
        before = len(sp.engine.definitions)
        Y = Derivation("Y", degree=0)
        h = Symbol("h")
        reg.declare(h, Graded(degree=0))
        sp.prove_hamiltonian_equality(Y, h)
        assert len(sp.engine.definitions) == before

    def test_independent_calls_dont_interfere(self, omega, f, reg):
        from jacopy.algebra.derivation import Derivation

        sp = SymplecticProblem(omega, (f,), registry=reg)
        h1 = Symbol("h1"); reg.declare(h1, Graded(degree=0))
        h2 = Symbol("h2"); reg.declare(h2, Graded(degree=0))
        Y1 = Derivation("Y1", degree=0)
        Y2 = Derivation("Y2", degree=0)
        c1 = sp.prove_hamiltonian_equality(Y1, h1)
        c2 = sp.prove_hamiltonian_equality(Y2, h2)
        assert c1.final == Integer(0)
        assert c2.final == Integer(0)

    def test_rejects_non_expr_Y(self, omega, f, reg):
        sp = SymplecticProblem(omega, (f,), registry=reg)
        with pytest.raises(TypeError, match="Y"):
            sp.prove_hamiltonian_equality("Y", Symbol("h"))  # type: ignore[arg-type]

    def test_rejects_non_expr_h(self, omega, f, reg):
        from jacopy.algebra.derivation import Derivation

        sp = SymplecticProblem(omega, (f,), registry=reg)
        with pytest.raises(TypeError, match="h"):
            sp.prove_hamiltonian_equality(
                Derivation("Y", degree=0), "h"  # type: ignore[arg-type]
            )


# --------------------------------------------------------------------- #
# NonDegenerate auto-declaration + engine rule (Faz 12.B #9)            #
# --------------------------------------------------------------------- #


class TestNonDegenerateAutoDeclare:
    def test_auto_declares_nondegenerate(self, omega, f, reg):
        assert not reg.has(omega, NonDegenerate)
        SymplecticProblem(omega, (f,), registry=reg)
        assert reg.has(omega, NonDegenerate)

    def test_does_not_overwrite_existing_nondegenerate(self, omega, f, reg):
        prior = NonDegenerate()
        reg.declare(omega, prior)
        SymplecticProblem(omega, (f,), registry=reg)
        assert reg.get(omega, NonDegenerate) is prior

    def test_engine_carries_nondegenerate_rule(self, omega, f, reg):
        sp = SymplecticProblem(omega, (f,), registry=reg)
        kinds = [type(d) for d in sp.engine.definitions]
        assert NonDegenerateInteriorEqualityDefinition in kinds


# --------------------------------------------------------------------- #
# prove_vector_field_equality                                           #
# --------------------------------------------------------------------- #


class TestProveVectorFieldEquality:
    def test_closes_when_y_equals_z(self, omega, f, reg):
        from jacopy.algebra.derivation import Derivation

        sp = SymplecticProblem(omega, (f,), registry=reg)
        Y = Derivation("Y", degree=0)
        chain = sp.prove_vector_field_equality(Y, Y)
        assert isinstance(chain, ProofChain)
        assert chain.final == Integer(0)

    def test_closes_for_registered_hamiltonian(self, omega, f, reg):
        sp = SymplecticProblem(omega, (f,), registry=reg)
        Xf = sp.hamiltonian(f)
        chain = sp.prove_vector_field_equality(Xf, Xf)
        assert chain.final == Integer(0)

    def test_fails_for_genuinely_distinct_vfs(self, omega, f, reg):
        """ι_Y ω - ι_Z ω peels to Y - Z; if those don't simplify the
        chain raises ProofFailure. The rule does what it should, it
        encodes injectivity, not magical agreement of distinct VFs."""
        from jacopy.algebra.derivation import Derivation
        from jacopy.proof.strategies import ProofFailure

        sp = SymplecticProblem(omega, (f,), registry=reg)
        Y = Derivation("Y", degree=0)
        Z = Derivation("Z", degree=0)
        with pytest.raises(ProofFailure):
            sp.prove_vector_field_equality(Y, Z)

    def test_requires_nondegenerate_flag(self, omega, f, reg):
        sp = SymplecticProblem(omega, (f,), registry=reg)
        # Strip the auto-declared flag.
        sp.registry.retract(omega, NonDegenerate)
        from jacopy.algebra.derivation import Derivation

        Y = Derivation("Y", degree=0)
        with pytest.raises(ValueError, match="NonDegenerate"):
            sp.prove_vector_field_equality(Y, Y)

    def test_rejects_non_expr_Y(self, omega, f, reg):
        sp = SymplecticProblem(omega, (f,), registry=reg)
        from jacopy.algebra.derivation import Derivation

        with pytest.raises(TypeError, match="Y"):
            sp.prove_vector_field_equality("Y", Derivation("Z", degree=0))  # type: ignore[arg-type]

    def test_rejects_non_expr_Z(self, omega, f, reg):
        sp = SymplecticProblem(omega, (f,), registry=reg)
        from jacopy.algebra.derivation import Derivation

        with pytest.raises(TypeError, match="Z"):
            sp.prove_vector_field_equality(Derivation("Y", degree=0), "Z")  # type: ignore[arg-type]
