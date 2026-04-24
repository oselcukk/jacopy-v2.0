"""Tests for the Hamiltonian vector field ``X_f``."""

import pytest

from gradalg.algebra.derivation import Act, Derivation
from gradalg.brackets.base import BracketApply
from gradalg.brackets.derived import VanishingCondition
from gradalg.brackets.schouten import sn
from gradalg.calculus.exterior_d import ExteriorDerivative, d as default_d
from gradalg.calculus.hamiltonian_vf import (
    HamiltonianVectorField,
    HamiltonianVfDerivedDefinition,
    equivalence_condition,
    hamiltonian_vf,
)
from gradalg.calculus.interior import InteriorProduct
from gradalg.calculus.musical import MusicalCompatibility
from gradalg.core.expr import Expr, Integer, Neg, Sum, Symbol
from gradalg.core.properties import Graded
from gradalg.core.registry import PropertyRegistry
from gradalg.core.symbolic_degree import Degree
from gradalg.proof.chain import ProofChain


# --------------------------------------------------------------------- #
# Fixtures                                                               #
# --------------------------------------------------------------------- #


@pytest.fixture
def reg():
    r = PropertyRegistry()
    f = Symbol("f")
    pi = Symbol("π")
    omega = Symbol("ω")
    r.declare(f, Graded(degree=-1))  # SN function
    r.declare(pi, Graded(degree=1))  # SN bivector
    r.declare(omega, Graded(degree=2))  # 2-form
    return r


# --------------------------------------------------------------------- #
# Construction and validation                                            #
# --------------------------------------------------------------------- #


class TestConstruction:
    def test_with_bivector_only(self):
        f = Symbol("f")
        pi = Symbol("π")
        Xf = hamiltonian_vf(f, bivector=pi)
        assert isinstance(Xf, HamiltonianVectorField)
        assert isinstance(Xf, Derivation)
        assert Xf.function is f
        assert Xf.bivector is pi
        assert Xf.symplectic_form is None

    def test_with_symplectic_form_only(self):
        f = Symbol("f")
        omega = Symbol("ω")
        Xf = hamiltonian_vf(f, symplectic_form=omega)
        assert Xf.symplectic_form is omega
        assert Xf.bivector is None

    def test_with_both(self):
        f = Symbol("f")
        pi = Symbol("π")
        omega = Symbol("ω")
        Xf = hamiltonian_vf(f, bivector=pi, symplectic_form=omega)
        assert Xf.bivector is pi
        assert Xf.symplectic_form is omega

    def test_degree_is_zero(self):
        Xf = hamiltonian_vf(Symbol("f"), bivector=Symbol("π"))
        assert Xf.degree == Degree.const(0)

    def test_default_name(self):
        Xf = hamiltonian_vf(Symbol("f"), bivector=Symbol("π"))
        assert Xf.name == "X_f"

    def test_custom_name(self):
        Xf = hamiltonian_vf(
            Symbol("f"), bivector=Symbol("π"), name="Ham_f"
        )
        assert Xf.name == "Ham_f"

    def test_requires_function_expr(self):
        with pytest.raises(TypeError, match="must be an Expr"):
            HamiltonianVectorField("f", bivector=Symbol("π"))  # type: ignore[arg-type]

    def test_requires_bivector_or_form(self):
        with pytest.raises(ValueError, match="at least one"):
            HamiltonianVectorField(Symbol("f"))

    def test_bivector_must_be_expr(self):
        with pytest.raises(TypeError, match="bivector"):
            HamiltonianVectorField(Symbol("f"), bivector="π")  # type: ignore[arg-type]

    def test_symplectic_form_must_be_expr(self):
        with pytest.raises(TypeError, match="symplectic_form"):
            HamiltonianVectorField(Symbol("f"), symplectic_form="ω")  # type: ignore[arg-type]


# --------------------------------------------------------------------- #
# Derived expansion                                                      #
# --------------------------------------------------------------------- #


class TestDerivedExpansion:
    def test_atomic_bivector_returns_neg_opaque(self, reg):
        """``X_f = −[f, π]_SN``; with atomic π the inner bracket stays
        as an opaque :class:`BracketApply` and the outer ``Neg`` is the
        shape the proof layer consumes."""
        f = Symbol("f")
        pi = Symbol("π")
        Xf = hamiltonian_vf(f, bivector=pi)
        out = Xf.derived_expansion(reg)
        assert isinstance(out, Neg)
        inner = out.arg
        assert isinstance(inner, BracketApply)
        assert inner.bracket is sn
        assert inner.a is f and inner.b is pi

    def test_matches_manual_sn_expansion(self, reg):
        """``X_f.derived_expansion()`` ≡ ``−sn.expand(f, π, registry)``."""
        f = Symbol("f")
        pi = Symbol("π")
        Xf = hamiltonian_vf(f, bivector=pi)
        assert Xf.derived_expansion(reg) == Neg(sn.expand(f, pi, reg))

    def test_raises_without_bivector(self):
        f = Symbol("f")
        omega = Symbol("ω")
        Xf = hamiltonian_vf(f, symplectic_form=omega)
        with pytest.raises(ValueError, match="bivector"):
            Xf.derived_expansion()


# --------------------------------------------------------------------- #
# Symplectic obstruction                                                 #
# --------------------------------------------------------------------- #


class TestSymplecticObstruction:
    def test_shape_is_sum_of_iota_and_df(self, reg):
        """``obstruction = ι_{X_f}(ω) + d(f)``."""
        f = Symbol("f")
        omega = Symbol("ω")
        Xf = hamiltonian_vf(f, symplectic_form=omega)
        out = Xf.symplectic_obstruction()
        assert isinstance(out, Sum)
        left, right = out.children
        assert isinstance(left, Act)
        assert isinstance(left.op, InteriorProduct)
        assert left.op.vector_field is Xf
        assert left.arg is omega
        assert isinstance(right, Act)
        assert isinstance(right.op, ExteriorDerivative)
        assert right.arg is f

    def test_defaults_to_singleton_d(self):
        f = Symbol("f")
        omega = Symbol("ω")
        Xf = hamiltonian_vf(f, symplectic_form=omega)
        out = Xf.symplectic_obstruction()
        right = out.children[1]
        assert right.op is default_d

    def test_custom_d_is_honoured(self):
        f = Symbol("f")
        omega = Symbol("ω")
        Xf = hamiltonian_vf(f, symplectic_form=omega)
        d_custom = ExteriorDerivative(name="d_E")
        out = Xf.symplectic_obstruction(d=d_custom)
        assert out.children[1].op is d_custom

    def test_custom_interior_factory(self):
        f = Symbol("f")
        omega = Symbol("ω")
        Xf = hamiltonian_vf(f, symplectic_form=omega)
        sentinel = Derivation("ι_custom", degree=-1)

        def factory(X: Expr) -> Derivation:
            assert X is Xf
            return sentinel

        out = Xf.symplectic_obstruction(interior=factory)
        assert out.children[0].op is sentinel

    def test_raises_without_symplectic_form(self):
        f = Symbol("f")
        pi = Symbol("π")
        Xf = hamiltonian_vf(f, bivector=pi)
        with pytest.raises(ValueError, match="symplectic_form"):
            Xf.symplectic_obstruction()


# --------------------------------------------------------------------- #
# Symplectic condition (VanishingCondition wrapper)                      #
# --------------------------------------------------------------------- #


class TestSymplecticCondition:
    def test_returns_vanishing_condition(self):
        f = Symbol("f")
        omega = Symbol("ω")
        Xf = hamiltonian_vf(f, symplectic_form=omega)
        cond = Xf.symplectic_condition()
        assert isinstance(cond, VanishingCondition)
        assert cond.obstruction == Xf.symplectic_obstruction()
        assert "symplectic definition" in cond.name
        assert Xf.name in cond.name


# --------------------------------------------------------------------- #
# Equivalence bridge                                                     #
# --------------------------------------------------------------------- #


class TestEquivalenceCondition:
    def test_returns_vanishing_condition_with_named_obstruction(self):
        f = Symbol("f")
        pi = Symbol("π")
        omega = Symbol("ω")
        cond = equivalence_condition(f, bivector=pi, symplectic_form=omega)
        assert isinstance(cond, VanishingCondition)
        assert "equivalence" in cond.name
        assert "X_f" in cond.name

    def test_obstruction_matches_symplectic_obstruction(self):
        """The equivalence bridge reuses the symplectic obstruction —
        vanishing there is exactly the musical-isomorphism condition."""
        f = Symbol("f")
        pi = Symbol("π")
        omega = Symbol("ω")
        cond = equivalence_condition(f, bivector=pi, symplectic_form=omega)
        Xf = hamiltonian_vf(f, bivector=pi, symplectic_form=omega)
        assert cond.obstruction == Xf.symplectic_obstruction()

    def test_custom_d_propagates(self):
        f = Symbol("f")
        pi = Symbol("π")
        omega = Symbol("ω")
        d_custom = ExteriorDerivative(name="d_E")
        cond = equivalence_condition(
            f, bivector=pi, symplectic_form=omega, d=d_custom
        )
        # The df term in the obstruction must use d_custom, not the default.
        right = cond.obstruction.children[1]
        assert right.op is d_custom


# --------------------------------------------------------------------- #
# prove_equivalence via musical compatibility                            #
# --------------------------------------------------------------------- #


class TestProveEquivalence:
    """Musical bridge closes the symplectic obstruction to ``0`` when a
    :class:`MusicalCompatibility` axiom declares ``ω`` and ``π`` as
    mutual musical inverses. The chain must fire the four registered
    rules — iota→flat, X_f → −π♯(df), Neg linearity, and compatibility
    — and land on an :class:`Integer` ``0`` residual under simplify.
    """

    @pytest.fixture
    def registry_and_symbols(self):
        r = PropertyRegistry()
        f = Symbol("f")
        pi = Symbol("π")
        omega = Symbol("ω")
        r.declare(f, Graded(degree=-1))
        r.declare(pi, Graded(degree=1))
        r.declare(omega, Graded(degree=2))
        return r, f, pi, omega

    def test_closes_to_proof_chain(self, registry_and_symbols):
        r, f, pi, omega = registry_and_symbols
        Xf = hamiltonian_vf(f, bivector=pi, symplectic_form=omega)
        compat = MusicalCompatibility.between(omega, pi)
        chain = Xf.prove_equivalence(compat, registry=r)
        assert isinstance(chain, ProofChain)
        # The chain's final step must land on Integer 0.
        assert chain.steps[-1].after == Integer(0)

    def test_fires_all_four_rules(self, registry_and_symbols):
        r, f, pi, omega = registry_and_symbols
        Xf = hamiltonian_vf(f, bivector=pi, symplectic_form=omega)
        compat = MusicalCompatibility.between(omega, pi)
        chain = Xf.prove_equivalence(compat, registry=r)
        rules = [s.rule for s in chain.steps]
        # Every musical rule must have fired at least once before simplify.
        assert any("ω♭" in r or "ω♭" in r for r in rules), rules
        assert any("π♯" in r or "π♯" in r for r in rules), rules
        assert any("-x" in r or "−x" in r for r in rules), rules
        assert any("musical compatibility" in r for r in rules), rules

    def test_rejects_mismatched_bivector(self, registry_and_symbols):
        r, f, _, omega = registry_and_symbols
        pi_used = Symbol("π_A")
        pi_axiom = Symbol("π_B")
        r.declare(pi_used, Graded(degree=1))
        r.declare(pi_axiom, Graded(degree=1))
        Xf = hamiltonian_vf(f, bivector=pi_used, symplectic_form=omega)
        compat = MusicalCompatibility.between(omega, pi_axiom)
        with pytest.raises(ValueError, match="bivector"):
            Xf.prove_equivalence(compat, registry=r)

    def test_rejects_mismatched_symplectic_form(self, registry_and_symbols):
        r, f, pi, _ = registry_and_symbols
        omega_used = Symbol("ω_A")
        omega_axiom = Symbol("ω_B")
        r.declare(omega_used, Graded(degree=2))
        r.declare(omega_axiom, Graded(degree=2))
        Xf = hamiltonian_vf(f, bivector=pi, symplectic_form=omega_used)
        compat = MusicalCompatibility.between(omega_axiom, pi)
        with pytest.raises(ValueError, match="symplectic"):
            Xf.prove_equivalence(compat, registry=r)

    def test_requires_bivector(self, registry_and_symbols):
        r, f, pi, omega = registry_and_symbols
        Xf = hamiltonian_vf(f, symplectic_form=omega)  # no bivector
        compat = MusicalCompatibility.between(omega, pi)
        with pytest.raises(ValueError, match="bivector"):
            Xf.prove_equivalence(compat, registry=r)

    def test_requires_symplectic_form(self, registry_and_symbols):
        r, f, pi, omega = registry_and_symbols
        Xf = hamiltonian_vf(f, bivector=pi)  # no symplectic_form
        compat = MusicalCompatibility.between(omega, pi)
        with pytest.raises(ValueError, match="symplectic_form"):
            Xf.prove_equivalence(compat, registry=r)


class TestHamiltonianVfDerivedDefinition:
    def test_rewrites_xf_to_neg_sharp_df(self):
        f = Symbol("f")
        pi = Symbol("π")
        omega = Symbol("ω")
        Xf = hamiltonian_vf(f, bivector=pi, symplectic_form=omega)
        compat = MusicalCompatibility.between(omega, pi)
        rule = HamiltonianVfDerivedDefinition(Xf, compat)
        assert rule.matches(Xf)
        out = rule.rewrite(Xf)
        # Expected shape: -Sharp(π)(d(f)).
        assert isinstance(out, Neg)
        inner = out.arg
        assert isinstance(inner, Act)
        assert inner.op is compat.sharp
        inner2 = inner.arg
        assert isinstance(inner2, Act)
        assert inner2.arg is f

    def test_ignores_other_hamiltonian(self):
        f1, f2 = Symbol("f"), Symbol("g")
        pi, omega = Symbol("π"), Symbol("ω")
        Xf = hamiltonian_vf(f1, bivector=pi, symplectic_form=omega)
        Xg = hamiltonian_vf(f2, bivector=pi, symplectic_form=omega)
        compat = MusicalCompatibility.between(omega, pi)
        rule = HamiltonianVfDerivedDefinition(Xf, compat)
        assert not rule.matches(Xg)
