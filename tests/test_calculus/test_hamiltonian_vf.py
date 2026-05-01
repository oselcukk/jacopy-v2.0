"""Tests for the Hamiltonian vector field ``X_f``."""

import pytest

from jacopy.algebra.derivation import Act, Derivation
from jacopy.brackets.base import BracketApply
from jacopy.brackets.derived import VanishingCondition
from jacopy.brackets.schouten import sn
from jacopy.calculus.exterior_d import ExteriorDerivative, d as default_d
from jacopy.calculus.hamiltonian_vf import (
    HamiltonianDefiningRelationDefinition,
    HamiltonianVectorField,
    HamiltonianVfDerivedDefinition,
    equivalence_condition,
    hamiltonian_vf,
    register_hamiltonian_defining_relation,
)
from jacopy.calculus.interior import interior
from jacopy.proof.expansion import ExpansionEngine
from jacopy.calculus.interior import InteriorProduct
from jacopy.calculus.musical import MusicalCompatibility
from jacopy.core.expr import Expr, Integer, Neg, Sum, Symbol
from jacopy.core.properties import Graded
from jacopy.core.registry import PropertyRegistry
from jacopy.core.symbolic_degree import Degree
from jacopy.proof.chain import ProofChain


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
        """The equivalence bridge reuses the symplectic obstruction,
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
    rules, iota→flat, X_f → −π♯(df), Neg linearity, and compatibility
   , and land on an :class:`Integer` ``0`` residual under simplify.
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


# --------------------------------------------------------------------- #
# Sign convention flag (Faz 12.C(b))                                     #
# --------------------------------------------------------------------- #


class TestSignConvention:
    def test_default_sign_minus(self):
        Xf = hamiltonian_vf(Symbol("f"), symplectic_form=Symbol("ω"))
        assert Xf.sign == "-"

    def test_explicit_plus(self):
        Xf = hamiltonian_vf(
            Symbol("f"), symplectic_form=Symbol("ω"), sign="+"
        )
        assert Xf.sign == "+"

    def test_invalid_sign_rejected(self):
        with pytest.raises(ValueError, match="sign"):
            hamiltonian_vf(
                Symbol("f"), symplectic_form=Symbol("ω"), sign="?"
            )

    def test_obstruction_minus_keeps_plus_df(self):
        f, omega = Symbol("f"), Symbol("ω")
        Xf = hamiltonian_vf(f, symplectic_form=omega, sign="-")
        out = Xf.symplectic_obstruction()
        assert isinstance(out, Sum)
        right = out.children[1]
        assert isinstance(right, Act)
        assert isinstance(right.op, ExteriorDerivative)
        assert right.arg is f

    def test_obstruction_plus_uses_neg_df(self):
        f, omega = Symbol("f"), Symbol("ω")
        Xf = hamiltonian_vf(f, symplectic_form=omega, sign="+")
        out = Xf.symplectic_obstruction()
        assert isinstance(out, Sum)
        right = out.children[1]
        assert isinstance(right, Neg)
        inner = right.arg
        assert isinstance(inner, Act)
        assert isinstance(inner.op, ExteriorDerivative)
        assert inner.arg is f

    def test_derived_rule_minus_emits_neg(self):
        f, pi, omega = Symbol("f"), Symbol("π"), Symbol("ω")
        Xf = hamiltonian_vf(
            f, bivector=pi, symplectic_form=omega, sign="-"
        )
        compat = MusicalCompatibility.between(omega, pi)
        rule = HamiltonianVfDerivedDefinition(Xf, compat)
        out = rule.rewrite(Xf)
        assert isinstance(out, Neg)

    def test_derived_rule_plus_omits_neg(self):
        f, pi, omega = Symbol("f"), Symbol("π"), Symbol("ω")
        Xf = hamiltonian_vf(
            f, bivector=pi, symplectic_form=omega, sign="+"
        )
        compat = MusicalCompatibility.between(omega, pi)
        rule = HamiltonianVfDerivedDefinition(Xf, compat)
        out = rule.rewrite(Xf)
        assert not isinstance(out, Neg)
        assert isinstance(out, Act)


# --------------------------------------------------------------------- #
# HamiltonianDefiningRelationDefinition + register helper (Faz 12.C(c)) #
# --------------------------------------------------------------------- #


class TestHamiltonianDefiningRelation:
    def test_match_default_minus_sign(self):
        X = Derivation("X_f", degree=0)
        f, omega = Symbol("f"), Symbol("ω")
        rule = HamiltonianDefiningRelationDefinition(X, f, omega)
        target = Act(interior(X), omega)
        assert rule.matches(target)

    def test_rewrite_default_minus_emits_neg_df(self):
        X = Derivation("X_f", degree=0)
        f, omega = Symbol("f"), Symbol("ω")
        rule = HamiltonianDefiningRelationDefinition(X, f, omega)
        out = rule.rewrite(Act(interior(X), omega))
        assert isinstance(out, Neg)
        inner = out.arg
        assert isinstance(inner, Act)
        assert isinstance(inner.op, ExteriorDerivative)
        assert inner.arg is f

    def test_rewrite_plus_emits_bare_df(self):
        X = Derivation("X_f", degree=0)
        f, omega = Symbol("f"), Symbol("ω")
        rule = HamiltonianDefiningRelationDefinition(
            X, f, omega, sign="+"
        )
        out = rule.rewrite(Act(interior(X), omega))
        assert isinstance(out, Act)
        assert isinstance(out.op, ExteriorDerivative)
        assert out.arg is f

    def test_match_skips_other_vf(self):
        X = Derivation("X_f", degree=0)
        Y = Derivation("X_g", degree=0)
        f, omega = Symbol("f"), Symbol("ω")
        rule = HamiltonianDefiningRelationDefinition(X, f, omega)
        assert not rule.matches(Act(interior(Y), omega))

    def test_match_skips_other_form(self):
        X = Derivation("X_f", degree=0)
        f, omega, eta = Symbol("f"), Symbol("ω"), Symbol("η")
        rule = HamiltonianDefiningRelationDefinition(X, f, omega)
        assert not rule.matches(Act(interior(X), eta))

    def test_match_skips_non_act(self):
        X = Derivation("X_f", degree=0)
        f, omega = Symbol("f"), Symbol("ω")
        rule = HamiltonianDefiningRelationDefinition(X, f, omega)
        assert not rule.matches(omega)

    def test_match_skips_non_iota_op(self):
        X = Derivation("X_f", degree=0)
        f, omega = Symbol("f"), Symbol("ω")
        rule = HamiltonianDefiningRelationDefinition(X, f, omega)
        # outer is d, not iota
        assert not rule.matches(Act(default_d, omega))

    def test_invalid_sign_rejected(self):
        X = Derivation("X_f", degree=0)
        f, omega = Symbol("f"), Symbol("ω")
        with pytest.raises(ValueError, match="sign"):
            HamiltonianDefiningRelationDefinition(
                X, f, omega, sign="?"
            )

    def test_non_expr_inputs_rejected(self):
        f, omega = Symbol("f"), Symbol("ω")
        X = Derivation("X_f", degree=0)
        with pytest.raises(TypeError, match="X"):
            HamiltonianDefiningRelationDefinition("X", f, omega)  # type: ignore[arg-type]
        with pytest.raises(TypeError, match="f"):
            HamiltonianDefiningRelationDefinition(X, "f", omega)  # type: ignore[arg-type]
        with pytest.raises(TypeError, match="omega"):
            HamiltonianDefiningRelationDefinition(X, f, "ω")  # type: ignore[arg-type]

    def test_register_helper_appends_rule(self):
        X = Derivation("X_f", degree=0)
        f, omega = Symbol("f"), Symbol("ω")
        engine = ExpansionEngine([])
        rule = register_hamiltonian_defining_relation(X, f, omega, engine)
        assert rule in engine.definitions
        assert isinstance(rule, HamiltonianDefiningRelationDefinition)

    def test_register_helper_threads_sign(self):
        X = Derivation("X_f", degree=0)
        f, omega = Symbol("f"), Symbol("ω")
        engine = ExpansionEngine([])
        rule = register_hamiltonian_defining_relation(
            X, f, omega, engine, sign="+"
        )
        out = rule.rewrite(Act(interior(X), omega))
        assert not isinstance(out, Neg)

    def test_engine_fires_via_default_engine(self):
        """End-to-end: notebook 2a-style proof that L_X ω = 0."""
        from jacopy.calculus.closed_axioms import ClosedFormDefinition
        from jacopy.calculus.lie_derivative import lie_derivative
        from jacopy.core.properties import Closed
        from jacopy.proof.expansion import default_engine
        from jacopy.proof.strategies import ExpandAndSimplify

        reg = PropertyRegistry()
        f = Symbol("f")
        omega = Symbol("ω")
        X = Derivation("X_f", degree=0)
        reg.declare(f, Graded(degree=0))
        reg.declare(omega, Graded(degree=2))
        reg.declare(omega, Closed())
        engine = default_engine(registry=reg, d_squared_mode="axiom")
        engine.register(ClosedFormDefinition(registry=reg))
        register_hamiltonian_defining_relation(
            X, f, omega, engine, sign="+"
        )
        L_X = lie_derivative(X)
        chain = ExpandAndSimplify().prove(
            Act(L_X, omega), Integer(0), registry=reg, engine=engine
        )
        assert chain.final == Integer(0)
        assert len(chain) > 0
