"""Tests for ``gradalg.library.poisson``."""

from __future__ import annotations

import pytest

from gradalg.algebra.derivation import Act
from gradalg.brackets.base import BracketApply
from gradalg.brackets.derived import DerivedBracket, VanishingCondition
from gradalg.brackets.schouten import sn
from gradalg.calculus.hamiltonian_vf import HamiltonianVectorField
from gradalg.core.expr import Symbol
from gradalg.core.properties import Graded
from gradalg.core.registry import PropertyRegistry
from gradalg.core.symbolic_degree import Degree
from gradalg.library import theorem_book
from gradalg.library.poisson import (
    THEOREM_POISSON_JACOBI,
    PoissonBracket,
    poisson_bracket,
)
from gradalg.proof.chain import ProofChain


# --------------------------------------------------------------------- #
# Fixtures                                                               #
# --------------------------------------------------------------------- #


@pytest.fixture
def registry():
    r = PropertyRegistry()
    pi = Symbol("π")
    f = Symbol("f")
    g = Symbol("g")
    h = Symbol("h")
    r.declare(pi, Graded(degree=1))
    r.declare(f, Graded(degree=-1))
    r.declare(g, Graded(degree=-1))
    r.declare(h, Graded(degree=-1))
    return r


# --------------------------------------------------------------------- #
# Construction                                                           #
# --------------------------------------------------------------------- #


class TestConstruction:
    def test_basic(self):
        pi = Symbol("π")
        P = PoissonBracket(pi)
        assert P.bivector is pi
        assert isinstance(P.derived, DerivedBracket)
        assert P.derived.base is sn
        assert P.derived.Q is pi

    def test_from_bivector_classmethod(self):
        pi = Symbol("π")
        P = PoissonBracket.from_bivector(pi)
        assert P.bivector is pi

    def test_factory(self):
        pi = Symbol("π")
        P = poisson_bracket(pi)
        assert isinstance(P, PoissonBracket)
        assert P.bivector is pi

    def test_default_name(self):
        pi = Symbol("π")
        P = PoissonBracket(pi)
        assert "π" in P.name

    def test_custom_name(self):
        pi = Symbol("π")
        P = PoissonBracket(pi, name="{·,·}_myPoisson")
        assert P.name == "{·,·}_myPoisson"

    def test_rejects_non_expr(self):
        with pytest.raises(TypeError, match="Expr"):
            PoissonBracket("π")  # type: ignore[arg-type]

    def test_derived_bracket_degree_is_minus_one(self):
        """``|{·,·}_π| = |π| − 2 = 1 − 2 = −1`` in the SN grading."""
        pi = Symbol("π")
        P = PoissonBracket(pi)
        assert P.derived.degree == Degree.const(-1)


# --------------------------------------------------------------------- #
# Three equivalent views                                                #
# --------------------------------------------------------------------- #


class TestViews:
    def test_expand_delegates_to_derived(self, registry):
        pi = Symbol("π")
        f, g = Symbol("f"), Symbol("g")
        P = PoissonBracket(pi)
        direct = P.expand(f, g, registry)
        via_db = P.derived.expand(f, g, registry)
        assert direct == via_db

    def test_via_hamiltonian_shape(self):
        """``{f, g}_π = X_f(g)`` — an :class:`Act` of ``X_f`` on ``g``."""
        pi = Symbol("π")
        f, g = Symbol("f"), Symbol("g")
        P = PoissonBracket(pi)
        out = P.via_hamiltonian(f, g)
        assert isinstance(out, Act)
        assert isinstance(out.op, HamiltonianVectorField)
        assert out.op.function is f
        assert out.op.bivector is pi
        assert out.arg is g

    def test_hamiltonian_vf(self):
        pi = Symbol("π")
        f = Symbol("f")
        P = PoissonBracket(pi)
        Xf = P.hamiltonian_vf(f)
        assert isinstance(Xf, HamiltonianVectorField)
        assert Xf.function is f
        assert Xf.bivector is pi

    def test_via_hamiltonian_rejects_non_expr(self):
        P = PoissonBracket(Symbol("π"))
        with pytest.raises(TypeError, match="Expr"):
            P.via_hamiltonian("f", Symbol("g"))  # type: ignore[arg-type]


# --------------------------------------------------------------------- #
# Jacobi                                                                 #
# --------------------------------------------------------------------- #


class TestJacobi:
    def test_jacobi_obstruction_matches_sn_self_bracket(self, registry):
        """The Poisson Jacobi obstruction is ``[π, π]_SN`` — for atomic
        ``π`` it stays opaque."""
        pi = Symbol("π")
        P = PoissonBracket(pi)
        assert P.jacobi_obstruction(registry) == sn.self_bracket(pi, registry)

    def test_jacobi_condition_is_vanishing_condition(self, registry):
        pi = Symbol("π")
        P = PoissonBracket(pi)
        cond = P.jacobi_condition(registry)
        assert isinstance(cond, VanishingCondition)
        assert cond.obstruction == sn.self_bracket(pi, registry)
        assert "Poisson" in cond.name

    def test_prove_jacobi_reduction_produces_theorem_step(self, registry):
        """The reduction chain's first step cites the Derived Bracket
        Theorem and rewrites the triple Jacobi sum to the raw
        obstruction ``[π, π]_SN``."""
        pi = Symbol("π")
        f, g, h = Symbol("f"), Symbol("g"), Symbol("h")
        P = PoissonBracket(pi)
        chain = P.prove_jacobi_reduction(f, g, h, registry=registry)
        assert isinstance(chain, ProofChain)
        assert len(chain) >= 1
        first = chain.steps[0]
        assert first.rule == "DerivedBracketTheorem"
        assert first.provenance_tag == "theorem"
        assert isinstance(first.after, BracketApply)
        assert first.after.bracket is sn

    def test_prove_jacobi_reduction_final_is_obstruction(self, registry):
        """Chain ends on ``[π, π]_SN``, not on zero — discharging the
        obstruction is the caller's job (the Poisson hypothesis)."""
        pi = Symbol("π")
        f, g, h = Symbol("f"), Symbol("g"), Symbol("h")
        P = PoissonBracket(pi)
        chain = P.prove_jacobi_reduction(f, g, h, registry=registry)
        assert chain.final == P.jacobi_obstruction(registry)


# --------------------------------------------------------------------- #
# Seeded theorem                                                         #
# --------------------------------------------------------------------- #


class TestSeededTheorem:
    def test_theorem_registered(self):
        """Importing :mod:`gradalg.library.poisson` seeds
        ``poisson_jacobi`` into the package-wide theorem book."""
        assert "poisson_jacobi" in theorem_book
        assert theorem_book.get("poisson_jacobi") is THEOREM_POISSON_JACOBI

    def test_theorem_from_axioms(self):
        thm = THEOREM_POISSON_JACOBI
        assert thm.name == "poisson_jacobi"
        assert "Derived Bracket Theorem" in thm.from_axioms
        assert any("[π, π]_SN" in ax for ax in thm.from_axioms)

    def test_theorem_proof_is_proofchain(self):
        thm = THEOREM_POISSON_JACOBI
        assert isinstance(thm.proof, ProofChain)
        assert len(thm.proof) >= 1
        assert thm.proof.steps[0].rule == "DerivedBracketTheorem"
